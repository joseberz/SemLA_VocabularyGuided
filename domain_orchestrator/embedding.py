from enum import Enum
from typing import Any

from sklearn.cluster import AgglomerativeClustering

from domain_orchestrator.hf_clip.modeling_clip import CLIPModel
from domain_orchestrator.hf_clip.processing_clip import CLIPProcessor
from transformers import CLIPTokenizer
from abc import abstractmethod
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
import open_clip
from domain_orchestrator.open_clip.model import VisionTransformer

# Erklärung im Paper warum Templates https://arxiv.org/abs/2103.00020
templates = [
    "a photo of a {}.",
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a photo of the {}.",
    "an image of a {}.",
]

class VocabEmbeddingMethod(Enum):
    NONE = "none"
    GLOBAL = "global"
    PATCH = "patch"
    OBJECTDETECTION = "objectdetection"

def filter_patches_by_area(patch_embeddings, n_clusters=16, min_area_pct=0.05, use_centroids=True):
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='cosine',
        linkage='average'
    )
    labels = clustering.fit_predict(patch_embeddings)

    filtered = []
    for c in range(n_clusters):
        mask = labels == c
        area_pct = mask.sum() / len(labels)
        if area_pct >= min_area_pct:
            if use_centroids:
                centroid = patch_embeddings[mask].mean(axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                filtered.append(centroid[np.newaxis, :])
            else:
                filtered.append(patch_embeddings[mask])

    if len(filtered) == 0:
        # Fallback: größten Cluster nehmen
        largest = np.argmax([np.sum(labels == c) for c in range(n_clusters)])
        mask = labels == largest
        if use_centroids:
            return patch_embeddings[mask].mean(axis=0, keepdims=True)
        return patch_embeddings[mask]

    return np.concatenate(filtered)

#abstract class
class EmbeddingModel:
    """Abstract class for embedding models."""
    @abstractmethod
    def embed_image(self, image_path, vocab_embedding_method=None):
        """Embed a single image."""
        pass

    @abstractmethod
    def embed_text(self, text):
        """Embed a single image."""
        pass


class ClipEmbeddingModel(EmbeddingModel):
    """Handles image and dataset embedding operations."""

    def __init__(self):
        self.embedding_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to("cuda")
        self.embedding_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

        # --- Hook: Value-Features der letzten ViT-Layer abgreifen ---
        last_attn = self.embedding_model.vision_model.encoder.layers[-1].self_attn
        self._value_store = {}

        def _hook(module, args, kwargs, output):
            hidden = kwargs['hidden_states']       # ← statt args[0]
            v = module.v_proj(hidden)
            v = last_attn.out_proj(v)
            self._value_store['patches'] = v[:, 1:]

        hook = last_attn.register_forward_hook(_hook, with_kwargs=True)

    def embed_image(self, image_path, vocab_embedding_method=None) -> tuple[Any, Any | None]:
        """Embed a single image."""
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file '{image_path}' not found.")
            raise
        except Exception as e:
            print(f"Error opening image '{image_path}': {e}")
            raise

        if self.embedding_processor is None or self.embedding_model is None:
            print("Error: CLIP model or processor is not initialized.")
            raise
        
        inputs = self.embedding_processor(images=image, return_tensors="pt").to("cuda")

        with torch.no_grad():
            vision_outputs = self.embedding_model.vision_model(**inputs)
            # Hole die Globalen Embeddings und die Patch Embeddings aus dem selben Forward Pass
            # Global
            cls_output = vision_outputs.pooler_output                          # (1, 1024)
            cls_output = self.embedding_model.visual_projection(cls_output)    # (1, 768)
            #cls_output = cls_output / cls_output.norm(dim=-1, keepdim=True)

            # Patch
            patch_features = self._value_store['patches'].squeeze(0)
            patch_features = self.embedding_model.visual_projection(patch_features)
            patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
            patch_embeddings = patch_features.detach().cpu().numpy()
            # TODO Irrelevante Patches herausfiltern ?
            # patch_embeddings = filter_patches_by_area(patch_embeddings, n_clusters=16, min_area_pct=0.05)

        return cls_output.squeeze(0).detach().cpu().numpy(), patch_embeddings

    def embed_text(self, text) -> npt.NDArray:
        """Embed a single text."""
        with torch.no_grad():
            texts = [t.format(text) for t in templates]
            inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to("cuda")
            text_embeddings = self.embedding_model.get_text_features(**inputs)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings = text_embeddings.mean(dim=0)
            text_embeddings = text_embeddings / text_embeddings.norm()
        return text_embeddings.detach().cpu().numpy()

    def embed_texts_batched(self, texts: list[str]) -> npt.NDArray:
        """Embed alle Texte in einem einzigen Forward Pass."""
        all_templated = [t.format(text) for text in texts for t in templates]

        with torch.no_grad():
            inputs = self.tokenizer(all_templated, padding=True, return_tensors="pt").to("cuda")
            embs = self.embedding_model.get_text_features(**inputs)
            embs = embs / embs.norm(dim=-1, keepdim=True)
        embs = embs.detach().cpu().numpy()
        # Pro Klasse die 5 Templates mitteln
        embs = embs.reshape(len(texts), len(templates), -1).mean(axis=1)
        embs = embs / np.linalg.norm(embs, axis=-1, keepdims=True)
        return embs

"""
NEU: Test von OpenClip als Embedding Modell!
"""
class OpenClipEmbeddingModel(EmbeddingModel):
    """Handles image and dataset embedding operations."""

    def __init__(self):
        self.pretrained_model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-quickgelu', pretrained='openai')
        # Visual in separates CUDA-Modell laden
        state_dict = self.pretrained_model.visual.state_dict()
        self.embedding_model = VisionTransformer(
            image_size=224,
            patch_size=14,
            width=1024,
            layers=24,
            heads=16,
            mlp_ratio=4.0,
            output_dim=768,
        ).to("cpu")
        self.embedding_model.load_state_dict(state_dict)
        del state_dict

        # Visual aus pretrained_model entfernen, wird nicht mehr gebraucht
        self.pretrained_model.visual = None
        self.pretrained_model = self.pretrained_model.to("cpu")

        self.embedding_processor = preprocess
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14-quickgelu")

    def embed_image(self, image_path) -> npt.NDArray:
        """Embed a single image."""
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file '{image_path}' not found.")
            raise
        except Exception as e:
            print(f"Error opening image '{image_path}': {e}")
            raise

        if self.embedding_processor is None or self.embedding_model is None:
            print("Error: CLIP model or processor is not initialized.")
            raise

        inputs = self.embedding_processor(image).unsqueeze(0).to("cpu")

        # Generate image embeddings
        with torch.no_grad():
            image_embeddings = (
                self.embedding_model(inputs).detach().cpu().numpy()
            )
        del inputs
        torch.cuda.empty_cache()
        return image_embeddings

    def embed_text(self, text) -> npt.NDArray:
        """Embed a single text."""
        with torch.no_grad():
            texts = [t.format(text) for t in templates]
            inputs = self.tokenizer(texts).to("cpu")
            text_embeddings = self.pretrained_model.encode_text(inputs)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings = text_embeddings.mean(dim=0)
            text_embeddings = text_embeddings / text_embeddings.norm()
        return text_embeddings


class EmbeddingManager:
    """Handles image and dataset embedding operations."""

    def __init__(self, embedding_model: EmbeddingModel = None):
        if embedding_model is None:
            embedding_model = ClipEmbeddingModel()
        self.embedding_model = embedding_model
    
    def embed_image(self, image_path, vocab_embedding_method=None) -> npt.NDArray:
        """Embed a single image."""
        return self.embedding_model.embed_image(image_path, vocab_embedding_method=vocab_embedding_method)

    """
    NEU: Text embeddings generieren
    """
    def embed_text(self, texts: list[str]) -> npt.NDArray:
        """Embed a text."""
        # TODO kann bei Python eine Methode überladen werden? Z.B. eine Methode die str annimmt und eine andere die Liste annimmt ?
        return self.embedding_model.embed_texts_batched(texts)

    """
    NEU: Vocabulary embeddings für Datensätze generieren und abspeichern
    """
    def calculate_vocabulary_embeddings(self, domain_name, domain_path, classnames):
        print(f"Embedding vocabulary for dataset '{domain_name}' ...")
        suffix = "_vocab_embeddings.npz"
        save_path = domain_path / f"{domain_name}{suffix}"

        if save_path.exists():
            print(f"Loading vocab embeddings from {save_path} ...")
            return np.load(save_path)["vocab_embeddings"]

        print(f"Computing vocab embeddings for {domain_name} ...")
        all_class_embeddings = []
        for classname in classnames:
            embeddings = self.embed_text(classname)
            all_class_embeddings.append(embeddings)
        all_class_embeddings = torch.stack(all_class_embeddings) # (N, dim)

        np.savez(save_path, vocab_embeddings=all_class_embeddings.cpu().numpy())
        torch.cuda.empty_cache()
        return all_class_embeddings

    def get_vocabulary_embeddings(self, domain_name, domain_path, classnames):
        print(f"Embedding vocabulary for dataset '{domain_name}' ...")
        suffix = "_vocab_embeddings.npz"
        save_path = domain_path / f"{domain_name}{suffix}"

        print(f"Loading vocab embeddings from {save_path} ...")
        return np.load(save_path)["vocab_embeddings"]
        
    def embed_dataset(self, dataset_path, debug=False) -> npt.NDArray:
        """Embed all images in a dataset."""
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path '{dataset_path}' not found.")

        print(f"Embedding dataset from '{dataset_path}' ...")
        dataset_embeddings = []

        # IDD has both png and jpg images in train set
        image_files = list(dataset_path.rglob("*.png")) + list(dataset_path.rglob("*.jpg"))
    
        if not image_files:
            print(f"Warning: No images found in dataset path '{dataset_path}'.")
            return []

        for img in image_files:
            embedding, _ = self.embed_image(img) # TODO testen, könnte Probleme machen
            if embedding is not None:
                dataset_embeddings.append(embedding)
            else:
                raise ValueError(f"Error embedding image '{img}'.")

        print("Finished embedding dataset.")
        torch.cuda.empty_cache()
        return dataset_embeddings
        
    def calculate_statistics(self, domain_name, domain_path, train_path):

        """
        Calculate or load domain statistics.
        Args:
            domain_name (str): The name of the domain.
            domain_path (Path): The path to the domain database where the statistics will be saved.
            train_path (Path): The path to the train set.
        Returns:
            dict: A dictionary containing the statistics.
        """
        suffix = "_statistics.npz"
        statistics_path = domain_path / f"{domain_name}{suffix}"
        stats_dict = {}

        print(f"Statistics file: {statistics_path}")
        if statistics_path.exists():  # Load the data if it exists
            try:
                print(f"Loading statistics from {domain_name}{suffix} ...")
                stats = np.load(statistics_path)
                stats_dict.update({
                    "train_average_embedding": stats["train_average_embedding"],
                })
                print(f"Statistics loaded from {domain_name}{suffix}")
                return stats_dict
            except Exception as e:
                print(f"Error loading statistics file '{statistics_path}': {e}")
                return None

        print(f"Statistics file {statistics_path} does not exist, calculating statistics for domain '{domain_name}' ...")
        train_dataset_embeddings = self.embed_dataset(train_path)

        if not train_dataset_embeddings:
            raise ValueError("No embeddings were generated for dataset.")

        try:
            train_average_embedding = np.mean(train_dataset_embeddings, axis=0)
        except Exception as e:
            print(f"Error computing mean embedding: {e}")
            raise

        stats_dict.update({
            "train_average_embedding": train_average_embedding
        })

        try:
            np.savez(
                statistics_path,
                train_average_embedding=train_average_embedding,
            )
            print(f"Statistics saved to {domain_name}{suffix}")
        except Exception as e:
            print(f"Error saving statistics file '{statistics_path}': {e}")
            raise
        return stats_dict
