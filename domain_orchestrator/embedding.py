# Geändert von Joshua Ritter, 2026, im Rahmen der Masterarbeit
# "Vokabulargeleitete Selektion von LoRA-Adaptern
# mittels CLIP für domänenadaptive Open-Vocabulary-Segmentierung"
# Ursprüngliche Datei: SemLA (Qorbani et al.), Apache-2.0-Lizenz

from enum import Enum
from typing import Any

from transformers import CLIPModel, CLIPProcessor
from transformers import CLIPTokenizer
from abc import abstractmethod
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

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
    NONE_NORMALIZED = "none_normalized"
    GLOBAL = "global"
    PATCH = "patch"
    OBJECTDETECTION = "objectdetection"

#abstract class
class EmbeddingModel:
    """Abstract class for embedding models."""
    @abstractmethod
    def embed_image(self, image_path):
        """Embed a single image."""
        pass

    @abstractmethod
    def embed_text(self, text):
        """Embed a single image."""
        pass


class ClipEmbeddingModel(EmbeddingModel):
    """
    TODO Doku
    Referenz: Zhou et al., "Extract Free Dense Labels from CLIP" (ECCV 2022)
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.embedding_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(self.device)
        self.embedding_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

        # --- Hook: Value-Features der letzten ViT-Layer abgreifen ---
        # TODO DOKU

        last_layer = self.embedding_model.vision_model.encoder.layers[-1]
        last_attn = last_layer.self_attn
        self._value_store: dict[str, torch.Tensor] = {}

        def _hook(module: torch.nn.Module, args: tuple, kwargs: dict, output: Any) -> None:
            hidden = kwargs.get('hidden_states')
            if hidden is None:
                hidden = args[0]

            # MaskCLIP angelehnt
            v = module.v_proj(hidden)
            v = module.out_proj(v)

            # patches sind alle tokens AUßER dem CLS Token (Index 0)
            # shape ist (batch, num_patches, hidden_dim)
            self._value_store['patches'] = v[:, 1:]

        self._hook_handle = last_attn.register_forward_hook(
            _hook, with_kwargs=True
        )

    def embed_image_original(self, image_path: str) -> npt.NDArray:
        """Gibt das globale Bild-Embedding zurück aber NICHT normalisiert (wie SemLA)."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.embedding_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_embeddings = (
                self.embedding_model.get_image_features(**inputs)
                .detach().cpu().numpy()
            )
        return image_embeddings  # nicht normalisiert

    def embed_image_dispatch(self, image_path, vocab_embedding_method):
        if vocab_embedding_method in (
                VocabEmbeddingMethod.NONE,
                VocabEmbeddingMethod.NONE_NORMALIZED,
                VocabEmbeddingMethod.GLOBAL,
                VocabEmbeddingMethod.OBJECTDETECTION,
        ):
            raw_embed = self.embed_image_original(image_path)
            norm_embed = raw_embed / np.linalg.norm(raw_embed)
            return norm_embed, raw_embed, None
        return self.embed_image(image_path)

    def embed_image(self, image_path) -> tuple[Any, Any, Any]:
        """
        TODO DOKU
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file '{image_path}' not found.")
            raise
        except Exception as e:
            print(f"Error opening image '{image_path}': {e}")
            raise

        if self.embedding_processor is None or self.embedding_model is None:
            raise RuntimeError("CLIP model or processor is not initialized.")

        inputs = self.embedding_processor(
            images=image, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            # Ein einzelner forward pass durch das Vision-Model
            # Der hook fängt dabei automatisch die patch eatures ab.
            vision_outputs = self.embedding_model.vision_model(**inputs)

            # --- Globales Embedding ---
            # TODO Doku
            pooled = vision_outputs.pooler_output
            global_embedding = self.embedding_model.visual_projection(pooled)

            raw_global = global_embedding.detach().cpu().numpy()
            global_embedding = global_embedding / global_embedding.norm(
                dim=-1, keepdim=True
            )

            # --- Patch-Embeddings  ---
            # TODO Doku
            patch_features = self._value_store['patches'].squeeze(0)
            patch_features = self.embedding_model.vision_model.post_layernorm(patch_features)
            patch_embeddings = self.embedding_model.visual_projection(
                patch_features
            )
            patch_embeddings = patch_embeddings / patch_embeddings.norm(
                dim=-1, keepdim=True
            )
            # TODO Evtl hier die Anzahl der Patches bereits reduzieren?

        global_np = global_embedding.detach().cpu().numpy()
        patches_np = patch_embeddings.detach().cpu().numpy()

        return global_np, raw_global, patches_np

    def __del__(self):
        if hasattr(self, '_hook_handle'):
            self._hook_handle.remove()

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
        """
        Embed alle Texte in einem einzigen Forward Pass.
        Berücksichtigt Synonyme aus komma-separierter Liste.
        """
        synonym_lists = [[s.strip() for s in text.split(",") if s.strip()] or [text]
                         for text in texts]
        flat_synonyms = [s for syns in synonym_lists for s in syns]

        all_templated = [t.format(s) for s in flat_synonyms for t in templates]

        with torch.no_grad():
            inputs = self.tokenizer(all_templated, padding=True, return_tensors="pt").to("cuda")
            embs = self.embedding_model.get_text_features(**inputs)
            embs = embs / embs.norm(dim=-1, keepdim=True)
        embs = embs.detach().cpu().numpy()

        # Pro Synonym über die Templates mitteln
        embs = embs.reshape(len(flat_synonyms), len(templates), -1).mean(axis=1)

        # Pro Klasse über ihre Synonyme mitteln
        out = np.zeros((len(texts), embs.shape[-1]), dtype=embs.dtype)
        idx = 0
        for i, syns in enumerate(synonym_lists):
            n = len(syns)
            out[i] = embs[idx:idx + n].mean(axis=0)
            idx += n

        out = out / np.linalg.norm(out, axis=-1, keepdims=True)
        return out


class EmbeddingManager:
    """Handles image and dataset embedding operations."""

    def __init__(self, embedding_model: EmbeddingModel = None):
        if embedding_model is None:
            embedding_model = ClipEmbeddingModel()
        self.embedding_model = embedding_model

    def embed_image(self, image_path, vocab_embedding_method=VocabEmbeddingMethod.NONE) -> tuple:
        """Embed a single image."""
        normalized_global, raw_global, patch_embeds = self.embedding_model.embed_image_dispatch(image_path, vocab_embedding_method)
        return normalized_global, raw_global, patch_embeds

    """
    NEU: Text embeddings generieren
    """
    def embed_text(self, texts: list[str]) -> npt.NDArray:
        """Embed a text."""
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


        all_class_embeddings = self.embedding_model.embed_texts_batched(classnames)  # schon numpy (N, 768)
        np.savez(save_path, vocab_embeddings=all_class_embeddings)
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
            _, embedding_raw, _ = self.embed_image(img, vocab_embedding_method=VocabEmbeddingMethod.NONE)
            if embedding_raw is not None:
                dataset_embeddings.append(embedding_raw)
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
