import json
import time
from typing import Union, Dict, Callable, List, Tuple
from typing import Any, Literal, Mapping
from pathlib import Path
from argparse import Namespace
from dataclasses import dataclass
from enum import Enum
import logging
import os

import numpy as np
import numpy.typing as npt

import torch
from torch import nn

import peft

from catseg.NovelSemSegEvaluator import NovelSemSegEvaluator
from .embedding import EmbeddingManager, VocabEmbeddingMethod
from .object_detection import ObjectDetector, ADAPTER_VOCAB_JSONS

logging.disable()

torch.set_float32_matmul_precision("high")

from .utils import custom_domain_args, get_domain_args, benchmark_catseg, load_catseg_model, get_classnames_for_domain
from .vocab_swap import swap_class_vocabulary, restore_class_vocabulary

# Normalization methods as Enums
# Are used in calculate_similarity_to_domains_voc_distance to
# normalize original SemLA distance and the new vocabulary distance
class NormalizationMethod(Enum):
    ZSCORE = "zscore"
    MINMAX = "minmax"

def softmax(x: list[float], softmax_temperature) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    if softmax_temperature == 0:
        softmax_temperature = 1e-6
    x = np.asarray(x, dtype=float)
    scaled = x / softmax_temperature
    scaled = scaled - np.max(scaled)  # stabilizing the softmax
    exp_x = np.exp(scaled)
    return exp_x / np.sum(exp_x, axis=0)

def min_max_normalize(x):
    if x.max() == x.min():
        return np.ones_like(x)
    return (x - x.min()) / (x.max() - x.min())

def z_score_normalize(x):
    if x.std() == 0:
        return np.zeros_like(x)
    return (x - x.mean()) / x.std()

# Dispatch function for the normalization methods
def normalize_scores(x: np.ndarray, method: NormalizationMethod) -> np.ndarray:
    if method == NormalizationMethod.ZSCORE:
        return z_score_normalize(x)
    elif method == NormalizationMethod.MINMAX:
        return min_max_normalize(x)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_miou(pred_mask: np.ndarray, gt_mask: np.ndarray, domain_name: str) -> Tuple[float, Dict[int, float], List[int]]:
    """Calculate a per image mIoU for all present classes"""

    # resize wenn nötig
    if pred_mask.shape != gt_mask.shape:
        from PIL import Image
        pred_mask = np.array(
            Image.fromarray(pred_mask.astype(np.int32)).resize(
                (gt_mask.shape[1], gt_mask.shape[0]),
                resample=Image.NEAREST  # NEAREST interpolation
            )
        )

    classes = np.unique(gt_mask)
    # ignore_label rausfiltern
    if domain_name == "mv":
        classes = classes[classes != 255]
    else:
        classes = classes[classes != 65]
    if len(classes) == 0:
        return float("nan")

    ious = []
    per_class_iou = {}
    for cls in classes:
        pred_cls = pred_mask == cls
        gt_cls   = gt_mask == cls
        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union        = np.logical_or(pred_cls, gt_cls).sum()
        if union > 0:
            class_iou = intersection / union
            per_class_iou[int(cls)] = {"miou": class_iou, "intersection": intersection, "union": union}
            ious.append(class_iou)

    return float(np.mean(ious)), per_class_iou, classes.tolist()

@dataclass
class ImageEmbedding:
    raw: npt.NDArray
    norm: npt.NDArray

    def get_embedding(self, normalize: bool) -> npt.NDArray:
        """returns either the normalized or raw embedding."""
        return self.norm if normalize else self.raw

@dataclass
class Domain:
    """A simple Domain class with a name attribute."""
    name: str
    args: Namespace
    train_dataset_path: Path
    train_average_embedding: npt.NDArray
    vocab_embeddings: npt.NDArray
    data_loader: Any
    evaluator: Any
    lora_path: Path = None


class DomainObserver:
    """Observer class that holds and manages a collection of Domains."""

    def __init__(
        self,
    ) -> None:
        self.domain_prototypes = {}
        self.domain_vocab_embeddings = {}
        self._debug_similarities_semla = []
        self._debug_similarities_voc = []

    def add_domain_prototypes(
        self,
        domain: Domain,
        average_embedding: npt.NDArray,
        vocab_embeddings: npt.NDArray
    ) -> None:
        """
        Add the average embedding of a domain to the observer.
        """
        self.domain_prototypes.update({domain.name: average_embedding})
        self.domain_vocab_embeddings.update({domain.name: vocab_embeddings}) # NEU

    def calculate_similarity_to_domains(
        self, 
        embedding: npt.NDArray, 
        domains: list[Domain],
        similarity_measure: Callable[[npt.NDArray, npt.NDArray], np.float64],
        sort_descending: bool = True
    ) -> Dict[str, float]:
        """
        Calculate the similarity between the target embedding and the domain prototypes.
        """
        similarities = []

        for domain in domains:
            prot = self.domain_prototypes[domain.name]
            similarity = similarity_measure(embedding, prot)
            similarities.append([domain.name, similarity])

        # sort similarities from lowest to highest
        similarities_dict = dict(sorted(similarities, key=lambda x: x[1], reverse=sort_descending))
        return similarities_dict

    def calculate_similarity_to_domains_normalized(
            self,
            embedding: npt.NDArray,
            domains: list[Domain],
            similarity_measure: Callable[[npt.NDArray, npt.NDArray], np.float64],
            sort_descending: bool = True,
            normalization_method: NormalizationMethod = NormalizationMethod.ZSCORE,
            top_k: int = 5,
    ) -> Dict[str, float]:
        """
        Same variant as the baseline domain distance method, but incorporating the same normalization procedure.
        Is used in experiments to exclude normalization as a confounding factor."""
        domain_names = []
        sem_sims = []

        for domain in domains:
            prot = self.domain_prototypes[domain.name]
            sem_sim = similarity_measure(embedding, prot)
            domain_names.append(domain.name)
            sem_sims.append(sem_sim)

        raw_sim = np.array(sem_sims, dtype=float)
        print(f"Rohe Distanzen: {raw_sim}")
        assert np.isfinite(raw_sim).all(), f"NaN/Inf in raw_sim: {dict(zip(domain_names, raw_sim))}"

        self._debug_similarities_semla.extend(raw_sim.tolist())

        # globale Normalisierung
        sim_norm_global = normalize_scores(raw_sim, normalization_method)

        order = np.argsort(sim_norm_global)
        if sort_descending:
            order = order[::-1]
        top_idx = order[:top_k]

        # finale Normalisierung nur auf den Top-K Rohwerten
        top_domain_names = [domain_names[i] for i in top_idx]
        top_raw_sim = raw_sim[top_idx]
        sim_norm_topk = normalize_scores(top_raw_sim, normalization_method)

        similarities_dict = dict(
            sorted(zip(top_domain_names, sim_norm_topk.tolist()), key=lambda x: x[1], reverse=sort_descending)
        )
        return similarities_dict

    def calculate_similarity_to_domains_voc_distance(
            self,
            embedding: npt.NDArray,
            domains: list[Domain],
            similarity_measure: Callable[[npt.NDArray, npt.NDArray], np.float64],
            sort_descending: bool = True,
            vocab_embedding_method: VocabEmbeddingMethod = VocabEmbeddingMethod.GLOBAL,
            target_vocab_embedding: npt.NDArray | None = None,
            top_q: int = 5,
            gamma: float = 0.5,
            normalization_method: NormalizationMethod = NormalizationMethod.ZSCORE,
            embedding_norm: npt.NDArray | None = None,
            top_q_frac: float | None = None,
            top_k: int = 5,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Calculate the similarity between the target embedding and the domain prototypes.
        """
        domain_names = []
        sem_sims = []
        voc_sims = []

        for domain in domains:
            # Domänendistanz (SemLA)
            prot = self.domain_prototypes[domain.name]
            sem_sim = similarity_measure(embedding, prot)

            # Vokabulardistanz
            if vocab_embedding_method is VocabEmbeddingMethod.OBJECTDETECTION and target_vocab_embedding is None:
                # Fallback falls das Yolo-World Modell keine Hinweise zu Klassen auf dem Bild entdeckt hat
                target_vocab_embedding = [embedding_norm]

            if vocab_embedding_method is not VocabEmbeddingMethod.GLOBAL:
                vocab_embeddings = self.domain_vocab_embeddings[domain.name]
                voc_sim = self.calculate_vocabulary_similarity(
                    target_vocab_embedding, vocab_embeddings, similarity_measure, top_q, top_q_frac
                )
            else:
                vocab_embeddings = self.domain_vocab_embeddings[domain.name]
                voc_sim = self.calculate_vocabulary_similarity(
                    [embedding_norm], vocab_embeddings, similarity_measure, top_q, top_q_frac
                )

            domain_names.append(domain.name)
            sem_sims.append(sem_sim)
            voc_sims.append(voc_sim)

        raw_sim = np.array(sem_sims, dtype=float)
        raw_voc_sims = np.array(voc_sims, dtype=float)

        assert np.isfinite(raw_sim).all(), f"NaN/Inf in raw_sim: {dict(zip(domain_names, raw_sim))}"
        assert np.isfinite(raw_voc_sims).all(), f"NaN/Inf in raw_voc_sims: {dict(zip(domain_names, raw_voc_sims))}"

        # Debug-Logs
        self._debug_similarities_semla.extend(raw_sim.tolist())
        self._debug_similarities_voc.extend(raw_voc_sims.tolist())

        # globale Normalisierung nur fürs Ranking
        sim_norm_global = normalize_scores(raw_sim, normalization_method)
        voc_norm_global = normalize_scores(raw_voc_sims, normalization_method)

        combined_global = gamma * sim_norm_global + (1 - gamma) * voc_norm_global

        order = np.argsort(combined_global)
        if sort_descending:
            order = order[::-1]
        top_idx = order[:top_k]

        # finale Normalisierung nur auf den Top-K Rohwerten
        top_domain_names = [domain_names[i] for i in top_idx]
        top_raw_sim = raw_sim[top_idx]
        top_raw_voc = raw_voc_sims[top_idx]

        sim_norm_topk = normalize_scores(top_raw_sim, normalization_method)
        voc_norm_topk = normalize_scores(top_raw_voc, normalization_method)

        combined_topk = gamma * sim_norm_topk + (1 - gamma) * voc_norm_topk

        similarities_dict = dict(
            sorted(zip(top_domain_names, combined_topk.tolist()), key=lambda x: x[1], reverse=sort_descending)
        )
        vis_similarities = dict(
            sorted(zip(top_domain_names, top_raw_sim.tolist()), key=lambda x: x[1], reverse=sort_descending)
        )
        voc_similarities = dict(
            sorted(zip(top_domain_names, top_raw_voc.tolist()), key=lambda x: x[1], reverse=sort_descending)
        )

        return similarities_dict, vis_similarities, voc_similarities

    @staticmethod
    def calculate_vocabulary_similarity(
            embedding: List[npt.NDArray],
            vocab_embeddings: List[npt.NDArray],
            similarity_measure: Callable[[npt.NDArray, npt.NDArray], np.float64],
            top_q: int = 5,
            top_q_frac: float | None = None,
            use_cosine: bool = True
    ) -> np.float64:
        E = np.stack([e.squeeze() for e in embedding])
        V = np.stack([v.squeeze() for v in vocab_embeddings])

        if use_cosine:
            E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
            V_norm = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
            distance_matrix = E_norm @ V_norm.T
        else:
            # Fallback: normale verschachtelte for-Schleife
            distance_matrix = np.zeros((len(embedding), len(vocab_embeddings)))
            for i, emb in enumerate(embedding):
                print(np.linalg.norm(emb))
                for j, vocab_emb in enumerate(vocab_embeddings):
                    print(np.linalg.norm(vocab_emb))
                    distance_matrix[i][j] = similarity_measure(emb, vocab_emb)

        min_dists_bild_adapter = np.max(distance_matrix, axis=1)
        dist_bild_adapter = np.mean(min_dists_bild_adapter)

        min_dists_per_vocab = np.max(distance_matrix, axis=0)
        n_classes = len(vocab_embeddings)

        if top_q_frac is not None:
            q = max(1, min(round(top_q_frac * n_classes), n_classes))
        else:
            q = min(top_q, n_classes)
        top_q_dists = np.sort(min_dists_per_vocab)[-q:]
        dist_adapter_bild = np.mean(top_q_dists)
        return dist_bild_adapter + dist_adapter_bild

    """
    Debugging Methode um die Wertebereiche der beiden Distanz Methoden zu vergleichen
    """
    def print_similarity_stats(self, save_path: str | None = None):
        A = np.array(self._debug_similarities_semla, dtype=float)
        B = np.array(self._debug_similarities_voc, dtype=float)

        print("\nSimilarity Statistics (global über alle Bilder)")

        if A.size == 0:
            print("Keine Domänen-Distanz-Werte geloggt, überspringe Statistik.")
            return

        print(f"Domänen-Distanz A:  min={A.min():.4f}, max={A.max():.4f}, mean={A.mean():.4f}, std={A.std():.4f}")

        if B.size == 0:
            print("Keine Vokabular-Distanz-Werte geloggt (z.B. bei NONE_NORMALIZED)")
        else:
            print(f"Vocab-Distanz  B:  min={B.min():.4f}, max={B.max():.4f}, mean={B.mean():.4f}, std={B.std():.4f}")

        if save_path is not None:
            if B.size == 0:
                np.savez(save_path, domain_distance=A)
            else:
                np.savez(save_path, domain_distance=A, vocab_distance=B)
            print(f"Rohwerte gespeichert unter: {save_path}")


class DomainOrchestrator:
    def __init__(
        self,
        domains: list[str],
        vocab_embedding_method: VocabEmbeddingMethod,
        lora_db_path: Union[str, Path] = "loradb/",
        embedding_manager: EmbeddingManager = EmbeddingManager(),
        normalize_centroids: bool = True,
        subset_fraction: float = 1.0,
        subset_seed: int = 42,
        val_test_split: float = 0.0,
        val_test_seed: int = 123,
        use_val_portion: bool = True,
    ) -> None:
        
        # TODO: Currently, to use catseg for experiments, we need to change the directory to the catseg directory
        # This can be fixed by refactoring the catseg repo
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        catseg_path = os.path.join(parent_dir, "catseg") # TODO: This is a hardcoded path, it should be a parameter
        
        print(f"Changing directory to '{catseg_path}' ...")

        try:
            os.chdir(catseg_path)
        except FileNotFoundError:
            print(f"Error: The specified path '{catseg_path}' does not exist.")
            exit(1)
        except PermissionError:
            print(f"Error: Insufficient permissions to access '{catseg_path}'.")
            exit(1)
        except Exception as e:
            print(f"Unexpected error while changing directory: {e}")
            exit(1)

        self.lora_db_path: Path = Path(lora_db_path)

        self.embedding_manager = embedding_manager

        self.normalize_centroids = normalize_centroids

        self._subset_fraction = subset_fraction
        self._subset_seed = subset_seed
        self._val_test_split = val_test_split
        self._val_test_seed = val_test_seed
        self._use_val_portion = use_val_portion

        self.object_detector = None
        if vocab_embedding_method is VocabEmbeddingMethod.OBJECTDETECTION:
            self.object_detector = ObjectDetector()

        self.current_model = None

        self.observer: DomainObserver = DomainObserver()

        print("Adding source domains ...")
        
        self._source_domains: Mapping[str, Domain] = self._add_domains(
            domains, split="train"
        )
        print("Source domains added. \n\n")

        print("Adding target domains ...")
        self._target_domains: Mapping[str, Domain] = self._add_domains(
            domains, split="val"
        )
        print("Target domains added. \n\n")

        self._setup_observer()


    def _benchmark_on_current_target_domain(self, name: str, target_domain: Domain) -> Any:
        print(
            f"Benchmarking {name} on the domain {target_domain.name} ...\n"
        )
        res = benchmark_catseg(self.current_model, target_domain.args)
        return res

    def _set_current_target_domain(
        self,
        target_domain: Domain,
    ) -> None:
        """
        Set the current target domain to the specified domain.
        """
        # We need to load all adapters each time the target domain changes because the same
        # config can't be used across datasets and PEFT does not allow us to change the base
        # model and keep the loaded adapters

        print(f"Setting current target domain to {target_domain.name}.\n")

        self.current_model = None  # This will ensure that the current PEFT model will be initialized using base model with new config
        self._load_adapters(target_domain)

    def _load_adapters(self, target_domain: Domain) -> None:
        """
        Load all adapters for the source domains.
        """
        for source_domain in self._source_domains.values():
            lora_path = source_domain.lora_path
            assert lora_path.exists(), lora_path
            print(f"Loading LoRA: '{lora_path}' ...")
            self._load_lora(target_domain, source_domain, lora_path)
            print(f"LoRA: '{lora_path}' loaded\n\n")

    def _load_lora(self, target_domain: Domain, source_domain: Domain, lora_path: Path) -> None:
        """
        Load the LoRA adapter for the specified domain.
        """
        if self.current_model is None:
            # Wrap the model in PeftModel class the first time an adapter is loaded
            # The base model should be loaded with target domain config to avoid label space mismatch
            self.current_model = peft.PeftModel.from_pretrained(
                load_catseg_model(target_domain.args), lora_path, source_domain.name
            )
        else:
            self.current_model.load_adapter(lora_path, source_domain.name)

    def _add_domains(
        self,
        source_domain_names: list[str],
        split: Literal["train", "val"],
    ) -> Dict[str, Domain]:
        """
        Add the specified domains to the orchestrator.
        """

        source_domains = {}

        for source_domain_name in source_domain_names:
            args, evaluator, data_loader = get_domain_args(
                source_domain_name, split=split,
                subset_fraction=self._subset_fraction,
                subset_seed=self._subset_seed,
                val_test_split=self._val_test_split if split == "val" else 0.0,
                val_test_seed=self._val_test_seed,
                use_val_portion=self._use_val_portion,
            )
            source_domains.update(
                {
                    source_domain_name: self._add_domain(
                        domain_name=source_domain_name,
                        args=args,
                        evaluator=evaluator,
                        data_loader=data_loader,
                    )
                }
            )

        return source_domains


    def _add_domain(
        self,
        domain_name: str,
        args: Namespace,
        evaluator,
        data_loader,
        lora_path: Union[str, Path, None] = None,
    ) -> Domain:
        """Adds a Domain instance to the domains list."""

        train_dataset_path = Path(args.train_dataset_path)
        assert train_dataset_path.exists(), train_dataset_path

        if lora_path is None:
            lora_path = self.lora_db_path / domain_name

        statistics: Dict[str, npt.NDArray] = self.embedding_manager.calculate_statistics(
            domain_name, lora_path, train_dataset_path,
        )

        voc_embeddings = self.embedding_manager.get_vocabulary_embeddings(domain_name, lora_path, train_dataset_path)

        train_average_embedding: npt.NDArray = statistics[
            "train_average_embedding"
        ]

        if self.normalize_centroids:
            norm = np.linalg.norm(train_average_embedding)
            if norm > 0:
                train_average_embedding = train_average_embedding / norm
            print(f"Normalized centroid for '{domain_name}'.")

        domain = Domain(
            domain_name,
            args,
            train_dataset_path=train_dataset_path,
            lora_path=lora_path,
            train_average_embedding=train_average_embedding,
            vocab_embeddings=voc_embeddings,
            evaluator=evaluator,
            data_loader=data_loader,
        )

        return domain
    
    def _batch_merge(
        self,
        target_domains: list[str],
        mode: Literal["uniform", "centroid"],
        remove_target_adapter: bool = False,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Merge the source domains and benchmark the merged adapter on the target domains.
        """

        results = {}
        weights = {}

        for current_target_domain_name in target_domains:

            current_target_domain = self._target_domains[current_target_domain_name]
            
            self._set_current_target_domain(
                current_target_domain,
            )

            weight_dict, merged_adpater_name, _, _, _ = self._merge(
                current_target_domain,
                remove_target_adapter,
                mode,
                top_k=len(self._source_domains) - 1 if remove_target_adapter else len(self._source_domains)
            )

            weights.update({current_target_domain.name: weight_dict})

            result_dict = self._benchmark_on_current_target_domain(
                name=merged_adpater_name,
                target_domain=current_target_domain
            )

            print(result_dict)

            result = self._get_result_from_dict(result_dict)

            results.update(
                {
                    current_target_domain.name: result
                }  # Different datasets have different evaluation methods so this won't always work
            )

            # Delete the adapter so we can add another with the same name but different weights (remove unused adapters)
            print(f"Deleting adapter {merged_adpater_name}.")
            self.current_model.delete_adapter(merged_adpater_name)

            print("\n")

        return results, weights


    def _merge(
        self,
        target_domain: Domain,
        remove_target_adapter: bool,
        mode: Literal["uniform", "centroid"],
        target_embedding: ImageEmbedding | None = None,
        softmax_temperature: int | None = 0.05,
        top_k: int = 5,  # number of domains to merge
        combination_type: str = "cat",
        similarity_measure: Callable[
            [npt.NDArray, npt.NDArray], np.float64
        ] = lambda v1, v2: np.linalg.norm(v1 - v2),
        sort_descending: bool = True,
        vocab_embedding_method: VocabEmbeddingMethod = VocabEmbeddingMethod.NONE,
        target_vocab_embedding=None,
        top_q: int = 5,
        gamma: float = 0.5,
        normalization_method: NormalizationMethod = NormalizationMethod.ZSCORE,
        top_q_frac: float | None = None
    ) -> tuple[dict[str, float], str, Dict[str, float], Dict[str, float], float]:
        """
        Merge the source domains and benchmark the merged adapter on the target domain.
        """
    
        source_domains = None
        if remove_target_adapter:
            print(f"Removing {target_domain.name} from source domains!")
            source_domains = [
                domain
                for _, domain in self._source_domains.items() if domain.name != target_domain.name
            ]
            # TODO schöner machen..
            source_domains = [
                domain
                for domain in source_domains if domain.name != "nyunovel" and domain.name != "iddnovel" and domain.name != "pc59novel"
            ]
            if target_domain.name == "nyunovel":
                source_domains = [
                    domain
                    for domain in source_domains if domain.name != "nyu"
                ]
            if target_domain.name == "iddnovel":
                source_domains = [
                    domain
                    for domain in source_domains if domain.name != "idd"
                ]
            if target_domain.name == "pc59novel":
                source_domains = [
                    domain
                    for domain in source_domains if domain.name != "pc59"
                ]
        else:
            source_domains = [
                domain
                for _, domain in self._source_domains.items()
            ]

        voc_distance = None
        vis_distance = None

        # Selektion Zeitmessung
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_selection_start = time.perf_counter()
        t_selection_end = 0
        if mode == "uniform":

            weights = [1 / len(source_domains) for _ in range(len(source_domains))]
            domain_weight_mapping = {domain.name: weight for domain, weight in zip(source_domains, weights)}

            merged_name = ""
            for n, w in domain_weight_mapping.items():
                merged_name += f"_{n}_{str(w).replace('.','_')}"
            merged_name += f"_{combination_type}_{target_domain.name}" # Create a unique name for merged adapter so that it does not override existing adapters

            self._merge_adapters(
                merge_domains=[domain.name for domain in source_domains],
                weights=weights,
                merged_name=merged_name,
                combination_type=combination_type
            )

        elif mode == "centroid":
            if vocab_embedding_method is VocabEmbeddingMethod.NONE:
                similarity_mapping = self.observer.calculate_similarity_to_domains(
                    embedding=target_embedding.get_embedding(self.normalize_centroids),
                    domains=source_domains,
                    similarity_measure=similarity_measure,
                    sort_descending=sort_descending
                )
                vis_distance = similarity_mapping
            elif vocab_embedding_method is VocabEmbeddingMethod.NONE_NORMALIZED:
                # Baseline mit gleicher Normalisierung wie die Voc-Methoden
                # Zur Isolierung des Normalisierungs-Effekts als konfundierenden Faktor
                similarity_mapping = self.observer.calculate_similarity_to_domains_normalized(
                    embedding=target_embedding.get_embedding(self.normalize_centroids),
                    domains=source_domains,
                    similarity_measure=similarity_measure,
                    sort_descending=sort_descending,
                    normalization_method=normalization_method,
                    top_k=top_k,
                )
                vis_distance = similarity_mapping
            else:
                similarity_mapping, vis_distance, voc_distance = self.observer.calculate_similarity_to_domains_voc_distance(
                    embedding=target_embedding.get_embedding(self.normalize_centroids),
                    domains=source_domains,
                    similarity_measure=similarity_measure,
                    sort_descending=sort_descending,
                    vocab_embedding_method=vocab_embedding_method,
                    target_vocab_embedding=target_vocab_embedding,
                    top_q=top_q,
                    gamma=gamma,
                    normalization_method=normalization_method,
                    embedding_norm=target_embedding.norm,  # For GLOBAL Voc distance
                    top_q_frac=top_q_frac,
                    top_k=top_k
                )

            k_closest_names = list(similarity_mapping.keys())[: top_k]
            k_closest_similarities = list(similarity_mapping.values())[: top_k]

            print(f"Similarities to {top_k} closest domains: ")
            for n, d in zip(k_closest_names, k_closest_similarities):
                print(f"{n}: {d}", end=", ")
            print("")

            weights = self._calculate_adapter_weights(k_closest_similarities, softmax_temperature)
            #weights = [1 / len(k_closest_names) for _ in k_closest_names]  # TEST: Gleichverteilung statt SemLA-Gewichte

            domain_weight_mapping = {
                k_closest_name: weight
                for k_closest_name, weight in zip(k_closest_names, weights)
            }

            merged_name = ""
            for n, w in domain_weight_mapping.items():
                merged_name += f"_{n}_{str(w).replace('.','_')}"
            merged_name += f"_{combination_type}_{target_domain.name}" # Create a unique name for merged adapter so that it does not override existing adapters

            # Selektion und Fusionierung Zeitmessung
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_selection_end = time.perf_counter()

            self._merge_adapters(
                merge_domains=k_closest_names,
                weights=weights,
                merged_name=merged_name,
                combination_type=combination_type
            )

        print(f"Setting {merged_name} as the active adapter.\n")
        self.current_model.set_adapter(merged_name)

        selection_ms = (t_selection_end - t_selection_start) * 1000.0

        return domain_weight_mapping, merged_name, vis_distance, voc_distance, selection_ms

    def _merge_adapters(
        self,
        merge_domains: list[str],
        weights: list[float],
        merged_name: str,
        combination_type: str,
    ) -> None:
        """
        Merge the specified adapters with the specified weights.
        """
        
        print(f"Merging domains with weights:")
        for n, w in zip(merge_domains, weights):
            print(f"{n}: {w}", end=", ")
        print("")

        self.current_model.add_weighted_adapter(
            merge_domains,
            weights,
            merged_name,
            combination_type=combination_type,
        )

    def _calculate_adapter_weights(self, similarities:list[float], temperature: float) -> list[float]:
        """
        Calculate the weights for the merged adapter based on the similarities to the source domains.
        """
        weights = softmax(similarities, temperature)
        return weights

    def _setup_observer(self):
        print("Adding domain prototypes to the observer.")
        for domain in self._source_domains.values():
            self.observer.add_domain_prototypes(
                domain=domain, average_embedding=domain.train_average_embedding, vocab_embeddings=domain.vocab_embeddings
            )

    def _get_result_from_dict(self, result_dict: Mapping) -> float:
        res = result_dict["sem_seg"].get("IoU", None)
        if res is None:
            res = result_dict["sem_seg"].get("mIoU")
        if res is None:
            res = result_dict["sem_seg"].get("mIoU-All-Classes")
        return res


    def benchmark_zeroshot(self, target_domains: list[str]) -> dict[str, float]:
        results = {}

        for current_target_domain_name in target_domains:
            current_target_domain = self._target_domains[current_target_domain_name]

            args: Namespace = custom_domain_args(
                config_file=current_target_domain.args.config_file,
                output_path="output/benchmark_zeroshot/",
                num_gpus=1,
                model_path="models/model_final.pth",
            )

            self.current_model = load_catseg_model(
                args, model_path=args.model_path
            )

            result_dict = self._benchmark_on_current_target_domain(name="zeroshot", target_domain=current_target_domain)

            print(f"Zeroshot results for {current_target_domain.name}:")
            print(result_dict)

            result = self._get_result_from_dict(result_dict)

            results.update(
                {
                    current_target_domain.name: result
                }  # res can look different from dataset to dataset
            )

        return results
    
    def benchmark_oracle(self, target_domains: list[str]) -> dict[str, float]:
        results = {}

        for current_target_domain_name in target_domains:
            
            current_target_domain = self._target_domains[current_target_domain_name]

            self._set_current_target_domain(
                current_target_domain,
            )

            self.current_model.set_adapter(current_target_domain.name)

            result_dict = self._benchmark_on_current_target_domain(
                name=current_target_domain.name,
                target_domain=current_target_domain
            )
            print(result_dict)

            result = self._get_result_from_dict(result_dict)

            results.update(
                {
                    current_target_domain.name: result
                }
            )

        return results

    def benchmark_uniform(
        self,
        target_domains: list[str],
        remove_target_adapter: bool,
    ) -> tuple[dict[str, float], dict[str, float]]:
        print(f"Starting uniform merge on domains {target_domains}")

        results, weights = self._batch_merge(
            target_domains=target_domains,
            mode="uniform",
            remove_target_adapter=remove_target_adapter,
        )

        print(f"Finished uniform merge on domains {target_domains}")

        return results, weights

    def benchmark_semla(
        self,
        target_domains: list[str],
        remove_target_adapter: bool = False,
        softmax_temperature: float | None = 0.05,
        top_k: int = 5,
        combination_type: str = "cat",
        similarity_measure: Callable[
            [npt.NDArray, npt.NDArray], np.float64
        ] = lambda v1, v2: 1 / np.linalg.norm(v1 - v2),
        vocab_embedding_method: VocabEmbeddingMethod = VocabEmbeddingMethod.NONE,
        sort_descending: bool = True,
        gamma: float = 0.5,
        top_q: int = 5,
        save_per_image_log: bool = False,
        normalization_method: NormalizationMethod = NormalizationMethod.ZSCORE,
        top_q_frac: float | None = None,
    ) -> tuple[dict[str, float], dict[str, float], list[dict[str, float] | None]]:

        from detectron2.evaluation import inference_context, SemSegEvaluator
        from contextlib import ExitStack

        results = {}
        weights = {}

        # Zeitmessung (Summe + Anzahl Bilder)
        timing_sum = {"embedding_ms": 0.0, "vocab_retrieval_ms": 0.0, "selection_ms": 0.0, "fusion_ms": 0.0}
        timing_count = 0

        t0 = time.time()

        for current_target_domain_name in target_domains:
            
            current_target_domain = self._target_domains[current_target_domain_name]

            self._set_current_target_domain(
                current_target_domain,
            )

            data_loader = current_target_domain.data_loader
            evaluator = current_target_domain.evaluator

            model = self.current_model

            # These lines are adopted from
            # https://github.com/facebookresearch/detectron2/blob/2a420edb307c9bdf640f036d3b196bed474b8593/detectron2/evaluation/evaluator.py#L103

            evaluator.reset()
            with (ExitStack() as stack):
                if isinstance(model, nn.Module):
                    stack.enter_context(inference_context(model))
                stack.enter_context(torch.no_grad())

                for _, inputs in enumerate(data_loader):

                    input_path = inputs[0]["file_name"]

                    print(f"Predicting image: {input_path}")

                    # Zeitmessung füßr das Bild-Embedding
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_embed_start = time.perf_counter()

                    current_embedding_norm, current_embedding_raw, current_patch_embedding = self.embedding_manager.embed_image(input_path, vocab_embedding_method)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    embedding_ms = (time.perf_counter() - t_embed_start) * 1000.0

                    current_embedding = ImageEmbedding(raw=current_embedding_raw, norm=current_embedding_norm)

                    # Zeitmessung füßr das Voc-Retrieval
                    vocab_retrieval_ms = 0.0

                    target_gt_names = None
                    if vocab_embedding_method is VocabEmbeddingMethod.OBJECTDETECTION:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        t_voc_start = time.perf_counter()

                        with torch.no_grad():
                            result_objects = self.object_detector.detect_objects(input_path)
                            detected_classes = []
                            boxes = result_objects[0].boxes
                            for i in range(len(boxes)):
                                class_id = int(boxes.cls[i].item())
                                detected_classes.append(result_objects[0].names[class_id])
                        del result_objects
                        del boxes

                        detected_classes = list(set(detected_classes))
                        print(detected_classes)

                        # Falls die Objektdetection nichts erkennt, bleiben die patch embeddings als Zielvokabular vorhanden
                        if len(detected_classes) != 0:
                            current_patch_embedding = self.embedding_manager.embed_text(detected_classes)

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        vocab_retrieval_ms = (time.perf_counter() - t_voc_start) * 1000.0

                    weight_dict, merged_adpater_name, vis_distance, voc_distance, merge_timing = self._merge(
                        target_domain=current_target_domain,
                        remove_target_adapter=remove_target_adapter,
                        mode="centroid",
                        target_embedding=current_embedding,
                        softmax_temperature=softmax_temperature,
                        top_k=top_k,
                        combination_type=combination_type,
                        similarity_measure=similarity_measure,
                        sort_descending=sort_descending,
                        vocab_embedding_method=vocab_embedding_method,
                        target_vocab_embedding=current_patch_embedding,
                        gamma=gamma,
                        top_q=top_q,
                        normalization_method=normalization_method,
                        top_q_frac=top_q_frac
                    )

                    # Zeiten zur Effizienzabschätzung
                    timing_sum["embedding_ms"] += embedding_ms
                    timing_sum["vocab_retrieval_ms"] += vocab_retrieval_ms
                    timing_sum["selection_ms"] += merge_timing
                    timing_count += 1

                    del current_patch_embedding
                    del current_embedding
                    del current_embedding_norm
                    if current_embedding_raw is not None:
                        del current_embedding_raw
                    torch.cuda.synchronize()  # warten bis alle CUDA ops fertig sind
                    torch.cuda.empty_cache()

                    if isinstance(evaluator, NovelSemSegEvaluator):
                        present_class = Path(input_path).parent.name
                        if present_class not in weights:
                            weights[present_class] = {}
                        for domain, weight in weight_dict.items():
                            weights[present_class].setdefault(domain, []).append(weight)
                    else:
                        for domain, weight in weight_dict.items():
                            weights.setdefault(domain, []).append(weight)

                    model = self.current_model
                    with torch.no_grad():
                        outputs = model(inputs)
                        pred_mask = outputs[0]["sem_seg"].argmax(dim=0).cpu().numpy()
                        gt_mask   = inputs[0]["sem_seg"].numpy()

                        # Logging für Korrelationsanalyse
                        if not hasattr(self, "_correlation_log"):
                            self._correlation_log = []

                        if save_per_image_log:
                            img_miou, class_mious, gt_class_ids = compute_miou(pred_mask, gt_mask, current_target_domain_name)

                            # IDs kriegen über Klassennamen über die Metadata des Evaluators.
                            # TODO prüfen ob das für jeden Evaluator funktioniert
                            stuff_classes = getattr(evaluator, "_class_names", None)
                            if stuff_classes is None:
                                meta_data = getattr(evaluator, "_metadata", None)
                                stuff_classes = getattr(meta_data, "stuff_classes", None)

                            class_mious_named = {stuff_classes[class_id]: miou for class_id, miou in class_mious.items() if class_id < len(stuff_classes)}
                            self._correlation_log.append({
                                "image_path":         input_path,
                                "domain":             current_target_domain_name,
                                "adapter_weights":    {k: float(v) for k, v in weight_dict.items()},
                                "adapter_distance_visual": vis_distance,
                                "adapter_distance_vocabulary": voc_distance,
                                "gt_ids":             gt_class_ids,
                                "per_image_miou":     img_miou,
                                "per_class_miou":     class_mious_named,
                                "vocab_method":       vocab_embedding_method if isinstance(vocab_embedding_method, str) else vocab_embedding_method.value,
                            })

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    if isinstance(evaluator, SemSegEvaluator) or isinstance(evaluator, NovelSemSegEvaluator):
                        _ = evaluator.process(inputs, outputs)
                        print(_)
                    else:
                        _ = evaluator.process_image(inputs, outputs)

                    self.current_model.delete_adapter(merged_adpater_name)

            print(f"Benchmarking on domain '{current_target_domain.name}' ...")
            result_dict = evaluator.evaluate()
            result = self._get_result_from_dict(result_dict)
            print(f"Result for domain '{current_target_domain.name}': {result}\n")

            if isinstance(evaluator, NovelSemSegEvaluator):
                results.update({current_target_domain.name: result_dict["sem_seg"]})
            else:
                results.update({current_target_domain.name: result})

            if not isinstance(evaluator, SemSegEvaluator):
                evaluator._working_dir.cleanup()

        total = time.time() - t0
        print(f"Experiment took {total} seconds to complete!")

        if vocab_embedding_method is not VocabEmbeddingMethod.NONE:
            parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            results_path = os.path.join(parent_dir, "results")
            os.makedirs(results_path, exist_ok=True)
            self.observer.print_similarity_stats(save_path=results_path)

        # Gemessene Zeiten für die Effizienzabschätzung
        if timing_count > 0:
            self._timing_summary = {
                k: v / timing_count for k, v in timing_sum.items()
            }
            self._timing_summary["n_images"] = timing_count
            print(f"Zusammenfassung Zeitmessung {self._timing_summary}")

        return results, weights, self._correlation_log

    def benchmark_semla_qualitative(
        self,
        target_domains: list[str],
        top_x: int,
        image_paths: list[str],
        output_dir: str,
        remove_target_adapter: bool = False,
        softmax_temperature: float | None = 0.05,
        top_k: int = 5,
        combination_type: str = "cat",
        similarity_measure: Callable = lambda v1, v2: 1 / np.linalg.norm(v1 - v2),
        vocab_embedding_method: VocabEmbeddingMethod = VocabEmbeddingMethod.GLOBAL,
        gamma: float = 0.5,
        top_q: int = 5,
        normalization_method: NormalizationMethod = NormalizationMethod.L1,
        top_q_frac: float | None = None,
    ) -> None:
        from detectron2.evaluation import inference_context
        from contextlib import ExitStack
        import scipy.spatial.distance as spdist

        os.makedirs(output_dir, exist_ok=True)

        for current_target_domain_name in target_domains:
            current_target_domain = self._target_domains[current_target_domain_name]
            self._set_current_target_domain(current_target_domain)

            gt_classnames = get_classnames_for_domain(current_target_domain_name)
            data_loader = current_target_domain.data_loader
            model = self.current_model

            with ExitStack() as stack:
                if isinstance(model, nn.Module):
                    stack.enter_context(inference_context(model))
                stack.enter_context(torch.no_grad())

                for inputs in data_loader:
                    input_path = inputs[0]["file_name"]

                    if image_paths is not None and input_path not in image_paths:
                        continue  # Bild nicht in der handverlesenen Liste, dann überspringen
                    print(f"Predicting: {input_path}")

                    # Adapterauswahl (wie benchmark_semla)
                    embedding_norm, embedding_raw, patch_embedding = self.embedding_manager.embed_image(
                        input_path, vocab_embedding_method
                    )
                    current_embedding = ImageEmbedding(raw=embedding_raw, norm=embedding_norm)

                    weight_dict, merged_adapter_name, vis_distance, voc_distance = self._merge(
                        target_domain=current_target_domain,
                        remove_target_adapter=remove_target_adapter,
                        mode="centroid",
                        target_embedding=current_embedding,
                        softmax_temperature=softmax_temperature,
                        top_k=top_k,
                        combination_type=combination_type,
                        similarity_measure=similarity_measure,
                        sort_descending=True,
                        vocab_embedding_method=vocab_embedding_method,
                        target_vocab_embedding=patch_embedding,
                        gamma=gamma,
                        top_q=top_q,
                        normalization_method=normalization_method,
                        top_q_frac=top_q_frac,
                    )


                    # Bei method=NONE: SemLA-Baseline-Segmentierung
                    # Bei method=PATCH/GLOBAL: Vokabular-Matching-Adapterauswahl, aber noch mit GT-Vokabular
                    selection_out = self.current_model(inputs)
                    selection_mask = selection_out[0]["sem_seg"].argmax(dim=0).cpu().numpy()

                    # Vokabular-Erweiterung NUR wenn method != NONE
                    extended_mask = None
                    extended_classnames = None

                    if vocab_embedding_method is not VocabEmbeddingMethod.NONE:
                        extended_classnames = list(gt_classnames)
                        query_embeddings = patch_embedding if patch_embedding is not None else [embedding_norm]

                        for adapter_name in weight_dict.keys():
                            adapter_vocab_embeddings = self.observer.domain_vocab_embeddings[adapter_name]
                            adapter_classnames = get_classnames_for_domain(adapter_name)

                            sims = np.array([
                                max(1 - spdist.cosine(q.squeeze(), c.squeeze()) for q in query_embeddings)
                                for c in adapter_vocab_embeddings
                            ])
                            top_idx = np.argsort(sims)[-top_x:]
                            for idx in top_idx:
                                name = adapter_classnames[idx]
                                if name not in extended_classnames:
                                    extended_classnames.append(name)

                        old_vocab = swap_class_vocabulary(self.current_model, extended_classnames)
                        extended_out = self.current_model(inputs)
                        extended_mask = extended_out[0]["sem_seg"].argmax(dim=0).cpu().numpy()
                        restore_class_vocabulary(self.current_model, old_vocab)

                    save_qualitative_result(
                        image_path=input_path,
                        vocab_embedding_method=vocab_embedding_method.value,
                        selection_mask=selection_mask,
                        selection_classnames=gt_classnames,
                        extended_mask=extended_mask,
                        extended_classnames=extended_classnames,
                        adapter_weights=weight_dict,
                        output_dir=output_dir,
                    )

                    self.current_model.delete_adapter(merged_adapter_name)
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

            print(f"Vokabular-Erweiterung für '{current_target_domain_name}' für qualitative Segmentierungsbeispiele abgeschlossen.")

def classname_to_color(name: str) -> tuple[int, int, int]:

    h = hashlib.md5(name.encode()).hexdigest()
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

def colorize_mask(mask: np.ndarray, classnames: list[str]) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, name in enumerate(classnames):
        color = classname_to_color(name)
        rgb[mask == idx] = color
    return rgb

def save_qualitative_result(image_path, vocab_embedding_method, selection_mask, selection_classnames,
                             extended_mask, extended_classnames, adapter_weights, output_dir):
    stem = Path(image_path).stem
    prefix = f"{stem}_{vocab_embedding_method}"

    Image.open(image_path).save(f"{output_dir}/{stem}_input.png")
    Image.fromarray(colorize_mask(selection_mask, selection_classnames)).save(f"{output_dir}/{prefix}_selection.png")

    meta = {
        "adapter_weights": {k: float(v) for k, v in adapter_weights.items()},
        "selection_classnames": selection_classnames,
    }

    if extended_mask is not None:
        Image.fromarray(colorize_mask(extended_mask, extended_classnames)).save(f"{output_dir}/{prefix}_extended.png")
        meta["extended_classnames"] = extended_classnames
        meta["new_classes"] = [c for c in extended_classnames if c not in selection_classnames]

    with open(f"{output_dir}/{prefix}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)