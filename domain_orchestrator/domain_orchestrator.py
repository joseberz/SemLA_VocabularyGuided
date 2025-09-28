import time
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Mapping, Union

import logging

import numpy as np
import numpy.typing as npt

import torch
from torch import nn

import peft

from .embedding import EmbeddingManager
from .model_adapters.protocol import ModelAdapter

logging.disable()

torch.set_float32_matmul_precision("high")


def softmax(x: list[float], softmax_temperature) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""

    if softmax_temperature == 0:
        softmax_temperature = 1e-6
    exp_x = np.exp(np.divide(x, softmax_temperature))
    return exp_x / np.sum(exp_x, axis=0)


@dataclass
class Domain:
    """A simple Domain class with a name attribute."""

    name: str
    args: Namespace
    train_dataset_path: Path
    train_average_embedding: npt.NDArray
    data_loader: Any = None
    evaluator: Any = None
    lora_path: Path = None


class DomainObserver:
    """Observer class that holds and manages a collection of Domains."""

    def __init__(
        self,
    ) -> None:
        self.domain_prototypes = {}

    def add_domain_prototypes(
        self,
        domain: Domain,
        average_embedding: npt.NDArray,
    ) -> None:
        """
        Add the average embedding of a domain to the observer.
        """
        self.domain_prototypes.update({domain.name: average_embedding})

    def calculate_similarity_to_domains(
        self,
        embedding: npt.NDArray,
        domains: list[Domain],
        similarity_measure: Callable[[npt.NDArray, npt.NDArray], np.float64],
        sort_descending: bool = True,
    ) -> Dict[str, float]:
        """
        Calculate the similarity between the target embedding and the domain prototypes.
        """
        similarities = []

        for domain in domains:
            prot = self.domain_prototypes[domain.name]
            similarity = similarity_measure(embedding, prot)
            similarities.append([domain.name, similarity])

        similarities_dict = dict(
            sorted(similarities, key=lambda x: x[1], reverse=sort_descending)
        )
        return similarities_dict


class DomainOrchestrator:
    def __init__(
        self,
        domains: list[str],
        lora_db_path: Union[str, Path] = "loradb/",
        embedding_manager: EmbeddingManager = EmbeddingManager(),
        model_adapter: ModelAdapter | None = None,
    ) -> None:
        self.model_adapter = model_adapter or self._default_adapter()
        self.model_adapter.setup()

        self.lora_db_path: Path = Path(lora_db_path)

        self.embedding_manager = embedding_manager

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

    @staticmethod
    def _default_adapter() -> ModelAdapter:
        from .model_adapters.catseg_adapter import CatSegAdapter

        return CatSegAdapter()

    def _benchmark_on_current_target_domain(self, name: str, target_domain: Domain) -> Any:
        print(
            f"Benchmarking {name} on the domain {target_domain.name} ...\n"
        )
        res = self.model_adapter.benchmark(self.current_model, target_domain.args)
        return res

    def _set_current_target_domain(
        self,
        target_domain: Domain,
    ) -> None:
        """
        Set the current target domain to the specified domain.
        """

        print(f"Setting current target domain to {target_domain.name}.\n")

        self.current_model = None
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
            self.current_model = peft.PeftModel.from_pretrained(
                self.model_adapter.load_base_model(target_domain.args),
                lora_path,
                source_domain.name,
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
            resources = self.model_adapter.get_domain_resources(
                source_domain_name, split=split
            )
            source_domains.update(
                {
                    source_domain_name: self._add_domain(
                        domain_name=source_domain_name,
                        args=resources.args,
                        evaluator=resources.evaluator,
                        data_loader=resources.data_loader,
                        train_dataset_path=resources.train_dataset_path,
                    )
                }
            )

        return source_domains

    def _add_domain(
        self,
        domain_name: str,
        args: Namespace,
        evaluator=None,
        data_loader=None,
        lora_path: Union[str, Path, None] = None,
        train_dataset_path: Path | None = None,
    ) -> Domain:
        """Adds a Domain instance to the domains list."""

        if train_dataset_path is None:
            if hasattr(args, "train_dataset_path"):
                train_dataset_path = Path(args.train_dataset_path)
            else:
                raise ValueError(
                    "Training dataset path is required to initialise Domain objects."
                )

        if evaluator and data_loader:
            assert train_dataset_path.exists(), train_dataset_path

        if not train_dataset_path.exists():
            raise FileNotFoundError(
                f"Path to training dataset '{train_dataset_path}' does not exist!"
            )

        resolved_lora_path = self._resolve_lora_path(domain_name, lora_path)

        statistics: Dict[str, npt.NDArray] = self.embedding_manager.calculate_statistics(
            domain_name, resolved_lora_path, train_dataset_path
        )

        train_average_embedding: npt.NDArray = statistics[
            "train_average_embedding"
        ]

        domain = Domain(
            domain_name,
            args,
            train_dataset_path=train_dataset_path,
            lora_path=resolved_lora_path,
            train_average_embedding=train_average_embedding,
            evaluator=evaluator,
            data_loader=data_loader,
        )

        return domain

    def _resolve_lora_path(
        self,
        domain_name: str,
        lora_path: Union[str, Path, None],
    ) -> Path:
        if lora_path is None:
            return self.lora_db_path / domain_name
        return Path(lora_path)

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

            weight_dict, merged_adpater_name = self._merge(
                current_target_domain,
                remove_target_adapter,
                mode,
                top_k=len(self._source_domains) - 1
                if remove_target_adapter
                else len(self._source_domains),
            )

            weights.update({current_target_domain.name: weight_dict})

            result_dict = self._benchmark_on_current_target_domain(
                name=merged_adpater_name, target_domain=current_target_domain
            )

            print(result_dict)

            result = self._get_result_from_dict(result_dict)

            results.update(
                {
                    current_target_domain.name: result
                }
            )

            print(f"Deleting adapter {merged_adpater_name}.")
            self.current_model.delete_adapter(merged_adpater_name)

            print("\n")

        return results, weights

    def _merge(
        self,
        target_domain: Domain,
        remove_target_adapter: bool,
        mode: Literal["uniform", "centroid"],
        target_embedding=None,
        softmax_temperature: int | None = 0.05,
        top_k: int = 5,
        combination_type: str = "cat",
        similarity_measure: Callable[
            [npt.NDArray, npt.NDArray], np.float64
        ] = lambda v1, v2: np.linalg.norm(v1 - v2),
        sort_descending: bool = True,
    ) -> tuple[dict[str, float], str]:
        """
        Merge the source domains and benchmark the merged adapter on the target domain.
        """

        if remove_target_adapter:
            print(f"Removing {target_domain.name} from source domains!")
            source_domains = [
                domain
                for domain in self._source_domains.values()
                if domain.name != target_domain.name
            ]
        else:
            source_domains = [domain for domain in self._source_domains.values()]

        if mode == "uniform":
            weights = [1 / len(source_domains) for _ in range(len(source_domains))]
            domain_weight_mapping = {
                domain.name: weight for domain, weight in zip(source_domains, weights)
            }

            merged_name = ""
            for n, w in domain_weight_mapping.items():
                merged_name += f"_{n}_{str(w).replace('.', '_')}"
            merged_name += f"_{combination_type}_{target_domain.name}"

            self._merge_adapters(
                merge_domains=[domain.name for domain in source_domains],
                weights=weights,
                merged_name=merged_name,
                combination_type=combination_type,
            )

        elif mode == "centroid":
            similarity_mapping = self.observer.calculate_similarity_to_domains(
                embedding=target_embedding,
                domains=source_domains,
                similarity_measure=similarity_measure,
                sort_descending=sort_descending,
            )

            k_closest_names = list(similarity_mapping.keys())[: top_k]
            k_closest_similarities = list(similarity_mapping.values())[: top_k]

            print(f"Similarities to {top_k} closest domains: ")
            for n, d in zip(k_closest_names, k_closest_similarities):
                print(f"{n}: {d}", end=", ")
            print("")

            weights = self._calculate_adapter_weights(
                k_closest_similarities, softmax_temperature
            )

            domain_weight_mapping = {
                k_closest_name: weight
                for k_closest_name, weight in zip(k_closest_names, weights)
            }

            merged_name = ""
            for n, w in domain_weight_mapping.items():
                merged_name += f"_{n}_{str(w).replace('.', '_')}"
            merged_name += f"_{combination_type}_{target_domain.name}"

            self._merge_adapters(
                merge_domains=k_closest_names,
                weights=weights,
                merged_name=merged_name,
                combination_type=combination_type,
            )

        print(f"Setting {merged_name} as the active adapter.\n")
        self.current_model.set_adapter(merged_name)

        return domain_weight_mapping, merged_name

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

        print("Merging domains with weights:")
        for n, w in zip(merge_domains, weights):
            print(f"{n}: {w}", end=", ")
        print("")

        self.current_model.add_weighted_adapter(
            merge_domains,
            weights,
            merged_name,
            combination_type=combination_type,
        )

    def _calculate_adapter_weights(self, similarities: list[float], temperature: float) -> list[float]:
        """
        Calculate the weights for the merged adapter based on the similarities to the source domains.
        """
        weights = softmax(similarities, temperature)
        return weights

    def _setup_observer(self):
        print("Adding domain prototypes to the observer.")
        for domain in self._source_domains.values():
            self.observer.add_domain_prototypes(
                domain=domain, average_embedding=domain.train_average_embedding
            )

    def _get_result_from_dict(self, result_dict: Mapping) -> float:
        res = result_dict["sem_seg"].get("IoU", None)
        if res is None:
            res = result_dict["sem_seg"].get("mIoU")
        return res

    def benchmark_zeroshot(self, target_domains: list[str]) -> dict[str, float]:
        results = {}

        for current_target_domain_name in target_domains:
            current_target_domain = self._target_domains[current_target_domain_name]

            args: Namespace = self.model_adapter.build_eval_args(
                config_file=current_target_domain.args.config_file,
                output_path="output/benchmark_zeroshot/",
                num_gpus=1,
                model_path="models/model_final.pth",
            )

            self.current_model = self.model_adapter.load_base_model(
                args, model_path=args.model_path
            )

            result_dict = self._benchmark_on_current_target_domain(
                name="zeroshot", target_domain=current_target_domain
            )

            print(f"Zeroshot results for {current_target_domain.name}:")
            print(result_dict)

            result = self._get_result_from_dict(result_dict)

            results.update(
                {
                    current_target_domain.name: result
                }
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
                name=current_target_domain.name, target_domain=current_target_domain
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
        softmax_temperature: int | None = 0.05,
        top_k: int = 5,
        combination_type: str = "cat",
        similarity_measure: Callable[
            [npt.NDArray, npt.NDArray], np.float64
        ] = lambda v1, v2: 1 / np.linalg.norm(v1 - v2),
        sort_descending: bool = True,
    ) -> tuple[dict[str, float], dict[str, float]]:
        from detectron2.evaluation import inference_context, SemSegEvaluator
        from contextlib import ExitStack

        results = {}
        weights = {}

        t0 = time.time()

        for current_target_domain_name in target_domains:
            current_target_domain = self._target_domains[current_target_domain_name]

            self._set_current_target_domain(
                current_target_domain,
            )

            data_loader = current_target_domain.data_loader
            evaluator = current_target_domain.evaluator

            model = self.current_model

            evaluator.reset()

            with ExitStack() as stack:
                if isinstance(model, nn.Module):
                    stack.enter_context(inference_context(model))
                stack.enter_context(torch.no_grad())

                for _, inputs in enumerate(data_loader):
                    input_path = inputs[0]["file_name"]

                    print(f"Predicting image: {input_path}")

                    current_embedding = self.embedding_manager.embed_image(input_path)

                    weight_dict, merged_adpater_name = self._merge(
                        target_domain=current_target_domain,
                        remove_target_adapter=remove_target_adapter,
                        mode="centroid",
                        target_embedding=current_embedding,
                        softmax_temperature=softmax_temperature,
                        top_k=top_k,
                        combination_type=combination_type,
                        similarity_measure=similarity_measure,
                        sort_descending=sort_descending,
                    )

                    for domain, weight in weight_dict.items():
                        weights.setdefault(domain, []).append(weight)

                    model = self.current_model

                    outputs = model(inputs)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    if isinstance(evaluator, SemSegEvaluator):
                        _ = evaluator.process(inputs, outputs)
                    else:
                        _ = evaluator.process_image(inputs, outputs)

                    self.current_model.delete_adapter(merged_adpater_name)

            print(f"Benchmarking on domain '{current_target_domain.name}' ...")
            result_dict = evaluator.evaluate()
            result = self._get_result_from_dict(result_dict)
            print(f"Result for domain '{current_target_domain.name}': {result}\n")

            results.update({current_target_domain.name: result})

            if not isinstance(evaluator, SemSegEvaluator):
                evaluator._working_dir.cleanup()

        total = time.time() - t0
        print(f"Experiment took {total} seconds to complete!")

        return results, weights
