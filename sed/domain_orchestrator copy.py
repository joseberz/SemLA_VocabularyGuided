import os
import sys
import time
from typing import Union, Dict, Callable, Sequence, Optional
from collections import abc, deque
from pathlib import Path
from argparse import Namespace
from dataclasses import dataclass
from contextlib import ExitStack
from typing import List, Union, Any, Literal, Mapping
from PIL import Image
import logging

import numpy as np
import numpy.typing

import torch
from torch import nn
import torch
import torchvision.transforms as transforms

from transformers import CLIPModel, CLIPProcessor

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation

from scipy.spatial.distance import cosine, euclidean, mahalanobis

import peft

from detectron2.checkpoint import DetectionCheckpointer
import detectron2.utils.comm as comm
from detectron2.evaluation import inference_context, DatasetEvaluators, SemSegEvaluator
from detectron2.evaluation.cityscapes_evaluation import CityscapesEvaluator

from train_net import Trainer, setup

from utils import custom_domain_args, domain_args, benchmark

import wandb
import copy

from train_net import Trainer, setup

logging.disable()

torch.set_float32_matmul_precision("high")


def softmax(x, temp):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(np.divide(x, temp)) / np.sum(np.exp(np.divide(x, temp)), axis=0)


#INDECIES = ["index-1000", "index-2000", "index-3000", "index-4000", "index-5000"]
INDECIES = ["index-1000"]


DOMAIN_MAPPING = {
    "cityscapes": {
        "normal": 0,
        "rain25mm": 1,
        "rain50mm": 2,
        "rain75mm": 3,
        "rain100mm": 4,
        "rain200mm": 5,
    },
    "acdc": {
        "acdc-rain": 0,
        "acdc-fog": 1,
        "acdc-snow": 2,
        "acdc-night": 3,
    },
}

BASE_MODEL_CONFIG = custom_domain_args(
    # config_file="configs/acdc/acdc_base.yaml",
    config_file="configs/vitl_336.yaml", #TODO: Changed this from config.yaml, might break things
    num_gpus=1,
    output_path="output/base/eval",
    model_path="models/model_final.pth",
)


@dataclass
class Domain:
    """A simple Domain class with a name attribute."""

    name: str
    args: Namespace
    train_dataset_path: Path
    val_dataset_path: Path
    train_dataset_embeddings: np.typing.NDArray
    val_dataset_embeddings: np.typing.NDArray
    data_loader: Any
    evaluator: Any
    lora_path: Path = None
    model_path: Path = None
    train_average_embedding: np.typing.NDArray = None
    val_average_embedding: np.typing.NDArray = None
    # standard_deviation: float = None
    # covariance_matrix: np.typing.NDArray = None
    # inverse_covariance_matrix: np.typing.NDArray = None


class DomainObserver:
    """Observer class that holds and manages a collection of Domains."""

    def __init__(
        self,
        distance_measure: Callable[[np.typing.NDArray, np.typing.NDArray], np.float64],
        distance_masure_name: str,
    ) -> None:
        self.distance_measure = distance_measure
        self.distance_measure_name = distance_masure_name
        self.reducer = PCA(n_components=2, svd_solver="full", random_state=42)
        self.domain_variations = {}
        self.domain_prototypes = {}

        # For visualization
        self.reduced_domain_prototypes = {}
        self.reduced_domain_variations = {}
        self.trajectory = []
        self.colors = [
            "salmon",
            "deepskyblue",
            "steelblue",
            "dodgerblue",
            "blue",
            "darkblue",
        ]

    def fit_reducer(self, all_embeddings: np.typing.NDArray):
        self.reducer = self.reducer.fit(all_embeddings)

    def reduce_embedding(self, embedding: np.typing.NDArray, domain: Domain):
        # self.reducer = self.reducer.fit(domain.dataset_embeddings.squeeze())
        reduced_embedding = self.reducer.transform(embedding)
        return reduced_embedding

    def add_domain_prototypes(
        self, average_embedding: np.typing.NDArray, domain: Domain
    ):
        # reduced_prototype = self.reduce_embedding(average_embedding, domain)

        self.domain_prototypes.update({domain.name: average_embedding})
        # self.reduced_domain_prototypes.update({domain.name: reduced_prototype})

    def add_domain_deviation(self, standard_variation, domain: Domain):
        self.domain_variations.update({domain.name: standard_variation})

    def add_reduced_domain_deviation(self, reduced_standard_variation, domain: Domain):
        self.reduced_domain_variations.update({domain.name: reduced_standard_variation})

    def calcualte_distance_to_domains(
        self, embedding: np.typing.NDArray, domains: list[Domain]
    ) -> Dict[str, float]:
        distances = []

        for domain in domains:
            prot = self.domain_prototypes[domain.name]
            if self.distance_measure_name == "mahalanobis":
                disance = self.distance_measure(
                    embedding, prot, domain.inverse_covariance_matrix
                )
            else:
                disance = self.distance_measure(embedding, prot)

            distances.append([domain.name, disance])

        # sort distances from lowest to highest
        distances_dict = dict(sorted(distances, key=lambda x: x[1]))
        return distances_dict

    def domain_shifted(
        self,
        current_embedding: np.typing.NDArray,
        lastest_embedding: np.typing.NDArray,
        threshold: float,
        # domain: Domain = None,
    ):
        # if self.distance_measure_name == "mahalanobis":
        #     difference = self.distance_measure(
        #         current_embedding, lastest_embedding, domain.inverse_covariance_matrix
        #     )
        # else:
        difference = self.distance_measure(current_embedding, lastest_embedding)
        print("Difference to previous domain: ", difference)
        return abs(difference) > threshold

    def draw_trajectory(self, embeddings, domain, name):
        trajectory = []
        for embng in embeddings:
            trajectory.append(self.reduce_embedding(embng, domain))

        trajectory = np.array(trajectory).squeeze()
        print("trajectory shape ", trajectory.shape)
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        print("len(x): ", len(x))

        fig, ax = plt.subplots()
        ax.axis([-10, 10, -10, 10])
        ax.set_title(f"Trajectory ({name})")
        ax.set_xlabel("PC 1 ")
        ax.set_ylabel("PC 2")

        for domain_name, embedding in self.reduced_domain_prototypes.items():
            plt.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c="blue",
                marker="o",
                edgecolor="k",
                s=70,
            )
            plt.annotate(domain_name, (embedding[:, 0], embedding[:, 1]), fontsize=12)

        scat = ax.scatter([], [], c="red", marker="o", edgecolor="k", alpha=0.4)

        def animate(i):
            # Update data for scatter points incrementally
            scat.set_offsets(np.c_[x[: i + 1], y[: i + 1]])
            return (scat,)

        ani = matplotlib.animation.FuncAnimation(
            fig, animate, frames=len(x), interval=20, blit=True
        )

        return ani

    def draw_trajectory_live(
        self,
        new_embedding: np.typing.NDArray,
        domain: Domain,
        draw_history: bool = False,
    ):

        if draw_history:
            self.trajectory.append(self.reduce_embedding(new_embedding, domain))

        new_embdng = self.reduce_embedding(new_embedding, domain)

        plt.figure()
        for domain_name, embedding in self.reduced_domain_prototypes.items():
            plt.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c="blue",
                marker="o",
                edgecolor="k",
                s=70,
            )
            # Annotating points
            plt.annotate(domain_name, (embedding[:, 0], embedding[:, 1]), fontsize=12)

        if draw_history:
            for embedding in self.trajectory:
                plt.scatter(
                    embedding[:, 0],
                    embedding[:, 1],
                    c="red",
                    marker="^",
                    edgecolor="k",
                    s=70,
                )

        plt.scatter(
            new_embdng[:, 0], new_embdng[:, 1], c="red", marker="^", edgecolor="k", s=70
        )

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Trajectory")
        plt.grid(True)
        plt.show()

    # Create a color map object
    def _custom_cmap_with_n_colors(self, n_colors: int):
        return mcolors.LinearSegmentedColormap.from_list(
            "custom_cmap", self.colors, N=n_colors
        )

    def visualize_domains_with_statistics(self, reduced=False):
        domain_names = list(self.reduced_domain_prototypes.keys())
        n_domains = len(domain_names)
        cmap = self._custom_cmap_with_n_colors(n_domains)

        plt.figure(figsize=(10, 6))
        for i, domain_name in enumerate(domain_names):
            color = cmap(i / n_domains)
            prototype = self.reduced_domain_prototypes[domain_name]
            if not reduced:
                variation = self.domain_variations[domain_name]
                plt.errorbar(
                    x=prototype[:, 0],
                    y=prototype[:, 1],
                    yerr=[variation],
                    fmt="o",
                    label=f"{domain_name} avg ± std",
                    capsize=5,
                    markersize=10,
                    linestyle="",
                    elinewidth=2,
                    color=color,
                )
            else:
                variation = self.reduced_domain_variations[domain_name]
                plt.errorbar(
                    x=prototype[:, 0],
                    y=prototype[:, 1],
                    xerr=variation[0],
                    yerr=variation[1],
                    fmt="o",
                    label=f"{domain_name} avg ± std",
                    capsize=5,
                    markersize=10,
                    linestyle="",
                    elinewidth=2,
                    color=color,
                )
            # Optionally add annotations
            # plt.annotate(domain_name, (prototype[:, 0], prototype[:, 1]), fontsize=12)

        plt.legend()
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA Projection of domain prototypes")
        plt.grid(True)
        plt.show()

    def visualize_domains_pca(self, domains: "list[Domain]"):
        plt.figure(figsize=(10, 6))
        domain_embeddings = [domain.dataset_embeddings for domain in domains]
        reduced_embeddings = [
            self.reducer.transform(embeddings.squeeze())
            for embeddings in domain_embeddings
        ]

        # Create a colormap that transitions from red to blue
        colors = [
            "salmon",
            "deepskyblue",
            "steelblue",
            "dodgerblue",
            "blue",
            "darkblue",
        ]
        labels = [domain.name for domain in domains]

        # Create a color map object
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "custom_cmap", colors, N=len(reduced_embeddings)
        )

        # Create a scatter plot for each set using the colormap

        for i, s in enumerate(reduced_embeddings):
            plt.scatter(s[:, 0], s[:, 1], color=cmap(i), label=f"{labels[i]}")

        # Add a legend
        plt.legend()

        # Add title and labels to the axis
        plt.title("PCA Projection of all training image embeddings")
        plt.xlabel("PC1")
        plt.ylabel("PC2")

        # Show the plot
        plt.savefig("visualize_domains.png")
        plt.show()

    def visualize_domains_tsne(self, domains: "list[Domain]"):
        plt.figure(figsize=(10, 6))
        reducer = TSNE(n_components=2, perplexity=40, random_state=42)

        domain_embeddings = []
        labels = []
        domain_names = [
            domain.name for domain in domains
        ]  # Assuming each domain has a 'name' attribute for the legend

        for i, domain in enumerate(domains):
            embeddings = np.array(domain.dataset_embeddings).squeeze()
            domain_embeddings.append(embeddings)
            labels.extend([i] * embeddings.shape[0])

        domain_embeddings = np.vstack(domain_embeddings)
        labels = np.array(labels)

        reduced_embeddings = reducer.fit_transform(domain_embeddings)

        # Create a colormap
        colors = plt.cm.get_cmap("tab10", len(domains))

        # Create a scatter plot for each set using the colormap
        for i in range(len(domains)):
            plt.scatter(
                reduced_embeddings[labels == i, 0],
                reduced_embeddings[labels == i, 1],
                color=colors(i),
                label=domain_names[i],
                alpha=0.6,
            )

        # Add a legend
        plt.legend()
        # Add title and labels to the axis
        plt.title("t-SNE Projection of all training image embeddings")
        plt.xlabel("t-SNE dimension 1")
        plt.ylabel("t-SNE dimension 2")
        plt.show()

    def visualize_domain_shift(
        self, detected_domains, gt_domains, window_size, dataset: str
    ):
        fig = plt.figure(figsize=(10, 6))
        print(len(gt_domains))
        print(len(detected_domains))

        # Convert y_values to numerical values
        y_detected = [DOMAIN_MAPPING[dataset][y] for y in detected_domains]
        y_gt = [DOMAIN_MAPPING[dataset][y] for y in gt_domains]

        # x values are indices, since we didn't explicitly provide them
        x_values = list(
            range(len(gt_domains))
        )  # Assuming both y_values1 and y_values2 have the same length

        # Create the first line plot
        plt.plot(
            x_values, y_detected, marker="o", linestyle="-", color="b", label="Detected"
        )

        # Create the second line plot
        plt.plot(
            x_values,
            y_gt,
            marker="x",
            linestyle="--",
            color="r",
            label="Actual",
            alpha=0.6,
        )

        # Customizing the graph
        plt.title(
            f"Detected vs. Actual domain ({self.distance_measure_name}, window size: {window_size})"
        )
        plt.xlabel("Step")
        plt.ylabel("Domains")

        # Set custom y-ticks
        plt.yticks(
            ticks=list(DOMAIN_MAPPING.values()), labels=list(DOMAIN_MAPPING.keys())
        )

        plt.legend()  # To show a legend distinguishing the two lines
        plt.grid(True)  # To show a grid

        # Display the graph
        plt.show()
        plt.savefig("acdc_domain_shift.png")

        return fig


class DomainOrchestrator:
    def __init__(
        self,
        source_domains: list[str],
        target_domains: list[str],
        temp: int = 0.03,
        k_adapters: str = 2,  # number of domains to merge
        window_size: Union[
            int, None
        ] = None,  # average the test data in a window of length n, if None don't use,
        distance_thresh=None,
        merge_type: str = "cat",
        distance_measure: Callable[
            [np.typing.NDArray, np.typing.NDArray], np.float64
        ] = lambda v1, v2: np.linalg.norm(v1 - v2),
        lora_db_path: Union[str, Path] = "loradb/",
        distance_measure_name: str = "euclidean",
        pruning_density: float = 0.0,
        no_lora: bool = False,
    ) -> None:

        self._target_domain_names: list[str] = target_domains
        self._source_domains_names: list[str] = source_domains

        self.lora_db_path: Path = Path(lora_db_path)

        self._domains: Dict[str, Domain] = {}
        self.current_target_domain: Domain = None

        self._base_model = self._load_base_model(BASE_MODEL_CONFIG)
        self.current_model = None
        self.current_model_copy = None

        self.merge_type: str = merge_type
        self.pruning_density: float = pruning_density

        self.distance_measure_name: str = distance_measure_name
        self.distance_measure: Callable[
            [np.typing.NDArray, np.typing.NDArray], np.float64
        ] = distance_measure

        self.observer: DomainObserver = DomainObserver(
            self.distance_measure, self.distance_measure_name
        )
        self.distance_thresh: float = distance_thresh

        self.k_adapters: int = k_adapters
        self.window_size: int = window_size
        self.temp: float = temp

        self.embedding_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to("cuda")

        self.embedding_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

        self.no_lora: bool = no_lora

        self._source_domains: Mapping[Domain] = self._add_source_domains(source_domains)
        self._target_domains: Mapping[Domain] = self._add_target_domains(target_domains)

        if no_lora:
            model_paths = {
                domain.name: domain.model_path
                for domain in self._source_domains.values()
            }
            print("Model paths: ", model_paths)
            self._models = {
                name: torch.load(path, map_location=torch.device("cpu"))["model"]
                for name, path in model_paths.items()
            }
            print("Loaded models: ", self._models.keys())

        self._setup_observer()

        print("\n\n\n")

    def _add_source_domains(
        self, source_domain_names: list[str], index: str = "index-1000"
    ):
        source_domains = {}
        for source_domain_name in source_domain_names:
            args, evaluator, data_loader = domain_args(source_domain_name, index=index)
            source_domains.update(
                {
                    source_domain_name: self._add_source_domain(
                        source_domain_name, args, evaluator, data_loader, index=index
                    )
                }
            )
        return source_domains

    def _add_target_domains(
        self, target_domain_names: list[str], index: str = "index-1000"
    ):
        target_domains = {}

        for target_domain_name in target_domain_names:
            args, evaluator, data_loader = domain_args(
                target_domain_name, index=index, split="val"
            )
            # for idx in INDECIES:
            target_domains.update(
                {
                    target_domain_name: self._add_target_domain(
                        target_domain_name,
                        args,
                        evaluator,
                        data_loader,
                        index=index,
                        save_embeddings_path="output/dataset_embeddings/",
                    )
                }
            )

        return target_domains

    def _add_source_domain(
        self,
        name: str,
        args: Namespace,
        evaluator: Any,
        data_loader: Any,
        index: Union[str, None] = None,
        lora_path: Union[str, Path] = None,
    ):
        assert name != ""
        domain: Domain = self._add_domain(
            domain_name=name,
            args=args,
            source=True,
            lora_path=lora_path,
            index=index,
            data_loader=data_loader,
            evaluator=evaluator,
        )

        return domain

    def _add_target_domain(
        self,
        name: str,
        args: Namespace,
        evaluator: Any,
        data_loader: Any,
        index: Union[str, None] = None,
        lora_path: Union[str, Path] = None,
        save_embeddings_path=None,
    ):
        assert name != ""
        domain = self._add_domain(
            domain_name=name,
            args=args,
            source=False,
            lora_path=lora_path,
            index=index,
            data_loader=data_loader,
            evaluator=evaluator,
            save_embeddings_path=save_embeddings_path,
        )
        return domain

    def benchmark_experts(self, is_adapter: bool, zeroshot: bool):

        if zeroshot:
            results = {"zeroshot": {}}
        else:
            results = {
                idx: {
                    source_domain_name: {}
                    for source_domain_name in self._source_domains_names
                }
                for idx in INDECIES
            }

        for current_source_domain_name in self._source_domains_names:

            for current_target_domain_name in self._target_domain_names:

                if zeroshot:

                    # Benchmark zeroshot model
                    self._set_current_target_domain(
                        current_target_domain_name,
                        self._target_domains[current_target_domain_name].args,
                        index="index-1000",
                    )

                    source_args = self._source_domains[current_source_domain_name].args
                    target_args = self.current_target_domain.args

                    # TODO: This might be buggy, we have to use the target domain config
                    # to load the base model to not have label space mismatch

                    args = custom_domain_args(
                        target_args.config_file,
                        "output/benchmark_zeroshot/",
                        num_gpus=1,
                        model_path="models/model_final.pth",
                    )

                    self.current_model = self._load_base_model(
                        args, model_path="models/model_final.pth"
                    )

                    result_dict = self._benchmark_on_current_target_domain(
                        name="zeroshot"
                    )
                    print("zeroshot")
                    print(result_dict)

                    result = self._get_result_from_dict(result_dict)

                    results["zeroshot"].update(
                        {
                            current_target_domain_name: result
                        }  # TODO: res can be different from dataset to dataset
                    )

                else:

                    for idx in INDECIES:

                        print(f"Current Index = {idx}")
                        self._set_current_target_domain(
                            current_target_domain_name,
                            self._target_domains[current_target_domain_name].args,
                            index=idx,
                        )

                        # Set the current model, load adapter or model depending on is_adapter
                        if is_adapter:
                            self.current_model.set_adapter(current_source_domain_name)
                        else:
                            source_args = self._source_domains[
                                current_source_domain_name
                            ].args
                            target_args = self.current_target_domain.args
                            # Change the args to the format that is able to load specific models
                            args = custom_domain_args(
                                target_args.config_file,
                                Path(target_args.model_path).parent.as_posix(),
                                num_gpus=1,
                                model_path=source_args.model_path,
                            )
                            self.current_model = self._load_base_model(args)

                        result_dict = self._benchmark_on_current_target_domain(
                            name=current_source_domain_name
                        )
                        print(result_dict)

                        result = self._get_result_from_dict(result_dict)

                        results[idx][current_source_domain_name].update(
                            {
                                current_target_domain_name: result
                            }  # TODO: res can be different from dataset to dataset
                        )

        return results

    def _setup_observer(self):

        # The following lines appends all the embedding from all the domains because PCA requires a 2D array
        # all_embeddings = []

        # for domain in self._source_domains.values():
        #     all_embeddings.extend(domain.train_dataset_embeddings.squeeze()) # TODO: this might cause problems

        # all_embeddings = np.array(all_embeddings).squeeze()
        # all_embeddings = all_embeddings.reshape(-1, 768)

        # self.observer.fit_reducer(all_embeddings)

        print("Adding domain prototypes to the observer.")
        for domain in self._source_domains.values():

            self.observer.add_domain_prototypes(
                average_embedding=domain.train_average_embedding, domain=domain
            )

            # self.observer.add_domain_deviation(
            #     standard_variation=domain.standard_deviation, domain=domain
            # )

            # if save_reduced:
            #     reduced_embeddings = self.observer.reducer.transform(
            #         domain.dataset_embeddings.squeeze()
            #     )
            # reduced_deviation = self._standard_deviation(
            #     domain.name, reduced_embeddings, axis=0
            # )
            # self.observer.add_reduced_domain_deviation(reduced_deviation, domain)

    def batch_uniform_merge(
        self, remove_target_adapter: bool, full_model_merging: bool
    ):
        print(f"Starting batch uniform merge on domains {self._target_domain_names}")

        results, weights = self._batch_merge(
            weight_type="uniform",
            remove_target_adapter=remove_target_adapter,
            full_model_merging=full_model_merging,
        )

        print(f"Finished batch uniform merge on domains {self._target_domain_names}")

        return results, weights

    def batch_centroid_merge(
        self, remove_target_adapter: bool = False, full_model_merging: bool = False
    ):
        print(f"Starting batch centroid merge on domains {self._target_domain_names}")

        results, weights = self._batch_merge(
            weight_type="centroid",
            remove_target_adapter=remove_target_adapter,
            full_model_merging=full_model_merging,
        )

        print(f"Finished batch centroid merge on domains {self._target_domain_names}")

        return results, weights

    def online_centroid_merge_full(
        self,
        remove_target_adapter: bool = False,
        full_model_merging: bool = True,
    ):
        results = {idx: {} for idx in INDECIES}
        weights = {idx: {} for idx in INDECIES}

        t0 = time.time()

        for idx in INDECIES:
            for current_target_domain_name in self._target_domain_names:

                print(f"Current Index = {idx}")
                self._set_current_target_domain(
                    current_target_domain_name,
                    self._target_domains[current_target_domain_name].args,
                    index=idx,
                )

                target_removed = False
                if remove_target_adapter == True:
                    if current_target_domain_name in self._source_domains_names:
                        target_removed = True
                        self._source_domains_names.remove(current_target_domain_name)

                data_loader = self.current_target_domain.data_loader
                evaluator = self.current_target_domain.evaluator

                # These lines are adopted from
                # https://github.com/facebookresearch/detectron2/blob/2a420edb307c9bdf640f036d3b196bed474b8593/detectron2/evaluation/evaluator.py#L103

                evaluator.reset()

                self.current_model_copy = self._load_base_model(
                    self.current_target_domain.args
                )

                for _, inputs in enumerate(data_loader):

                    input_path = inputs[0]["file_name"]

                    print(f"Predicting image: {input_path}")

                    print(input_path)

                    current_embedding = self._embed_single_image(input_path)

                    weight_dict, _ = self._merge(
                        weight_type="centroid",
                        full_model_merging=True,
                        target_average_embedding=current_embedding,
                    )

                    # Weights for all adapting on all target domain are appened for each index
                    for domain, weight in weight_dict.items():
                        weights[idx].setdefault(domain, []).append(weight)

                    with ExitStack() as stack:
                        if isinstance(self.current_model, nn.Module):
                            stack.enter_context(inference_context(self.current_model))
                        stack.enter_context(torch.no_grad())

                        outputs = self.current_model(inputs)

                        if isinstance(evaluator, SemSegEvaluator):
                            _ = evaluator.process(inputs, outputs)
                        else:
                            _ = evaluator.process_image(inputs, outputs)

                        torch.cuda.empty_cache()

                        del self.current_model

                print(f"Benchmarking on domains {current_target_domain_name}")
                result_dict = evaluator.evaluate()
                result = self._get_result_from_dict(result_dict)
                print(f"The result is {result}\n")

                results[idx].update({current_target_domain_name: result})

                if not isinstance(evaluator, SemSegEvaluator):
                    evaluator._working_dir.cleanup()

                if target_removed:
                    self._source_domains_names.append(current_target_domain_name)
                    target_removed = False

        total = time.time() - t0
        print(f"Experiment took {total} seconds to complete!")
        return results, weights


    def online_centroid_merge(
        self,
        remove_target_adapter: bool = False,
        full_model_merging: bool = False,
    ):
        results = {idx: {} for idx in INDECIES}
        weights = {idx: {} for idx in INDECIES}

        t0 = time.time()

        for idx in INDECIES:
            for current_target_domain_name in self._target_domain_names:

                print(f"Current Index = {idx}")
                self._set_current_target_domain(
                    current_target_domain_name,
                    self._target_domains[current_target_domain_name].args,
                    index=idx,
                )

                target_removed = False
                if remove_target_adapter == True:
                    if current_target_domain_name in self._source_domains_names:
                        target_removed = True
                        self._source_domains_names.remove(current_target_domain_name)

                data_loader = self.current_target_domain.data_loader
                evaluator = self.current_target_domain.evaluator

                model = self.current_model

                # These lines are adopted from
                # https://github.com/facebookresearch/detectron2/blob/2a420edb307c9bdf640f036d3b196bed474b8593/detectron2/evaluation/evaluator.py#L103

                evaluator.reset()

                with ExitStack() as stack:
                    if isinstance(model, nn.Module):
                        stack.enter_context(inference_context(model))
                    stack.enter_context(torch.no_grad())

                    for _, inputs in enumerate(data_loader):

                        input_path = inputs[0]["file_name"]
                                                
                        print(f"Predicting image: {input_path}")

                        current_embedding = self._embed_single_image(input_path)

                        weight_dict, merged_adpater_name = self._merge(
                            "centroid", full_model_merging, current_embedding
                        )

                        # Weights for all adapting on all target domain are appened for each index
                        for domain, weight in weight_dict.items():
                            weights[idx].setdefault(domain, []).append(weight)

                        outputs = model(inputs)

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()

                        if isinstance(evaluator, SemSegEvaluator):
                            _ = evaluator.process(inputs, outputs)
                        else:
                            _ = evaluator.process_image(inputs, outputs)

                        self.current_model.delete_adapter(merged_adpater_name)

                print(f"Benchmarking on domains {current_target_domain_name}")
                result_dict = evaluator.evaluate()
                result = self._get_result_from_dict(result_dict)
                print(f"The result is {result}\n")

                results[idx].update({current_target_domain_name: result})

                if not isinstance(evaluator, SemSegEvaluator):
                    evaluator._working_dir.cleanup()

                if target_removed:
                    self._source_domains_names.append(current_target_domain_name)
                    target_removed = False

        total = time.time() - t0
        print(f"Experiment took {total} seconds to complete!")

        return results, weights

    def miou_per_image(
        self,
        zeroshot: bool = False,
    ):
        results = {idx: {} for idx in INDECIES}
        embeddings = {idx: {} for idx in INDECIES}
        
        results = {}
        embeddings = {}

        t0 = time.time()

        for idx in INDECIES:
            for current_target_domain_name in self._target_domain_names:

                print(f"Current Index = {idx}")
                self._set_current_target_domain(
                    current_target_domain_name,
                    self._target_domains[current_target_domain_name].args,
                    index=idx,
                )

                data_loader = self.current_target_domain.data_loader
                evaluator = self.current_target_domain.evaluator
                
                if zeroshot:

                    # Benchmark zeroshot model
                    self._set_current_target_domain(
                        current_target_domain_name,
                        self._target_domains[current_target_domain_name].args,
                        index="index-1000",
                    )

                    target_args = self.current_target_domain.args

                    # TODO: This might be buggy, we have to use the target domain config
                    # to load the base model to not have label space mismatch

                    args = custom_domain_args(
                        target_args.config_file,
                        "output/benchmark_zeroshot/",
                        num_gpus=1,
                        model_path="models/model_final.pth",
                    )

                    self.current_model = self._load_base_model(
                        args, model_path="models/model_final.pth"
                    )

                model = self.current_model

                # These lines are adopted from
                # https://github.com/facebookresearch/detectron2/blob/2a420edb307c9bdf640f036d3b196bed474b8593/detectron2/evaluation/evaluator.py#L103

                evaluator.reset()
                result_dict = {}
                embed = {}

                with ExitStack() as stack:
                    if isinstance(model, nn.Module):
                        stack.enter_context(inference_context(model))
                    stack.enter_context(torch.no_grad())

                    for _, inputs in enumerate(data_loader):
                        
                        input_path = inputs[0]["file_name"]

                        print(f"Predicting image: {input_path}")
                        
                        print(f"Source domain: {self._source_domains[self._source_domains_names[0]].name}")
                        
                        adapter_embedding = self._source_domains[self._source_domains_names[0]].train_average_embedding

                        current_embedding = self._embed_single_image(input_path)

                        embed.update({input_path: {
                            'adapter_embedding': adapter_embedding,
                            'current_embedding': current_embedding,
                        }})

                        outputs = model(inputs)

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()

                        if isinstance(evaluator, SemSegEvaluator):
                            _ = evaluator.process(inputs, outputs)
                        else:
                            _ = evaluator.process_image(inputs, outputs)
                        
                        if not isinstance(evaluator, CityscapesEvaluator):
                            result_dict.update({input_path: evaluator.evaluate()})
    
                print(f"Benchmarking on domains {current_target_domain_name}")
                if not isinstance(evaluator, SemSegEvaluator):
                    result_dict = evaluator.evaluate_image()

                results.update({current_target_domain_name: result_dict})
                embeddings.update({current_target_domain_name: embed})

        total = time.time() - t0
        print(f"Experiment took {total} seconds to complete!")

        return results, embeddings
    
    def miou_per_image_merged(
            self,
            remove_target_adapter: bool = False,
        ):

            results = {idx: {} for idx in INDECIES}
            embeddings = {idx: {} for idx in INDECIES}
            
            results = {}
            embeddings = {}

            t0 = time.time()

            for idx in INDECIES:
                for current_target_domain_name in self._target_domain_names:

                    print(f"Current Index = {idx}")
                    self._set_current_target_domain(
                        current_target_domain_name,
                        self._target_domains[current_target_domain_name].args,
                        index=idx,
                    )

                    target_removed = False
                    if remove_target_adapter == True:
                        if current_target_domain_name in self._source_domains_names:
                            target_removed = True
                            self._source_domains_names.remove(current_target_domain_name)

                    data_loader = self.current_target_domain.data_loader
                    evaluator = self.current_target_domain.evaluator

                    model = self.current_model

                    # These lines are adopted from
                    # https://github.com/facebookresearch/detectron2/blob/2a420edb307c9bdf640f036d3b196bed474b8593/detectron2/evaluation/evaluator.py#L103

                    evaluator.reset()
                    result_dict = {}
                    embed = {}

                    with ExitStack() as stack:
                        if isinstance(model, nn.Module):
                            stack.enter_context(inference_context(model))
                        stack.enter_context(torch.no_grad())

                        for _, inputs in enumerate(data_loader):
                            
                            input_path = inputs[0]["file_name"]

                            print(f"Predicting image: {input_path}")
                            
                            print(f"Source domain: {self._source_domains[self._source_domains_names[0]].name}")
                            
                            adapter_embedding = self._source_domains[self._source_domains_names[0]].train_average_embedding

                            current_embedding = self._embed_single_image(input_path)
                            
                            weight_dict, merged_adpater_name = self._merge(
                                "centroid", False, current_embedding
                            )

                            embed.update({input_path: {
                                'adapter_embedding': adapter_embedding,
                                'current_embedding': current_embedding,
                            }})

                            outputs = model(inputs)

                            if torch.cuda.is_available():
                                torch.cuda.synchronize()

                            if isinstance(evaluator, SemSegEvaluator):
                                _ = evaluator.process(inputs, outputs)
                            else:
                                _ = evaluator.process_image(inputs, outputs)
                            
                            if not isinstance(evaluator, CityscapesEvaluator):
                                result_dict.update({input_path: evaluator.evaluate()})
                            
                            self.current_model.delete_adapter(merged_adpater_name)
        
                    print(f"Benchmarking on domains {current_target_domain_name}")
                    if not isinstance(evaluator, SemSegEvaluator):
                        result_dict = evaluator.evaluate_image()

                    results.update({current_target_domain_name: result_dict})
                    embeddings.update({current_target_domain_name: embed})
                    
                    if target_removed:
                        self._source_domains_names.append(current_target_domain_name)
                        target_removed = False

            total = time.time() - t0
            print(f"Experiment took {total} seconds to complete!")

            return results, embeddings
    
    def image_prediction(
        self,
        weights: Optional[list[float]] = None,
        full_model_merging: bool = False,
        max_images: int = 10,
    ):
        for idx in INDECIES:
            for current_target_domain_name in self._target_domain_names:

                print(f"Current Index = {idx}")
                self._set_current_target_domain(
                    current_target_domain_name,
                    self._target_domains[current_target_domain_name].args,
                    index=idx,
                )

                data_loader = self.current_target_domain.data_loader
                evaluator = self.current_target_domain.evaluator

                model = self.current_model
                predictions = []

                # These lines are adopted from
                # https://github.com/facebookresearch/detectron2/blob/2a420edb307c9bdf640f036d3b196bed474b8593/detectron2/evaluation/evaluator.py#L103

                evaluator.reset()

                with ExitStack() as stack:
                    if isinstance(model, nn.Module):
                        stack.enter_context(inference_context(model))
                    stack.enter_context(torch.no_grad())

                    for i, inputs in enumerate(data_loader):
                        if i >= max_images:
                            break
                        input_path = inputs[0]["file_name"]                      
                        current_embedding = self._embed_single_image(input_path)

                        weight_dict, _ = self._merge(
                            "centroid",full_model_merging,current_embedding,weights
                        )
                        prediction = model(inputs)
                        predictions.append({
                            "semseg": prediction[0]["sem_seg"],
                            "file_name": input_path,
                            "weights": weight_dict
                        })

        return predictions

    def online_centroid_threshold_merge(
        self,
        test_domains: list[str],
        data_loaders,
        evaluators,
        include_target_adapter: bool = True,
    ):
        window = deque()

        latest_embedding = None
        latest_not_set = True  # Indicates whether a latest embedding is set or not

        all_results = {}
        all_weights = []

        for current_target_domain_name in test_domains:

            merged_adpater_name = None

            data_loader = data_loaders[current_target_domain_name]
            evaluator = evaluators[current_target_domain_name]

            target_removed = False
            if include_target_adapter == False:
                if current_target_domain_name in self._source_domains_names:
                    self._source_domains_names.remove(current_target_domain_name)
                    target_removed = True

            model = self.current_model

            # These lines are adopted from
            # https://github.com/facebookresearch/detectron2/blob/2a420edb307c9bdf640f036d3b196bed474b8593/detectron2/evaluation/evaluator.py#L103

            evaluator.reset()

            with ExitStack() as stack:

                if isinstance(model, nn.Module):
                    stack.enter_context(inference_context(model))
                stack.enter_context(torch.no_grad())

                for idx, inputs in enumerate(data_loader):

                    input_path = inputs[0]["file_name"]

                    print(input_path)

                    new_embedding = self._embed_single_image(input_path)

                    current_embedding = self._sliding_average(new_embedding, window)

                    if latest_not_set is True:
                        latest_embedding = current_embedding
                        latest_not_set = False

                    # Set the target domain to test on
                    target_domain = current_target_domain_name
                    print(f"current target domain: {target_domain}")
                    # self.set_current_target_domain(target_domain)

                    if self.observer.domain_shifted(
                        current_embedding=current_embedding,
                        lastest_embedding=latest_embedding,
                        threshold=self.distance_thresh,
                        # domain=self._domains[current_target_domain],
                    ):
                        print("Domain shifted!")

                        if merged_adpater_name is not None:
                            self.current_model.delete_adapter(merged_adpater_name)

                        latest_embedding = current_embedding

                        weight_dict, merged_adpater_name = self._centroid_merge(
                            current_embedding, current_target_domain_name
                        )

                        all_weights.append(weight_dict)

                    outputs = model(inputs)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    evaluator.process_image(inputs, outputs)

            print(f"Benchmarking on domains {current_target_domain_name}")
            results = evaluator.evaluate()
            print(f"Result: {results}")
            current_result = results["sem_seg"]["IoU"]

            all_results.update({current_target_domain_name: current_result})

            evaluator._working_dir.cleanup()

            if target_removed:
                self._source_domains_names.append(current_target_domain_name)
                target_removed = False

        return (
            all_results,
            all_weights,
        )

    def _batch_merge(
        self,
        weight_type: Literal["uniform", "centroid"],
        remove_target_adapter: bool = False,
        full_model_merging: bool = False,
    ):

        results = {idx: {} for idx in INDECIES}
        weights = {idx: {} for idx in INDECIES}

        for current_target_domain_name in self._target_domain_names:
            for idx in INDECIES:

                print(f"Current Index = {idx}")
                self._set_current_target_domain(
                    current_target_domain_name,
                    self._target_domains[current_target_domain_name].args,
                    index=idx,
                )

                target_removed = False
                if remove_target_adapter == True:
                    if current_target_domain_name in self._source_domains_names:
                        print(
                            f"Removing {current_target_domain_name} from the list of source domains."
                        )
                        print(self._source_domains_names)
                        self._source_domains_names.remove(current_target_domain_name)
                        print(f"Remaining domains: {self._source_domains_names}")
                        target_removed = True

                if full_model_merging:
                    self.current_model_copy = self._load_base_model(
                        self.current_target_domain.args
                    )
                
                weight_dict, merged_adpater_name = self._merge(
                    weight_type, full_model_merging
                )

                weights[idx].update({current_target_domain_name: weight_dict})

                result_dict = self._benchmark_on_current_target_domain(
                    name=merged_adpater_name
                )

                print(result_dict)

                result = self._get_result_from_dict(result_dict)

                results[idx].update(
                    {
                        current_target_domain_name: result
                    }  # TODO: Different datasets have different evaluation methods so this won't always work
                )

                if not full_model_merging:
                    # Delete the adapter so we can add another with the same name but different weights (remove unused adapters)
                    print(f"Deleting adapter {merged_adpater_name}.")
                    self.current_model.delete_adapter(merged_adpater_name)

                if target_removed:
                    print(
                        f"Adding {current_target_domain_name} back to the list of source domains."
                    )
                    self._source_domains_names.append(current_target_domain_name)
                    target_removed = False

                print("\n\n\n")

        return results, weights


    def _merge(
        self,
        weight_type: Literal["uniform", "centroid"],
        full_model_merging: bool = False,
        target_average_embedding = None,
        weights: list[float] = None,
    ):

        if target_average_embedding is None:
            target_average_embedding = self.current_target_domain.val_average_embedding

        domain_distance_mapping = self.observer.calcualte_distance_to_domains(
            target_average_embedding,
            [
                self._source_domains[domain_name]
                for domain_name in self._source_domains_names
            ],  # Important to iterate over _source_domains_names and not _source_domains since target domain might be removed from the the source_domain_names but not from _source_domains
        )

        k_closest_distances = list(domain_distance_mapping.values())[: self.k_adapters]
        k_closest_names = list(domain_distance_mapping.keys())[: self.k_adapters]

        print(f"Distances to {self.k_adapters} closest domains: ")
        for n, d in zip(k_closest_names, k_closest_distances):
            print(f"{n}: {d}", end=", ")
        print("")

        if weights is None:
            if weight_type == "uniform":
                current_k = len(k_closest_distances)
                weights = [1 / current_k for _ in range(current_k)]
            elif weight_type == "centroid":
                weights = self._calcualte_adapter_weights(k_closest_distances)

        # print(f"Weights of {self.k_adapters} closest domains: ")
        # for n, w in zip(k_closest_names, weights):
        #     print(f"{n}: {w}", end=", ")
        # print("")

        weights_dict = {
            k_closest_name: weight
            for k_closest_name, weight in zip(k_closest_names, weights)
        }

        merged_name = ""
        for n, w in zip(k_closest_names, weights):
            merged_name += f"_{n}_{str(w).replace('.','_')}"
        merged_name += f"_{self.merge_type}_{self.current_target_domain.name}"

        if not full_model_merging:
            self._merge_adapters(
                merge_domains=k_closest_names,
                weights=weights,
                merged_name=merged_name,
            )

            print(f"Setting {merged_name} as the active adapter.\n")
            self.current_model.set_adapter(merged_name)

        else:
            self._merge_models(merge_domains=k_closest_names, weights=weights)

        return weights_dict, merged_name

    def _benchmark_on_current_target_domain(self, name: str):
        print(
            f"Benchmarking {name} on the domain {self.current_target_domain.name} ...\n"
        )
        res = benchmark(self.current_model, self.current_target_domain.args)
        return res

    def _get_result_from_dict(self, result_dict: Mapping):

        res = result_dict["sem_seg"].get("IoU", None)
        if res is None:
            res = result_dict["sem_seg"].get("mIoU")
        return res

    # def _get_domain_args(
    #     self, dataset: str, domain: str, sub_domain: Union[str, None] = None
    # ):
    #     args = None

    #     if dataset == "cityscapes":
    #         if "normal" in domain:
    #             args = domain_args("cityscapes", "normal", split="val")
    #         else:
    #             args = domain_args("cityscapes", "rain", sub_domain, split="val")
    #     elif dataset == "muses":
    #         args = domain_args("muses", domain.split("-")[1], sub_domain, split="val")
    #     elif dataset == "acdc":
    #         args = domain_args("acdc", domain.split("-")[1], split="val")

    #     return args

    def _calcualte_adapter_weights(self, distances):

        distances_inverted = [1 / distance for distance in distances]
        weights = softmax(distances_inverted, self.temp)

        return weights

    def _add_domain(
        self,
        domain_name: str,
        args: Namespace,
        evaluator,
        data_loader,
        source: bool,  # whether it is a source domain or a target domain
        lora_path: Union[str, Path, None] = None,
        index: Union[str, None] = None,
        save_embeddings_path=None,
    ) -> Domain:
        """Adds a Domain instance to the domains list."""

        # TODO: Dark zurich can't be trained on, only save the validation embeddings!

        train_dataset_path = Path(args.train_dataset_path)
        assert train_dataset_path.exists(), train_dataset_path

        val_dataset_path = Path(args.val_dataset_path)
        assert val_dataset_path.exists(), val_dataset_path

        model_path = Path(args.model_path)

        domain = None

        # if source == True:
        if lora_path is None:
            lora_path = self.lora_db_path / domain_name
        if index is not None:
            lora_path = lora_path.parent / (lora_path.name + "_" + index)

        statistics: Dict[str, np.typing.NDArray] = self._calculate_statistics(
            domain_name, lora_path, train_dataset_path, val_dataset_path
        )

        train_average_embedding: np.typing.NDArray = statistics[
            "train_average_embedding"
        ]
        val_average_embedding: np.typing.NDArray = statistics["val_average_embedding"]
        train_dataset_embeddings: np.typing.NDArray = statistics[
            "train_dataset_embeddings"
        ]
        val_dataset_embeddings: np.typing.NDArray = statistics["val_dataset_embeddings"]
        # standard_deviation: np.typing.NDArray = statistics["standard_deviation"]
        # covarinace_matrix: np.typing.NDArray = statistics["covariance_matrix"]
        # inverse_covariance_matrix: np.typing.NDArray = statistics[
        #     "inverse_covariance_matrix"
        # ]

        if not self.no_lora:
            assert lora_path.exists(), lora_path
            print(f"Loading LoRA from: {lora_path}")
            self._load_lora(domain_name, lora_path)
            print("LoRA Loaded.")

        domain = Domain(
            domain_name,
            args,
            train_dataset_path=train_dataset_path,
            val_dataset_path=val_dataset_path,
            lora_path=lora_path,
            train_dataset_embeddings=train_dataset_embeddings,
            val_dataset_embeddings=val_dataset_embeddings,
            train_average_embedding=train_average_embedding,
            val_average_embedding=val_average_embedding,
            evaluator=evaluator,
            data_loader=data_loader,
            model_path=model_path,
            # standard_deviation=standard_deviation,
            # covariance_matrix=covarinace_matrix,
            # inverse_covariance_matrix=inverse_covariance_matrix,
        )

        # else:  # Target domain
        #     if save_embeddings_path is not None:
        #         save_embeddings_path = Path(save_embeddings_path)
        #         embeddings_path = (
        #             save_embeddings_path / f"{domain_name}_dataset_embeddings.npz"
        #         )

        #         if embeddings_path.exists():
        #             dataset_embeddings = np.load(embeddings_path)["dataset_embeddings"]
        #             print(f"Target {domain_name} dataset embeddings loaded.")
        #         else:
        #             dataset_embeddings = np.array(
        #                 self._embed_dataset(
        #                     domain_name,
        #                     dataset_path,
        #                 )
        #             )
        #             np.savez(
        #                 save_embeddings_path / f"{domain_name}_dataset_embeddings.npz",
        #                 dataset_embeddings=dataset_embeddings,
        #             )
        #             print(f"Target {domain_name} dataset embeddings saved.")

        #     else:
        #         dataset_embeddings = np.array(
        #             self._embed_dataset(
        #                 domain_name,
        #                 dataset_path,
        #             )
        #         )

        #     domain = Domain(
        #         name=domain_name,
        #         args=args,
        #         dataset_path=dataset_path,
        #         dataset_embeddings=dataset_embeddings,
        #     )

        self._domains.update({domain_name: domain})

        print(f"Added domain {domain_name}. \n")

        return domain

    def _load_lora(self, domain_name: str, lora_path: Path):
        if self.current_model is None:
            # Wrap the model in PeftModel class the first time an adapter is loaded
            self.current_model = peft.PeftModel.from_pretrained(
                self._base_model, lora_path, domain_name
            )
        else:
            self.current_model.load_adapter(lora_path, domain_name)

    def _set_active_lora(self, domain_name):
        assert (
            domain_name in self._source_domains_names
        ), f"Can't activate a LoRA for domain '{domain_name}' because it is not one of the source domains!"
        self.current_model.set_adapter(domain_name)

    def _set_current_target_domain(
        self, domain_name: str, domain_args: Namespace, index: str
    ):

        # We need to load all adapters each time the target domain changes because the same config can't be used across datasets
        # and PEFT does not allow us to change the base model and keep the loaded adapters
        print(f"Setting current target domain to {domain_name}.\n")

        self._base_model = self._load_base_model(domain_args)
        self.current_model = None  # This will ensure that the current PEFT model will be initialized using base model with new config
        self._source_domains = self._add_source_domains(
            self._source_domains_names, index=index
        )
        self.current_target_domain = self._target_domains[domain_name]

    def _merge_adapters(
        self,
        merge_domains: list[str],
        weights: list[float],
        merged_name: str,
    ):
        print(f"Merging domains with weights:")
        for n, w in zip(merge_domains, weights):
            print(f"{n}: {w}", end=", ")
        print("")

        self.current_model.add_weighted_adapter(
            merge_domains,
            weights,
            merged_name,
            combination_type=self.merge_type,
            density=self.pruning_density,
        )

    def _merge_models(self, merge_domains: list[str], weights: list[float]) -> dict:

        print(f"Merging models with weights:")
        for n, w in zip(merge_domains, weights):
            print(f"{n}: {w}", end=", ")
        print("")

        agg_state = self.current_model_copy

        for par_name, _ in agg_state.named_parameters():
            agg_state.get_parameter(par_name).data = torch.stack(
                [
                    self._models[domain][par_name].data * weight
                    for domain, weight in zip(merge_domains, weights)
                ]
            ).sum(axis=0)

        self.current_model = agg_state
        self.current_model.to("cuda:0")

        # del models
        # del agg_state

    def _get_domain(self, name: str) -> Union[Domain, None]:
        """Retrieves a Domain by name. Returns None if not found."""
        for domain in self._domains:
            if domain.name == name:
                return domain
        return None

    def _load_base_model(self, args, model_path: str = None):
        print("Loading base model ...")

        cfg = setup(args)

        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS if model_path is None else model_path, resume=args.resume
        )
        print("Base model loaded.\n")
        return model

    def _calculate_statistics(
        self,
        domain_name: str,
        lora_path: Path,
        train_dataset_path: Path,
        val_dataset_path: Path,
    ) -> Dict[str, torch.Tensor]:

        stats_dict = {}
        stats_path = lora_path / f"{domain_name}_statistics.npz"

        if stats_path.exists():
            print(f"Loading statistics from {domain_name}_statistics.npz.")
            stats = np.load(lora_path / f"{domain_name}_statistics.npz")
            stats_dict.update(
                {"train_average_embedding": stats["train_average_embedding"]}
            )
            stats_dict.update({"val_average_embedding": stats["val_average_embedding"]})
            stats_dict.update(
                {"train_dataset_embeddings": stats["train_dataset_embeddings"]}
            )
            stats_dict.update(
                {"val_dataset_embeddings": stats["val_dataset_embeddings"]}
            )
            # stats_dict.update({"standard_deviation": stats["standard_deviation"]})
            # stats_dict.update({"covariance_matrix": stats["covariance_matrix"]})
            # stats_dict.update(
            #     {"inverse_covariance_matrix": np.linalg.inv(stats["covariance_matrix"])}
            # )
            print(f"Statistics loaded.")

        else:
            print(f"Calculating statistics for domain '{domain_name}' ...")

            train_dataset_embeddings = np.array(
                self._embed_dataset(domain_name, train_dataset_path)
            )

            val_dataset_embeddings = np.array(
                self._embed_dataset(domain_name, val_dataset_path)
            )

            stats_dict.update({"train_dataset_embeddings": train_dataset_embeddings})
            stats_dict.update({"val_dataset_embeddings": val_dataset_embeddings})

            train_average_embedding = self._average_embedding(
                domain_name, train_dataset_embeddings
            )
            val_average_embedding = self._average_embedding(
                domain_name, val_dataset_embeddings
            )

            stats_dict.update({"train_average_embedding": train_average_embedding})
            stats_dict.update({"val_average_embedding": val_average_embedding})

            # standard_deviation = self._standard_deviation(
            #     domain_name, dataset_embeddings
            # )
            # stats_dict.update({"standard_deviation": standard_deviation})

            # covariance_matrix = self._covariance_matrix(domain_name, dataset_embeddings)

            # stats_dict.update({"covariance_matrix": covariance_matrix})
            # stats_dict.update(
            #     {"inverse_covariance_matrix": np.linalg.inv(covariance_matrix)}
            # )

            np.savez(
                lora_path / f"{domain_name}_statistics.npz",
                train_average_embedding=train_average_embedding,
                val_average_embedding=val_average_embedding,
                train_dataset_embeddings=train_dataset_embeddings,
                val_dataset_embeddings=val_dataset_embeddings,
                # standard_deviation=standard_deviation,
                # covariance_matrix=covariance_matrix,
            )

            print(f"Saving statistics to {domain_name}_statistics.npz.")

        return stats_dict

    def _sliding_average(
        self,
        embdng: list[np.typing.NDArray],
        window: list[np.typing.NDArray],
    ):
        if self.window_size is not None:
            window.append(embdng)
            if len(window) > self.window_size:
                window.popleft()
            current_embedding = np.mean(window, axis=0)
            return current_embedding
        else:
            current_embedding = embdng
            return current_embedding

    def _average_embedding(self, domain_name: str, dataset_embeddings):
        print(f"Calculating average embedding for domain {domain_name} ...")
        average_embedding = np.array(dataset_embeddings).mean(axis=0)
        print("Finished calcualting average embedding.")
        return average_embedding

    def _standard_deviation(self, domain_name: str, dataset_embeddings, axis=None):
        print(f"Calculating standard deviation for domain {domain_name} ...")
        std = np.array(dataset_embeddings).std(axis=axis)
        print("Finished calcualting standard deviation.")
        return std

    def _covariance_matrix(self, domain_name: str, dataset_embeddings):
        print(f"Calculating covariance matrix for domain {domain_name} ...")
        cov_mtrx = np.cov(np.array(dataset_embeddings).squeeze().T, bias=True)
        print("Finished calcualting covariance matrix.")
        return cov_mtrx

    def _embed_dataset(
        self,
        domain_name: str,
        dataset_path: Path,
    ):
        assert dataset_path.exists()
        print(f"Embeddeing dataset of domain '{domain_name}' ...")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        dataset_embeddings = []

        # IDD has both png and jpg images in train set
        image_files = list(dataset_path.rglob("*.png"))
        image_files += list(dataset_path.rglob("*.jpg"))

        for img in image_files:

            embedding = self._embed_image(img, model, processor).numpy()

            # print(f"Embedded image: {img}")

            # TODO: use generator in case embeddings are large, currently not working
            # yield embedding

            dataset_embeddings.append(embedding)
        print("Finished embedding dataset")

        return dataset_embeddings

    def _embed_image(self, image_path, model, processor):
        # Load and process the image
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to("cuda")

        # Generate image embeddings
        image_embeddings = model.get_image_features(**inputs).detach().cpu()
        return image_embeddings

    def _embed_single_image(self, image_path):
        # Load and process the image
        image = Image.open(image_path).convert("RGB")
        inputs = self.embedding_processor(images=image, return_tensors="pt").to("cuda")

        # Generate image embeddings
        image_embedding = (
            self.embedding_model.get_image_features(**inputs).detach().cpu()
        )
        return image_embedding
