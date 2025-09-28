from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol

from torch import nn


@dataclass(slots=True)
class DomainResources:
    """Container holding adapter specific resources for a domain."""

    args: Namespace
    train_dataset_path: Path | None
    val_dataset_path: Path | None
    evaluator: Any | None = None
    data_loader: Any | None = None


class ModelAdapter(Protocol):
    """Protocol that model-specific adapters must implement."""

    def setup(self) -> None:
        """Perform any one-time environment setup required by the model."""

    def get_domain_resources(
        self,
        domain_name: str,
        split: Literal["train", "val"],
        *,
        load_data: bool = True,
    ) -> DomainResources:
        """Return configuration and dataloading artefacts for a particular domain."""

    def load_base_model(
        self,
        args: Namespace,
        model_path: str | None = None,
    ) -> nn.Module:
        """Instantiate and load the underlying base model for the provided args."""

    def benchmark(self, model: nn.Module, args: Namespace) -> Mapping[str, Any]:
        """Evaluate *model* using the configuration encoded in *args*."""

    def build_eval_args(
        self,
        *,
        config_file: str,
        output_path: str,
        num_gpus: int = 1,
        model_path: str = "models/model_final.pth",
        dataset_path: str | None = None,
        seed: int | None = None,
    ) -> Namespace:
        """Construct arguments for running an evaluation over a given config."""
