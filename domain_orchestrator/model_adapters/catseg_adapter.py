from __future__ import annotations

import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Literal, Mapping

from torch import nn

from domain_orchestrator.model_adapters.protocol import DomainResources, ModelAdapter
from domain_orchestrator.utils import get_device


class CatSegAdapter(ModelAdapter):
    """Model adapter that exposes CAT-Seg to the SemLA orchestrator."""

    _LOGGER_NAMES = ("detectron2", "d2", "fvcore")

    def __init__(
        self,
        *,
        catseg_root: Path | None = None,
        dataset_root: str | Path | None = None,
        default_mode: str = "lora",
        default_base_model_path: str = "models/model_final.pth",
        default_num_gpus: int = 1,
    ) -> None:
        self.catseg_root = (
            Path(catseg_root)
            if catseg_root is not None
            else Path(__file__).resolve().parent.parent.parent / "catseg"
        )
        dataset_root = dataset_root if dataset_root is not None else os.getenv("DETECTRON2_DATASETS", "")
        dataset_root = Path(dataset_root)
        self.dataset_root = dataset_root
        root_str = os.fspath(dataset_root)
        if root_str and not root_str.endswith(os.sep):
            root_str = f"{root_str}{os.sep}"
        self._dataset_root_prefix = root_str
        self.default_mode = default_mode
        self.default_base_model_path = default_base_model_path
        self.default_num_gpus = default_num_gpus
        self._is_setup = False

    def setup(self) -> None:
        if self._is_setup:
            return

        catseg_path = self.catseg_root
        print(f"Changing directory to '{catseg_path}' ...")
        try:
            os.chdir(catseg_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"The specified CAT-Seg path '{catseg_path}' does not exist.") from exc
        except PermissionError as exc:
            raise PermissionError(f"Insufficient permissions to access '{catseg_path}'.") from exc
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Unexpected error while changing directory to '{catseg_path}': {exc}") from exc

        self._is_setup = True

    def get_domain_resources(
        self,
        domain_name: str,
        split: Literal["train", "val"],
        *,
        load_data: bool = True,
    ) -> DomainResources:
        self.setup()
        self._reset_detectron_loggers()

        dataset, domain, sub_domain = self._parse_domain_name(domain_name)
        mode = self.default_mode

        config_file = self._resolve_config(dataset, domain, sub_domain, mode)
        train_dataset_path_str, val_dataset_path_str = self._resolve_dataset_paths(dataset, domain, sub_domain)
        output_path = self._build_output_path(dataset, domain, sub_domain, mode)

        args = Namespace(
            config_file=config_file,
            eval_only=True,
            num_gpus=self.default_num_gpus,
            train_dataset_path=train_dataset_path_str,
            val_dataset_path=val_dataset_path_str,
            opts=[
                "OUTPUT_DIR",
                output_path,
                "TEST.SLIDING_WINDOW",
                "True",
                "MODEL.SEM_SEG_HEAD.POOLING_SIZES",
                "[1,1]",
                "MODEL.WEIGHTS",
                self.default_base_model_path,
            ],
            resume=True,
        )

        train_dataset_path = Path(train_dataset_path_str) if train_dataset_path_str else None
        val_dataset_path = Path(val_dataset_path_str) if val_dataset_path_str else None

        evaluator: Any | None = None
        data_loader: Any | None = None

        if load_data:
            if train_dataset_path is None or not train_dataset_path.exists():
                raise FileNotFoundError(
                    f"Path to training dataset '{train_dataset_path}' does not exist!"
                )

            from catseg.train_net import Trainer, setup

            cfg = setup(args)
            data_loader = Trainer.build_test_loader(
                cfg, f"{domain_name}_sem_seg_{split}"
            )
            evaluator = Trainer.build_evaluator(
                cfg, f"{domain_name}_sem_seg_{split}"
            )

        return DomainResources(
            args=args,
            train_dataset_path=train_dataset_path,
            val_dataset_path=val_dataset_path,
            evaluator=evaluator,
            data_loader=data_loader,
        )

    def load_base_model(
        self,
        args: Namespace,
        model_path: str | None = None,
    ) -> nn.Module:
        self.setup()
        from catseg.train_net import Trainer, setup
        from detectron2.checkpoint import DetectionCheckpointer

        print("Loading base model ...")
        try:
            cfg = setup(args)
            cfg.defrost()
            cfg.MODEL.DEVICE = get_device()
            cfg.freeze()
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS if model_path is None else model_path, resume=args.resume
            )
            print("Base model loaded.\n")
            return model
        except AttributeError as exc:
            raise AttributeError(f"Invalid model configuration: {exc}") from exc
        except FileNotFoundError as exc:
            raise FileNotFoundError("Model weights not found at the specified path.") from exc
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Unexpected error while loading the base model: {exc}") from exc

    def benchmark(self, model: nn.Module, args: Namespace) -> Mapping[str, Any]:
        self.setup()
        import detectron2.utils.comm as comm
        from detectron2.evaluation import verify_results
        from catseg.train_net import Trainer, set_random_seed, setup

        cfg = setup(args)
        set_random_seed(cfg.SEED)
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

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
        self.setup()

        args = Namespace(
            config_file=config_file,
            eval_only=True,
            num_gpus=num_gpus,
            dataset_path=dataset_path,
            opts=[
                "OUTPUT_DIR",
                output_path,
                "TEST.SLIDING_WINDOW",
                "True",
                "MODEL.SEM_SEG_HEAD.POOLING_SIZES",
                "[1,1]",
                "MODEL.WEIGHTS",
                model_path,
            ],
            resume=True,
            model_path=model_path,
        )

        if seed is not None:
            args.opts.extend(["SEED", seed])

        return args

    def _reset_detectron_loggers(self) -> None:
        for name in self._LOGGER_NAMES:
            logger = logging.getLogger(name)
            if logger.hasHandlers():
                logger.handlers.clear()

    def _parse_domain_name(self, domain_name: str) -> tuple[str, str, str]:
        parts = domain_name.split("-")
        split_list = ["", "", ""]
        for index, part in enumerate(parts[:3]):
            split_list[index] = part
        return tuple(split_list)  # type: ignore[return-value]

    def _resolve_config(self, dataset: str, domain: str, sub_domain: str, mode: str) -> str:
        mode_check = {"lora"}
        if mode not in mode_check:
            raise AssertionError(f"Mode '{mode}' not supported")

        dataset_check = {
            "cs",
            "acdc",
            "muses",
            "bdd",
            "mv",
            "a150",
            "idd",
            "pc59",
            "nyu",
            "coconutL",
        }
        if dataset not in dataset_check:
            raise AssertionError(f"Dataset '{dataset}' not supported")

        cs_domain_check = {"normal", "rain"}
        cs_sub_domain_check = {"25mm", "50mm", "75mm", "100mm", "200mm"}
        acdc_domain_check = {"fog", "night", "snow", "rain"}
        muses_domain_check = {"clear", "rain", "fog", "snow"}
        muses_sub_domain_check = {"day", "night"}

        if dataset == "cs":
            if domain not in cs_domain_check:
                raise AssertionError(f"Domain '{domain}' not supported for Cityscapes")
            if domain == "rain" and sub_domain not in cs_sub_domain_check:
                raise AssertionError(
                    f"Given volume '{sub_domain}' is not supported for Cityscapes"
                )
            if domain == "normal" and sub_domain != "":
                raise AssertionError(
                    f"Volume '{sub_domain}' is not supported for this domain in Cityscapes"
                )
        elif dataset == "muses":
            if domain not in muses_domain_check:
                raise AssertionError(f"Domain '{domain}' not supported for MUSES")
            if sub_domain not in muses_sub_domain_check:
                raise AssertionError(
                    f"Given illumination '{sub_domain}' is not supported for MUSES"
                )
        elif dataset == "acdc":
            if domain not in acdc_domain_check:
                raise AssertionError(f"Domain '{domain}' is not supported for ACDC")
            if sub_domain != "":
                raise AssertionError("Volume is not supported in ACDC")

        configs: dict[str, Any] = {
            "cs": {
                "rain": f"configs/cityscapes/rain/{sub_domain}/{mode}-{domain}-{sub_domain}.yaml",
                "normal": f"configs/cityscapes/normal/{mode}-{domain}.yaml",
            },
            "acdc": {f"{domain}": f"configs/acdc/{domain}/{mode}-{domain}-acdc.yaml"},
            "muses": {
                f"{domain}": f"configs/muses/{domain}/muses-{domain}-{sub_domain}.yaml"
            },
            "bdd": "configs/bdd/bdd.yaml",
            "mv": "configs/mv/mv.yaml",
            "nyu": "configs/nyu/nyu.yaml",
            "a150": "configs/a150/a150.yaml",
            "idd": "configs/idd/idd.yaml",
            "pc59": "configs/pc59/pc59.yaml",
            "coconutL": "configs/coconutL/coconutL.yaml",
        }

        if domain == "" and sub_domain == "":
            config_file = configs[dataset]
        else:
            config_file = configs[dataset][domain]
        return config_file  # type: ignore[return-value]

    def _resolve_dataset_paths(
        self,
        dataset: str,
        domain: str,
        sub_domain: str,
    ) -> tuple[str, str]:
        root = self._dataset_root_prefix

        datasets = {
            "cs": {
                "normal": {
                    "train": f"{root}cityscapes/leftImg8bit/train/",
                    "val": f"{root}cityscapes/leftImg8bit/val/",
                },
                "rain": {
                    "train": f"{root}cityscapes/leftImg8bit/train/",
                    "val": f"{root}cityscapes/leftImg8bit/val/",
                },
            },
            "acdc": {
                "train": f"{root}acdc/rgb_anon/{domain}/train/",
                "val": f"{root}acdc/rgb_anon/{domain}/val/",
            },
            "muses": {
                "train": f"{root}muses/frame_camera/train/{domain}/{sub_domain}/",
                "val": f"{root}muses/frame_camera/val/{domain}/{sub_domain}/",
            },
            "bdd": {
                "train": f"{root}bdd100k/images/10k/train/",
                "val": f"{root}bdd100k/images/10k/val/",
            },
            "mv": {
                "train": f"{root}mapillary_vistas/train/images/",
                "val": f"{root}mapillary_vistas/val/images/",
            },
            "a150": {
                "train": f"{root}ADE20k/images/training/",
                "val": f"{root}ADE20k/images/validation/",
            },
            "idd": {
                "train": f"{root}IDD_Segmentation/leftImg8bit/train/",
                "val": f"{root}IDD_Segmentation/leftImg8bit/val/",
            },
            "pc59": {
                "train": f"{root}pascal_ctx_d2/images/training",
                "val": f"{root}pascal_ctx_d2/images/validation",
            },
            "nyu": {
                "train": f"{root}nyudv2_splitted/train/rgb",
                "val": f"{root}nyudv2_splitted/test/rgb",
            },
            "coconutL": {
                "train": f"{root}coconut-l/train2017/",
                "val": f"{root}coconut-l/val2017",
            },
        }

        if dataset == "cs":
            dataset_entry = datasets[dataset][domain]
            train_dataset_path = dataset_entry["train"]
            val_dataset_path = dataset_entry["val"]
        else:
            dataset_entry = datasets[dataset]
            train_dataset_path = dataset_entry["train"]
            val_dataset_path = dataset_entry["val"]

        return train_dataset_path, val_dataset_path

    @staticmethod
    def _build_output_path(dataset: str, domain: str, sub_domain: str, mode: str) -> str:
        suffix = ""
        if domain:
            suffix += f"-{domain}"
        if sub_domain:
            suffix += f"-{sub_domain}"
        return f"output/{dataset}/{mode}-{dataset}{suffix}/eval/"
