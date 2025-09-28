import os

import torch
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.utils.comm as comm
from detectron2.evaluation import verify_results


import peft

# from cat_seg.third_party.model_vpt import PlainMultiHeadAttention
from train_net import Trainer, setup

import wandb

from argparse import Namespace
# logging.disable()

torch.set_float32_matmul_precision("high")  

def get_domain_config(
    dataset: str,
    domain: str,
    volume: str = "",
    mode: str = "lora",
    base_model_path: str = "models/model_final.pth",
    num_gpus: int = 1,
):
    # Supported configurations
    DATASET_CHECK = {"cityscapes", "acdc"}
    CS_DOMAIN_CHECK = {"normal", "rain"}
    ACDC_DOMAIN_CHECK = {"fog", "night", "snow", "rain"}
    CS_VOLUME_CHECK = ["25mm", "50mm", "75mm", "100mm", "200mm"]
    MODE_CHECK = {"lora", "finetune", "zeroshot"}

    # Configurations assertions
    assert dataset in DATASET_CHECK

    if dataset == "cityscapes":
        assert (
            domain in CS_DOMAIN_CHECK
        ), "Domain '{domain}' not supported for Cityscapes"
        if domain == "rain":
            assert (
                volume in CS_VOLUME_CHECK
            ), "Given volume '{volume}' is not supported for Cityscapes"
        elif domain == "normal":
            assert volume == "", "Volume is not supported for this domain in Cityscapes"
    elif dataset == "acdc":
        assert (
            domain in ACDC_DOMAIN_CHECK
        ), "Domain '{domain}' is not supported for ACDC"
        assert volume == "", "Volume is not supported in ACDC"

    assert mode in MODE_CHECK, "Mode '{mode}' not supported"

    # Paths configurations
    configs = {
        "cityscapes": {
            "rain": f"configs/cityscapes/rain/{volume}/{mode}-{domain}-{volume}.yaml",
            "normal": f"configs/cityscapes/normal/{mode}-{domain}.yaml",
        },
        "acdc": {f"{domain}": f"configs/acdc/{domain}/{mode}-{domain}-acdc.yaml"},
    }

    datasets = {
        "cityscapes": {
            "normal": f"/home/reza/datasets/cityscapes/leftImg8bit/train/",
            "rain": f"/home/reza/datasets/weather_datasets/weather_cityscapes/leftImg8bit/train/rain/{volume}/rainy_image/",
        },
        "acdc": f"/home/reza/datasets/acdc/rgb_anon/{domain}/train/",
    }

    # Output path configuration
    output_path = (
        f"output/{dataset}/{mode}-{domain}"
        + (f"-{volume}" if (dataset == "cityscapes" and domain != "normal") else "")
        + "/eval/"
    )

    # Constructing the return values
    config_file = configs[dataset][domain]
    dataset_path = datasets[dataset] if dataset == "acdc" else datasets[dataset][domain]

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
            base_model_path,
        ],
        resume=True,
    )
    return args

    
def benchmark(model, args):  
    cfg = setup(args)
    res = Trainer.test(cfg, model)
    if cfg.TEST.AUG.ENABLED:
        res.update(Trainer.test_with_TTA(cfg, model))
    if comm.is_main_process():
        verify_results(cfg, res)
    return res

args = Namespace(
    config_file="./configs/cityscapes/cityscapes_base.yaml",
    eval_only=True,
    num_gpus=1,
    opts=[
        "OUTPUT_DIR",
        "outputs/cityscapes/",
        "TEST.SLIDING_WINDOW",
        "True",
        "MODEL.SEM_SEG_HEAD.POOLING_SIZES",
        "[1,1]",
        "MODEL.WEIGHTS",
        "models/sed_model_large.pth",
    ],
    resume=True,
)

cfg = setup(args)
model = Trainer.build_model(cfg)
ret = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    cfg.MODEL.WEIGHTS, resume=args.resume
)


for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(f"{name}")