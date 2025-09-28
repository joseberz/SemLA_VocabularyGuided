from argparse import Namespace

import logging
import copy
import os
from pathlib import Path

from detectron2.checkpoint import DetectionCheckpointer
import detectron2.utils.comm as comm
from detectron2.evaluation import verify_results
from detectron2.engine import default_setup
from detectron2.projects.deeplab import add_deeplab_config
from sed import (
    add_sed_config,
    add_lora_config,
)

from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg as get_d2_cfg

from train_net import Trainer, set_random_seed

detectron2_datasets_path = os.getenv("DETECTRON2_DATASETS")

import peft


def domain_args(
    domain_name: str,
    index: str,
    mode: str = "lora",
    split: str = "train",
    base_model_path: str = "models/model_final.pth",
    multidomain=False,
    num_gpus: int = 1,
):
    logger_names = ["detectron2", "d2", "fvcore"]
    for name in logger_names:
        logger = logging.getLogger(name)

    if logger.hasHandlers():
        logger.handlers.clear()

    parts = domain_name.split("-")

    split_list = ["", "", ""]

    for i, part in enumerate(parts):
        split_list[i] = part

    dataset, domain, sub_domain = split_list

    # Supported configurations
    MODE_CHECK = {"lora", "finetune", "zeroshot"}

    DATASET_CHECK = {
        "cs",
        "acdc",
        "muses",
        "bdd",
        "mv",
        "a150",
        "idd",
        "pc59",
        "nyu",
        "coconutL"
    }

    CS_DOMAIN_CHECK = {"normal", "rain"}
    CS_SUB_DOMAIN_CHECK = ["25mm", "50mm", "75mm", "100mm", "200mm"]

    ACDC_DOMAIN_CHECK = {"fog", "night", "snow", "rain"}

    MUSES_DOMAIN_CHECK = {"clear", "rain", "fog", "snow"}
    MUSES_SUB_DOMAIN_CHECK = {"day", "night"}

    # Configurations assertions
    assert dataset in DATASET_CHECK

    if dataset == "cs":
        assert (
            domain in CS_DOMAIN_CHECK
        ), "Domain '{domain}' not supported for Cityscapes"
        if domain == "rain":
            assert (
                sub_domain in CS_SUB_DOMAIN_CHECK
            ), "Given volume '{volume}' is not supported for Cityscapes"
        elif domain == "normal":
            assert (
                sub_domain == ""
            ), f"Volume '{sub_domain}' is not supported for this domain in Cityscapes"
    elif dataset == "muses":
        assert (
            domain in MUSES_DOMAIN_CHECK
        ), f"Domain '{domain}' not supported for MUSES"
        assert (
            sub_domain in MUSES_SUB_DOMAIN_CHECK
        ), f"Given illumination '{sub_domain}' is not supported for MUSES"
    elif dataset == "acdc":
        assert (
            domain in ACDC_DOMAIN_CHECK
        ), f"Domain '{domain}' is not supported for ACDC"
        assert sub_domain == "", "Volume is not supported in ACDC"

    assert mode in MODE_CHECK, "Mode '{mode}' not supported"

    # Paths configurations
    if multidomain == False:
        configs = {
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
            'pc59': 'configs/pc59/pc59.yaml',
            'nyu': 'configs/nyu/nyu.yaml',
            'coconutL': 'configs/coconutL/coconutL.yaml'
        }
    else:
        configs = {
            "cs": {
                "rain": f"configs/cityscapes/rain/{sub_domain}/{mode}-{domain}-{sub_domain}-multi.yaml",
                "normal": f"configs/cityscapes/normal/{mode}-{domain}-multi.yaml",
            },
            "acdc": {
                f"{domain}": f"configs/acdc/{domain}/{mode}-{domain}-acdc-multi.yaml"
            },
            "muses": {
                f"{domain}": f"configs/muses/{domain}/{mode}-{domain}-muses-multi.yaml"
            },
        }

    if "cluster" not in domain_name:
        datasets = {
            "cs": {
                "normal": {
                    "train": f"{detectron2_datasets_path}cityscapes/leftImg8bit/train/",
                    "val": f"{detectron2_datasets_path}cityscapes/leftImg8bit/val/",
                },
                "rain": {
                    "train": f"{detectron2_datasets_path}weather_datasets/weather_cityscapes/leftImg8bit/train/rain/{sub_domain}/rainy_image/",
                    "val": f"{detectron2_datasets_path}weather_datasets/weather_cityscapes/leftImg8bit/val/rain/{sub_domain}/rainy_image/",
                },
            },
            "acdc": {
                "train": f"{detectron2_datasets_path}acdc/rgb_anon/{domain}/train/",
                "val": f"{detectron2_datasets_path}acdc/rgb_anon/{domain}/val/",
            },
            "muses": {
                "train": f"{detectron2_datasets_path}muses/frame_camera/train/{domain}/{sub_domain}/",
                "val": f"{detectron2_datasets_path}muses/frame_camera/val/{domain}/{sub_domain}/",
            },
            "bdd": {
                "train": f"{detectron2_datasets_path}bdd100k/images/10k/train/",
                "val": f"{detectron2_datasets_path}bdd100k/images/10k/val/",
            },
            "mv": {
                "train": f"{detectron2_datasets_path}mapillary_vistas/train/images/",
                "val": f"{detectron2_datasets_path}mapillary_vistas/val/images/",
            },
            "a150": {
                "train": f"{detectron2_datasets_path}ADE20k/images/training/",
                "val": f"{detectron2_datasets_path}ADE20k/images/validation/",
            },
            "idd": {
                "train": f"{detectron2_datasets_path}IDD_Segmentation/leftImg8bit/train/",
                "val": f"{detectron2_datasets_path}IDD_Segmentation/leftImg8bit/val/",
            },
            "pc59": {
                "train": f"{detectron2_datasets_path}pascal_ctx_d2/images/training",
                "val": f"{detectron2_datasets_path}pascal_ctx_d2/images/validation",
            },
            "nyu": {
                "train": f"{detectron2_datasets_path}nyudv2_splitted/train/rgb",
                "val": f"{detectron2_datasets_path}nyudv2_splitted/test/rgb",
            },
            "coconutL": {
                "train": f"{detectron2_datasets_path}coconut-l/train2017/",
                "val": f"{detectron2_datasets_path}coconut-l/val2017",
            },
        }
    else:  # Sharded domains
        datasets = {
            "bdd": {
                "train": f"{detectron2_datasets_path}sharded_datasets/bdd100k/{domain_name}/images/",
                "val": f"{detectron2_datasets_path}bdd100k/images/10k/val/",
            },
            "mv": {
                "train": f"{detectron2_datasets_path}sharded_datasets/mv/{domain_name}/images/",
                "val": f"{detectron2_datasets_path}mapillary_vistas/val/images/",
            },
        }

    # Output path configuration
    output_path = (
        f"output/{dataset}/{mode}-{dataset}"
        + (f"-{domain}" if  domain != "" else "")
        + (f"-{sub_domain}" if sub_domain != "" else "")
        + "/eval/"
    )

    model_path = f"output/{dataset}/{domain_name}_fullfinetune_{index}/model_final.pth"

    # Constructing the return values
    print("domain", sub_domain)
    if (domain == "" and sub_domain == "") or "cluster" in domain:
        config_file = configs[dataset]
    else:
        config_file = configs[dataset][domain]

    train_dataset_path = (
        datasets[dataset]["train"]
        if dataset != "cs"
        else datasets[dataset][domain]["train"]
    )

    val_dataset_path = (
        datasets[dataset]["val"]
        if dataset != "cs"
        else datasets[dataset][domain]["val"]
    )

    args = Namespace(
        config_file=config_file,
        eval_only=True,
        num_gpus=num_gpus,
        train_dataset_path=train_dataset_path,
        val_dataset_path=val_dataset_path,
        model_path=model_path,
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

    if "cluster" in domain:
        data_loader = Trainer.build_test_loader(
            setup(args), f"{domain_name}_sem_seg"
        )
        
        evaluator = Trainer.build_evaluator(setup(args), f"{domain_name}_sem_seg")
    else:
        data_loader = Trainer.build_test_loader(
            setup(args), f"{domain_name}_sem_seg_{split}"
        )

        evaluator = Trainer.build_evaluator(setup(args), f"{domain_name}_sem_seg_{split}")

    return args, evaluator, data_loader


def custom_domain_args(
    config_file,
    output_path,
    num_gpus=1,
    model_path="models/model_final.pth",
    dataset_path: str = None,
    seed=None,
):

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
    )

    if seed != None:
        args.opts.extend(["SEED", seed])

    return args


def setup(args):
    """
    Create configs and perform basic setups.
    """

    logger_names = ["detectron2", "d2", "fvcore"]
    for name in logger_names:
        logger = logging.getLogger(name)
        if logger.hasHandlers():
            logger.handlers.clear()

    cfg = get_d2_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_sed_config(cfg)
    add_lora_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(
        output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask_former"
    )

    return cfg


def add_lora(cfg, model):

    logger = logging.getLogger("detectron2.trainer")
    config = peft.LoraConfig(
        r=cfg.MODEL.LORA.RANK,
        lora_alpha=cfg.MODEL.LORA.ALPHA,
        lora_dropout=cfg.MODEL.LORA.DROPOUT,
        target_modules=cfg.MODEL.LORA.MODULES,
        bias=cfg.MODEL.LORA.BIAS,
        use_rslora=cfg.MODEL.LORA.USE_RSLORA,
        use_dora=cfg.MODEL.LORA.USE_DORA,
    )

    peft_model = peft.get_peft_model(
        copy.deepcopy(model), config, adapter_name=cfg.MODEL.LORA.NAME
    )

    logger.info("LoRAs injected for training.")

    return peft_model


def init_peft_model_from_config(cfg, model):

    logger = logging.getLogger("detectron2.trainer")
    model = peft.PeftModel.from_pretrained(
        model, cfg.MODEL.LORA.DB_PATH + cfg.MODEL.LORA.NAME, cfg.MODEL.LORA.NAME
    )

    logger.info("Pre-trained LoRA loaded.")
    return


def benchmark(model, args):

    cfg = setup(args)
    set_random_seed(cfg.SEED)
    res = Trainer.test(cfg, model)
    if cfg.TEST.AUG.ENABLED:
        res.update(Trainer.test_with_TTA(cfg, model))
    if comm.is_main_process():
        verify_results(cfg, res)
    return res


def load_base_model(args):

    cfg = setup(args)
    set_random_seed(cfg.SEED)
    model = Trainer.build_model(cfg)
    ret = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    return model


def init_peft_model(model, lora_db_path="loradb/", lora_name="normal"):

    logger = logging.getLogger("detectron2.trainer")
    model = peft.PeftModel.from_pretrained(model, lora_db_path + lora_name, lora_name)
    logger.info("Pre-trained LoRA loaded.")
    return model
