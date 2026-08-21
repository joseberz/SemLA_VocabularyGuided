# Geändert von Joshua Ritter, 2026, im Rahmen der Masterarbeit
# "Vokabulargeleitete Selektion von LoRA-Adaptern
# mittels CLIP für domänenadaptive Open-Vocabulary-Segmentierung"
# Ursprüngliche Datei: SemLA (Qorbani et al.), Apache-2.0-Lizenz

from argparse import Namespace
import json

import logging
from typing import Literal
from catseg.train_net import setup

DETECTRON2_DATASET_PATH = ""

_classnames_cache: dict[str, list[str]] = {}

def get_classnames_for_domain(domain_name: str) -> list[str]:
    """Lädt die Klassennamen einer Domäne"""
    if domain_name in _classnames_cache:
        return _classnames_cache[domain_name]

    domain_args = get_domain_args(domain_name, "val", get_cofing_only=True)
    cfg = setup(domain_args)
    json_path = cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON
    with open(json_path) as f:
        classnames = json.load(f)

    _classnames_cache[domain_name] = classnames
    return classnames

def get_domain_args(
    domain_name: str,
    split: Literal['train', 'val'],
    mode: str = "lora",
    base_model_path: str = "models/model_final.pth",
    num_gpus: int = 1,
    get_cofing_only: bool = False,
    subset_fraction: float = 1.0,
    subset_seed: int = 42,
    val_test_split: float = 0.0,
    val_test_seed: int = 123,
    use_val_portion: bool = True,   # True = Val-Anteil nutzen für Parameter-Suche und wenn False = Test-Anteil für Evaluation
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
    MODE_CHECK = {"lora"}

    DATASET_CHECK = {
        "cs",
        "acdc",
        "muses",
        "bdd",
        "mv",
        "a150",
        "a133",
        "idd",
        "pc59",
        "nyu",
        "coconutL",
        "iddnovel",
        "nyunovel",
        "pc59novel"
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
        "a133": "configs/a133/a133.yaml",
        "idd": "configs/idd/idd.yaml",
        'pc59': 'configs/pc59/pc59.yaml',
        'coconutL': 'configs/coconutL/coconutL.yaml',
        "iddnovel": "configs/novel_eval/idd_novel.yaml",
        "nyunovel": "configs/novel_eval/nyu_novel.yaml",
        "pc59novel": "configs/novel_eval/pc59_novel.yaml",
    }

    datasets = {
        "cs": {
            "normal": {
                "train": f"{DETECTRON2_DATASET_PATH}cityscapes/leftImg8bit/train/",
                "val": f"{DETECTRON2_DATASET_PATH}cityscapes/leftImg8bit/val/",
            },
        },
        "acdc": {
            "train": f"{DETECTRON2_DATASET_PATH}acdc/rgb_anon/{domain}/train/",
            "val": f"{DETECTRON2_DATASET_PATH}acdc/rgb_anon/{domain}/val/",
        },
        "muses": {
            "train": f"{DETECTRON2_DATASET_PATH}muses/frame_camera/train/{domain}/{sub_domain}/",
            "val": f"{DETECTRON2_DATASET_PATH}muses/frame_camera/val/{domain}/{sub_domain}/",
        },
        "bdd": {
            "train": f"{DETECTRON2_DATASET_PATH}bdd100k/images/10k/train/",
            "val": f"{DETECTRON2_DATASET_PATH}bdd100k/images/10k/val/",
        },
        "mv": {
            "train": f"{DETECTRON2_DATASET_PATH}mapillary_vistas/train/images/",
            "val": f"{DETECTRON2_DATASET_PATH}mapillary_vistas/val/images/",
        },
        "a150": {
            "train": f"{DETECTRON2_DATASET_PATH}ADEChallengeData2016/images/training/",
            "val": f"{DETECTRON2_DATASET_PATH}ADEChallengeData2016/images/validation/",
        },
        "a133": {
            "train": f"{DETECTRON2_DATASET_PATH}ADE133_ignore_thre_0_1/images/training/", # TODO
            "val": f"{DETECTRON2_DATASET_PATH}ADE133_ignore_thre_0_1/images/validation/", # TODO
        },
        "idd": {
            "train": f"{DETECTRON2_DATASET_PATH}IDD_Segmentation/leftImg8bit/train/",
            "val": f"{DETECTRON2_DATASET_PATH}IDD_Segmentation/leftImg8bit/val/",
        },
        "pc59": {
            "train": f"{DETECTRON2_DATASET_PATH}pascal_ctx_d2/images/training",
            "val": f"{DETECTRON2_DATASET_PATH}pascal_ctx_d2/images/validation",
        },
        "nyu": {
            "train": f"{DETECTRON2_DATASET_PATH}nyudv2_splitted/train/rgb",
            "val": f"{DETECTRON2_DATASET_PATH}nyudv2_splitted/test/rgb",
        },
        "coconutL": {
            "train": f"{DETECTRON2_DATASET_PATH}coco/train2017/",
            "val": f"{DETECTRON2_DATASET_PATH}coco/val2017/",
        },
        "iddnovel": {
            "train": f"{DETECTRON2_DATASET_PATH}IDD_novel/image/",
            "val":   f"{DETECTRON2_DATASET_PATH}IDD_novel/image/",
        },
        "nyunovel": {
            "train": f"{DETECTRON2_DATASET_PATH}NYU_novel/image/",
            "val":   f"{DETECTRON2_DATASET_PATH}NYU_novel/image/",
        },
        "pc59novel": {
            "train": f"{DETECTRON2_DATASET_PATH}PC59_novel/image/",
            "val":   f"{DETECTRON2_DATASET_PATH}PC59_novel/image/",
        },
    }

    # Output path configuration
    output_path = (
        f"output/{dataset}/{mode}-{dataset}"
        + (f"-{domain}" if  domain != "" else "")
        + (f"-{sub_domain}" if sub_domain != "" else "")
        + "/eval/"
    )

    # Constructing the return values
    if domain == "" and sub_domain == "":
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

    if get_cofing_only:
        return args
    else:
        from catseg.train_net import Trainer, setup
        from catseg.data_utils import register_subset_dataset
        from catseg.data_utils import register_val_test_split

        dataset_name = f"{domain_name}_sem_seg_{split}"

        # Bei Val-Durchläufen und bei Angabe eines splits bzw. einer datenreduzierung, wird der datensatz auf ein Subset reduziert
        if split == "val" and subset_fraction < 1.0:
            dataset_name = register_subset_dataset(dataset_name, subset_fraction, subset_seed)

        if split == "val" and val_test_split > 0.0:
            val_name, test_name = register_val_test_split(
                dataset_name, val_test_split, val_test_seed
            )
            dataset_name = val_name if use_val_portion else test_name

        data_loader = Trainer.build_test_loader(
            setup(args), dataset_name
        )

        evaluator = Trainer.build_evaluator(setup(args), dataset_name)

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
        model_path=model_path,
    )

    if seed != None:
        args.opts.extend(["SEED", seed])

    return args


def benchmark_catseg(model, args):

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

def load_catseg_model(args, model_path: str = None):
    from catseg.train_net import Trainer, setup
    from detectron2.checkpoint import DetectionCheckpointer

    print("Loading base model ...")
    
    try:
        cfg = setup(args)
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS if model_path is None else model_path, resume=args.resume
        )
        print("Base model loaded.\n")
        return model
    except AttributeError as e:
        print(f"Error: Invalid model configuration: {e}")
        raise
    except FileNotFoundError:
        print(f"Error: Model weights not found at the specified path.")
        raise
    except Exception as e:
        print(f"Unexpected error while loading the base model: {e}")
        raise