"""
TODO Documentation!
"""
import os
import logging

from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.utils.file_io import PathManager

from cat_seg.data.datasets.register_idd import IDD_SEM_SEG_CATEGORIES
from cat_seg.data.datasets.register_nyudv2 import classes as NYU_CLASSES
from cat_seg.data.datasets.register_pascal_ctx_59_sem_seg import PASCAL_CTX_59_CATEGORIES

logger = logging.getLogger(__name__)

NOVEL_DATASETS = {
    "iddnovel": {
        "class_names": ["bag", "bench", "flag", "river", "trash can", "umbrella", "backpack"],
        "novel_ids": [26, 27, 28, 29, 30, 31, 32],
        "root_subdir": "IDD_novel",
        "img_suffix":  "_leftImg8bit",
        "mask_suffix": "_gtFine_labellevel3Ids.png",
    },
    "nyunovel": {
        "class_names": ["dog", "bicycle", "keyboard", "potted plant", "teddy bear", "dishwasher", "microwave", "trash can"],
        "novel_ids": [40, 42, 43, 44, 45, 46, 47, 48],
        "root_subdir": "NYU_novel",
        "img_suffix":  None,
        "mask_suffix": None,
    },
    "pc59novel": {
        "class_names": ["bridge", "fan", "picture", "refrigerator", "traffic light", "pillow", "rug"],
        "novel_ids": [59, 60, 61, 62, 63, 64, 65],
        "root_subdir": "PC59_novel",
        "img_suffix":  None,
        "mask_suffix": None,
    },
}

def _find_mask(img_fname, mask_dir, img_suffix, mask_suffix):
    stem = os.path.splitext(img_fname)[0]

    if img_suffix is not None:
        if stem.endswith(img_suffix):
            base_id = stem[: -len(img_suffix)]
        else:
            logger.warning(
                f"Expected img_suffix '{img_suffix}' not found in stem '{stem}', "
                "falling back to full stem."
            )
            base_id = stem
    else:
        base_id = stem

    if mask_suffix is not None:
        candidates = [os.path.join(mask_dir, base_id + mask_suffix)]
    else:
        candidates = [
            os.path.join(mask_dir, base_id + ext)
            for ext in (".png", ".tif", ".tiff", ".jpg")
        ]

    for c in candidates:
        if os.path.isfile(c):
            return c

    return None


def load_novel_sem_seg(image_root, mask_root, id_map, img_suffix, mask_suffix):
    ret = []
    object_dirs = sorted(
        d for d in os.listdir(image_root)
        if os.path.isdir(os.path.join(image_root, d))
    )

    for obj_dir in object_dirs:
        img_dir = os.path.join(image_root, obj_dir)
        msk_dir = os.path.join(mask_root,  obj_dir)

        if not os.path.isdir(msk_dir):
            logger.warning(f"Mask directory missing: {msk_dir}")
            continue

        for fname in sorted(os.listdir(img_dir)):
            if fname.startswith("."):
                continue

            img_path  = os.path.join(img_dir, fname)
            mask_path = _find_mask(fname, msk_dir, img_suffix, mask_suffix)

            if mask_path is None:
                logger.warning(
                    f"No mask found for '{img_path}' "
                    f"(img_suffix={img_suffix!r}, mask_suffix={mask_suffix!r})"
                )
                continue

            ret.append({
                "file_name":         img_path,
                "sem_seg_file_name": mask_path,
                "novel_id_map":      id_map,
            })

    assert len(ret), f"No image/mask pairs found under {image_root}"
    logger.info(f"Loaded {len(ret)} novel-eval samples from {image_root}")
    return ret


def register_novel_eval_datasets(root):
    for key, meta in NOVEL_DATASETS.items():
        current_class_names = meta["class_names"]
        for key2 in ["train", "val"]:
            dataset_name = f"{key}_sem_seg_{key2}"

            if dataset_name in DatasetCatalog:
                continue

            ds_root    = os.path.join(root, meta["root_subdir"])
            image_root = os.path.join(ds_root, "image")
            mask_root  = os.path.join(ds_root, "new_masks")

            DatasetCatalog.register(
                dataset_name,
                lambda ir=image_root,
                       mr=mask_root,
                       idm=meta["novel_ids"],
                       ims=meta["img_suffix"],
                       msk=meta["mask_suffix"]: load_novel_sem_seg(ir, mr, idm, ims, msk),
            )

            if key == "iddnovel":
                IDD_SEM_SEG_CATEGORIES.update(
                    {o: i for o, i in zip(meta["class_names"], meta["novel_ids"])}
                )
                meta["class_names"] = list(IDD_SEM_SEG_CATEGORIES)
            elif key == "nyunovel":
                meta["class_names"] = NYU_CLASSES + current_class_names
            elif key == "pc59novel":
                meta["class_names"] = [k["name"] for k in PASCAL_CTX_59_CATEGORIES][:] + current_class_names

            MetadataCatalog.get(dataset_name).set(
                stuff_classes=meta["class_names"], # TODO müssen alle klassen sein / Prüfen ob TODO noch aktuell
                image_root=image_root,
                sem_seg_root=mask_root,
                evaluator_type="novel_sem_seg",
                ignore_label=255,
                novel_ids=meta["novel_ids"],
            )

            logger.info(
                f"Registered '{dataset_name}': "
                f"{len(meta['class_names'])} novel classes, "
                f"img_suffix={meta['img_suffix']!r}"
            )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_novel_eval_datasets("/home/joshi/Desktop/HSB/SemLA_VocabularyGuided/catseg/datasets")

