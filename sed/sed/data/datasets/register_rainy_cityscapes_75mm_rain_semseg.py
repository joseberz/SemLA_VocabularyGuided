import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.cityscapes import load_cityscapes_semantic
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata, CITYSCAPES_CATEGORIES

RAINY_CITYSCAPES_75MM_SPLITS = {
    "cs-rain-75mm_{task}_train": ("weather_datasets/weather_cityscapes/leftImg8bit/train/rain/75mm/rainy_image/",
                                    "cityscapes/gtFine/train/"),
    "cs-rain-75mm_{task}_val": ("weather_datasets/weather_cityscapes/leftImg8bit/val/rain/75mm/rainy_image/",
                                    "cityscapes/gtFine/val/"),
}

def register_rainy_cityscapes_75mm(root):
    for key, (image_dir, gt_dir) in RAINY_CITYSCAPES_75MM_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        sem_key = key.format(task="sem_seg")

        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )

        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="resized_cityscapes_sem_seg",
            ignore_label=255,
            thing_colors=[k["color"] for k in CITYSCAPES_CATEGORIES if k["name"] in meta["thing_classes"]],
            stuff_colors=[k["color"] for k in CITYSCAPES_CATEGORIES if k["name"] in meta["stuff_classes"]],
            **meta,
        )
        
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_rainy_cityscapes_75mm(_root)