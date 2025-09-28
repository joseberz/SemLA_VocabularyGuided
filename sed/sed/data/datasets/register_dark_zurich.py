import os
import logging

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata, CITYSCAPES_CATEGORIES


logger = logging.getLogger(__name__)


DARKZURICH_COLORED_SPLITS = {
    "dz_{task}_val": ("dark_zurich_val/rgb_anon/val/night", "dark_zurich_val/gt/val/night/"),
}


def _get_dark_zurich_files(image_dir, gt_dir):
    files = []
    # scan through the directory
    cities = PathManager.ls(image_dir)
    logger.info(f"{len(cities)} cities found in '{image_dir}'.")
    for city in cities:
        city_img_dir = os.path.join(image_dir, city)
        city_gt_dir = os.path.join(gt_dir, city)
        for basename in PathManager.ls(city_img_dir):
            image_file = os.path.join(city_img_dir, basename)

            suffix = "rgb_anon.png"
            assert basename.endswith(suffix), basename
            basename = basename[: -len(suffix)]

            # instance_file = os.path.join(city_gt_dir, basename + "gtFine_instanceIds.png")
            label_file = os.path.join(city_gt_dir, basename + "gt_labelIds.png")
            # json_file = os.path.join(city_gt_dir, basename + "gtFine_polygons.json")

            files.append((image_file, label_file))
    assert len(files), "No images found in {}".format(image_dir)
    for f in files[0]:
        assert PathManager.isfile(f), f
    return files

def load_dark_zurich_semantic(image_dir, gt_dir):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".

    Returns:
        list[dict]: a list of dict, each has "file_name" and
            "sem_seg_file_name".
    """
    ret = []
    # gt_dir is small and contain many small files. make sense to fetch to local first
    gt_dir = PathManager.get_local_path(gt_dir)
    for image_file, label_file in _get_dark_zurich_files(image_dir, gt_dir):
        label_file = label_file.replace("labelIds", "labelTrainIds")


        ret.append(
            {
                "file_name": image_file,
                "sem_seg_file_name": label_file,
            }
        )
        
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(
        ret[0]["sem_seg_file_name"]
    ), "Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py"  # noqa
    return ret


def register_colored_dark_zurich(root):
    for key, (image_dir, gt_dir) in DARKZURICH_COLORED_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            "dz", lambda x=image_dir, y=gt_dir: load_dark_zurich_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="acdc_sem_seg",
            ignore_label=255,
            thing_colors=[k["color"] for k in CITYSCAPES_CATEGORIES if k["name"] in meta["thing_classes"]],
            stuff_colors=[k["color"] for k in CITYSCAPES_CATEGORIES if k["name"] in meta["stuff_classes"]],
            **meta,
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_colored_dark_zurich(_root)