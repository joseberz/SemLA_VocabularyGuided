# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

from pathlib import Path

def list_subdirectories(directory):
    try:
        # Create a Path object for the directory
        path = Path(directory)
        # Use the glob method to list all sub-directories
        subdirectories = [subdir.name for subdir in path.iterdir() if subdir.is_dir()]
        return subdirectories
    except FileNotFoundError:
        print(f"The directory {directory} does not exist.")
        return []
    except PermissionError:
        print(f"Permission denied for accessing {directory}.")
        return []


BDD100K_SEM_SEG_CATEGORIES = [
    {"name": "road", "id": 0, "trainId": 0},
    {"name": "sidewalk", "id": 1, "trainId": 1},
    {"name": "building", "id": 2, "trainId": 2},
    {"name": "wall", "id": 3, "trainId": 3},
    {"name": "fence", "id": 4, "trainId": 4},
    {"name": "pole", "id": 5, "trainId": 5},
    {"name": "traffic light", "id": 6, "trainId": 6},
    {"name": "traffic sign", "id": 7, "trainId": 7},
    {"name": "vegetation", "id": 8, "trainId": 8},
    {"name": "terrain", "id": 9, "trainId": 9},
    {"name": "sky", "id": 10, "trainId": 10},
    {"name": "person", "id": 11, "trainId": 11},
    {"name": "rider", "id": 12, "trainId": 12},
    {"name": "car", "id": 13, "trainId": 13},
    {"name": "truck", "id": 14, "trainId": 14},
    {"name": "bus", "id": 15, "trainId": 15},
    {"name": "train", "id": 16, "trainId": 16},
    {"name": "motorcycle", "id": 17, "trainId": 17},
    {"name": "bicycle", "id": 18, "trainId": 18},
]

def _get_bdd100k_sem_seg_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing, so all ids are shifted by 1.
    stuff_ids = [k["id"] for k in BDD100K_SEM_SEG_CATEGORIES]
    assert len(stuff_ids) == 19, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in BDD100K_SEM_SEG_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }

    return ret

def register_bdd100k_sharded_sem_seg(root):
    
    root = os.path.join(root, "sharded_datasets", "bdd100k",)
    meta = _get_bdd100k_sem_seg_meta()
    
    subdirs = list_subdirectories(root)
    for dir in subdirs:
        # for name, dirname in [("train", "train"), ("val", "val")]:
        image_dir = os.path.join(root, dir, "images",)
        gt_dir = os.path.join(root, dir, "labels")
        name = f"{dir}_sem_seg_train"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
# _root = "/leonardo_work/EUHPC_D11_069/sharded_datasets"
register_bdd100k_sharded_sem_seg(_root)