from typing import Union, List, Any, Callable, Dict, Optional

import torch.utils.data as torchdata

from detectron2.config import configurable
from detectron2.data import get_detection_dataset_dicts, DatasetMapper, DatasetFromList, MapDataset
from detectron2.data.build import trivial_batch_collator
from detectron2.data.samplers import InferenceSampler


def _test_loader_from_config(cfg, dataset_name, mapper=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        proposal_files=(
            [
                cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)]
                for x in dataset_name
            ]
            if cfg.MODEL.LOAD_PROPOSALS
            else None
        ),
    )
    if cfg.TEST.MAX_IMAGES > -1:
        dataset = dataset[:cfg.TEST.MAX_IMAGES]
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    return {
        "dataset": dataset,
        "mapper": mapper,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "sampler": (
            InferenceSampler(len(dataset))
            if not isinstance(dataset, torchdata.IterableDataset)
            else None
        ),
    }

@configurable(from_config=_test_loader_from_config)
def build_custom_test_loader(dataset: Union[List[Any], torchdata.Dataset],
                          *,
                          mapper: Callable[[Dict[str, Any]], Any],
                          sampler: Optional[torchdata.Sampler] = None,
                          batch_size: int = 1,
                          num_workers: int = 0,
                          collate_fn: Optional[Callable[[List[Any]], Any]] = None,
                          ):
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = InferenceSampler(len(dataset))
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
    )