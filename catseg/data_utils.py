# Autor: Joshua Ritter
# Teil der Masterarbeit "Vokabulargeleitete Selektion von LoRA-Adaptern
# mittels CLIP für domänenadaptive Open-Vocabulary-Segmentierung" (2026)

import random
from typing import Union, List, Any, Callable, Dict, Optional, Tuple

import torch.utils.data as torchdata

from detectron2.config import configurable
from detectron2.data import get_detection_dataset_dicts, DatasetMapper, DatasetFromList, MapDataset, DatasetCatalog, MetadataCatalog
from detectron2.data.build import trivial_batch_collator
from detectron2.data.samplers import InferenceSampler


def register_subset_dataset(
        dataset_name: str,
        subset_fraction: float,
        subset_seed: int = 42,
) -> str:
    """
    Registriert eine Teilmenge eines bereits registrierten
    Datasets unter neuem Namen im DatasetCatalog.

    Bei subset_fraction >= 1.0 wird keine Teilmenge gebildet und stattdessen
    wird der ursprüngliche Datasetname unverändert zurückgegeben.
    """
    if subset_fraction >= 1.0:
        return dataset_name  # kein Subset

    full_dicts = DatasetCatalog.get(dataset_name)
    n_total = len(full_dicts)
    n_subset = max(1, int(n_total * subset_fraction))

    rng = random.Random(subset_seed)
    indices = sorted(rng.sample(range(n_total), n_subset))
    subset_dicts = [full_dicts[i] for i in indices]

    subset_name = f"{dataset_name}_sub{subset_fraction}_seed{subset_seed}"

    # Nur einmal regsitrieren, weil DatasetCatalog keine Duplikate erlaubt
    if subset_name not in DatasetCatalog.list():
        DatasetCatalog.register(subset_name, lambda d=subset_dicts: d)
        _copy_metadata(dataset_name, subset_name)

    print(
        f"SUBSET: '{dataset_name}': {n_subset}/{n_total} Bilder "
        f"(seed={subset_seed}) => '{subset_name}'"
    )
    return subset_name


def _copy_metadata(source_name: str, target_name: str) -> None:
    """Kopiert Metadaten (Klassen, ignore_label, usw) vom Original."""
    meta = MetadataCatalog.get(source_name)
    new_meta = MetadataCatalog.get(target_name)
    for k, v in meta.as_dict().items():
        try:
            new_meta.set(**{k: v})
        except Exception:
            pass


def register_val_test_split(
        dataset_name: str,
        val_fraction: float = 0.5,
        split_seed: int = 123,
) -> Tuple[str, str]:
    """
    Teilt ein bereits registriertes Dataset anhand eines Seeds in eine
    Validierungs und eine Testmenge auf und registriert beide unter
    neuem Namen im DatasetCatalog.
    """
    assert 0.0 < val_fraction < 1.0, (
        f"val_fraction muss zwischen 0 und 1 liegen, ist aber {val_fraction}"
    )

    full_dicts = DatasetCatalog.get(dataset_name)
    n_total = len(full_dicts)
    n_val = max(1, int(n_total * val_fraction))

    # Indizes mischen und aufteilen
    rng = random.Random(split_seed)
    all_indices = list(range(n_total))
    rng.shuffle(all_indices)

    val_indices = sorted(all_indices[:n_val])
    test_indices = sorted(all_indices[n_val:])

    val_dicts = [full_dicts[i] for i in val_indices]
    test_dicts = [full_dicts[i] for i in test_indices]

    # Namen für die neuen Datensätze
    val_name = f"{dataset_name}__val_{val_fraction}_seed{split_seed}"
    test_name = f"{dataset_name}__test_{val_fraction}_seed{split_seed}"

    # Registrieren
    if val_name not in DatasetCatalog.list():
        DatasetCatalog.register(val_name, lambda d=val_dicts: d)
        _copy_metadata(dataset_name, val_name)

    if test_name not in DatasetCatalog.list():
        DatasetCatalog.register(test_name, lambda d=test_dicts: d)
        _copy_metadata(dataset_name, test_name)

    print(
        f"VAL/TEST SPLIT: '{dataset_name}' => "
        f"val={len(val_dicts)}, test={len(test_dicts)} "
        f"(total={n_total}, seed={split_seed})"
    )
    return val_name, test_name


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