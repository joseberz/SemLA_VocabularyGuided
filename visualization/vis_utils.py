from typing import List, Dict

import numpy as np

from class_mapping import CANONICAL_CLASS_MAP

# Für vokabularübergreifenden Testdatensatz: Welche Adapter enthalten das Objekt im Trainings-Vokabular?
GOOD_ADAPTERS_FOR_OBJECT = {
    # IDD
    "bag":           ["a133", "coconutL", "pc59", "nyu"], #coconut hat nur handbag
    "bench":         ["a133", "coconutL", "mv", "pc59"],
    "flag":          ["a133"],
    "river":         ["a133", "coconutL"],
    "trash can":     ["mv"],
    "umbrella":      ["coconutL"],

    # PC59
    "bridge":        ["a133", "coconutL", "mv", "idd"],
    "fan":           ["a133"],
    "picture":       ["nyu"],
    "refrigerator":  ["a133", "coconutL", "nyu"],
    "traffic light": ["a133", "coconutL", "mv", "acdc", "cs", "muses", "bdd", "idd"],

    # NYU
    "bicycle":       ["a133", "coconutL", "mv", "acdc", "cs", "muses", "bdd", "idd", "pc59"],
    "dog":           ["coconutL", "pc59"],
    "keyboard":      ["coconutL", "pc59"],
    "potted plant":  ["coconutL", "pc59"],
    "teddy bear":    ["coconutL"],
}

ADAPTER_VOCAB_JSONS = {
    "mv":              "catseg/datasets/mv65.json",
    "cs-normal":       "catseg/datasets/cs19.json",
    "pc59":            "catseg/datasets/pc59.json",
    "a133":            "catseg/datasets/ade133.json",
    "coconutL":        "catseg/datasets/coconut.json",
    "nyu":             "catseg/datasets/nyu40.json",
    "idd":             "catseg/datasets/idd30.json",
    "iddnovel":        "catseg/datasets/idd30.json",
    "nyunovel":        "catseg/datasets/nyu40.json",
    "pc59novel":       "catseg/datasets/pc59.json",
    "acdc-night":      "catseg/datasets/cs19.json",
    "acdc-rain":       "catseg/datasets/cs19.json",
    "acdc-snow":       "catseg/datasets/cs19.json",
    "acdc-fog":        "catseg/datasets/cs19.json",
    "muses-clear-day": "catseg/datasets/cs19.json",
    "muses-clear-night": "catseg/datasets/cs19.json",
    "muses-rain-day": "catseg/datasets/cs19.json",
    "muses-rain-night": "catseg/datasets/cs19.json",
    "muses-fog-day": "catseg/datasets/cs19.json",
    "muses-fog-night": "catseg/datasets/cs19.json",
    "muses-snow-day": "catseg/datasets/cs19.json",
    "muses-snow-night": "catseg/datasets/cs19.json",
    "bdd": "catseg/datasets/cs19.json"
}

def get_class_names_from_ids(domain: str, gt_ids: List[int], all_adapters_and_classes: Dict[str, List[str]]) -> List[str]:
    # Liest die Masken-IDs (gt_ids) aus und holt sich daraus die zugehörigen
    # Klassenbezeichnungen aus dem bereits geladenen Adapter-Vokabular
    vocab = all_adapters_and_classes.get(domain)
    if vocab is None:
        return []

    class_names = []
    for i in gt_ids:
        if i >= len(vocab):
            pass
        else:
            class_names.append(vocab[i])

    return class_names

def normalize_class_name(raw: str) -> list[str]:
    # immer lower case
    # entferne Leerzeichen
    # Komma getrennt
    # Synonyme mappen
    cleaned = raw.strip().lower()
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if not parts:
        return []
    return [CANONICAL_CLASS_MAP.get(p, p) for p in parts]


def normalize_class_list(raw_classes: list[str]) -> list[str]:
    result: set[str] = set()
    for raw in raw_classes:
        result.update(normalize_class_name(raw))
    return list(result)

def aggregate_weights_cross_vocab(weights):
    # Zusammenfassen von Subdomänen (wie acdc-fog, acdc-rain...)
    aggregated = {}
    for obj, adapter_dict in weights.items():
        aggregated[obj] = {}
        for adapter, vals in adapter_dict.items():
            agg_name = adapter
            for prefix in ["acdc", "muses"]:
                if adapter == prefix or adapter.startswith(prefix + "-") or adapter.startswith(prefix + "_"):
                    agg_name = prefix
                    break
            aggregated[obj].setdefault(agg_name, [])
            aggregated[obj][agg_name].extend(vals)
    return aggregated

def is_good_adapter(adapter, obj):
    for g in GOOD_ADAPTERS_FOR_OBJECT.get(obj, []):
        if adapter == g or adapter.startswith(g + "-") or adapter.startswith(g + "_"):
            return True
    return False

def avg_per_adapter(obj_weights):
    return {a: float(np.mean(vals)) for a, vals in obj_weights.items() if vals}