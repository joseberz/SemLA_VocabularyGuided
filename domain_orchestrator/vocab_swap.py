# Autor: Joshua Ritter
# Teil der Masterarbeit "Vokabulargeleitete Selektion von LoRA-Adaptern
# mittels CLIP für domänenadaptive Open-Vocabulary-Segmentierung" (2026)
# Dient dem dynamischen Austauschen des CAT-Seg Vokabulars

from typing import List
from cat_seg.modeling.transformer.cat_seg_predictor import CATSegPredictor

def find_predictor(model) -> CATSegPredictor:
    for module in model.modules():
        if isinstance(module, CATSegPredictor):
            return module
    raise RuntimeError("Predictor nicht gefunden")

def swap_class_vocabulary(model, new_classnames: List[str]) -> List[str]:
    """Setzt ein neues Testvokabular / Open-Vokabular für Cat Seg"""
    predictor = find_predictor(model)
    old_classnames = predictor.test_class_texts
    predictor.test_class_texts = new_classnames
    predictor.cache = None # erzwingt neuladen des Vokabulars
    predictor.tokens = None #  erzwingt neuladen des Vokabulars
    return old_classnames

def restore_class_vocabulary(model, old_classnames: List[str]) -> None:
    predictor = find_predictor(model)
    predictor.test_class_texts = old_classnames
    predictor.cache = None
    predictor.tokens = None