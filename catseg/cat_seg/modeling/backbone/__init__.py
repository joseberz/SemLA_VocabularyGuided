# Copyright (c) Facebook, Inc. and its affiliates.

from detectron2.modeling import BACKBONE_REGISTRY
from .swin import D2SwinTransformer

_name = "D2SwinTransformer"

# fvcore Registry speichert in _obj_map
if _name not in BACKBONE_REGISTRY._obj_map:
    BACKBONE_REGISTRY.register(D2SwinTransformer)
else:

    if BACKBONE_REGISTRY._obj_map[_name] is not D2SwinTransformer:
        raise AssertionError(
            f"{_name} already registered with a different object: "
            f"{BACKBONE_REGISTRY._obj_map[_name]} vs {D2SwinTransformer}"
        )