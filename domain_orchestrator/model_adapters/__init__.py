"""Model adapter implementations for SemLA."""

from .protocol import DomainResources, ModelAdapter

__all__ = ["DomainResources", "ModelAdapter"]

try:  # pragma: no cover - optional import for default adapter
    from .catseg_adapter import CatSegAdapter
except ModuleNotFoundError:  # pragma: no cover - CatSeg dependencies might be optional
    CatSegAdapter = None  # type: ignore[assignment]
else:  # pragma: no cover - executed when CatSeg is available
    __all__.append("CatSegAdapter")
