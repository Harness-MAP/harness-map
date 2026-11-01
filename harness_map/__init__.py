"""Harness-MAP: quality-diversity evolution of agent harnesses with CVT-MAP-Elites."""
from __future__ import annotations

__all__ = ["run", "MODES", "Genome", "Fitness", "EpisodeResult"]

# facade name -> (submodule, attribute); resolved on first access (PEP 562), not at import.
_FACADE = {
    "run": ("harness_map.main", "run"),
    "MODES": ("harness_map.main", "MODES"),
    "Genome": ("harness_map.main", "Genome"),
    "Fitness": ("harness_map.main", "Fitness"),
    "EpisodeResult": ("harness_map.main", "EpisodeResult"),
}


def __getattr__(name: str):
    import importlib
    target = _FACADE.get(name)
    if target is None:
        raise AttributeError(f"module 'harness_map' has no attribute {name!r}")
    mod, attr = target
    return getattr(importlib.import_module(mod), attr)


def __dir__():
    return sorted(list(globals()) + __all__)
