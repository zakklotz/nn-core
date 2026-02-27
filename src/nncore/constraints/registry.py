from __future__ import annotations

from typing import Callable

from nncore.constraints.base import Constraint


_REGISTRY: dict[str, Callable[..., Constraint]] = {}


def register_constraint(name: str):
    """Register a constraint builder by name.

    Convention: builders are called as `builder(**params)`.
    """

    def _decorator(builder: Callable[..., Constraint]):
        if name in _REGISTRY:
            raise ValueError(f"Constraint {name!r} is already registered.")
        _REGISTRY[name] = builder
        return builder

    return _decorator


def build_constraint(name: str, params: dict[str, object]) -> Constraint:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown constraint: {name}. Available: [{available}]")
    builder = _REGISTRY[name]
    return builder(**params)


def list_constraints() -> list[str]:
    return sorted(_REGISTRY.keys())
