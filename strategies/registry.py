"""Small registry used by the backtest CLI to discover strategies."""

from __future__ import annotations

from typing import Callable, Dict

from .base import StrategyBase


_REGISTRY: Dict[str, Callable[..., StrategyBase]] = {}


def register(name: str):
    """Register a StrategyBase subclass under a command-line name."""
    key = name.strip()
    if not key:
        raise ValueError("Strategy name cannot be empty")

    def decorator(cls):
        if not issubclass(cls, StrategyBase):
            raise TypeError("Registered class must inherit StrategyBase")
        _REGISTRY[key] = cls
        cls.name = key
        return cls

    return decorator


def create_strategy(name: str, **params) -> StrategyBase:
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "none"
        raise KeyError(f"Strategy '{name}' is not registered. Available: {available}") from exc
    return factory(**params)


def available_strategies() -> list[str]:
    return sorted(_REGISTRY)
