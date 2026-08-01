"""Public API: get_model(name: str) -> BaseForecaster."""

from prep_presisi.models.base import BaseForecaster
from prep_presisi.models.baseline import SeasonalNaiveForecaster

_REGISTRY: dict[str, type[BaseForecaster]] = {
    "seasonal_naive": SeasonalNaiveForecaster,
}


def get_model(name: str) -> BaseForecaster:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Model '{name}' tidak dikenal. Tersedia: {available}")
    return _REGISTRY[name]()


__all__ = ["BaseForecaster", "get_model"]
