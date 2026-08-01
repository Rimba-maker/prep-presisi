"""Public API: get_model(name: str) -> BaseForecaster."""

from prep_presisi.models.base import BaseForecaster
from prep_presisi.models.baseline import SeasonalNaiveForecaster
from prep_presisi.models.statsforecast_model import StatsForecastForecaster
from prep_presisi.models.xgb_model import XGBoostForecaster

_REGISTRY: dict[str, type[BaseForecaster]] = {
    "seasonal_naive": SeasonalNaiveForecaster,
    "xgboost_global": XGBoostForecaster,
    "statsforecast_sample": StatsForecastForecaster,
}


def get_model(name: str, **kwargs) -> BaseForecaster:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Model '{name}' tidak dikenal. Tersedia: {available}")
    return _REGISTRY[name](**kwargs)


__all__ = ["BaseForecaster", "get_model"]
