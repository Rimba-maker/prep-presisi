"""Public API: evaluate(), compute_waste_avoided(), total_waste_avoided_rupiah()."""

import pandas as pd

from prep_presisi.evaluation.metrics import mae, mape, wmape
from prep_presisi.evaluation.waste_calculator import (
    compute_waste_avoided,
    total_waste_avoided_rupiah,
)


def evaluate(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    return {
        "mae": mae(actual, predicted),
        "mape": mape(actual, predicted),
        "wmape": wmape(actual, predicted),
    }


__all__ = ["compute_waste_avoided", "evaluate", "total_waste_avoided_rupiah"]
