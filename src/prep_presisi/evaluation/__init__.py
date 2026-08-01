"""Public API: evaluate()."""

import pandas as pd

from prep_presisi.evaluation.metrics import mae, mape, wmape


def evaluate(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    return {
        "mae": mae(actual, predicted),
        "mape": mape(actual, predicted),
        "wmape": wmape(actual, predicted),
    }


__all__ = ["evaluate"]
