import pandas as pd


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    mask = actual.notna() & predicted.notna()
    return float((actual[mask] - predicted[mask]).abs().mean())


def mape(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean Absolute Percentage Error. Baris dengan actual == 0 di-drop (pembagi nol) —
    WMAPE (lihat wmape()) yang jadi metric utama untuk kombinasi low-volume (PRD §3)."""
    mask = actual.notna() & predicted.notna() & (actual != 0)
    return float(((actual[mask] - predicted[mask]).abs() / actual[mask]).mean())


def wmape(actual: pd.Series, predicted: pd.Series) -> float:
    """Weighted MAPE = total absolute error / total actual — tidak meledak di kombinasi
    low-volume seperti MAPE biasa (PRD §3), jadi metric sekunder wajib."""
    mask = actual.notna() & predicted.notna()
    total_actual = actual[mask].sum()
    if total_actual == 0:
        raise ValueError("wmape: total actual = 0, tidak bisa dihitung")
    return float((actual[mask] - predicted[mask]).abs().sum() / total_actual)
