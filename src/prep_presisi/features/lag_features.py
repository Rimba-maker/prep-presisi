import pandas as pd

_DEFAULT_LAGS = (1, 7, 14)
_DEFAULT_ROLLING_WINDOWS = (7, 14)


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "qty_sold",
    lags: tuple[int, ...] = _DEFAULT_LAGS,
    rolling_windows: tuple[int, ...] = _DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Lag & rolling-mean dari `target_col`, dihitung per outlet x menu (bukan lintas
    series) — pakai .shift() dulu sebelum .rolling() supaya tidak ada leakage dari hari
    yang sedang diprediksi. Baris paling awal tiap series wajar NaN (belum ada histori
    sepanjang window itu) — dibiarkan NaN, XGBoost menangani missing value secara native,
    tidak perlu imputasi manual."""
    result = df.sort_values(["outlet_id", "menu_item_id", "date"]).copy()
    grouped = result.groupby(["outlet_id", "menu_item_id"], sort=False)[target_col]

    for lag in lags:
        result[f"lag_{lag}"] = grouped.shift(lag)

    for window in rolling_windows:
        result[f"rolling_mean_{window}"] = grouped.transform(
            lambda s, w=window: s.shift(1).rolling(window=w, min_periods=1).mean()
        )

    return result.sort_index()
