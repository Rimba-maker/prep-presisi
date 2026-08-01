"""Public API: build_features(), split_by_date()."""

from datetime import date

import pandas as pd

from prep_presisi.features.lag_features import add_lag_features
from prep_presisi.features.time_features import add_time_features, encode_categoricals


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Urutan wajib (TRD §7.1): panggil di atas dataset PENUH yang terurut waktu,
    baru split_by_date() setelah ini — kalau dibalik, baris awal val/test kehilangan
    lag value karena histori 7/14-hari-lalunya ada di subset train yang terpisah."""
    result = add_time_features(df)
    result = encode_categoricals(result)
    result = add_lag_features(result)
    return result


def split_by_date(
    df: pd.DataFrame, train_end: date, val_end: date, test_end: date
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(df["date"]).dt.date
    train = df[dates <= train_end]
    val = df[(dates > train_end) & (dates <= val_end)]
    test = df[(dates > val_end) & (dates <= test_end)]
    return train, val, test


__all__ = ["build_features", "split_by_date"]
