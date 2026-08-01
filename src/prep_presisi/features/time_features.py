import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Perkaya kolom waktu di atas flag yang sudah ada dari data generator
    (day_of_week, is_weekend, is_payday_week, is_ramadan, is_lebaran_week)."""
    result = df.copy()
    date_col = pd.to_datetime(result["date"])
    result["month"] = date_col.dt.month
    result["day_of_month"] = date_col.dt.day
    result["week_of_year"] = date_col.dt.isocalendar().week.astype(int)
    result["quarter"] = date_col.dt.quarter
    result["year"] = date_col.dt.year
    return result


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """outlet_id & menu_item_id -> dtype category, dipakai XGBoost native categorical
    split (enable_categorical=True) — bukan one-hot manual (lihat TRD §3)."""
    result = df.copy()
    result["outlet_id"] = result["outlet_id"].astype("category")
    result["menu_item_id"] = result["menu_item_id"].astype("category")
    return result
