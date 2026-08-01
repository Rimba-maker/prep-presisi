"""Helper internal dashboard — load & cache resource/data yang dipakai lintas halaman
(overview & detail). Bukan public API prep_presisi, cuma dipakai di dalam dashboard/."""

from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from prep_presisi.config import BusinessRulesConfig, load_business_rules
from prep_presisi.datagen.calendar_rules import (
    is_lebaran_week,
    is_payday_week,
    is_ramadan,
    is_weekend,
)
from prep_presisi.entities import InsightContext
from prep_presisi.evaluation import compute_waste_avoided
from prep_presisi.features import build_features, split_by_date
from prep_presisi.insights import generate_insight

ROOT_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT_DIR / "data" / "raw"
MODELS_DIR = ROOT_DIR / "artifacts" / "models"

# Sesuai PRD §5.1
TRAIN_END = date(2025, 8, 31)
VAL_END = date(2025, 10, 31)
TEST_END = date(2025, 12, 31)

_NEXT_DAY_LOOKBACK_DAYS = (
    21  # cukup buat lag_14/rolling_mean_14, tak perlu histori penuh
)


@st.cache_resource
def load_latest_model(model_name: str = "xgboost_global"):
    candidates = sorted(MODELS_DIR.glob(f"{model_name}_*.joblib"))
    if not candidates:
        raise FileNotFoundError(
            f"Tidak ada model artifact '{model_name}' di {MODELS_DIR} — jalankan "
            f"`uv run python -m prep_presisi.models {model_name}` dulu."
        )
    return joblib.load(candidates[-1])


@st.cache_data
def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales = pd.read_parquet(RAW_DIR / "sales_records.parquet")
    outlets = pd.read_parquet(RAW_DIR / "outlets.parquet")
    menu_items = pd.read_parquet(RAW_DIR / "menu_items.parquet")
    return sales, outlets, menu_items


@st.cache_data
def load_rules() -> BusinessRulesConfig:
    return load_business_rules()


@st.cache_data
def compute_backtest_results(
    _model, sales: pd.DataFrame, menu_items: pd.DataFrame
) -> pd.DataFrame:
    """Prediksi model di test set (2025-11-01 s/d 2025-12-31) + waste avoided per baris —
    dasar ringkasan performa di Overview & estimasi range di Detail."""
    features = build_features(sales)
    _train, _val, test = split_by_date(features, TRAIN_END, VAL_END, TEST_END)
    result = _model.predict(test)
    return compute_waste_avoided(result, menu_items)


def _build_next_day_stub(
    sales: pd.DataFrame, rules: BusinessRulesConfig
) -> pd.DataFrame:
    last_date = pd.to_datetime(sales["date"]).max().date()
    next_date = last_date + timedelta(days=1)

    stub = sales[["outlet_id", "menu_item_id"]].drop_duplicates().reset_index(drop=True)
    stub["date"] = next_date
    stub["qty_prepared"] = 0
    stub["qty_sold"] = 0.0
    stub["day_of_week"] = next_date.weekday()
    stub["is_weekend"] = is_weekend(next_date)
    stub["is_payday_week"] = is_payday_week(next_date, rules.payday.days_before_after)
    stub["is_ramadan"] = is_ramadan(next_date, rules.ramadan_periods)
    stub["is_lebaran_week"] = is_lebaran_week(next_date, rules.lebaran_periods)
    return stub


@st.cache_data
def compute_next_day_predictions(
    _model, sales: pd.DataFrame, rules: BusinessRulesConfig
) -> pd.DataFrame:
    """1 baris prediksi per outlet x menu untuk hari setelah tanggal terakhir di data —
    lag/rolling dihitung dari histori asli (cukup lookback singkat, bukan dataset penuh)
    lewat build_features(); baris stub cuma nge-set tanggal & flag kalender (qty_sold
    belum diketahui, memang belum terjadi)."""
    last_date = pd.to_datetime(sales["date"]).max().date()
    next_date = last_date + timedelta(days=1)
    cutoff = last_date - timedelta(days=_NEXT_DAY_LOOKBACK_DAYS)

    recent = sales[pd.to_datetime(sales["date"]).dt.date >= cutoff]
    stub = _build_next_day_stub(sales, rules)

    combined = pd.concat([recent, stub], ignore_index=True)
    features = build_features(combined)

    next_day_mask = pd.to_datetime(features["date"]).dt.date == next_date
    next_day_input = features[next_day_mask].reset_index(drop=True)

    return _model.predict(next_day_input)


@st.cache_data
def cached_insight(context: InsightContext) -> str | None:
    """Cache key = isi InsightContext (outlet, menu, tanggal via flag kalender, dst) —
    generate_insight() cuma dipanggil ulang kalau datanya benar-benar berubah (TRD §3.2)."""
    return generate_insight(context)
