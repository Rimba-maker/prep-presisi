"""Jalankan: `uv run python -m prep_presisi.models <model_name>` — latih & evaluasi 1
model terhadap test set, cetak metrik + log ke artifacts/experiments.jsonl (TRD §6.2).
Default model_name: seasonal_naive."""

import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import joblib
import pandas as pd

from prep_presisi.evaluation import (
    compute_waste_avoided,
    evaluate,
    total_waste_avoided_rupiah,
)
from prep_presisi.features import build_features, split_by_date
from prep_presisi.models import get_model

ROOT_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT_DIR / "data" / "raw"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
EXPERIMENTS_LOG = ARTIFACTS_DIR / "experiments.jsonl"

# Sesuai PRD §5.1
TRAIN_END = date(2025, 8, 31)
VAL_END = date(2025, 10, 31)
TEST_END = date(2025, 12, 31)


def _log_experiment(model_name: str, metrics: dict[str, float], params: dict) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model_name": model_name,
        **metrics,
        "params": params,
    }
    with EXPERIMENTS_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _save_model(model, model_name: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = MODELS_DIR / f"{model_name}_{timestamp}.joblib"
    joblib.dump(model, path)
    return path


def _print_baseline_comparison(model_name: str, metrics: dict[str, float]) -> None:
    if not EXPERIMENTS_LOG.exists():
        return
    baseline_entries = [
        json.loads(line)
        for line in EXPERIMENTS_LOG.read_text().splitlines()
        if json.loads(line)["model_name"] == "seasonal_naive"
    ]
    if not baseline_entries:
        return
    baseline = baseline_entries[-1]
    print(f"\nPerbandingan vs baseline ({baseline['model_name']}):")
    for metric in ("mape", "wmape"):
        base_val = baseline[metric]
        model_val = metrics[metric]
        improvement = (base_val - model_val) / base_val
        print(
            f"  {metric}: baseline={base_val:.4f} vs {model_name}={model_val:.4f} "
            f"({improvement:+.1%})"
        )


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "seasonal_naive"

    sales = pd.read_parquet(RAW_DIR / "sales_records.parquet")
    menu_items = pd.read_parquet(RAW_DIR / "menu_items.parquet")
    features = build_features(sales)
    train, val, test = split_by_date(features, TRAIN_END, VAL_END, TEST_END)

    if model_name == "xgboost_global":
        model = get_model(model_name, val_df=val)
        fit_df = train
    elif model_name == "statsforecast_sample":
        # Butuh histori kontinu sampai tepat sebelum test (StatsForecast meramal h hari
        # ke depan dari akhir data fit, bukan ke tanggal spesifik) — gabung train+val.
        model = get_model(model_name)
        fit_df = pd.concat([train, val], ignore_index=True)
    else:
        model = get_model(model_name)
        fit_df = train

    model.fit(fit_df)
    result = model.predict(test)

    metrics = evaluate(result["qty_sold"], result["predicted_qty"])
    print(f"[{model.name}] test set metrics: {metrics}")

    waste_df = compute_waste_avoided(result, menu_items)
    total_waste = total_waste_avoided_rupiah(waste_df)
    metrics["waste_avoided_rupiah"] = total_waste
    print(
        f"Estimasi waste avoided (test set, {test['date'].nunique()} hari): Rp {total_waste:,.0f}"
    )

    if hasattr(model, "feature_importance"):
        print("\nFeature importance:")
        print(model.feature_importance().to_string())

    if model_name != "seasonal_naive":
        _print_baseline_comparison(model_name, metrics)

    saved_path = _save_model(model, model.name)
    print(f"\nModel disimpan ke {saved_path}")

    _log_experiment(model.name, metrics, params={})


if __name__ == "__main__":
    main()
