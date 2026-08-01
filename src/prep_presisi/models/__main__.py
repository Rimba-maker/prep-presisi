"""Jalankan: `uv run python -m prep_presisi.models <model_name>` — latih & evaluasi 1
model terhadap test set, cetak metrik + log ke artifacts/experiments.jsonl (TRD §6.2).
Default model_name: seasonal_naive."""

import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from prep_presisi.evaluation import evaluate
from prep_presisi.features import build_features, split_by_date
from prep_presisi.models import get_model

ROOT_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT_DIR / "data" / "raw"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
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


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "seasonal_naive"

    sales = pd.read_parquet(RAW_DIR / "sales_records.parquet")
    features = build_features(sales)
    train, _val, test = split_by_date(features, TRAIN_END, VAL_END, TEST_END)

    model = get_model(model_name)
    model.fit(train)
    result = model.predict(test)

    metrics = evaluate(result["qty_sold"], result["predicted_qty"])
    print(f"[{model.name}] test set metrics: {metrics}")

    _log_experiment(model.name, metrics, params={})


if __name__ == "__main__":
    main()
