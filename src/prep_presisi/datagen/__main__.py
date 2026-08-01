"""Jalankan: `uv run python -m prep_presisi.datagen` — generate dataset & simpan ke data/raw/*.parquet."""

from pathlib import Path

import pandas as pd

from prep_presisi.config import load_business_rules, load_simulation_config
from prep_presisi.datagen import generate_dataset

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"


def main() -> None:
    sim_config = load_simulation_config()
    rules = load_business_rules()

    print(
        f"Generating {sim_config.scope.num_outlets} outlets x "
        f"{len(sim_config.scope.menu_items)} menu items x "
        f"{sim_config.scope.start_date}..{sim_config.scope.end_date} ..."
    )
    outlets, menu_items, sales_records = generate_dataset(sim_config, rules)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([o.model_dump() for o in outlets]).to_parquet(
        RAW_DIR / "outlets.parquet", index=False
    )
    pd.DataFrame([m.model_dump() for m in menu_items]).to_parquet(
        RAW_DIR / "menu_items.parquet", index=False
    )
    pd.DataFrame([r.model_dump() for r in sales_records]).to_parquet(
        RAW_DIR / "sales_records.parquet", index=False
    )

    print(
        f"Saved {len(outlets)} outlets, {len(menu_items)} menu items, "
        f"{len(sales_records)} sales records to {RAW_DIR}"
    )


if __name__ == "__main__":
    main()
