from datetime import timedelta

import pandas as pd
import streamlit as st

from prep_presisi.dashboard._components import data_table, entity_picker
from prep_presisi.dashboard._data import (
    cached_insight,
    compute_backtest_results,
    compute_next_day_predictions,
    load_latest_model,
    load_raw_data,
    load_rules,
)
from prep_presisi.entities import InsightContext
from prep_presisi.evaluation.metrics import mape as mape_fn

st.header("Detail outlet")

model = load_latest_model()
sales, outlets, menu_items = load_raw_data()
rules = load_rules()

selected_outlet_id, selected_name = entity_picker(
    outlets, id_col="outlet_id", name_col="name", label="Pilih outlet"
)

next_date = pd.to_datetime(sales["date"]).max().date() + timedelta(days=1)
st.caption(f"Rekomendasi qty prep untuk {next_date.isoformat()}")

next_day = compute_next_day_predictions(model, sales, rules)
next_day = next_day.copy()
next_day["outlet_id"] = next_day["outlet_id"].astype(str)
next_day["menu_item_id"] = next_day["menu_item_id"].astype(str)
outlet_next_day = next_day[next_day["outlet_id"] == selected_outlet_id]

backtest = compute_backtest_results(model, sales, menu_items)
backtest = backtest.copy()
backtest["outlet_id"] = backtest["outlet_id"].astype(str)
backtest["menu_item_id"] = backtest["menu_item_id"].astype(str)
outlet_backtest = backtest[backtest["outlet_id"] == selected_outlet_id]

per_item_stats = (
    outlet_backtest.groupby("menu_item_id", observed=True)
    .apply(
        lambda g: pd.Series(
            {
                "avg_waste_avoided_rupiah": g["waste_avoided_rupiah"].mean(),
                "avg_actual_qty": g["qty_sold"].mean(),
                "mape": mape_fn(g["qty_sold"], g["predicted_qty"]),
            }
        ),
        include_groups=False,
    )
    .reset_index()
)

table = outlet_next_day.merge(
    menu_items[["menu_item_id", "name"]], on="menu_item_id", how="left"
).merge(per_item_stats, on="menu_item_id", how="left")

table["recommended_qty_prep"] = table["predicted_qty"].round().astype(int)
table["error_range"] = (table["predicted_qty"] * table["mape"]).round().astype(int)

data_table(
    table,
    rename={
        "name": "Menu item",
        "recommended_qty_prep": "Qty prep disarankan",
        "error_range": "± error (estimasi historis)",
        "avg_waste_avoided_rupiah": "Estimasi waste avoided (Rp)",
    },
    number_formats={"Estimasi waste avoided (Rp)": "Rp %d"},
)

st.caption(
    "Estimasi waste avoided & error range dihitung dari rata-rata performa model di test "
    "set historis (2025-11-01 s/d 2025-12-31) untuk kombinasi outlet-menu ini — bukan "
    "jaminan angka pasti untuk besok."
)

st.divider()
show_insights = st.checkbox(
    "Tampilkan insight naratif (opsional — butuh OPENROUTER_API_KEY di .env)",
    value=False,
)
if show_insights:
    for row in table.itertuples():
        context = InsightContext(
            outlet_name=selected_name,
            menu_item_name=row.name,
            predicted_qty=row.recommended_qty_prep,
            historical_avg_qty=float(row.avg_actual_qty)
            if pd.notna(row.avg_actual_qty)
            else float(row.recommended_qty_prep),
            waste_avoided_rupiah=float(row.avg_waste_avoided_rupiah)
            if pd.notna(row.avg_waste_avoided_rupiah)
            else 0.0,
            is_weekend=bool(row.is_weekend),
            is_payday_week=bool(row.is_payday_week),
            is_ramadan=bool(row.is_ramadan),
            is_lebaran_week=bool(row.is_lebaran_week),
        )
        insight = cached_insight(context)
        if insight:
            st.write(f"**{row.name}:** {insight}")
        else:
            st.write(
                f"**{row.name}:** _(insight tidak tersedia — cek OPENROUTER_API_KEY)_"
            )
