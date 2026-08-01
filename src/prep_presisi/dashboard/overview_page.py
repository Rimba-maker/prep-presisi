import pandas as pd
import streamlit as st

from prep_presisi.dashboard._data import (
    compute_backtest_results,
    load_latest_model,
    load_raw_data,
)
from prep_presisi.evaluation import evaluate
from prep_presisi.evaluation.metrics import mape as mape_fn

st.header("Overview")
st.caption(
    "Ringkasan performa forecast lintas seluruh outlet — backtest terhadap test set "
    "(2025-11-01 s/d 2025-12-31)."
)

model = load_latest_model()
sales, outlets, menu_items = load_raw_data()
backtest = compute_backtest_results(model, sales, menu_items)

overall_metrics = evaluate(backtest["qty_sold"], backtest["predicted_qty"])
total_waste_avoided = backtest["waste_avoided_rupiah"].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total waste avoided (test set)", f"Rp {total_waste_avoided:,.0f}")
col2.metric("MAPE", f"{overall_metrics['mape']:.1%}")
col3.metric("WMAPE", f"{overall_metrics['wmape']:.1%}")

st.subheader("Performa per outlet")

backtest = backtest.copy()
backtest["outlet_id"] = backtest["outlet_id"].astype(str)

per_outlet_waste = backtest.groupby("outlet_id", observed=True)[
    "waste_avoided_rupiah"
].sum()
per_outlet_mape = backtest.groupby("outlet_id", observed=True).apply(
    lambda g: mape_fn(g["qty_sold"], g["predicted_qty"]), include_groups=False
)

outlet_summary = (
    pd.DataFrame(
        {
            "waste_avoided_rupiah": per_outlet_waste,
            "mape": per_outlet_mape,
        }
    )
    .reset_index()
    .merge(outlets[["outlet_id", "name", "region"]], on="outlet_id", how="left")
    .sort_values("waste_avoided_rupiah", ascending=False)
)
outlet_summary["mape"] = outlet_summary["mape"].map(lambda x: f"{x:.1%}")

display_table = outlet_summary[
    ["name", "region", "waste_avoided_rupiah", "mape"]
].rename(
    columns={
        "name": "Outlet",
        "region": "Wilayah",
        "waste_avoided_rupiah": "Waste avoided (Rp)",
        "mape": "MAPE",
    }
)

st.dataframe(
    display_table,
    width="stretch",
    hide_index=True,
    column_config={
        "Waste avoided (Rp)": st.column_config.NumberColumn(format="Rp %d"),
    },
)
