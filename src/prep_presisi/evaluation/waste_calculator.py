import pandas as pd


def compute_waste_avoided(
    result_df: pd.DataFrame, menu_items_df: pd.DataFrame
) -> pd.DataFrame:
    """Bandingkan waste 'tanpa forecasting' (qty_prepared aktual di data — hasil keputusan
    operator historis naif, lihat datagen/demand_simulator.py) vs waste 'dengan
    forecasting' (predicted_qty dipakai sebagai qty_prepared baru). Resolusi PRD §11 open
    question: waste avoided dihitung terhadap qty_prepared aktual (skenario tanpa
    forecasting), bukan terhadap baseline model. Cuma menghitung sisi over-produksi
    (waste); risiko understock/lost-sales di luar scope metric ini."""
    merged = result_df.copy()
    merged["menu_item_id"] = merged["menu_item_id"].astype(str)

    price_lookup = menu_items_df[["menu_item_id", "unit_price"]].copy()
    price_lookup["menu_item_id"] = price_lookup["menu_item_id"].astype(str)
    merged = merged.merge(price_lookup, on="menu_item_id", how="left")

    historical_waste_qty = (merged["qty_prepared"] - merged["qty_sold"]).clip(lower=0)
    new_waste_qty = (merged["predicted_qty"] - merged["qty_sold"]).clip(lower=0)
    waste_avoided_qty = historical_waste_qty - new_waste_qty

    merged["historical_waste_qty"] = historical_waste_qty
    merged["new_waste_qty"] = new_waste_qty
    merged["waste_avoided_qty"] = waste_avoided_qty
    merged["waste_avoided_rupiah"] = waste_avoided_qty * merged["unit_price"]
    return merged


def total_waste_avoided_rupiah(waste_df: pd.DataFrame) -> float:
    return float(waste_df["waste_avoided_rupiah"].sum())
