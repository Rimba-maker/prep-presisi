import numpy as np
import pandas as pd

from prep_presisi.config import BusinessRulesConfig
from prep_presisi.entities import MenuCategory, MenuItem, Outlet

# ponytail: bobot kategori adalah heuristik tetap (menu utama laku lebih banyak dari
# side/minuman) — bukan business rule per PRD §5, tidak perlu tuning per-item untuk
# project latihan ini. Naikkan ke config kalau nanti butuh kalibrasi lebih presisi.
_CATEGORY_WEIGHT: dict[MenuCategory, float] = {
    MenuCategory.MAIN: 1.0,
    MenuCategory.SIDE: 0.5,
    MenuCategory.BEVERAGE: 0.6,
}
_BASE_DAILY_QTY = 20.0
_ROLLING_WINDOW_DAYS = 7


def _ramadan_multiplier(rules: BusinessRulesConfig) -> float:
    """Grain data ini harian (bukan per-jam), jadi efek 'siang turun' + 'jelang buka
    puasa naik' dari config digabung jadi satu multiplier harian berbobot 40/60 —
    soto lebih ramai jam makan malam/buka puasa daripada siang hari."""
    return (
        0.4 * rules.ramadan.daytime_multiplier
        + 0.6 * rules.ramadan.pre_iftar_multiplier
    )


def compute_true_demand(
    df: pd.DataFrame,
    outlets_by_id: dict[str, Outlet],
    menu_items_by_id: dict[str, MenuItem],
    rules: BusinessRulesConfig,
    rng: np.random.Generator,
) -> pd.Series:
    """Demand harian 'sebenarnya' (float, belum dibulatkan) per baris outlet x menu x hari.
    `df` harus sudah punya kolom outlet_id, menu_item_id, is_weekend, is_payday_week,
    is_ramadan, is_lebaran_week."""
    outlet_multiplier = {
        oid: o.base_demand_multiplier for oid, o in outlets_by_id.items()
    }
    category_weight = {
        mid: _CATEGORY_WEIGHT[m.category] for mid, m in menu_items_by_id.items()
    }

    base_mult = df["outlet_id"].map(outlet_multiplier)
    cat_weight = df["menu_item_id"].map(category_weight)
    base_demand = _BASE_DAILY_QTY * base_mult * cat_weight

    weekend_mult = np.where(df["is_weekend"], rules.weekend.uplift_multiplier, 1.0)
    payday_mult = np.where(df["is_payday_week"], rules.payday.uplift_multiplier, 1.0)
    ramadan_mult = np.where(df["is_ramadan"], _ramadan_multiplier(rules), 1.0)
    lebaran_mult = np.where(df["is_lebaran_week"], rules.lebaran.week_multiplier, 1.0)

    noise = np.clip(rng.normal(1.0, rules.noise.daily_std_pct, size=len(df)), 0.0, None)

    true_demand = (
        base_demand.to_numpy()
        * weekend_mult
        * payday_mult
        * ramadan_mult
        * lebaran_mult
        * noise
    )
    return pd.Series(true_demand, index=df.index)


def simulate_qty_prepared(
    df: pd.DataFrame,
    true_demand: pd.Series,
    rules: BusinessRulesConfig,
    rng: np.random.Generator,
) -> pd.Series:
    """Operator historis menyiapkan stok berdasar rata-rata demand N hari terakhir (bukan
    demand hari ini yang belum diketahui) plus noise — inilah sumber gap qty_prepared vs
    qty_sold yang merepresentasikan waste (lihat PRD §5 Aturan Generation)."""
    working = df[["outlet_id", "menu_item_id", "date"]].copy()
    working["true_demand"] = true_demand.to_numpy()
    working = working.sort_values(["outlet_id", "menu_item_id", "date"])

    grouped = working.groupby(["outlet_id", "menu_item_id"], sort=False)["true_demand"]
    rolling_avg = grouped.transform(
        lambda s: s.shift(1).rolling(window=_ROLLING_WINDOW_DAYS, min_periods=1).mean()
    )
    # Hari pertama tiap series belum punya histori — fallback ke true_demand hari itu sendiri.
    rolling_avg = rolling_avg.fillna(working["true_demand"])

    operator_noise = np.clip(
        rng.normal(1.0, rules.noise.daily_std_pct, size=len(working)), 0.0, None
    )

    qty_prepared = np.round(rolling_avg.to_numpy() * operator_noise).astype(int)
    qty_prepared = np.clip(qty_prepared, 0, None)

    result = pd.Series(qty_prepared, index=working.index)
    return result.reindex(df.index)


def simulate_sales(
    df: pd.DataFrame,
    outlets_by_id: dict[str, Outlet],
    menu_items_by_id: dict[str, MenuItem],
    rules: BusinessRulesConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Tambah kolom qty_prepared & qty_sold ke df. `df` harus sudah punya outlet_id,
    menu_item_id, date, is_weekend, is_payday_week, is_ramadan, is_lebaran_week."""
    true_demand = compute_true_demand(df, outlets_by_id, menu_items_by_id, rules, rng)
    qty_prepared = simulate_qty_prepared(df, true_demand, rules, rng)
    qty_sold = np.minimum(
        qty_prepared.to_numpy(), np.round(true_demand.to_numpy()).astype(int)
    )
    qty_sold = np.clip(qty_sold, 0, None)

    result = df.copy()
    result["qty_prepared"] = qty_prepared.to_numpy()
    result["qty_sold"] = qty_sold
    return result
