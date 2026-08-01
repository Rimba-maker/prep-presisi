from datetime import date, timedelta

import numpy as np
import pandas as pd

from prep_presisi.config import BusinessRulesConfig, SimulationConfig
from prep_presisi.datagen.calendar_rules import (
    is_lebaran_week,
    is_payday_week,
    is_ramadan,
    is_weekend,
)
from prep_presisi.datagen.demand_simulator import simulate_sales
from prep_presisi.entities import MenuCategory, MenuItem, Outlet, SalesRecord, SizeTier

# Katalog menu tetap (nama/kategori/harga) — metadata referensi, bukan business rule,
# jadi tidak perlu masuk config/*.toml (yang menampung parameter yang bisa berubah).
_MENU_CATALOG: dict[str, tuple[str, MenuCategory, float]] = {
    "soto_ayam": ("Soto Ayam", MenuCategory.MAIN, 15000.0),
    "soto_daging": ("Soto Daging", MenuCategory.MAIN, 18000.0),
    "sate": ("Sate", MenuCategory.MAIN, 20000.0),
    "perkedel": ("Perkedel", MenuCategory.SIDE, 3000.0),
    "tempe_goreng": ("Tempe Goreng", MenuCategory.SIDE, 2500.0),
    "kerupuk": ("Kerupuk", MenuCategory.SIDE, 2000.0),
    "nasi_putih": ("Nasi Putih", MenuCategory.SIDE, 5000.0),
    "es_teh": ("Es Teh", MenuCategory.BEVERAGE, 5000.0),
    "es_jeruk": ("Es Jeruk", MenuCategory.BEVERAGE, 7000.0),
    "teh_hangat": ("Teh Hangat", MenuCategory.BEVERAGE, 4000.0),
}

_REGIONS = ["Boyolali", "Solo", "Semarang", "Klaten", "Sragen"]
_SIZE_TIER_BASE_RANGE = {
    SizeTier.SMALL: (0.5, 0.9),
    SizeTier.MEDIUM: (0.9, 1.3),
    SizeTier.LARGE: (1.3, 1.9),
}
_SIZE_TIER_WEIGHTS = {SizeTier.SMALL: 0.5, SizeTier.MEDIUM: 0.35, SizeTier.LARGE: 0.15}


def _build_outlets(num_outlets: int, rng: np.random.Generator) -> list[Outlet]:
    tiers = list(_SIZE_TIER_WEIGHTS.keys())
    weights = list(_SIZE_TIER_WEIGHTS.values())
    outlets = []
    for i in range(num_outlets):
        tier = tiers[rng.choice(len(tiers), p=weights)]
        low, high = _SIZE_TIER_BASE_RANGE[tier]
        multiplier = round(float(rng.uniform(low, high)), 3)
        region = _REGIONS[i % len(_REGIONS)]
        outlets.append(
            Outlet(
                outlet_id=f"OUT{i + 1:03d}",
                name=f"Soto Boyolali {region} #{i + 1}",
                region=region,
                size_tier=tier,
                base_demand_multiplier=multiplier,
            )
        )
    return outlets


def _build_menu_items(menu_item_ids: list[str]) -> list[MenuItem]:
    items = []
    for mid in menu_item_ids:
        if mid not in _MENU_CATALOG:
            raise ValueError(
                f"menu_item_id '{mid}' tidak ada di katalog internal _MENU_CATALOG"
            )
        name, category, price = _MENU_CATALOG[mid]
        items.append(
            MenuItem(menu_item_id=mid, name=name, category=category, unit_price=price)
        )
    return items


def _date_range(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _build_calendar_flags(
    dates: list[date], rules: BusinessRulesConfig
) -> pd.DataFrame:
    """Flag kalender dihitung sekali per tanggal unik (bukan per baris outlet x menu)."""
    return pd.DataFrame(
        {
            "date": dates,
            "day_of_week": [d.weekday() for d in dates],
            "is_weekend": [is_weekend(d) for d in dates],
            "is_payday_week": [
                is_payday_week(d, rules.payday.days_before_after) for d in dates
            ],
            "is_ramadan": [is_ramadan(d, rules.ramadan_periods) for d in dates],
            "is_lebaran_week": [
                is_lebaran_week(d, rules.lebaran_periods) for d in dates
            ],
        }
    )


def generate_dataset(
    sim_config: SimulationConfig, rules: BusinessRulesConfig
) -> tuple[list[Outlet], list[MenuItem], list[SalesRecord]]:
    rng = np.random.default_rng(sim_config.reproducibility.random_seed)

    outlets = _build_outlets(sim_config.scope.num_outlets, rng)
    menu_items = _build_menu_items(sim_config.scope.menu_items)
    dates = _date_range(sim_config.scope.start_date, sim_config.scope.end_date)
    calendar_flags = _build_calendar_flags(dates, rules)

    outlet_ids = [o.outlet_id for o in outlets]
    menu_item_ids = [m.menu_item_id for m in menu_items]

    grid = pd.MultiIndex.from_product(
        [outlet_ids, menu_item_ids, dates], names=["outlet_id", "menu_item_id", "date"]
    ).to_frame(index=False)
    grid = grid.merge(calendar_flags, on="date", how="left")

    outlets_by_id = {o.outlet_id: o for o in outlets}
    menu_items_by_id = {m.menu_item_id: m for m in menu_items}

    grid = simulate_sales(grid, outlets_by_id, menu_items_by_id, rules, rng)

    records = [
        SalesRecord(
            date=row.date,
            outlet_id=row.outlet_id,
            menu_item_id=row.menu_item_id,
            qty_prepared=int(row.qty_prepared),
            qty_sold=int(row.qty_sold),
            day_of_week=int(row.day_of_week),
            is_weekend=bool(row.is_weekend),
            is_payday_week=bool(row.is_payday_week),
            is_ramadan=bool(row.is_ramadan),
            is_lebaran_week=bool(row.is_lebaran_week),
        )
        for row in grid.itertuples(index=False)
    ]

    return outlets, menu_items, records
