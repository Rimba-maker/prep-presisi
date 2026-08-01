from datetime import date

import pytest
from pydantic import ValidationError

from prep_presisi.entities import SalesRecord


def _base_kwargs(**overrides):
    kwargs = {
        "date": date(2024, 1, 1),
        "outlet_id": "OUT001",
        "menu_item_id": "soto_ayam",
        "qty_prepared": 10,
        "qty_sold": 8,
        "day_of_week": 0,
        "is_weekend": False,
        "is_payday_week": False,
        "is_ramadan": False,
        "is_lebaran_week": False,
    }
    kwargs.update(overrides)
    return kwargs


def test_valid_record_accepted():
    record = SalesRecord(**_base_kwargs())
    assert record.qty_waste == 2


def test_qty_sold_over_prepared_rejected():
    with pytest.raises(ValidationError):
        SalesRecord(**_base_kwargs(qty_prepared=5, qty_sold=10))


def test_negative_qty_prepared_rejected():
    with pytest.raises(ValidationError):
        SalesRecord(**_base_kwargs(qty_prepared=-1))


def test_negative_qty_sold_rejected():
    with pytest.raises(ValidationError):
        SalesRecord(**_base_kwargs(qty_sold=-1))
