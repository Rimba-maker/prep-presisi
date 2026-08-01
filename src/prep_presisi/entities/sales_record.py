from datetime import date

from pydantic import BaseModel


class SalesRecord(BaseModel):
    date: date
    outlet_id: str
    menu_item_id: str
    qty_prepared: int  # >= 0
    qty_sold: int  # >= 0, <= qty_prepared
    day_of_week: int  # 0-6
    is_weekend: bool
    is_payday_week: bool
    is_ramadan: bool
    is_lebaran_week: bool

    @property
    def qty_waste(self) -> int:
        return self.qty_prepared - self.qty_sold
