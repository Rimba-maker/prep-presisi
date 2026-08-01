from datetime import date

from pydantic import BaseModel, Field, model_validator


class SalesRecord(BaseModel):
    date: date
    outlet_id: str
    menu_item_id: str
    qty_prepared: int = Field(ge=0)
    qty_sold: int = Field(ge=0)
    day_of_week: int = Field(ge=0, le=6)
    is_weekend: bool
    is_payday_week: bool
    is_ramadan: bool
    is_lebaran_week: bool

    @model_validator(mode="after")
    def _qty_sold_not_over_prepared(self) -> "SalesRecord":
        if self.qty_sold > self.qty_prepared:
            raise ValueError(
                f"qty_sold ({self.qty_sold}) tidak boleh melebihi qty_prepared ({self.qty_prepared})"
            )
        return self

    @property
    def qty_waste(self) -> int:
        return self.qty_prepared - self.qty_sold
