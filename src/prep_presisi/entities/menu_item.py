from enum import Enum

from pydantic import BaseModel, Field


class MenuCategory(str, Enum):
    MAIN = "main"
    SIDE = "side"
    BEVERAGE = "beverage"


class MenuItem(BaseModel):
    menu_item_id: str
    name: str
    category: MenuCategory
    unit_price: float = Field(gt=0)  # dalam Rupiah
