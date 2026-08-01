from enum import Enum

from pydantic import BaseModel, Field


class SizeTier(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class Outlet(BaseModel):
    outlet_id: str
    name: str
    region: str
    size_tier: SizeTier
    base_demand_multiplier: float = Field(gt=0)
