from pydantic import BaseModel


class InsightContext(BaseModel):
    """Input terstruktur ke LLM — satu-satunya cara insights/ menerima data."""

    outlet_name: str
    menu_item_name: str
    predicted_qty: int
    historical_avg_qty: float
    waste_avoided_rupiah: float
    is_weekend: bool
    is_payday_week: bool
    is_ramadan: bool
    is_lebaran_week: bool
