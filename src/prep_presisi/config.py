"""Public API: load_business_rules(), load_simulation_config()."""

import tomllib
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "config"


class WeekendRule(BaseModel):
    uplift_multiplier: float = Field(gt=0)


class PaydayRule(BaseModel):
    uplift_multiplier: float = Field(gt=0)
    days_before_after: int = Field(ge=0)


class RamadanRule(BaseModel):
    daytime_multiplier: float = Field(gt=0)
    pre_iftar_multiplier: float = Field(gt=0)


class LebaranRule(BaseModel):
    week_multiplier: float = Field(gt=0)


class NoiseRule(BaseModel):
    daily_std_pct: float = Field(ge=0)


class DatePeriod(BaseModel):
    start: date
    end: date


class BusinessRulesConfig(BaseModel):
    weekend: WeekendRule
    payday: PaydayRule
    ramadan: RamadanRule
    lebaran: LebaranRule
    noise: NoiseRule
    ramadan_periods: list[DatePeriod]
    lebaran_periods: list[DatePeriod]


class SimulationScope(BaseModel):
    num_outlets: int = Field(gt=0)
    menu_items: list[str]
    start_date: date
    end_date: date


class ReproducibilityConfig(BaseModel):
    random_seed: int


class SimulationConfig(BaseModel):
    scope: SimulationScope
    reproducibility: ReproducibilityConfig


def load_business_rules(path: Path | None = None) -> BusinessRulesConfig:
    path = path or CONFIG_DIR / "business_rules.toml"
    with path.open("rb") as f:
        data = tomllib.load(f)
    return BusinessRulesConfig.model_validate(data)


def load_simulation_config(path: Path | None = None) -> SimulationConfig:
    path = path or CONFIG_DIR / "simulation.toml"
    with path.open("rb") as f:
        data = tomllib.load(f)
    return SimulationConfig.model_validate(data)
