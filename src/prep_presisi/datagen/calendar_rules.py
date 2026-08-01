from calendar import monthrange
from datetime import date

from prep_presisi.config import DatePeriod


def is_weekend(d: date) -> bool:
    return d.weekday() >= 5  # Saturday=5, Sunday=6


def is_payday_week(d: date, days_before_after: int) -> bool:
    """True kalau `d` dalam N hari dari akhir bulan, atau N hari dari awal bulan —
    dua sisi ini bersama-sama mencakup jendela di sekitar pergantian bulan (gajian)."""
    month_end_day = monthrange(d.year, d.month)[1]
    days_from_month_end = month_end_day - d.day
    days_from_month_start = d.day - 1
    return (
        days_from_month_end <= days_before_after
        or days_from_month_start <= days_before_after
    )


def _in_any_period(d: date, periods: list[DatePeriod]) -> bool:
    return any(p.start <= d <= p.end for p in periods)


def is_ramadan(d: date, ramadan_periods: list[DatePeriod]) -> bool:
    return _in_any_period(d, ramadan_periods)


def is_lebaran_week(d: date, lebaran_periods: list[DatePeriod]) -> bool:
    return _in_any_period(d, lebaran_periods)
