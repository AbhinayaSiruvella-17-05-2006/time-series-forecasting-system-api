"""Feature engineering for tree / NN models.

Lag features (t-1, t-4, t-8 for weekly data; the spec asks for t-1, t-7, t-30
which are day-level — for weekly cadence we use the equivalent week lags),
rolling mean & std, calendar features, and a US holiday flag.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import holidays

from src.config import DATE_COL, TARGET

# Weekly equivalents of the requested daily lags
LAGS = (1, 4, 7, 8, 13, 26, 30, 52)
ROLL_WINDOWS = (4, 8, 13)

_US_HOLS = holidays.country_holidays("US")


def _holiday_in_week(end_date: pd.Timestamp) -> int:
    start = end_date - pd.Timedelta(days=6)
    days = pd.date_range(start, end_date, freq="D")
    return int(any(d in _US_HOLS for d in days))


def add_calendar(df: pd.DataFrame, date_col: str = DATE_COL) -> pd.DataFrame:
    out = df.copy()
    d = pd.to_datetime(out[date_col])
    out["weekofyear"] = d.dt.isocalendar().week.astype(int)
    out["month"] = d.dt.month
    out["quarter"] = d.dt.quarter
    out["year"] = d.dt.year
    out["dayofweek"] = d.dt.dayofweek  # constant for W-SUN but kept for completeness
    out["holiday_flag"] = d.apply(_holiday_in_week)
    # cyclical encodings
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["woy_sin"] = np.sin(2 * np.pi * out["weekofyear"] / 52)
    out["woy_cos"] = np.cos(2 * np.pi * out["weekofyear"] / 52)
    return out


def add_lags(df: pd.DataFrame, target: str = TARGET,
             lags=LAGS, rolls=ROLL_WINDOWS) -> pd.DataFrame:
    out = df.copy()
    for L in lags:
        out[f"lag_{L}"] = out[target].shift(L)
    for w in rolls:
        out[f"rmean_{w}"] = out[target].shift(1).rolling(w).mean()
        out[f"rstd_{w}"] = out[target].shift(1).rolling(w).std()
    return out


def build_supervised(series: pd.Series) -> pd.DataFrame:
    """Series indexed by date -> supervised DataFrame with features + target."""
    df = pd.DataFrame({DATE_COL: series.index, TARGET: series.values})
    df = add_calendar(df)
    df = add_lags(df)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (DATE_COL, TARGET)]
