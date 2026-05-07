"""Robust loader for the Forecasting Case-Study CSV.

The raw file mixes ``MM/DD/YYYY`` and ``DD-MM-YYYY`` date formats and has an
irregular cadence per state. We:

1. Parse both date formats.
2. Coerce ``Total`` (a comma-formatted string) to float.
3. Resample every (State, Category) series to a regular weekly grid
   (``W-SUN``) and linearly interpolate small gaps so downstream models see a
   clean, evenly spaced series.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from src.config import RAW_CSV, FREQ, DATE_COL, TARGET, GROUP_COL


def _parse_date(s: str) -> pd.Timestamp:
    s = str(s).strip()
    if "-" in s:
        return pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")
    return pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")


def load_raw(path: Path | str = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["Total"] = (
        df["Total"].astype(str).str.replace(",", "", regex=False).str.strip().astype(float)
    )
    df["Date"] = df["Date"].apply(_parse_date)
    if df["Date"].isna().any():
        bad = df[df["Date"].isna()]
        raise ValueError(f"Failed to parse {len(bad)} dates")
    return df.rename(columns={"Date": DATE_COL, "Total": TARGET})


def to_weekly_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Return a long DataFrame with columns [State, ds, y] on a weekly grid."""
    out = []
    for state, g in df.groupby(GROUP_COL):
        g = g.sort_values(DATE_COL).set_index(DATE_COL)
        # Average duplicate dates (defensive); resample to weekly
        s = g[TARGET].groupby(level=0).mean().asfreq("D")
        s = s.resample(FREQ).mean()
        # Interpolate short internal gaps; forward/back fill at edges
        s = s.interpolate("linear", limit_direction="both")
        out.append(pd.DataFrame({GROUP_COL: state,
                                 DATE_COL: s.index,
                                 TARGET: s.values}))
    panel = pd.concat(out, ignore_index=True)
    return panel


def load_panel(path: Path | str = RAW_CSV) -> pd.DataFrame:
    return to_weekly_panel(load_raw(path))


def get_series(panel: pd.DataFrame, state: str) -> pd.Series:
    g = panel[panel[GROUP_COL] == state].sort_values(DATE_COL)
    return pd.Series(g[TARGET].values, index=pd.DatetimeIndex(g[DATE_COL]), name=state)
