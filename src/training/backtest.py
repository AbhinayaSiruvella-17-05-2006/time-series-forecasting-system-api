"""Walk-forward holdout backtest (no leakage)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import pandas as pd

from src.models.base import ForecastModel
from src.utils.metrics import all_metrics


@dataclass
class BacktestResult:
    model: str
    smape: float
    mae: float
    rmse: float


def holdout_backtest(model: ForecastModel, y: pd.Series, horizon: int) -> BacktestResult:
    train, test = y.iloc[:-horizon], y.iloc[-horizon:]
    model.fit(train)
    preds = model.predict(horizon)
    # Align indexes defensively
    preds = preds.iloc[:horizon].values
    m = all_metrics(test.values, preds)
    return BacktestResult(model=model.name, **m)


def to_dict(r: BacktestResult) -> dict:
    return asdict(r)
