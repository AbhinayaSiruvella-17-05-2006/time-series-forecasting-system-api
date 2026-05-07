"""Facebook Prophet wrapper."""
from __future__ import annotations
import logging
import pandas as pd

from src.models.base import ForecastModel
from src.config import FREQ

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


class ProphetModel(ForecastModel):
    name = "Prophet"

    def __init__(self, **kwargs):
        # weekly cadence => disable daily, keep yearly
        defaults = dict(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            interval_width=0.8,
        )
        defaults.update(kwargs)
        self._kwargs = defaults
        self._model = None
        self._last_index = None

    def fit(self, y: pd.Series) -> "ProphetModel":
        from prophet import Prophet  # lazy import (slow)
        df = pd.DataFrame({"ds": y.index, "y": y.values})
        self._model = Prophet(**self._kwargs)
        self._model.add_country_holidays(country_name="US")
        self._model.fit(df)
        self._last_index = y.index[-1]
        return self

    def predict(self, horizon: int) -> pd.Series:
        future_idx = pd.date_range(self._last_index, periods=horizon + 1, freq=FREQ)[1:]
        future = pd.DataFrame({"ds": future_idx})
        fc = self._model.predict(future)
        return pd.Series(fc["yhat"].values, index=future_idx, name=self.name)
