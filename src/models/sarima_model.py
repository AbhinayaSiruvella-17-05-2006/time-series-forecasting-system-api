"""SARIMA wrapper using statsmodels.

We use a small order set tuned for noisy weekly data with yearly seasonality
(period 52). Order is fixed for speed; for production this would be chosen
per-series via AIC search inside ``fit``.
"""
from __future__ import annotations
import warnings
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.models.base import ForecastModel
from src.config import FREQ


class SarimaModel(ForecastModel):
    name = "SARIMA"

    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 1, 1, 52)):
        self.order = order
        self.seasonal_order = seasonal_order
        self._res = None
        self._last_index = None

    def fit(self, y: pd.Series) -> "SarimaModel":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = SARIMAX(y, order=self.order,
                                seasonal_order=self.seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                self._res = model.fit(disp=False, maxiter=200)
            except Exception:
                # fallback: drop seasonality if too short / unstable
                model = SARIMAX(y, order=self.order,
                                seasonal_order=(0, 0, 0, 0),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                self._res = model.fit(disp=False, maxiter=200)
        self._last_index = y.index[-1]
        return self

    def predict(self, horizon: int) -> pd.Series:
        future_idx = pd.date_range(self._last_index, periods=horizon + 1, freq=FREQ)[1:]
        fc = self._res.forecast(steps=horizon)
        return pd.Series(fc.values, index=future_idx, name=self.name)
