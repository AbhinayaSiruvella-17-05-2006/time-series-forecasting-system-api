"""Common interface every forecasting model must implement."""
from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd


class ForecastModel(ABC):
    name: str = "Base"

    @abstractmethod
    def fit(self, y: pd.Series) -> "ForecastModel":
        """Fit on a Series indexed by a DatetimeIndex (regular freq)."""

    @abstractmethod
    def predict(self, horizon: int) -> pd.Series:
        """Return a Series of length ``horizon`` indexed by future dates."""

    def fit_predict(self, y: pd.Series, horizon: int) -> pd.Series:
        self.fit(y)
        return self.predict(horizon)
