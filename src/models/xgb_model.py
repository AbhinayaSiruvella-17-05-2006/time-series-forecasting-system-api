"""XGBoost regressor with lag + calendar features and recursive forecasting."""
from __future__ import annotations
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.models.base import ForecastModel
from src.features.build import build_supervised, feature_columns, add_calendar, LAGS, ROLL_WINDOWS
from src.config import FREQ, DATE_COL, TARGET, SEED


class XGBModel(ForecastModel):
    name = "XGBoost"

    def __init__(self, **kwargs):
        defaults = dict(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=SEED,
            n_jobs=1,
            verbosity=0,
        )
        defaults.update(kwargs)
        self._params = defaults
        self._model = None
        self._history: pd.Series | None = None
        self._feat_cols: list[str] | None = None

    def fit(self, y: pd.Series) -> "XGBModel":
        sup = build_supervised(y).dropna().reset_index(drop=True)
        self._feat_cols = feature_columns(sup)
        X = sup[self._feat_cols].values
        target = sup[TARGET].values
        self._model = XGBRegressor(**self._params)
        self._model.fit(X, target)
        self._history = y.copy()
        return self

    def _row_for(self, future_date: pd.Timestamp, history: pd.Series) -> pd.DataFrame:
        row = {DATE_COL: future_date, TARGET: np.nan}
        df = pd.DataFrame([row])
        df = add_calendar(df)
        for L in LAGS:
            df[f"lag_{L}"] = history.iloc[-L] if len(history) >= L else np.nan
        for w in ROLL_WINDOWS:
            tail = history.iloc[-w:] if len(history) >= w else history
            df[f"rmean_{w}"] = tail.mean()
            df[f"rstd_{w}"] = tail.std() if len(tail) > 1 else 0.0
        return df[self._feat_cols].ffill(axis=1).fillna(0)

    def predict(self, horizon: int) -> pd.Series:
        history = self._history.copy()
        future_idx = pd.date_range(history.index[-1], periods=horizon + 1, freq=FREQ)[1:]
        preds = []
        for d in future_idx:
            X = self._row_for(d, history).values
            yhat = float(self._model.predict(X)[0])
            preds.append(yhat)
            history = pd.concat([history, pd.Series([yhat], index=[d])])
        return pd.Series(preds, index=future_idx, name=self.name)
