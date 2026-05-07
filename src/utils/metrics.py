"""Forecast accuracy metrics."""
from __future__ import annotations
import numpy as np


def _arr(a):
    return np.asarray(a, dtype=float)


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in percent. Robust to zeros."""
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / np.where(denom == 0, 1.0, denom)
    return float(np.mean(diff) * 100.0)


def all_metrics(y_true, y_pred) -> dict:
    return {"smape": smape(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "rmse": rmse(y_true, y_pred)}
