"""LSTM forecaster (PyTorch). Sequence-to-one with recursive multi-step.

Trained on standardized differences for better stability on raw revenue.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.models.base import ForecastModel
from src.config import FREQ, SEED

WINDOW = 16  # weeks of look-back


class _LSTMNet(nn.Module):
    def __init__(self, hidden: int = 32, layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers,
                            batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMModel(ForecastModel):
    name = "LSTM"

    def __init__(self, window: int = WINDOW, hidden: int = 32,
                 epochs: int = 80, lr: float = 1e-2):
        self.window = window
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self._net: _LSTMNet | None = None
        self._mu = 0.0
        self._sd = 1.0
        self._history: pd.Series | None = None

    @staticmethod
    def _windows(arr: np.ndarray, window: int):
        X, y = [], []
        for i in range(len(arr) - window):
            X.append(arr[i: i + window])
            y.append(arr[i + window])
        return np.array(X), np.array(y)

    def fit(self, y: pd.Series) -> "LSTMModel":
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        vals = y.values.astype(np.float32)
        self._mu = float(vals.mean())
        self._sd = float(vals.std() + 1e-8)
        z = (vals - self._mu) / self._sd
        X, t = self._windows(z, self.window)
        if len(X) < 8:
            # Not enough data; fall back to predicting the last value (handled in predict).
            self._net = None
            self._history = y.copy()
            return self
        X = torch.tensor(X[:, :, None], dtype=torch.float32)
        t = torch.tensor(t, dtype=torch.float32)
        self._net = _LSTMNet(hidden=self.hidden)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self._net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            pred = self._net(X)
            loss = loss_fn(pred, t)
            loss.backward()
            opt.step()
        self._history = y.copy()
        return self

    def predict(self, horizon: int) -> pd.Series:
        future_idx = pd.date_range(self._history.index[-1], periods=horizon + 1, freq=FREQ)[1:]
        if self._net is None:
            last = float(self._history.iloc[-1])
            return pd.Series([last] * horizon, index=future_idx, name=self.name)
        z = (self._history.values.astype(np.float32) - self._mu) / self._sd
        z = list(z[-self.window:])
        self._net.eval()
        preds_z = []
        with torch.no_grad():
            for _ in range(horizon):
                x = torch.tensor(np.array(z[-self.window:])[None, :, None],
                                 dtype=torch.float32)
                yhat = float(self._net(x).item())
                preds_z.append(yhat)
                z.append(yhat)
        preds = np.array(preds_z) * self._sd + self._mu
        return pd.Series(preds, index=future_idx, name=self.name)
