import pandas as pd, numpy as np
from src.data.loader import load_panel, get_series
from src.config import GROUP_COL, HORIZON_WEEKS
from src.models.sarima_model import SarimaModel
from src.models.xgb_model import XGBModel
from src.models.lstm_model import LSTMModel


def _series():
    panel = load_panel()
    state = sorted(panel[GROUP_COL].unique())[0]
    return get_series(panel, state)


def _check(model):
    y = _series()
    model.fit(y)
    fc = model.predict(HORIZON_WEEKS)
    assert len(fc) == HORIZON_WEEKS
    assert fc.index[0] > y.index[-1]
    assert np.isfinite(fc.values).all()


def test_sarima(): _check(SarimaModel())
def test_xgb():    _check(XGBModel(n_estimators=50))
def test_lstm():   _check(LSTMModel(epochs=5))
