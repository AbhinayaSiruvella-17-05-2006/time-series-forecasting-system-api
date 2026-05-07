import numpy as np
from src.utils.metrics import mae, rmse, smape


def test_metrics_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0
    assert rmse(y, y) == 0
    assert smape(y, y) == 0


def test_smape_bounded():
    y = np.array([100.0, 200.0])
    p = np.array([110.0, 180.0])
    s = smape(y, p)
    assert 0 < s < 200
