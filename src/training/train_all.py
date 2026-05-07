"""Train every model for every state, pick the per-state champion, persist.

Run with:  python -m src.training.train_all
"""
from __future__ import annotations
import logging
import sys
import traceback
import pandas as pd

from src.config import (ARTIFACT_DIR, HORIZON_WEEKS, MIN_TRAIN_WEEKS,
                        ENABLED_MODELS, GROUP_COL, SEED)
from src.data.loader import load_panel, get_series
from src.models.sarima_model import SarimaModel
from src.models.prophet_model import ProphetModel
from src.models.xgb_model import XGBModel
from src.models.lstm_model import LSTMModel
from src.training.backtest import holdout_backtest, to_dict
from src.utils.io import save_pickle, save_json
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


def _model_factory(name: str):
    return {
        "SARIMA": SarimaModel,
        "Prophet": ProphetModel,
        "XGBoost": XGBModel,
        "LSTM": LSTMModel,
    }[name]


def train_state(state: str, y: pd.Series, horizon: int = HORIZON_WEEKS) -> dict:
    if len(y) < MIN_TRAIN_WEEKS + horizon:
        log.warning("Skipping %s: only %d weeks", state, len(y))
        return {}

    results: list[dict] = []
    fitted_full = {}

    for name in ENABLED_MODELS:
        try:
            cls = _model_factory(name)
            bt_model = cls()
            r = holdout_backtest(bt_model, y, horizon)
            results.append(to_dict(r))
            # Refit on full history for production forecasts
            full_model = cls()
            full_model.fit(y)
            fitted_full[name] = full_model
            log.info("[%s] %s -> SMAPE=%.2f MAE=%.0f", state, name, r.smape, r.mae)
        except Exception as exc:  # noqa: BLE001
            log.error("[%s] %s FAILED: %s", state, name, exc)
            log.debug(traceback.format_exc())

    if not results:
        return {}

    results.sort(key=lambda d: d["smape"])
    champion_name = results[0]["model"]
    champion_metrics = results[0]
    champion_model = fitted_full[champion_name]

    # Persist
    state_dir = ARTIFACT_DIR / "models" / state.replace(" ", "_")
    state_dir.mkdir(parents=True, exist_ok=True)
    # LSTM contains a torch.nn.Module; joblib handles it fine.
    save_pickle(champion_model, state_dir / "champion.joblib")
    save_json({"state": state, "champion": champion_name,
               "metrics": champion_metrics, "all_results": results},
              state_dir / "report.json")
    return {"state": state, "champion": champion_name,
            "metrics": champion_metrics, "all_results": results}


def main() -> int:
    set_seed(SEED)
    panel = load_panel()
    states = sorted(panel[GROUP_COL].unique())
    log.info("Training on %d states", len(states))

    summary = []
    for state in states:
        y = get_series(panel, state)
        rep = train_state(state, y)
        if rep:
            summary.append(rep)

    save_json({"horizon_weeks": HORIZON_WEEKS, "states": summary},
              ARTIFACT_DIR / "champions.json")
    log.info("Wrote %d champion models", len(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
