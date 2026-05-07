"""FastAPI service exposing trained per-state forecasters."""
from __future__ import annotations
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from src.config import ARTIFACT_DIR, HORIZON_WEEKS
from src.utils.io import load_pickle, load_json

log = logging.getLogger("api")
app = FastAPI(title="State-Level Sales Forecasting API", version="1.0")

CHAMPIONS_PATH = ARTIFACT_DIR / "champions.json"


# ---------- Schemas -------------------------------------------------

class ForecastPoint(BaseModel):
    date: str
    yhat: float


class ForecastResponse(BaseModel):
    state: str
    model: str
    horizon_weeks: int
    metrics: dict
    forecast: List[ForecastPoint]


class ModelInfo(BaseModel):
    state: str
    champion: str
    metrics: dict


# ---------- Loader cache --------------------------------------------

_CACHE: dict = {"champions": None, "models": {}}


def _load_champions() -> dict:
    if _CACHE["champions"] is None:
        if not CHAMPIONS_PATH.exists():
            raise HTTPException(503, "Models not trained yet. POST /retrain or run training.")
        _CACHE["champions"] = load_json(CHAMPIONS_PATH)
    return _CACHE["champions"]


def _model_for(state: str):
    if state not in _CACHE["models"]:
        path = ARTIFACT_DIR / "models" / state.replace(" ", "_") / "champion.joblib"
        if not path.exists():
            raise HTTPException(404, f"No model for state '{state}'")
        _CACHE["models"][state] = load_pickle(path)
    return _CACHE["models"][state]


def _state_report(state: str) -> dict:
    champs = _load_champions()
    for s in champs["states"]:
        if s["state"] == state:
            return s
    raise HTTPException(404, f"State '{state}' not found")


# ---------- Routes --------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "models_trained": CHAMPIONS_PATH.exists()}


@app.get("/states", response_model=List[str])
def list_states():
    champs = _load_champions()
    return [s["state"] for s in champs["states"]]


@app.get("/models", response_model=List[ModelInfo])
def list_models():
    champs = _load_champions()
    return [ModelInfo(state=s["state"], champion=s["champion"],
                      metrics=s["metrics"]) for s in champs["states"]]


@app.get("/forecast/{state}", response_model=ForecastResponse)
def forecast_state(state: str, horizon: Optional[int] = None):
    horizon = horizon or HORIZON_WEEKS
    rep = _state_report(state)
    model = _model_for(state)
    preds = model.predict(horizon)
    return ForecastResponse(
        state=state,
        model=rep["champion"],
        horizon_weeks=horizon,
        metrics=rep["metrics"],
        forecast=[ForecastPoint(date=str(pd.Timestamp(d).date()), yhat=float(v))
                  for d, v in preds.items()],
    )


@app.get("/forecast", response_model=List[ForecastResponse])
def forecast_all(horizon: Optional[int] = None):
    champs = _load_champions()
    return [forecast_state(s["state"], horizon) for s in champs["states"]]


def _retrain_job():
    log.info("Retraining started in background...")
    subprocess.run([sys.executable, "-m", "src.training.train_all"], check=False)
    _CACHE["champions"] = None
    _CACHE["models"].clear()
    log.info("Retraining finished")


@app.post("/retrain")
def retrain(bg: BackgroundTasks):
    bg.add_task(_retrain_job)
    return {"status": "retraining_scheduled"}
