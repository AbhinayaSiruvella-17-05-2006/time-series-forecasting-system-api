"""Tiny IO helpers (joblib + json)."""
from __future__ import annotations
import json
from pathlib import Path
import joblib


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(obj, path)


def load_pickle(path: Path):
    return joblib.load(path)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)
