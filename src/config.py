"""Global config for the forecasting project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)

RAW_CSV = DATA_DIR / "sales.csv"

# Forecast settings
HORIZON_WEEKS = 8                 # forecast horizon
FREQ = "W-SUN"                    # resample frequency (week ending Sunday)
TARGET = "y"                      # canonical target column
DATE_COL = "ds"                   # canonical date column
GROUP_COL = "State"

# Backtest
TEST_SIZE = HORIZON_WEEKS         # last 8 weeks for validation
MIN_TRAIN_WEEKS = 52              # need >= 1 year of history to train

# Models that participate in the bake-off
ENABLED_MODELS = ("SARIMA", "Prophet", "XGBoost", "LSTM")

# Random seed
SEED = 42
