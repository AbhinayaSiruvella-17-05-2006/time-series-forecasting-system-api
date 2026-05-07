"""Quick sanity demo notebook (script form)."""
from src.data.loader import load_panel, get_series
from src.config import GROUP_COL, HORIZON_WEEKS
from src.models.xgb_model import XGBModel

panel = load_panel()
print("States:", panel[GROUP_COL].nunique(), "Rows:", len(panel))
y = get_series(panel, "California")
m = XGBModel().fit(y)
print(m.predict(HORIZON_WEEKS))
