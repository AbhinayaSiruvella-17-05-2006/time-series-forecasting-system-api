from src.data.loader import load_panel
from src.config import RAW_CSV, GROUP_COL, DATE_COL, TARGET


def test_panel_loads_and_is_weekly():
    panel = load_panel(RAW_CSV)
    assert {GROUP_COL, DATE_COL, TARGET}.issubset(panel.columns)
    one = panel[panel[GROUP_COL] == panel[GROUP_COL].iloc[0]].sort_values(DATE_COL)
    diffs = one[DATE_COL].diff().dropna().dt.days.unique().tolist()
    assert diffs == [7], f"Expected weekly cadence, got {diffs}"
    assert one[TARGET].notna().all()
