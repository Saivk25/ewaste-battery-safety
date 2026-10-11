"""Headless smoke test of dashboard.py: every tab renders without an exception, and the Results tab
shows the held-out tables. Slow interactive runs (race, replay) are exercised by scripts/verify_dashboard.py."""
from pathlib import Path
import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
ROOT = Path(__file__).resolve().parents[1]


def test_all_tabs_render():
    """Tabs are lazy (only the open one runs), so open each in turn and check it renders without an exception."""
    import sys; sys.path.insert(0, str(ROOT))
    import dashboard
    at = st_testing.AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == len(dashboard.TABS)
    for label in dashboard.TABS:
        at.session_state["tab"] = label
        at.run()
        assert not at.exception, (label, [e.value for e in at.exception])
        if "Results" in label and (ROOT / "results" / "pub" / "eval.csv").exists():
            assert len(at.dataframe) >= 3          # FNR, margin and battery-free tables in the Results tab
