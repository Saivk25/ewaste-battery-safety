"""Clicks the slow dashboard paths headlessly and checks their output (run manually, takes a few minutes):
    python tests/verify_dashboard_interactive.py
One sidebar item drives Live Monitor, Baseline Race and Conveyor Line. Baseline Race (gas spikes): all 8 systems reach the metrics table. Real Measured Cell: the replay
produces the three metrics. Upload tab: the YOLO detector loads and runs on a test photo."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from streamlit.testing.v1 import AppTest


def select(at, name):
    """AppTest cannot click a tab, so set the lazy tabs' widget state; it must be set again before every run."""
    import dashboard
    at.session_state["tab"] = next(l for l in dashboard.TABS if name in l)


def tab(at, name):
    """Open the tab whose label contains `name` (tabs are lazy: only the open one renders) and return it."""
    select(at, name); at.run()
    return [t for t in at.tabs if name in t.label][0]


at = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=900).run()
assert not at.exception, [e.value for e in at.exception]
at.selectbox(key="item").set_value("Gas spikes, no battery"); at.run()     # one sidebar item drives every tab
race = tab(at, "Baseline race")
race.select_slider(key="race_speed").set_value(50)
race.button(key="race_run").click()
select(at, "Baseline race"); at.run()
assert not at.exception, [e.value for e in at.exception]
m = [t for t in at.tabs if "Baseline race" in t.label][0].dataframe[-1].value
print(m.to_string())
assert len(m) == 8, len(m)

at.selectbox(key="item").set_value("Failing cell hidden in a device (60 C, slow)")
at.select_slider(key="repeat").set_value(3); at.run()
lm = tab(at, "Live monitor"); lm.select_slider(key="lm_speed").set_value(50); lm.button(key="lm_run").click(); select(at, "Live monitor"); at.run()
assert not at.exception, [e.value for e in at.exception]
assert any("Diverted (hood)" in m.value for m in tab(at, "Conveyor line").markdown), "belt outcome missing"
at.selectbox(key="item").set_value("Open belt (no hood) + hidden cell"); at.run()
assert any("belt-only" in i.value for i in tab(at, "Live monitor").info)

rc = tab(at, "Real measured cell")
if rc.button:
    rc.button(key="rc_run").click()
    select(at, "Real measured cell"); at.run()
    assert not at.exception, [e.value for e in at.exception]
    rcm = [t for t in at.tabs if "Real measured cell" in t.label][0].metric
    print([(x.label, x.value, x.delta) for x in rcm])
    assert len(rcm) == 3

import dashboard
det = dashboard.get_detector()
assert det is not None, "train the detector first (src/detector_yolo.py train)"
img = next((ROOT / "data_det" / "images" / "val").glob("*"))
from PIL import Image
r = det.predict(Image.open(img).convert("RGB"), conf=0.25, verbose=False)[0]
print("detector boxes on", img.name, ":", len(r.boxes))
assert r.plot().ndim == 3
print("dashboard interactive checks passed")
