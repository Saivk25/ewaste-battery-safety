import numpy as np
import conveyor_sim as c

# fixed camera pool so the test needs no checkpoint
POOL = {"test_battery_coarse": np.array([0.9]), "test_battery_recybat": np.array([0.9]),
        "test_nonbattery": np.array([0.05]), "test_electronics": np.array([0.02])}


def test_camera_only_shreds_hidden_failing_cell():
    r = c.trial("fail80_hidden", 100, "camera_only", POOL)
    assert r["entered_shredder"] and r["hazardous_at_entry"]


def test_ours_keeps_hidden_failing_cell_out_of_shredder():
    for kind in ("fail60_hidden", "fail80_hidden"):
        r = c.trial(kind, 100, "ours", POOL)
        assert not r["entered_shredder"], kind


def test_intact_battery_diverted_and_bin_quiet():
    r = c.trial("intact", 100, "ours", POOL)
    assert r["diverted"] and r["divert_reason"] == "camera" and r["bin_alarm"] is False
    assert c.trial("intact", 100, "rule", POOL)["entered_shredder"]   # heat/gas rule cannot see it


def test_benign_item_passes():
    r = c.trial("benign", 100, "ours", POOL)
    assert r["entered_shredder"] and not r["diverted"] and not r["belt_stopped"]


def test_undilute_inverts_dilution():
    h2, co2 = c.undilute(0.2 * 1000.0, 400.0 + 0.2 * 2000.0, 0.2)
    assert abs(h2 - 1000.0) < 1e-9 and abs(co2 - 2400.0) < 1e-9


def test_divert_threshold_is_the_validated_detection_threshold():
    import re, pathlib
    src = (pathlib.Path(c.__file__).parent / "evaluate_model.py").read_text(encoding="utf-8")
    assert c.P_DIVERT == float(re.search(r"^BATTERY_THRESHOLD = ([0-9.]+)", src, re.M).group(1))
