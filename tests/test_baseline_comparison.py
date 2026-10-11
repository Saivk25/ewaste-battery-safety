"""
Tests for baseline_comparison.py: the B1 seeding fix (a real bug found
and fixed in a prior session -- baseline_vision_only() used to draw
noise from the unseeded global np.random state, so its numbers, T_stop_s
especially, differed on every rerun) and compute_metrics()'s core
aggregation logic (Hazard_FNR, FPR, CFNC).
"""
import pandas as pd
import pytest

from baseline_comparison import baseline_vision_only, compute_metrics


def make_sensor_df(rows):
    """rows: list of dicts with time, temperature_C, battery_present, phase."""
    return pd.DataFrame(rows)


class TestBaselineVisionOnlySeeding:
    def test_reproducible_with_same_seed(self):
        df = make_sensor_df([
            {"time": t * 0.1, "temperature_C": 25.0 + t, "battery_present": 1, "phase": "Phase1_SEI"}
            for t in range(50)
        ])
        result_a = baseline_vision_only(df, seed=0)
        result_b = baseline_vision_only(df, seed=0)
        assert (result_a["risk"] == result_b["risk"]).all()
        assert (result_a["state"] == result_b["state"]).all()

    def test_different_seeds_give_different_draws(self):
        df = make_sensor_df([
            {"time": t * 0.1, "temperature_C": 60.0, "battery_present": 1, "phase": "Phase2_Separator"}
            for t in range(30)
        ])
        result_a = baseline_vision_only(df, seed=0)
        result_b = baseline_vision_only(df, seed=1)
        assert not (result_a["risk"] == result_b["risk"]).all()

    def test_default_seed_is_zero(self):
        df = make_sensor_df([
            {"time": 0.0, "temperature_C": 100.0, "battery_present": 1, "phase": "Phase3_Runaway"}
        ])
        default_result = baseline_vision_only(df)
        explicit_result = baseline_vision_only(df, seed=0)
        assert (default_result["risk"] == explicit_result["risk"]).all()


class TestComputeMetrics:
    def test_perfect_detection_gives_zero_hazard_fnr(self):
        df = make_sensor_df([
            {"state": "Warning", "battery_present": 1, "phase": "Phase1_SEI", "time": 0.0},
            {"state": "Critical", "battery_present": 1, "phase": "Phase2_Separator", "time": 0.1},
            {"state": "Emergency_Stop", "battery_present": 1, "phase": "Phase3_Runaway", "time": 20.2},
        ])
        m = compute_metrics(df, battery_arrives=0.0, system_name="test")
        assert m["Hazard_FNR"] == 0.0

    def test_complete_miss_gives_full_hazard_fnr(self):
        df = make_sensor_df([
            {"state": "Normal", "battery_present": 1, "phase": "Phase1_SEI", "time": 0.0},
            {"state": "Normal", "battery_present": 1, "phase": "Phase2_Separator", "time": 0.1},
            {"state": "Normal", "battery_present": 1, "phase": "Phase3_Runaway", "time": 0.2},
        ])
        m = compute_metrics(df, battery_arrives=0.0, system_name="test")
        assert m["Hazard_FNR"] == 1.0

    def test_false_positive_rate_counts_emergency_stops_without_battery(self):
        df = make_sensor_df([
            {"state": "Emergency_Stop", "battery_present": 0, "phase": "Normal", "time": 0.0},
            {"state": "Normal", "battery_present": 0, "phase": "Normal", "time": 0.1},
        ])
        m = compute_metrics(df, battery_arrives=999.0, system_name="test")
        assert m["FPR"] == 0.5

    def test_no_false_positives_when_all_normal(self):
        df = make_sensor_df([
            {"state": "Normal", "battery_present": 0, "phase": "Normal", "time": t * 0.1}
            for t in range(20)
        ])
        m = compute_metrics(df, battery_arrives=999.0, system_name="test")
        assert m["FPR"] == 0.0

    def test_t_stop_measures_time_from_battery_arrival(self):
        df = make_sensor_df([
            {"state": "Normal", "battery_present": 0, "phase": "Normal", "time": 0.0},
            {"state": "Normal", "battery_present": 1, "phase": "Phase1_SEI", "time": 10.0},
            {"state": "Emergency_Stop", "battery_present": 1, "phase": "Phase3_Runaway", "time": 35.0},
        ])
        m = compute_metrics(df, battery_arrives=10.0, system_name="test")
        assert m["T_stop_s"] == pytest.approx(25.0)

    def test_t_stop_is_infinity_marker_if_never_triggered(self):
        df = make_sensor_df([
            {"state": "Warning", "battery_present": 1, "phase": "Phase1_SEI", "time": t * 0.1}
            for t in range(10)
        ])
        m = compute_metrics(df, battery_arrives=0.0, system_name="test")
        assert m["T_stop_s"] == ">400"

    def test_cfnc_weights_late_phase_misses_more_heavily(self):
        df_late_miss = make_sensor_df([
            {"state": "Normal", "battery_present": 1, "phase": "Phase4_Fire", "time": 0.0},
        ])
        df_early_miss = make_sensor_df([
            {"state": "Normal", "battery_present": 1, "phase": "Phase1_SEI", "time": 0.0},
        ])
        m_late = compute_metrics(df_late_miss, battery_arrives=0.0, system_name="test")
        m_early = compute_metrics(df_early_miss, battery_arrives=0.0, system_name="test")
        assert m_late["CFNC"] > m_early["CFNC"]

    def test_system_name_passed_through(self):
        df = make_sensor_df([
            {"state": "Normal", "battery_present": 0, "phase": "Normal", "time": 0.0}
        ])
        m = compute_metrics(df, battery_arrives=0.0, system_name="MySystem")
        assert m["System"] == "MySystem"
