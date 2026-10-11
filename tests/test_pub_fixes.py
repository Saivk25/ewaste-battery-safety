"""Checks for the journal-fix code: temperature slope, rate-of-rise gate, calibrated thermal model and
measured-trace replay (synthetic curve, so no download is needed)."""
import numpy as np
import pytest
from pub_fixes import slope_series, gated_rate, cal_sim
from pub_realtrace import trace_df
from adaptive_risk import M_PERSIST

DT = 0.1


def test_slope_ignores_arrival_jump_and_recovers_ramp():
    n = 200
    T = np.r_[np.full(50, 25.0), 80.0 + 0.5 * DT * np.arange(n - 50)]   # arrives hot, then rises 0.5 C/s
    obj = np.r_[np.zeros(50), np.ones(n - 50)].astype(bool)
    sl = slope_series(T, obj, W=5.0)
    assert np.isnan(sl[:60]).all()                       # no slope during the first second after arrival
    assert sl[-1] == pytest.approx(0.5, abs=1e-6)        # the 25 -> 80 C arrival step is not counted


def test_rate_gate_smooths_burst_on_warm_flat_item_but_follows_heating_cell():
    n = 300
    S = np.full(n, 0.2); S[150:150 + M_PERSIST - 5] = 0.9  # gas-only burst shorter than the persistence window
    obj = np.ones(n, bool)
    flat = np.full(n, 50.0)                                # warm benign item: hot enough, but not rising
    R_flat = gated_rate(S, flat, obj, W=5.0, r=0.2)
    assert R_flat[150:150 + M_PERSIST - 5].max() < 0.3     # slow memory: the burst is smoothed
    rising = 50.0 + 1.0 * DT * np.arange(n)                # heating cell, 1 C/s
    R_rise = gated_rate(S, rising, obj, W=5.0, r=0.2)
    assert R_rise[150] == pytest.approx(0.9)               # corroborated: followed at once


def test_calibrated_model_uses_published_landmarks():
    df, arrive = cal_sim(seed=100, duration_s=600.0)
    first = lambda ph: df.temperature_C[df.phase == ph].iloc[0]
    assert 90 <= first("Phase1_SEI") < 95 and 130 <= first("Phase2_Separator") < 135
    assert 200 <= first("Phase3_Runaway") < 260
    assert df.temperature_C[df.mq2_ppm > 0].iloc[0] >= 115 - 2     # H2 first appears at venting (~115 C)


def test_trace_replay_labels_phases_from_measured_temperature():
    tt = np.arange(0.0, 600.0, DT); T = 30 + 0.5 * tt               # synthetic 0.5 C/s curve up to ~330 C
    df, arrive = trace_df(tt, T, seed=1)
    assert arrive == 20.0 and df.object_present[df.time < 20].sum() == 0
    after = df[df.time >= 20]
    assert (after.phase[after.temperature_C < 85] == "Normal").all()
    assert "Phase3_Runaway" in set(after.phase)
