"""
Tests for adaptive_risk.py's AdaptiveRiskModel (the beta(t) temporal
update and object-presence reset) and SimulatedCNN's RNG independence.
Does not exercise RealCNN here -- that needs real checkpoints/test images
on disk and is covered by the manual verification workflow
(generate_roc_pr.py / generate_gradcam.py), not unit tests.
"""
import numpy as np
import pytest

from adaptive_risk import (AdaptiveRiskModel, SimulatedCNN, BETA_MAX, LAMBDA_B, RESET_TIME, DT,
                           BETA_FAST, BETA_SLOW, C_GATE, M_PERSIST, T_CORR)


class TestComputeBeta:
    def test_stable_signal_gives_beta_near_max(self):
        model = AdaptiveRiskModel()
        beta = model.compute_beta(S_current=0.5, S_previous=0.5)
        assert beta == pytest.approx(BETA_MAX, abs=1e-9)

    def test_larger_change_gives_smaller_beta(self):
        model = AdaptiveRiskModel()
        beta_small_change = model.compute_beta(S_current=0.51, S_previous=0.50)
        beta_large_change = model.compute_beta(S_current=0.90, S_previous=0.10)
        assert beta_large_change < beta_small_change

    def test_beta_always_between_zero_and_beta_max(self):
        model = AdaptiveRiskModel()
        for delta in [0.0, 0.1, 0.5, 1.0, 2.0]:
            beta = model.compute_beta(S_current=delta, S_previous=0.0)
            assert 0.0 <= beta <= BETA_MAX


class TestAdaptiveRiskModelUpdate:
    def test_output_bounded_zero_to_one(self):
        model = AdaptiveRiskModel()
        rng = np.random.RandomState(0)
        for _ in range(100):
            S_t = rng.uniform(0, 1)
            P_battery = rng.uniform(0, 1)
            R_t, beta, R_fused = model.update(S_t, P_battery, object_present=True)
            assert 0.0 <= R_fused <= 1.0

    def test_cnn_confidence_amplifies_risk(self):
        # Same fuzzy risk, higher CNN battery confidence -> higher R_fused.
        model_low_conf = AdaptiveRiskModel()
        model_low_conf.update(0.5, P_battery=0.0, object_present=True)
        R_t_low, _, R_fused_low = model_low_conf.update(0.5, P_battery=0.0, object_present=True)

        model_high_conf = AdaptiveRiskModel()
        model_high_conf.update(0.5, P_battery=0.0, object_present=True)
        R_t_high, _, R_fused_high = model_high_conf.update(0.5, P_battery=0.95, object_present=True)

        assert R_fused_high > R_fused_low

    def test_object_absence_resets_after_reset_time(self):
        # needs a filter with memory below the gate, so use the original exponential rule
        model = AdaptiveRiskModel(mode="exp")
        # Drive risk up while an object is present.
        for _ in range(20):
            model.update(S_t=0.9, P_battery=0.9, object_present=True)
        assert model.R_prev > 0.5

        # Object leaves. Before RESET_TIME elapses, R_prev should not
        # have been force-reset yet.
        steps_before_reset = int(RESET_TIME / DT) - 1
        for _ in range(steps_before_reset):
            model.update(S_t=0.0, P_battery=0.0, object_present=False)
        assert model.R_prev > 0.0

        # After RESET_TIME total elapses with no object, R_prev must be 0.
        model.update(S_t=0.0, P_battery=0.0, object_present=False)
        assert model.R_prev == 0.0

    def test_object_reappearing_does_not_inherit_stale_risk(self):
        # A prior high-risk object leaves, enough time passes to reset,
        # then a new (low-risk) object arrives -- its early readings must
        # not be contaminated by the previous object's risk.
        model = AdaptiveRiskModel()
        for _ in range(20):
            model.update(S_t=0.95, P_battery=0.95, object_present=True)

        gap_steps = int(RESET_TIME / DT) + 5
        for _ in range(gap_steps):
            model.update(S_t=0.0, P_battery=0.0, object_present=False)

        R_t, beta, R_fused = model.update(S_t=0.05, P_battery=0.0, object_present=True)
        assert R_fused < 0.3


class TestCorroborationGatedMemory:
    """The default memory rule: fast below the gate or when temperature corroborates,
    slow for a short gas-only excursion, fast again once that excursion persists."""

    def test_below_gate_follows_immediately(self):
        m = AdaptiveRiskModel()
        R_t, beta, _ = m.update(S_t=C_GATE - 0.1, P_battery=0.0, object_present=True, temperature=25.0)
        assert beta == BETA_FAST and R_t == pytest.approx(C_GATE - 0.1)

    def test_short_gas_only_burst_is_held_back(self):
        m = AdaptiveRiskModel()
        m.update(S_t=0.2, P_battery=0.0, object_present=True, temperature=25.0)
        for _ in range(M_PERSIST - 1):
            R_t, beta, _ = m.update(S_t=0.9, P_battery=0.0, object_present=True, temperature=25.0)
            assert beta == BETA_SLOW
        assert R_t < 0.55   # never reaches the Critical threshold during the burst

    def test_persistent_gas_only_rise_is_followed(self):
        m = AdaptiveRiskModel()
        for _ in range(M_PERSIST):
            m.update(S_t=0.9, P_battery=0.0, object_present=True, temperature=25.0)
        R_t, beta, _ = m.update(S_t=0.9, P_battery=0.0, object_present=True, temperature=25.0)
        assert beta == BETA_FAST and R_t == pytest.approx(0.9)

    def test_hot_cell_is_followed_at_once(self):
        m = AdaptiveRiskModel()
        R_t, beta, _ = m.update(S_t=0.9, P_battery=0.0, object_present=True, temperature=T_CORR + 5)
        assert beta == BETA_FAST and R_t == pytest.approx(0.9)

    def test_missing_temperature_counts_as_uncorroborated(self):
        m = AdaptiveRiskModel()
        _, beta, _ = m.update(S_t=0.9, P_battery=0.0, object_present=True)
        assert beta == BETA_SLOW


class TestSimulatedCNNRNGIndependence:
    """Regression guard: SimulatedCNN previously called the global
    np.random.seed(seed), which meant constructing multiple random-using
    objects in sequence would clobber each other's streams (see
    sensor_simulation.py's fix for the same underlying bug pattern)."""

    def test_construction_order_does_not_affect_output(self):
        cnn_a = SimulatedCNN(seed=77)
        # Simulate "other objects" being constructed afterward, each of
        # which might touch the global RNG if they still had the bug.
        _ = np.random.RandomState(1)
        _ = np.random.RandomState(2)
        result_a = cnn_a.predict(temperature=100.0, mq2_ppm=50.0,
                                   object_present=True, metal_present=True)

        cnn_b = SimulatedCNN(seed=77)
        result_b = cnn_b.predict(temperature=100.0, mq2_ppm=50.0,
                                   object_present=True, metal_present=True)

        assert result_a["battery_prob"] == result_b["battery_prob"]

    def test_smoothing_resets_on_object_exit(self):
        cnn = SimulatedCNN(seed=77)
        for _ in range(10):
            cnn.predict(temperature=200.0, mq2_ppm=500.0, object_present=True, metal_present=True)
        assert cnn._smooth > 0.0

        cnn.predict(temperature=25.0, mq2_ppm=0.0, object_present=False, metal_present=False)
        assert cnn._smooth == 0.0
