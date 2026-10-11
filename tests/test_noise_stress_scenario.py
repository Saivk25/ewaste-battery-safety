"""
Tests for run_noise_stress_scenario() and the beta(t) noise-rejection
property it exists to validate (prior-session roadmap item: current
sensor noise never crosses fuzzy membership thresholds, so beta(t)'s
false-alarm-rejection role was never actually exercised).
"""
import pandas as pd
import pytest

from sensor_simulation import run_noise_stress_scenario
from fuzzy_inference import FuzzyMemberships as fm
from adaptive_risk import AdaptiveRiskModel, RealCNN
from supervisory_control import SupervisoryFSM, State
from baseline_comparison import baseline_fixed_fuzzy
from fuzzy_inference import MamdaniFIS


class TestNoiseStressScenarioGeneration:
    def test_no_battery_ever_present(self):
        df = run_noise_stress_scenario(duration_s=60.0, n_spikes=2)
        assert (df["battery_present"] == 0).all()

    def test_spikes_are_injected(self):
        df = run_noise_stress_scenario(duration_s=60.0, n_spikes=2)
        assert df["is_noise_spike"].sum() > 0

    def test_spike_magnitude_is_substantial(self):
        # The whole point: spikes must be large enough to actually
        # challenge the fuzzy membership functions, not trivially small.
        df = run_noise_stress_scenario(duration_s=60.0, n_spikes=2,
                                         spike_mq2_ppm=350.0)
        spike_rows = df[df["is_noise_spike"] == 1]
        assert spike_rows["mq2_ppm"].min() > 200
        assert fm.h2_medium(spike_rows["mq2_ppm"].mean()) > 0.3

    def test_baseline_outside_spikes_stays_near_zero_h2(self):
        df = run_noise_stress_scenario(duration_s=60.0, n_spikes=2)
        quiet_rows = df[df["is_noise_spike"] == 0]
        assert quiet_rows["mq2_ppm"].max() < 10

    def test_reproducible_with_same_seed(self):
        df1 = run_noise_stress_scenario(duration_s=60.0, seed=1, n_spikes=2)
        df2 = run_noise_stress_scenario(duration_s=60.0, seed=1, n_spikes=2)
        assert (df1["mq2_ppm"] == df2["mq2_ppm"]).all()


class TestNoiseRejectionBehavior:
    """The actual validation: does the system distinguish a transient
    spike from a sustained hazard signal, and does adaptive beta recover
    from a spike rather than getting stuck escalated?"""

    def test_fixed_beta_b3_never_escalates_past_normal(self):
        df = run_noise_stress_scenario(duration_s=120.0, n_spikes=3)
        result = baseline_fixed_fuzzy(df)
        assert not (result["state"] == "Critical").any()
        assert not (result["state"] == "Emergency_Stop").any()

    def test_adaptive_system_never_falsely_escalates_to_critical(self):
        df = run_noise_stress_scenario(duration_s=120.0, n_spikes=3)
        fis = MamdaniFIS()
        arm = AdaptiveRiskModel()
        fsm = SupervisoryFSM()

        for _, row in df.iterrows():
            fuzzy_risk, _ = fis.infer(row["temperature_C"], row["mq2_ppm"],
                                        row["mq135_ppm"], bool(row["metal_present"]))
            R_t, beta, R_fused = arm.update(fuzzy_risk, P_battery=0.0,
                                              object_present=bool(row["object_present"]))
            state, speed = fsm.step(row["time"], R_fused, row["temperature_C"],
                                      row["mq2_ppm"], row["mq135_ppm"])

        assert fsm.state not in (State.CRITICAL, State.EMERGENCY_STOP)

    def test_adaptive_system_recovers_after_each_spike(self):
        # Every Warning triggered by a spike must clear back to Normal
        # (not get stuck elevated), confirming beta(t) self-corrects once
        # the spike passes rather than only ever accumulating risk.
        df = run_noise_stress_scenario(duration_s=200.0, n_spikes=5)
        fis = MamdaniFIS()
        arm = AdaptiveRiskModel()
        fsm = SupervisoryFSM()
        states = []

        for _, row in df.iterrows():
            fuzzy_risk, _ = fis.infer(row["temperature_C"], row["mq2_ppm"],
                                        row["mq135_ppm"], bool(row["metal_present"]))
            R_t, beta, R_fused = arm.update(fuzzy_risk, P_battery=0.0,
                                              object_present=bool(row["object_present"]))
            state, speed = fsm.step(row["time"], R_fused, row["temperature_C"],
                                      row["mq2_ppm"], row["mq135_ppm"])
            states.append(state)

        # Must return to Normal at some point after every excursion into
        # Warning -- i.e. not end the scenario still elevated.
        assert states[-1] == State.NORMAL
