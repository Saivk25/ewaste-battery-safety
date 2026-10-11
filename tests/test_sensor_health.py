"""Fault monitor and fault-state behaviour of the FSM (a sensor can fail while a battery heats up)."""
import numpy as np
from sensor_health import SensorHealthMonitor, STUCK_S, GAS_CHECK_S
from supervisory_control import SupervisoryFSM, State, DT
import sensor_simulation as ss


def run(mon, T, h2, obj=True, cam=True):
    return [mon.step(t, h, 400.0, obj, cam) for t, h in zip(T, h2)]


def test_healthy_runs_raise_no_fault():
    for df in (ss.run_simulation(duration_s=300, battery_arrives_at=20, seed=101),
               ss.run_noise_stress_scenario(seed=1101, n_spikes=10, spike_mq2_ppm=900, spike_mq135_ppm=3000, spike_duration_steps=15)):
        mon = SensorHealthMonitor()
        assert not any(mon.step(r.temperature_C, r.mq2_ppm, r.mq135_ppm, r.object_present)[0]["fault"] for r in df.itertuples())


def test_frozen_ir_is_flagged_after_window_and_overrides_corroboration():
    n = int(STUCK_S / DT)
    out = run(SensorHealthMonitor(), np.full(n + 5, 60.0), np.zeros(n + 5))
    assert not out[n - 2][0]["ir_stuck"] and out[n + 2][0]["ir_stuck"]
    assert out[n + 2][0]["corroboration_override"]


def test_dead_gas_sensor_detected_on_hot_cell_and_dropout_held():
    n = int(GAS_CHECK_S / DT)
    T = 150.0 + 0.3 * np.random.RandomState(0).randn(n + 5)
    out = run(SensorHealthMonitor(), T, np.zeros(n + 5))
    assert out[-1][0]["gas_dead"] and not out[n - 5][0]["gas_dead"]
    mon = SensorHealthMonitor(); mon.step(70.0, 100.0, 400.0, True)
    fl, T_held, h2_held, _ = mon.step(float("nan"), float("nan"), 400.0, True)
    assert fl["dropout"] and fl["fault"] and T_held == 70.0


def test_gas_drift_flagged_and_subtracted():
    mon = SensorHealthMonitor()
    for _ in range(100): fl, _, h2c, _ = mon.step(25.0, 500.0, 400.0, False)
    assert fl["gas_drift"] and h2c == 0.0


def test_fsm_fault_never_raises_speed_and_can_still_stop():
    f = SupervisoryFSM(); t = 0.0
    for _ in range(40):                                                         # R 0.6: Warning, then Critical (below E-stop 0.75)
        f.step(t, 0.6, 200.0, 2000.0, 2000.0); t += DT
    assert f.state == State.CRITICAL
    s, speed = f.step(t, 0.6, 200.0, 2000.0, 2000.0, external_fault=True)
    assert s == State.CRITICAL and speed == 10                                 # fault does not lift 10 % to 50 %
    g = SupervisoryFSM(); t = 0.0
    s, _ = g.step(t, 0.1, 25.0, 0.0, 400.0, external_fault=True)
    assert s == State.SENSOR_FAULT
    for _ in range(5): s, speed = g.step(t, 0.8, 25.0, 0.0, 400.0, external_fault=True); t += DT
    assert s == State.EMERGENCY_STOP and speed == 0                            # remaining channels can stop it
