"""
Tests for supervisory_control.py's SupervisoryFSM.

test_sensor_fault_cannot_override_latched_emergency_stop is a direct
regression guard for a real safety-critical bug found and fixed in this
project: sensor-fault detection ran unconditionally before the state
dispatch, so 10 consecutive implausible sensor readings while latched in
Emergency_Stop would silently transition to Sensor_Fault (50% shredder
speed) with no manual reset involved -- contradicting the module's own
"LATCHING, requires manual reset" design intent, and most likely to
happen exactly when it matters least: during a real fire, when sensors
are most likely to peg out of range.
"""
import pytest

from supervisory_control import SupervisoryFSM, State, THRESHOLDS, MIN_DWELL, DT


def drive_to_emergency_stop(fsm: SupervisoryFSM, t_start: float = 0.0) -> float:
    """Push a fresh FSM through Normal -> Warning -> Critical -> Emergency_Stop
    with plausible sensor readings throughout, returning the time reached."""
    t = t_start
    fsm.step(t, R_fused=THRESHOLDS["normal_to_warning"] + 0.01,
              temperature=100.0, h2_ppm=100.0, co2_ppm=500.0)
    t += DT
    for _ in range(int(MIN_DWELL[State.WARNING] / DT) + 2):
        fsm.step(t, R_fused=THRESHOLDS["warning_to_critical"] + 0.01,
                  temperature=100.0, h2_ppm=100.0, co2_ppm=500.0)
        t += DT
    for _ in range(int(MIN_DWELL[State.CRITICAL] / DT) + 2):
        fsm.step(t, R_fused=THRESHOLDS["critical_to_emergency"] + 0.01,
                  temperature=100.0, h2_ppm=100.0, co2_ppm=500.0)
        t += DT
    assert fsm.state == State.EMERGENCY_STOP
    return t


class TestBasicTransitions:
    def test_starts_in_normal(self):
        fsm = SupervisoryFSM()
        assert fsm.state == State.NORMAL

    def test_normal_to_warning_on_threshold_crossing(self):
        fsm = SupervisoryFSM()
        fsm.step(0.0, R_fused=THRESHOLDS["normal_to_warning"] + 0.01,
                  temperature=25.0, h2_ppm=0.0, co2_ppm=400.0)
        assert fsm.state == State.WARNING

    def test_stays_normal_below_threshold(self):
        fsm = SupervisoryFSM()
        fsm.step(0.0, R_fused=THRESHOLDS["normal_to_warning"] - 0.01,
                  temperature=25.0, h2_ppm=0.0, co2_ppm=400.0)
        assert fsm.state == State.NORMAL

    def test_warning_dwell_time_enforced_before_escalation(self):
        fsm = SupervisoryFSM()
        fsm.step(0.0, R_fused=0.35, temperature=25.0, h2_ppm=0.0, co2_ppm=400.0)
        assert fsm.state == State.WARNING
        # Immediately try to escalate on the very next step -- dwell time
        # (2.0s) hasn't elapsed yet, so it must not jump straight to Critical.
        fsm.step(DT, R_fused=THRESHOLDS["warning_to_critical"] + 0.1,
                  temperature=25.0, h2_ppm=0.0, co2_ppm=400.0)
        assert fsm.state == State.WARNING

    def test_full_escalation_chain(self):
        fsm = SupervisoryFSM()
        drive_to_emergency_stop(fsm)
        assert fsm.state == State.EMERGENCY_STOP


class TestEmergencyStopLatching:
    def test_emergency_stop_does_not_clear_on_low_risk_alone(self):
        fsm = SupervisoryFSM()
        t = drive_to_emergency_stop(fsm)
        fsm.step(t, R_fused=0.0, temperature=25.0, h2_ppm=0.0, co2_ppm=400.0)
        assert fsm.state == State.EMERGENCY_STOP

    def test_manual_reset_clears_emergency_stop(self):
        fsm = SupervisoryFSM()
        t = drive_to_emergency_stop(fsm)
        fsm.step(t, R_fused=0.0, temperature=25.0, h2_ppm=0.0, co2_ppm=400.0,
                  manual_reset=True)
        assert fsm.state == State.WARNING

    def test_manual_reset_has_no_effect_outside_emergency_stop(self):
        fsm = SupervisoryFSM()
        fsm.step(0.0, R_fused=0.0, temperature=25.0, h2_ppm=0.0, co2_ppm=400.0,
                  manual_reset=True)
        assert fsm.state == State.NORMAL

    def test_sensor_fault_cannot_override_latched_emergency_stop(self):
        """Regression guard: implausible sensor readings while latched in
        Emergency_Stop must not transition to Sensor_Fault. See module
        docstring for why this matters (a real fire is exactly when
        sensors are most likely to read implausible values)."""
        fsm = SupervisoryFSM()
        t = drive_to_emergency_stop(fsm)

        # Feed 15 consecutive implausible readings (past the 10-reading
        # fault-count threshold) while still latched in Emergency_Stop.
        for _ in range(15):
            state, speed = fsm.step(t, R_fused=0.0, temperature=2000.0,
                                      h2_ppm=99999.0, co2_ppm=99999.0)
            t += DT
            assert state == State.EMERGENCY_STOP
            assert speed == 0

    def test_sensor_fault_does_trigger_from_non_latched_states(self):
        # Confirms the fix is scoped correctly: Sensor_Fault detection
        # should still work normally from Normal/Warning/Critical, only
        # Emergency_Stop is exempt.
        fsm = SupervisoryFSM()
        t = 0.0
        for _ in range(15):
            state, speed = fsm.step(t, R_fused=0.0, temperature=2000.0,
                                      h2_ppm=99999.0, co2_ppm=99999.0)
            t += DT
        assert state == State.SENSOR_FAULT
        assert speed == 50


class TestSensorFaultRecovery:
    def test_recovers_to_warning_once_readings_are_plausible_again(self):
        fsm = SupervisoryFSM()
        t = 0.0
        for _ in range(15):
            fsm.step(t, R_fused=0.0, temperature=2000.0, h2_ppm=99999.0, co2_ppm=99999.0)
            t += DT
        assert fsm.state == State.SENSOR_FAULT

        for _ in range(15):
            fsm.step(t, R_fused=0.0, temperature=25.0, h2_ppm=0.0, co2_ppm=400.0)
            t += DT
        assert fsm.state == State.WARNING


class TestShredderSpeed:
    def test_speed_matches_state(self):
        fsm = SupervisoryFSM()
        _, speed = fsm.step(0.0, R_fused=0.0, temperature=25.0, h2_ppm=0.0, co2_ppm=400.0)
        assert speed == 100
        t = drive_to_emergency_stop(fsm, t_start=DT)
        _, speed = fsm.step(t, R_fused=0.9, temperature=100.0, h2_ppm=100.0, co2_ppm=500.0)
        assert speed == 0
