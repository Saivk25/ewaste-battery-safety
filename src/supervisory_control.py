"""
supervisory_control.py — Finite State Machine for Shredder Supervisory Control

States:      Normal → Warning → Critical → Emergency_Stop
             + Sensor_Fault (from any state)

Transitions driven by R_fused threshold crossing + time-in-state hysteresis.
Emergency Stop is LATCHING — requires manual reset (safety requirement).
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from enum import Enum

DT = 0.1  # seconds

class State(Enum):
    NORMAL        = "Normal"
    WARNING       = "Warning"
    CRITICAL      = "Critical"
    EMERGENCY_STOP = "Emergency_Stop"
    SENSOR_FAULT  = "Sensor_Fault"

# Thresholds — tuned conservatively for safety
THRESHOLDS = {
    "normal_to_warning":   0.30,
    "warning_to_critical": 0.55,
    "critical_to_emergency": 0.75,
    "emergency_to_critical": 0.50,  # only after manual reset (hysteresis)
    "critical_to_warning": 0.45,
    "warning_to_normal":   0.20,
}

# Minimum dwell time before transition (prevents chattering)
MIN_DWELL = {
    State.NORMAL:         0.0,
    State.WARNING:        2.0,   # 2s before escalating
    State.CRITICAL:       1.0,   # 1s before emergency stop
    State.EMERGENCY_STOP: 999.0, # LATCHING — no auto-transition
    State.SENSOR_FAULT:   0.0,
}

# Shredder speed for each state (% of nominal)
SHREDDER_SPEED = {
    State.NORMAL:          100,
    State.WARNING:          60,
    State.CRITICAL:         10,
    State.EMERGENCY_STOP:    0,
    State.SENSOR_FAULT:     50,  # reduced but not stopped
}


class SupervisoryFSM:
    def __init__(self):
        self.state          = State.NORMAL
        self.prev_state     = State.NORMAL
        self.dwell_time     = 0.0
        self.emergency_latch = False
        self.transitions    : List[Dict] = []
        self.state_history  : List[str]  = []
        self.speed_history  : List[int]  = []
        self._sensor_fault_count = 0

    def _check_sensor_fault(self, T: float, h2: float, co2: float) -> bool:
        """Detect implausible/stuck sensor values.

        Ceilings sit well above what the sensor models can produce from
        noise alone (MQ-2 caps at 10,000ppm and can read up to roughly
        10,900ppm at a 3-sigma noise draw; thermal model caps at 900C),
        so a legitimate extreme reading during a real emergency doesn't
        get misread as a faulty sensor.
        """
        if T < -10 or T > 1000: return True
        if h2 < 0 or h2 > 15000: return True
        if co2 < 300 or co2 > 7000: return True
        return False

    def step(self, t: float, R_fused: float,
             temperature: float, h2_ppm: float, co2_ppm: float,
             manual_reset: bool = False, external_fault: bool = False) -> Tuple[State, int]:
        """
        Advance FSM one DT step.
        Returns (current_state, shredder_speed_percent)
        """
        self.dwell_time += DT

        # Check sensor health. Emergency_Stop is explicitly excluded: it's a
        # latching safety state that must only clear via manual_reset (see
        # module docstring). Without this guard, implausible readings during
        # an actual fire (exactly when sensors are most likely to peg out of
        # range) would silently transition Emergency_Stop -> Sensor_Fault,
        # which runs the shredder at 50% speed with no human involved.
        # external_fault: stuck / dropped-out / implausible channel reported by sensor_health.SensorHealthMonitor
        # A fault never raises the shredder speed: Critical (10 %) and Emergency_Stop (0 %) are kept, only
        # Normal (100 %) and Warning (60 %) drop to Sensor_Fault (50 %).
        if self.state not in (State.EMERGENCY_STOP, State.CRITICAL) and (external_fault or self._check_sensor_fault(temperature, h2_ppm, co2_ppm)):
            self._sensor_fault_count += 1
            if external_fault or self._sensor_fault_count >= 10:  # monitor fault: at once; out of range: 1 s
                self._transition(t, State.SENSOR_FAULT, R_fused)
        else:
            self._sensor_fault_count = max(0, self._sensor_fault_count - 1)

        # Manual reset from Emergency Stop
        if manual_reset and self.state == State.EMERGENCY_STOP:
            self.emergency_latch = False
            self._transition(t, State.WARNING, R_fused)

        # State machine transitions
        if self.state == State.NORMAL:
            if R_fused >= THRESHOLDS["normal_to_warning"]:
                self._transition(t, State.WARNING, R_fused)

        elif self.state == State.WARNING:
            if self.dwell_time >= MIN_DWELL[State.WARNING]:
                if R_fused >= THRESHOLDS["warning_to_critical"]:
                    self._transition(t, State.CRITICAL, R_fused)
                elif R_fused < THRESHOLDS["warning_to_normal"]:
                    self._transition(t, State.NORMAL, R_fused)

        elif self.state == State.CRITICAL:
            if self.dwell_time >= MIN_DWELL[State.CRITICAL]:
                if R_fused >= THRESHOLDS["critical_to_emergency"]:
                    self._transition(t, State.EMERGENCY_STOP, R_fused)
                    self.emergency_latch = True
                elif R_fused < THRESHOLDS["critical_to_warning"]:
                    self._transition(t, State.WARNING, R_fused)

        elif self.state == State.EMERGENCY_STOP:
            pass  # LATCHING — only manual reset can clear

        elif self.state == State.SENSOR_FAULT:
            # A sensor can fail while a battery is heating: the remaining channels must still be able to stop the
            # shredder (before this, Sensor_Fault ignored R_fused and kept the shredder at 50% until the fault cleared).
            if R_fused >= THRESHOLDS["critical_to_emergency"]:
                self._transition(t, State.EMERGENCY_STOP, R_fused)
                self.emergency_latch = True
            elif R_fused >= THRESHOLDS["warning_to_critical"]:
                self._transition(t, State.CRITICAL, R_fused)
            elif self._sensor_fault_count == 0:
                self._transition(t, State.WARNING, R_fused)  # cautious recovery

        speed = SHREDDER_SPEED[self.state]
        self.state_history.append(self.state.value)
        self.speed_history.append(speed)
        return self.state, speed

    def _transition(self, t: float, new_state: State, risk: float):
        if new_state != self.state:
            self.transitions.append({
                "time": t,
                "from": self.state.value,
                "to":   new_state.value,
                "risk_at_transition": round(risk, 4),
            })
            self.prev_state = self.state
            self.state = new_state
            self.dwell_time = 0.0


def run_supervisory(df: pd.DataFrame, manual_reset_times=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run FSM over full adaptive risk dataframe.

    manual_reset_times: optional set/list of `time` values at which to pass
    manual_reset=True to fsm.step(), simulating an operator clearing a
    latched Emergency_Stop. Without this, Emergency_Stop can never clear
    once triggered, since nothing else in the simulation can produce a
    manual reset -- which is correct real-world behavior, but means a
    multi-item scenario that crosses an Emergency_Stop needs an explicit
    reset injected to test what happens to items arriving afterward.
    """
    fsm = SupervisoryFSM()
    states, speeds = [], []
    reset_times = set(manual_reset_times) if manual_reset_times else set()

    for _, row in df.iterrows():
        state, speed = fsm.step(
            t           = row["time"],
            R_fused     = row["R_fused"],
            temperature = row["temperature_C"],
            h2_ppm      = row["mq2_ppm"],
            co2_ppm     = row["mq135_ppm"],
            manual_reset= row["time"] in reset_times,
        )
        states.append(state.value)
        speeds.append(speed)

    df_out = df.copy()
    df_out["fsm_state"]       = states
    df_out["shredder_speed"]  = speeds

    df_transitions = pd.DataFrame(fsm.transitions)
    return df_out, df_transitions


if __name__ == "__main__":
    import sys; sys.path.insert(0, "outputs/ewaste_system")
    df_risk = pd.read_csv("outputs/ewaste_system/adaptive_risk_output.csv")
    df_full, df_trans = run_supervisory(df_risk)
    df_full.to_csv("outputs/ewaste_system/full_pipeline_output.csv", index=False)
    df_trans.to_csv("outputs/ewaste_system/state_transitions.csv", index=False)

    print("FSM State distribution:")
    print(df_full["fsm_state"].value_counts().to_string())
    print("\nState Transitions:")
    print(df_trans.to_string(index=False) if len(df_trans) > 0 else "  No transitions.")
    if "Emergency_Stop" in df_full["fsm_state"].values:
        t_emg = df_full[df_full["fsm_state"]=="Emergency_Stop"]["time"].iloc[0]
        t_bat = 20.0
        print(f"\nEmergency Stop triggered at t={t_emg}s")
        print(f"Time from battery arrival to Emergency Stop: {t_emg - t_bat:.1f}s")
