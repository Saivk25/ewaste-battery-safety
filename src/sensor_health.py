"""
sensor_health.py -- detects the sensor failures a cheap sensor set will have in a plant (fault handling beyond the
FSM's out-of-range check).

Checks, every 0.1 s step:
  ir_stuck       IR temperature has not moved for STUCK_S seconds (a live IR sensor always shows noise)
  dropout        a channel returned NaN (missing reading); the last good value is held meanwhile
  gas_dead       physics cross-check: the cell is hot (T >= GAS_CHECK_T) for GAS_CHECK_S s but H2 stays near zero,
                 although H2 is released from ~70 C (Section 8.2) -> the MQ-2 is not responding
  camera_dropout an object is on the belt but the camera produced no classification
  gas_drift      MQ-2 baseline (median while the belt is empty and cool) above DRIFT_PPM -> recalibration request;
                 the learned baseline is subtracted from H2 (auto-zero), so drift does not raise the risk. It is learned
                 only on an EMPTY belt: learning it while items pass would subtract a real gas rise whenever the IR
                 reading lags, so a continuously full belt needs scheduled zero checks instead.

`fault` (any of the first four) puts the FSM into Sensor_Fault (50 % speed, operator alarm) via
SupervisoryFSM.step(external_fault=...). While the IR channel is faulty, gas-only rises are treated as corroborated
(`corroboration_override`), so the memory never slows the reaction to the remaining sensors (fail-safe).
MQ sensors drift and need recalibration every few months; gas_drift tells the operator when.
"""
from collections import deque
import numpy as np

DT = 0.1
STUCK_S, STUCK_EPS = 5.0, 0.02        # s, C: max-min range below which the IR reading counts as frozen
GAS_CHECK_T, GAS_CHECK_S, GAS_MIN_PPM = 140.0, 10.0, 10.0   # H2 is released from venting well below 140 C [Gardner 2025]
DRIFT_PPM, DRIFT_WIN = 50.0, 300       # ppm; baseline window (samples while empty and cool)


class SensorHealthMonitor:
    def __init__(self):
        self.T_hist = deque(maxlen=int(STUCK_S / DT))
        self.hot_no_gas = 0
        self.baseline = deque(maxlen=DRIFT_WIN)
        self.last = {"T": 25.0, "h2": 0.0, "co2": 400.0}

    def step(self, T, h2, co2, object_present, camera_ok=True):
        flags = {"dropout": False}
        vals = {"T": T, "h2": h2, "co2": co2}
        for k, v in vals.items():                      # hold the last good value through a missing reading
            if v is None or not np.isfinite(v):
                vals[k] = self.last[k]; flags["dropout"] = True
            else:
                self.last[k] = v
        T, h2, co2 = vals["T"], vals["h2"], vals["co2"]
        # only judged while an object is in view: the simulator's empty-belt temperature carries no noise, so a frozen
        # reading there is not evidence of a fault (a real IR sensor is noisy all the time)
        if object_present: self.T_hist.append(T)
        else: self.T_hist.clear()
        flags["ir_stuck"] = len(self.T_hist) == self.T_hist.maxlen and (max(self.T_hist) - min(self.T_hist)) < STUCK_EPS
        self.hot_no_gas = self.hot_no_gas + 1 if (T >= GAS_CHECK_T and h2 < GAS_MIN_PPM) else 0
        flags["gas_dead"] = self.hot_no_gas * DT >= GAS_CHECK_S
        flags["camera_dropout"] = bool(object_present) and not camera_ok
        if not object_present and T < 40.0:
            self.baseline.append(h2)
        base = float(np.median(self.baseline)) if len(self.baseline) >= 50 else 0.0
        flags["gas_drift"] = base > DRIFT_PPM
        h2_corrected = max(h2 - base, 0.0)
        flags["fault"] = flags["dropout"] or flags["ir_stuck"] or flags["gas_dead"] or flags["camera_dropout"]
        flags["corroboration_override"] = flags["ir_stuck"] or flags["dropout"]
        return flags, T, h2_corrected, co2
