"""
sensor_simulation.py  — Physics-based sensor simulation for E-Waste Safety System
Thermal model: Phenomenological 4-phase model (hand-tuned, NOT fitted to any dataset;
               onset temperatures sit below published ranges, see paper Section IX-G)
Gas model:     Freundlich adsorption (MQ-series datasheet constants)
"""
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
from typing import Tuple, List, Dict

DT         = 0.1
T_AMBIENT  = 25.0
T_H2_ONSET  = 70.0
T_CO2_ONSET = 120.0
MQ2_A, MQ2_B, MQ2_R0       = 0.3934, -0.547, 10.0
MQ135_A, MQ135_B, MQ135_R0 = 110.47, -2.862, 76.63


class ThermalRunawayModel:
    """
    Phenomenological 4-phase model, hand-tuned (not fitted to measured data).
    Phase 1: 25->45C  over ~100s  (SEI slow decomposition)
    Phase 2: 45->90C  over ~60s   (separator softening)
    Phase 3: 90->400C over ~30s   (~10C/s, rapid runaway onset)
    Phase 4: 400->850C peak, then exponential decay
    """
    PHASES = [
        (25,  45,  0.08, 1.2),
        (45,  90,  0.35, 1.5),
        (90,  400, 8.0,  2.0),
        (400, 850, 12.0, 1.8),
    ]
    COOLING_TAU = 180.0

    def __init__(self, seed=42):
        self._rng = np.random.RandomState(seed)
        self.T = T_AMBIENT
        self.t_batt = 0.0
        self.past_peak = False
        self.T_peak = T_AMBIENT
        self.phase_str = "Normal"

    def _rate(self, T):
        for T_lo, T_hi, base, exp in self.PHASES:
            if T_lo <= T < T_hi:
                frac = (T - T_lo) / (T_hi - T_lo)
                return base * (1 + frac) ** exp
        return 0.0

    def step(self, battery_present):
        if not battery_present:
            self.T = max(self.T - (self.T - T_AMBIENT) * 0.03 * DT, T_AMBIENT)
            self.t_batt = 0.0; self.past_peak = False
            self.phase_str = "Normal"
            return round(self.T, 2), self.phase_str

        self.t_batt += DT
        if not self.past_peak:
            dT = self._rate(self.T)
            q_cool = 5.0 * 0.01 * (self.T - T_AMBIENT) / (0.045 * 1100)
            self.T = min(self.T + (dT - q_cool) * DT + self._rng.normal(0, 0.15), 900.0)
            if self.T >= 850:
                self.past_peak = True; self.T_peak = self.T
        else:
            self.T = T_AMBIENT + (self.T_peak - T_AMBIENT) * np.exp(-self.t_batt / self.COOLING_TAU)
            self.T += self._rng.normal(0, 0.5)

        if self.T < 45: self.phase_str = "Normal"
        elif self.T < 90: self.phase_str = "Phase1_SEI"
        elif self.T < 200: self.phase_str = "Phase2_Separator"
        elif self.T < 500: self.phase_str = "Phase3_Runaway"
        else: self.phase_str = "Phase4_Fire"
        return round(self.T, 2), self.phase_str


class MQ2Sensor:
    def __init__(self, seed=43):
        self._rng = np.random.RandomState(seed); self._warmup = 0.0; self._last = 0.0

    def _temp_to_ppm(self, T):
        if T < T_H2_ONSET: return 0.0
        return min(60.0 * np.exp(0.042 * (T - T_H2_ONSET)), 10000.0)

    def _ppm_to_v(self, ppm):
        rs_r0 = max(MQ2_A * max(ppm, 1.0) ** MQ2_B, 0.05)
        return float(np.clip(5.0 * 10.0 / (10.0 + rs_r0 * MQ2_R0), 0, 5))

    def read(self, T):
        self._warmup += DT; wf = min(self._warmup / 30.0, 1.0)
        tgt = self._temp_to_ppm(T); tau = 45.0 if tgt < self._last else 5.0
        self._last += (tgt - self._last) * (DT / tau)
        ppm = max(self._last * wf * self._rng.normal(1.0, 0.03), 0.0)
        return round(ppm, 2), round(self._ppm_to_v(ppm), 4)


class MQ135Sensor:
    def __init__(self, seed=44):
        self._rng = np.random.RandomState(seed); self._last = 400.0

    def _temp_to_ppm(self, T):
        if T < T_CO2_ONSET: return 400.0
        return 400.0 + 4600.0 / (1 + np.exp(-0.035 * (T - T_CO2_ONSET - 80)))

    def _ppm_to_v(self, ppm):
        rs_r0 = max(MQ135_A * max(ppm, 10.0) ** MQ135_B, 0.1)
        return float(np.clip(5.0 * 20.0 / (20.0 + rs_r0 * MQ135_R0), 0, 5))

    def read(self, T):
        tgt = self._temp_to_ppm(T); tau = 60.0 if tgt < self._last else 8.0
        self._last += (tgt - self._last) * (DT / tau)
        ppm = max(self._last * self._rng.normal(1.0, 0.02), 0.0)
        return round(ppm, 2), round(self._ppm_to_v(ppm), 4)


class InductiveProximitySensor:
    """LJ18A3 eddy-current. 2% FNR + 10ms debounce."""
    def __init__(self, seed=45):
        self._rng = np.random.RandomState(seed); self._state = False; self._tdeb = 0.0

    def read(self, metal_present):
        raw = False if (metal_present and self._rng.random_sample() < 0.02) else metal_present
        if raw != self._state:
            self._tdeb += DT
            if self._tdeb >= 0.010: self._state = raw; self._tdeb = 0.0
        else: self._tdeb = 0.0
        return self._state


class UltrasonicSensor:
    """HC-SR04. Poisson arrivals, log-normal size distribution."""
    def __init__(self, seed=46):
        self._rng = np.random.RandomState(seed)
        self._next = self._rng.exponential(30.0)
        self._end = -1.0; self._sz = 0.0

    def read(self, t):
        if t >= self._next and t > self._end:
            self._end  = t + self._rng.exponential(8.0)
            self._sz   = self._rng.lognormal(2.7, 0.5)
            self._next = self._end + self._rng.exponential(30.0)
        present = t <= self._end
        size = self._sz + self._rng.normal(0, 0.3) if present else 0.0
        return present, max(size, 0.0)


def run_simulation(duration_s=400.0, battery_arrives_at=20.0, battery_leaves_at=400.0, seed=42):
    thermal = ThermalRunawayModel(seed=seed)
    mq2 = MQ2Sensor(seed=seed+1)
    mq135 = MQ135Sensor(seed=seed+2)
    inductive = InductiveProximitySensor(seed=seed+3)
    ultrasonic = UltrasonicSensor(seed=seed+4)
    rows = []
    for t in np.arange(0.0, duration_s, DT):
        batt = battery_arrives_at <= t < battery_leaves_at
        temp_C, phase = thermal.step(batt)
        mq2_ppm, mq2_v = mq2.read(temp_C)
        mq135_ppm, mq135_v = mq135.read(temp_C)
        metal_det = inductive.read(batt)
        obj_p, obj_sz = ultrasonic.read(t)
        if batt: obj_p = True; obj_sz = max(obj_sz, 10.0)
        rows.append(dict(time=round(t,2), temperature_C=temp_C, mq2_ppm=mq2_ppm,
            mq2_voltage=mq2_v, mq135_ppm=mq135_ppm, mq135_voltage=mq135_v,
            metal_present=int(metal_det), object_present=int(obj_p),
            object_size_cm=round(obj_sz,2), phase=phase, battery_present=int(batt)))
    return pd.DataFrame(rows)


def run_normal_scenario(duration_s=200.0, seed=99):
    rng = np.random.RandomState(seed)
    mq2 = MQ2Sensor(seed=seed+1); mq135 = MQ135Sensor(seed=seed+2)
    inductive = InductiveProximitySensor(seed=seed+3)
    ultrasonic = UltrasonicSensor(seed=seed+4)
    rows = []
    for t in np.arange(0.0, duration_s, DT):
        temp_C = T_AMBIENT + rng.normal(0,0.5) + 1.5*np.sin(0.05*t)
        mq2_ppm,mq2_v = mq2.read(temp_C); mq135_ppm,mq135_v = mq135.read(temp_C)
        metal_det = inductive.read(t % 25 < 5)
        obj_p, obj_sz = ultrasonic.read(t)
        rows.append(dict(time=round(t,2), temperature_C=round(temp_C,2), mq2_ppm=mq2_ppm,
            mq2_voltage=mq2_v, mq135_ppm=mq135_ppm, mq135_voltage=mq135_v,
            metal_present=int(metal_det), object_present=int(obj_p),
            object_size_cm=round(obj_sz,2), phase="Normal", battery_present=0))
    return pd.DataFrame(rows)


def run_multi_item_scenario(seed=42, item3_gap_s=900.0):
    """
    Three discrete objects in sequence, with idle gaps between them:
      t=0-15s              idle (nothing on the belt)
      t=15-45s             Item 1: benign metal object (e.g. a PCB/can) --
                            metal detected, no thermal/gas signature, never
                            a battery
      t=45-60s             idle gap (> RESET_TIME=3s, enough for beta(t)/
                            R_prev and the CNN's per-object cache to reset)
      t=60-460s             Item 2: the battery -- same thermal runaway
                            timeline as run_simulation()'s default scenario,
                            just time-shifted
      t=460 to 460+item3_gap_s   idle gap
      then a 30s window     Item 3: a second benign metal object

    Exists because the paper claims multi-item robustness ("the proposed
    system resets beta(t) via the LJ18A3 inductive sensor signal on object
    exit") without any scenario that actually puts more than one object on
    the belt to test it. Checks: (1) Item 2's risk doesn't get diluted by
    Item 1's presence beforehand, and (2) once the environment has actually
    cooled/cleared, Item 3's risk assessment isn't stuck contaminated by
    Item 2's aftermath.

    item3_gap_s defaults to 900s (5x the thermal model's COOLING_TAU=180s)
    because a full runaway peaking near 850C is still reading ~114C and
    ~2000ppm H2 the instant it leaves the belt (confirmed empirically) --
    a short gap (e.g. 15s) correctly keeps risk elevated for Item 3, since
    the ambient area genuinely is still hazardous, not because the reset
    mechanism failed. Pass a shorter gap to specifically test that
    still-elevated-ambient case instead.
    """
    thermal    = ThermalRunawayModel(seed=seed)
    mq2        = MQ2Sensor(seed=seed+1)
    mq135      = MQ135Sensor(seed=seed+2)
    inductive  = InductiveProximitySensor(seed=seed+3)

    item3_start = 460.0 + item3_gap_s
    item3_end   = item3_start + 30.0
    total_duration = item3_end + 5.0

    # (start, end, metal_present, battery_present) per item; gaps are the
    # space between entries where object_present is False.
    items = [
        (15.0,  45.0,  True,  False),   # Item 1: benign metal object
        (60.0,  460.0, True,  True),    # Item 2: the battery
        (item3_start, item3_end, True, False),   # Item 3: benign metal object
    ]

    rows = []
    for t in np.arange(0.0, total_duration, DT):
        item = next(((s, e, m, b) for s, e, m, b in items if s <= t < e), None)
        obj_present  = item is not None
        metal_target = item[2] if item else False
        batt         = item[3] if item else False

        temp_C, phase = thermal.step(batt)
        mq2_ppm, mq2_v = mq2.read(temp_C)
        mq135_ppm, mq135_v = mq135.read(temp_C)
        metal_det = inductive.read(metal_target)

        rows.append(dict(time=round(t,2), temperature_C=temp_C, mq2_ppm=mq2_ppm,
            mq2_voltage=mq2_v, mq135_ppm=mq135_ppm, mq135_voltage=mq135_v,
            metal_present=int(metal_det), object_present=int(obj_present),
            object_size_cm=(10.0 if obj_present else 0.0), phase=phase,
            battery_present=int(batt)))
    return pd.DataFrame(rows)


def run_noise_stress_scenario(duration_s=300.0, seed=123, n_spikes=8,
                               spike_mq2_ppm=350.0, spike_mq135_ppm=1800.0,
                               spike_duration_steps=3):
    """
    A battery-free scenario (ground truth: should never need Emergency_Stop)
    with occasional SHARP, BRIEF sensor noise spikes injected directly into
    mq2_ppm/mq135_ppm -- not derived from temperature, representing sensor-
    level interference (EMI, dust, a momentary glitch) rather than a real
    gas event.

    Exists because run_normal_scenario()'s noise is too mild to test
    anything: mq2_ppm never leaves 0.0 there (H2 onset requires T > 70C,
    which never happens without a battery), so beta(t)'s claimed
    noise-rejection role -- stay smooth/high-memory under transient spikes,
    respond fast to sustained real changes -- has never actually been
    exercised against noise large enough to matter. Confirmed directly:
    h2_medium(300) = 0.6875, a substantial membership degree, so a spike
    at this scale is exactly the kind of input the fuzzy rule base and
    beta(t) need to handle correctly, not something so small it is
    automatically smoothed away, or so absent it never gets tested.

    Adds an `is_noise_spike` column marking injected timesteps, so a
    caller can directly check the FSM's behavior during/after each spike
    (correctly transient rather than escalating like a sustained hazard).
    """
    rng = np.random.RandomState(seed)
    mq2 = MQ2Sensor(seed=seed+1)
    mq135 = MQ135Sensor(seed=seed+2)
    inductive = InductiveProximitySensor(seed=seed+3)

    n_steps = int(duration_s / DT)
    spike_starts = set(rng.choice(
        np.arange(int(20/DT), n_steps - int(20/DT)),  # avoid the very edges
        size=n_spikes, replace=False
    ).tolist())

    rows = []
    active_spike_remaining = 0
    for i, t in enumerate(np.arange(0.0, duration_s, DT)):
        temp_C = T_AMBIENT + rng.normal(0, 0.5) + 1.5 * np.sin(0.05 * t)
        mq2_ppm, mq2_v = mq2.read(temp_C)
        mq135_ppm, mq135_v = mq135.read(temp_C)
        metal_det = inductive.read(False)

        if i in spike_starts:
            active_spike_remaining = spike_duration_steps
        is_spike = active_spike_remaining > 0
        if is_spike:
            mq2_ppm = spike_mq2_ppm + rng.normal(0, 20)
            mq135_ppm = spike_mq135_ppm + rng.normal(0, 80)
            active_spike_remaining -= 1

        rows.append(dict(time=round(t, 2), temperature_C=round(temp_C, 2),
            mq2_ppm=round(max(mq2_ppm, 0.0), 2), mq2_voltage=mq2_v,
            mq135_ppm=round(max(mq135_ppm, 0.0), 2), mq135_voltage=mq135_v,
            metal_present=int(metal_det), object_present=1,
            object_size_cm=10.0, phase="Normal", battery_present=0,
            is_noise_spike=int(is_spike)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Running physics-based sensor simulation...")
    df_h = run_simulation(duration_s=400.0, battery_arrives_at=20.0)
    df_n = run_normal_scenario(duration_s=200.0)
    df_h.to_csv("outputs/ewaste_system/sensor_data_hazardous.csv", index=False)
    df_n.to_csv("outputs/ewaste_system/sensor_data_normal.csv", index=False)
    print(f"Hazardous: {len(df_h)} steps | Normal: {len(df_n)} steps")
    print(f"Peak temp: {df_h['temperature_C'].max():.1f}C | Peak H2: {df_h['mq2_ppm'].max():.1f}ppm | Peak CO2: {df_h['mq135_ppm'].max():.1f}ppm")
    print("Phase distribution:"); print(df_h['phase'].value_counts().to_string())
    print("\nSample readings (every 20s):")
    print(df_h[["time","temperature_C","mq2_ppm","mq135_ppm","metal_present","phase"]].iloc[::200].to_string(index=False))
