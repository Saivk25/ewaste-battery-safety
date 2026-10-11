"""
gap_prewarmed.py -- time-to-stop for batteries that arrive ALREADY failing.

The main scenario heats a cell from ambient after arrival (192 s to stop). A real belt
moves an item to the shredder in seconds, so here the cell reaches the belt at T0 and runs
away faster (rate multiplier). T0 is applied at the moment of arrival (not before).
Compares the adaptive filter (0.85, 0.15), a fixed beta = 0.85 filter, and rule-based B2.
Run from the repo root:  python src/gap_prewarmed.py   -> results/gaps/prewarmed.csv
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from sensor_simulation import ThermalRunawayModel, MQ2Sensor, MQ135Sensor, InductiveProximitySensor
from gap_temporal_sweeps import fuzzy_series, fixed_filter, adaptive_filter, fsm_frame, ALERT, DT

OUT = Path("results/gaps")


class Prewarmed(ThermalRunawayModel):
    def __init__(self, seed, T0, scale):
        super().__init__(seed); self.T0 = T0; self.scale = scale; self._started = False

    def _rate(self, T):
        return super()._rate(T) * self.scale

    def step(self, battery_present):
        if battery_present and not self._started:
            self._started = True; self.T = self.T0
        return super().step(battery_present)


def sim(seed, T0, scale, dur=120.0, arrive=20.0):
    th = Prewarmed(seed, T0, scale); mq2 = MQ2Sensor(seed + 1); mq135 = MQ135Sensor(seed + 2); ind = InductiveProximitySensor(seed + 3)
    rows = []
    for t in np.arange(0.0, dur, DT):
        b = t >= arrive
        T, ph = th.step(b); p2, v2 = mq2.read(T); p135, v135 = mq135.read(T); m = ind.read(b)
        rows.append(dict(time=round(t, 2), temperature_C=T, mq2_ppm=p2, mq2_voltage=v2, mq135_ppm=p135, mq135_voltage=v135,
                         metal_present=int(m), object_present=int(b), object_size_cm=10.0 if b else 0.0, phase=ph, battery_present=int(b)))
    return pd.DataFrame(rows)


def one(k):
    seed = 100 + k; out = []
    for T0, sc in ((25, 1), (60, 3), (80, 6), (90, 10)):
        d = sim(seed, T0, sc, dur=400.0 if sc == 1 else 120.0)
        S = fuzzy_series(d); obj = d.object_present.values
        tb2 = d[((d.temperature_C > 90) | (d.mq2_ppm > 500) | (d.mq135_ppm > 2500)) & (d.time >= 20.0)]
        for name, R in (("adaptive_0.85_0.15", adaptive_filter(S, obj, 0.85, 0.15)), ("fixed_0.85", fixed_filter(S, 0.85))):
            fr = fsm_frame(d, R)
            w = fr[fr.state.isin(ALERT) & (fr.time >= 20.0)]; s = fr[(fr.state == "Emergency_Stop") & (fr.time >= 20.0)]
            pre = float(fr[fr.time < 20.0].state.isin(ALERT).mean())
            out.append(dict(seed=seed, T0=T0, scale=sc, variant=name, T_warn_s=(w.time.iloc[0] - 20) if len(w) else np.nan,
                            T_stop_s=(s.time.iloc[0] - 20) if len(s) else np.nan, B2_stop_s=(tb2.time.iloc[0] - 20) if len(tb2) else np.nan,
                            alert_before_arrival=pre, peak_T=float(d.temperature_C.max())))
    return out


if __name__ == "__main__":
    from multiprocessing import Pool
    rows = []
    with Pool(8) as p:
        for r in p.imap_unordered(one, range(30)):
            rows += r
    df = pd.DataFrame(rows); df.to_csv(OUT / "prewarmed.csv", index=False)
    for (T0, sc), g in df.groupby(["T0", "scale"]):
        for v in ("adaptive_0.85_0.15", "fixed_0.85"):
            x = g[g.variant == v]
            print(f"T0={T0:>2}C rate x{sc:<2} {v:<19} T_warn {x.T_warn_s.mean():6.1f} (n={x.T_warn_s.notna().sum()}) T_stop {x.T_stop_s.mean():6.1f} (n={x.T_stop_s.notna().sum()}) B2 {x.B2_stop_s.mean():6.1f} alert-before-arrival {x.alert_before_arrival.mean():.3f}")
