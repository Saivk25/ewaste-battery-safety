"""
pub_faults.py -- sensor-failure injection on held-out seeds 100-129 (what happens when a cheap sensor dies).

Faults (injected into the measured channels; ground truth unchanged; IR noise of 0.2 C is added to the temperature
channel of every run, for both systems, because the simulator's empty-belt temperature is noise-free):
  ir_stuck      IR temperature frozen at its value 5 s after the battery arrives
  gas_dead      MQ-2 reads 0 ppm for the whole run
  gas_drift     MQ-2 baseline drift +500 ppm (battery-free hard-noise run, and single battery)
  camera_drop   camera gives no classification while the battery is on the belt
  dropout       all channels missing (NaN) for 3 s every 60 s

Systems: "Proposed" (as in the paper, no health monitor; missing values held at the last reading so it can run)
and "Proposed+monitor" (sensor_health.SensorHealthMonitor: Sensor_Fault on stuck / dead / dropout / camera loss,
remaining channels can still stop the shredder, fail-safe corroboration while IR is faulty, MQ-2 auto-zero).
Hazard FNR is reported with the usual alert set (Warning, Critical, E-stop) and also counting Sensor_Fault
(shredder at 50 % with an operator alarm) as an alert.

Run from the repo root:  python src/pub_faults.py -> results/pub/faults_eval.csv, faults_summary.txt
"""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from pub_experiments import scen, Camera, fsm_run, metrics, DT, ALPHA, EVAL, OUT, HAZ
from sensor_health import SensorHealthMonitor

FAULTS = [("ir_stuck", "single"), ("gas_dead", "single"), ("gas_drift", "single"), ("gas_drift", "hard"),
          ("camera_drop", "single"), ("dropout", "single"), ("none", "single"), ("none", "hard")]


IR_NOISE = 0.2   # C; the simulator's empty-belt / cold-item temperature has no noise, a real IR sensor always does


def inject(df, fault, arrive, seed=0):
    d = df.copy(); t = d.time.values
    d["temperature_C"] = d.temperature_C + np.random.RandomState(seed + 7).normal(0, IR_NOISE, len(d))
    if fault == "ir_stuck":
        k = np.searchsorted(t, arrive + 5.0); d.loc[k:, "temperature_C"] = d.temperature_C.iloc[k]
    elif fault == "gas_dead":
        d["mq2_ppm"] = 0.0
    elif fault == "gas_drift":
        d["mq2_ppm"] = d.mq2_ppm + 500.0
    elif fault == "dropout":
        m = (t % 60.0) >= 57.0
        for c in ("temperature_C", "mq2_ppm", "mq135_ppm"): d.loc[m, c] = np.nan
    return d


def gated_ovr(S, T, obj, ovr):
    from adaptive_risk import BETA_FAST, BETA_SLOW, C_GATE, M_PERSIST, T_CORR, RESET_TIME
    R = np.zeros(len(S)); r = 0.0; run = 0; no = 0.0
    for i, s in enumerate(S):
        if not obj[i]:
            no += DT
            if no >= RESET_TIME: r = 0.0; run = 0
        else: no = 0.0
        if s <= C_GATE: run = 0; b = BETA_FAST
        else:
            run += 1; b = BETA_FAST if (T[i] >= T_CORR or ovr[i] or run >= M_PERSIST) else BETA_SLOW
        r = b * r + (1 - b) * s; R[i] = r
    return R


def fsm_ext(df, R, fault):
    import supervisory_control as sc
    f = sc.SupervisoryFSM(); T, h, c, t = df.temperature_C.values, df.mq2_ppm.values, df.mq135_ppm.values, df.time.values
    return np.array([f.step(t[i], float(R[i]), T[i], h[i], c[i], external_fault=bool(fault[i]))[0].value for i in range(len(R))])


def work(task):
    fault, base, seed, P = task
    from gap_temporal_sweeps import fuzzy_series
    from pub_experiments import gated
    df0, arrive = scen(base, seed)
    df = inject(df0, fault, arrive, seed)
    obj = df.object_present.values
    cam_ok = ~((fault == "camera_drop") & (df.battery_present.values == 1))
    Pm = np.where(cam_ok, P, 0.0)
    # no monitor: hold the last reading through gaps so the pipeline can run at all
    dh = df.copy(); dh[["temperature_C", "mq2_ppm", "mq135_ppm"]] = dh[["temperature_C", "mq2_ppm", "mq135_ppm"]].ffill().fillna({"temperature_C": 25.0, "mq2_ppm": 0.0, "mq135_ppm": 400.0})
    S0 = fuzzy_series(dh)
    st0 = fsm_run(dh, np.clip(gated(S0, dh.temperature_C.values, obj) * (1 + ALPHA * Pm), 0, 1))
    # with monitor
    mon = SensorHealthMonitor(); rows = []
    for r, ok in zip(df.itertuples(), cam_ok):
        fl, T, h2, co2 = mon.step(r.temperature_C, r.mq2_ppm, r.mq135_ppm, r.object_present, camera_ok=bool(ok))
        rows.append((T, h2, co2, fl["fault"], fl["corroboration_override"], fl["gas_drift"]))
    a = np.array(rows, dtype=float)
    dm = df.copy(); dm["temperature_C"], dm["mq2_ppm"], dm["mq135_ppm"] = a[:, 0], a[:, 1], a[:, 2]
    S1 = fuzzy_series(dm)
    st1 = fsm_ext(dm, np.clip(gated_ovr(S1, dm.temperature_C.values, obj, a[:, 4].astype(bool)) * (1 + ALPHA * Pm), 0, 1), a[:, 3].astype(bool))
    first_fault = np.argmax(a[:, 3] > 0) if a[:, 3].any() else None
    out = []
    for name, st in (("Proposed", st0), ("Proposed+monitor", st1)):
        mt = metrics(df0, st, arrive)
        haz = np.isin(df0.phase.values, HAZ)
        fnr2 = float((~np.isin(st[haz], ["Warning", "Critical", "Emergency_Stop", "Sensor_Fault"])).mean()) if haz.any() else np.nan
        out.append(dict(fault=fault, base=base, seed=seed, system=name, **mt, Hazard_FNR_incl_fault=fnr2,
                        fault_alarm_s=(df.time.values[first_fault] - arrive) if (name.endswith("monitor") and first_fault is not None) else np.nan,
                        drift_flag=bool(a[:, 5].any()) if name.endswith("monitor") else np.nan))
    return out


if __name__ == "__main__":
    from multiprocessing import Pool
    cam = Camera(); tasks = []
    for fault, base in FAULTS:
        for s in EVAL:
            df, _ = scen(base, s); tasks.append((fault, base, s, cam.series(df)))
    with Pool(2) as pool:
        ev = pd.DataFrame(sum(pool.map(work, tasks), []))
    ev.to_csv(OUT / "faults_eval.csv", index=False)
    lines = ["Sensor-failure injection, held-out seeds 100-129. Hazard FNR (alerts = Warning/Critical/E-stop) | same counting "
             "Sensor_Fault | mean stop margin (s) | false escalations (battery-free) | seconds from arrival to fault alarm"]
    for (fault, base), g in ev.groupby(["fault", "base"], sort=False):
        lines.append(f"-- {fault} on {base}")
        for s, x in g.groupby("system", sort=False):
            lines.append(f"   {s:<18} FNR {x.Hazard_FNR.mean():.4f} | incl. fault {x.Hazard_FNR_incl_fault.mean():.4f} | margin {x.stop_margin_s.mean():7.1f} | "
                         f"false esc {int(x.false_escalation.sum())}/{len(x)} | E-stop runs {int(x.T_stop_s.notna().sum())} | fault alarm {x.fault_alarm_s.mean():.1f} s"
                         + (f" | drift flagged {int(x.drift_flag.sum())}/{len(x)}" if s.endswith("monitor") else ""))
    print("\n".join(lines)); open(OUT / "faults_summary.txt", "w", encoding="utf-8").write("\n".join(lines))
