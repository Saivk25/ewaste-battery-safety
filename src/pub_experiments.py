"""
pub_experiments.py -- experiments requested for journal submission (no hardware).

Fix 2  assumption-breaking scenarios: IR temperature lagging behind the gas (venting before the
       surface heats), hot benign objects with no gas, warm benign item + gas bursts.
Fix 3  stronger baselines: B0 camera-only stop-on-detection (real CNN), B2-debounce (rule
       thresholds that must persist k s, k tuned), CUSUM on the fuzzy risk (h tuned).
Fix 6  metrics: stop margin before runaway onset (Phase 3), false-alarm episodes per hour.
Fix 7  statistics: paired bootstrap 95% CIs, Cohen's dz, Holm-corrected p-values.
Fix 9  dwell-time sensitivity of the FSM (stop speed vs false alarms).

Protocol: tuning seeds 200-229 (B2-debounce k, CUSUM h; admissible = no false Critical /
E-stop in mild or hard battery-free noise), evaluation on held-out seeds 100-129.
The CNN runs only in the main process: camera probabilities are computed once per
(scenario, seed) with the real checkpoints (same draw logic as RealCNN, seed 77) and
handed to light workers that run fuzzy inference, memory, fusion and the FSM.

Run from the repo root:  python src/pub_experiments.py   -> results/pub/
"""
import os, sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd

OUT = Path("results/pub")
DT = 0.1
ALPHA = 0.35
TUNE, EVAL = list(range(200, 230)), list(range(100, 130))
HAZ = ["Phase1_SEI", "Phase2_Separator", "Phase3_Runaway", "Phase4_Fire"]
ALERT = ["Warning", "Critical", "Emergency_Stop"]


# ----------------------------------------------------------------------------- scenarios
def scen(name, seed):
    import sensor_simulation as ss
    from gap_memory_redesign import NOISE
    from gap_prewarmed import sim as prewarmed
    if name == "single":
        return ss.run_simulation(duration_s=400.0, battery_arrives_at=20.0, seed=seed), 20.0
    if name == "multi":
        d = ss.run_multi_item_scenario(seed=seed, item3_gap_s=0.0)
        return d[d.time < 460.0].reset_index(drop=True), 60.0
    if name.startswith("lag"):                       # IR reading lags the true cell temperature
        lag = int(name[3:]); d = ss.run_simulation(duration_s=400.0, battery_arrives_at=20.0, seed=seed)
        n = int(lag / DT); T = d.temperature_C.values.copy()
        d["temperature_C"] = np.r_[np.full(n, T[0]), T[:-n]] if n else T
        return d, 20.0
    if name.startswith("pre"):                       # pre60, pre80, pre90
        T0 = int(name[3:]); sc = {60: 3, 80: 6, 90: 10}[T0]
        return prewarmed(seed, T0, sc), 20.0
    if name.startswith("hot"):                       # hot benign metal object, no gas, no battery
        Tobj = int(name[3:]); rng = np.random.RandomState(seed)
        mq2 = ss.MQ2Sensor(seed + 1); mq135 = ss.MQ135Sensor(seed + 2); ind = ss.InductiveProximitySensor(seed + 3)
        rows = []
        for t in np.arange(0.0, 120.0, DT):
            on = 20.0 <= t < 100.0
            T = (Tobj if on else ss.T_AMBIENT) + rng.normal(0, 0.5)
            # gas sensors read ambient: a hot object without a failing cell releases no H2/CO2
            p2, v2 = mq2.read(ss.T_AMBIENT); p135, v135 = mq135.read(ss.T_AMBIENT)
            rows.append(dict(time=round(t, 2), temperature_C=round(T, 2), mq2_ppm=p2, mq2_voltage=v2, mq135_ppm=p135,
                             mq135_voltage=v135, metal_present=int(ind.read(on)), object_present=int(on),
                             object_size_cm=10.0 if on else 0.0, phase="Normal", battery_present=0))
        return pd.DataFrame(rows), 20.0
    if name in NOISE:
        return ss.run_noise_stress_scenario(seed=seed + 1000, **NOISE[name]), 0.0
    if name == "warm_bursts":                        # warm (50 C) benign item + hard gas bursts
        d = ss.run_noise_stress_scenario(seed=seed + 1000, **NOISE["hard"])
        d["temperature_C"] = d.temperature_C + 25.0
        return d, 0.0
    raise ValueError(name)


BATTERY_FREE = {"mild", "hard", "harder", "hot60", "hot90", "hot120", "warm_bursts"}
EVAL_SCEN = ["single", "multi", "lag5", "lag10", "lag20", "pre60", "pre80", "pre90",
             "mild", "hard", "harder", "hot60", "hot90", "hot120", "warm_bursts"]
TUNE_SCEN = ["single", "mild", "hard"]


# ----------------------------------------------------------------------------- camera (main process)
class Camera:
    def __init__(self):
        from adaptive_risk import RealCNN
        self.cnn = RealCNN(seed=77)

    def series(self, df):
        c = self.cnn
        c._rng = np.random.RandomState(77); c._was_present = False
        out = np.zeros(len(df))
        for i, r in enumerate(df.itertuples()):
            o = c.predict(temperature=r.temperature_C, mq2_ppm=r.mq2_ppm, object_present=bool(r.object_present),
                          metal_present=bool(r.metal_present), battery_present=bool(r.battery_present))
            out[i] = o["battery_prob"]
        return out


# ----------------------------------------------------------------------------- systems (workers)
def fsm_run(df, R, dwell=None):
    import supervisory_control as sc
    if dwell is not None:
        saved = dict(sc.MIN_DWELL)
        sc.MIN_DWELL[sc.State.WARNING], sc.MIN_DWELL[sc.State.CRITICAL] = dwell
    try:
        f = sc.SupervisoryFSM(); T, h, c, t = df.temperature_C.values, df.mq2_ppm.values, df.mq135_ppm.values, df.time.values
        st = np.array([f.step(t[i], float(R[i]), T[i], h[i], c[i])[0].value for i in range(len(R))])
    finally:
        if dwell is not None:
            sc.MIN_DWELL.update(saved)
    return st


def gated(S, T, obj):
    from adaptive_risk import BETA_FAST, BETA_SLOW, C_GATE, M_PERSIST, T_CORR, RESET_TIME
    R = np.zeros(len(S)); r = 0.0; run = 0; no = 0.0
    for i, s in enumerate(S):
        if not obj[i]:
            no += DT
            if no >= RESET_TIME: r = 0.0; run = 0
        else:
            no = 0.0
        if s <= C_GATE: run = 0; b = BETA_FAST
        else:
            run += 1; b = BETA_FAST if (T[i] >= T_CORR or run >= M_PERSIST) else BETA_SLOW
        r = b * r + (1 - b) * s; R[i] = r
    return R


def b2_raw(df):
    T, h2, co2, m = df.temperature_C.values, df.mq2_ppm.values, df.mq135_ppm.values, df.metal_present.values
    stop = (T > 90) | (h2 > 500) | (co2 > 2500)
    warn = (T > 50) | (h2 > 80) | ((m == 1) & (T > 35))
    return stop, warn


def b2_debounce(df, k):
    stop, warn = b2_raw(df); n = int(round(k / DT))
    st = []; latched = False; cs = cw = 0
    for s, w in zip(stop, warn):
        cs = cs + 1 if s else 0; cw = cw + 1 if (w or s) else 0
        if latched or cs > n: latched = True; st.append("Emergency_Stop")
        elif cw > n: st.append("Warning")
        else: st.append("Normal")
    return np.array(st)


def cusum(S, h, k=0.10, mu=0.17):
    g = 0.0; st = []; latched = False
    for s in S:
        g = max(0.0, g + s - mu - k)
        if latched or g > h: latched = True; st.append("Emergency_Stop")
        elif g > h / 2: st.append("Warning")
        else: st.append("Normal")
    return np.array(st)


def b0_camera_only(P, obj):
    from evaluate_model import BATTERY_THRESHOLD
    st = []; latched = False
    for p, o in zip(P, obj):
        if latched or (o and p >= BATTERY_THRESHOLD): latched = True; st.append("Emergency_Stop")
        else: st.append("Normal")
    return np.array(st)


def metrics(df, st, arrive):
    t = df.time.values; ph = df.phase.values
    out = {}
    haz = np.isin(ph, HAZ)
    out["Hazard_FNR"] = float((~np.isin(st[haz], ALERT)).mean()) if haz.any() else np.nan
    w = np.where(np.isin(st, ALERT) & (t >= arrive))[0]; s = np.where((st == "Emergency_Stop") & (t >= arrive))[0]
    out["T_warn_s"] = float(t[w[0]] - arrive) if len(w) else np.nan
    out["T_stop_s"] = float(t[s[0]] - arrive) if len(s) else np.nan
    p3 = np.where(np.isin(ph, ["Phase3_Runaway", "Phase4_Fire"]))[0]
    out["stop_margin_s"] = float(t[p3[0]] - t[s[0]]) if (len(p3) and len(s)) else np.nan   # >0: stopped before runaway
    esc = np.isin(st, ["Critical", "Emergency_Stop"])
    out["false_escalation"] = bool(esc.any()); out["false_estop"] = bool((st == "Emergency_Stop").any())
    ep = int(np.sum(esc[1:] & ~esc[:-1]) + int(esc[0]))
    out["escalation_episodes_per_h"] = ep / (len(st) * DT / 3600.0)
    return out


def systems(df, S, P, extra):
    """extra: dict with tuned 'b2_k' and 'cusum_h' (None while tuning)."""
    from baseline_comparison import baseline_vision_only, baseline_rule_based, baseline_fixed_fuzzy
    T, obj = df.temperature_C.values, df.object_present.values
    R = gated(S, T, obj); Rf = np.clip(R * (1 + ALPHA * P), 0, 1)
    out = {"Proposed": fsm_run(df, Rf),
           "Proposed_occluded": fsm_run(df, np.clip(R, 0, 1)),
           "B0_CameraOnly": b0_camera_only(P, obj),
           "B1_VisionProxy": baseline_vision_only(df, seed=0).state.values,
           "B2_Rule": baseline_rule_based(df).state.values,
           "B3_FixedFuzzy": baseline_fixed_fuzzy(df).state.values}
    for k in ([extra["b2_k"]] if extra.get("b2_k") is not None else [0.5, 1, 2, 3, 4, 5]):
        out[f"B2_Debounce_{k:g}s"] = b2_debounce(df, k)
    for h in ([extra["cusum_h"]] if extra.get("cusum_h") is not None else [1, 2, 4, 8, 16, 32]):
        out[f"CUSUM_h{h:g}"] = cusum(S, h)
    if extra.get("dwell"):
        for dw in [(2.0, 1.0), (1.0, 0.5), (0.5, 0.2), (0.0, 0.0)]:
            out[f"Proposed_dwell_{dw[0]:g}_{dw[1]:g}"] = fsm_run(df, Rf, dwell=dw)
    return out


def work(task):
    name, seed, P, extra = task
    from gap_temporal_sweeps import fuzzy_series
    df, arrive = scen(name, seed)
    S = fuzzy_series(df)
    rows = []
    for sysname, st in systems(df, S, P, extra).items():
        rows.append(dict(scenario=name, seed=seed, system=sysname, **metrics(df, st, arrive)))
    return rows


# ----------------------------------------------------------------------------- statistics
def paired_stats(a, b, n_boot=10000, rng=np.random.RandomState(0)):
    d = np.asarray(a) - np.asarray(b); d = d[~np.isnan(d)]
    if len(d) < 2 or np.allclose(d, 0):
        return dict(n=len(d), mean_diff=float(d.mean()) if len(d) else np.nan, ci_lo=np.nan, ci_hi=np.nan, dz=np.nan, p=np.nan)
    from scipy import stats
    boots = rng.choice(d, (n_boot, len(d))).mean(1)
    t, p = stats.ttest_rel(np.asarray(a)[~np.isnan(np.asarray(a) - np.asarray(b))], np.asarray(b)[~np.isnan(np.asarray(a) - np.asarray(b))])
    return dict(n=len(d), mean_diff=float(d.mean()), ci_lo=float(np.percentile(boots, 2.5)), ci_hi=float(np.percentile(boots, 97.5)),
                dz=float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan, p=float(p))


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(np.where(np.isnan(ps), 2, ps)); adj = np.full(len(ps), np.nan); m = np.sum(~np.isnan(ps)); run = 0.0
    for rank, i in enumerate(order):
        if np.isnan(ps[i]): continue
        run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return adj


# ----------------------------------------------------------------------------- main
def camera_tasks(cam, names, seeds, extra):
    tasks = []
    for name in names:
        for seed in seeds:
            df, _ = scen(name, seed)
            tasks.append((name, seed, cam.series(df), extra))
    return tasks


if __name__ == "__main__":
    from multiprocessing import Pool
    OUT.mkdir(parents=True, exist_ok=True)
    cam = Camera()
    log = []
    def L(s=""): print(s, flush=True); log.append(s)

    # --- tuning (B2-debounce k, CUSUM h)
    tasks = camera_tasks(cam, TUNE_SCEN, TUNE, {})
    with Pool(2) as pool:
        tune = pd.DataFrame(sum(pool.map(work, tasks), []))
    tune.to_csv(OUT / "tuning.csv", index=False)
    def pick(prefix):
        g = tune[tune.system.str.startswith(prefix)]
        bad = g[g.scenario.isin(["mild", "hard"])].groupby("system").false_escalation.sum()
        fnr = g[g.scenario == "single"].groupby("system").Hazard_FNR.mean()
        ok = [s for s in fnr.index if bad.get(s, 0) == 0]
        return (min(ok, key=lambda s: fnr[s]) if ok else None), fnr.to_dict(), bad.to_dict()
    b2_best, b2_fnr, b2_bad = pick("B2_Debounce"); cu_best, cu_fnr, cu_bad = pick("CUSUM")
    L(f"B2-debounce tuning: FNR {b2_fnr} | false escalations {b2_bad} -> chosen {b2_best}")
    L(f"CUSUM tuning:      FNR {cu_fnr} | false escalations {cu_bad} -> chosen {cu_best}")
    extra = {"b2_k": float(b2_best.split("_")[-1][:-1]) if b2_best else 5.0,
             "cusum_h": float(cu_best.split("_h")[-1]) if cu_best else 32.0, "dwell": True}

    # --- evaluation
    tasks = camera_tasks(cam, EVAL_SCEN, EVAL, extra)
    with Pool(2) as pool:
        ev = pd.DataFrame(sum(pool.map(work, tasks), []))
    ev.to_csv(OUT / "eval.csv", index=False)

    main_sys = ["Proposed", "B0_CameraOnly", "B1_VisionProxy", "B2_Rule", f"B2_Debounce_{extra['b2_k']:g}s",
                "B3_FixedFuzzy", f"CUSUM_h{extra['cusum_h']:g}"]
    L("\nHELD-OUT (seeds 100-129). Hazard scenarios: Hazard FNR mean | T_stop mean | stop margin before runaway mean (s)")
    for sc_ in ["single", "multi", "lag5", "lag10", "lag20", "pre60", "pre80", "pre90"]:
        L(f"-- {sc_}")
        for s in main_sys + ["Proposed_occluded"]:
            x = ev[(ev.scenario == sc_) & (ev.system == s)]
            L(f"   {s:<24} FNR {x.Hazard_FNR.mean():.4f}  T_stop {x.T_stop_s.mean():6.1f} (n={x.T_stop_s.notna().sum()})  margin {x.stop_margin_s.mean():6.1f}")
    L("\nBattery-free scenarios: runs with false Critical/E-stop (of 30) | false E-stop runs | escalation episodes per hour")
    for sc_ in sorted(BATTERY_FREE):
        L(f"-- {sc_}")
        for s in main_sys:
            x = ev[(ev.scenario == sc_) & (ev.system == s)]
            L(f"   {s:<24} {int(x.false_escalation.sum()):>2} | {int(x.false_estop.sum()):>2} | {x.escalation_episodes_per_h.mean():7.1f}/h")
    L("\nDwell-time sensitivity (Proposed): mean T_stop on failing cells | false-escalation runs in hard / harder / warm_bursts")
    for s in [c for c in ev.system.unique() if c.startswith("Proposed_dwell")]:
        x = ev[ev.system == s]
        stops = " / ".join(f"{x[x.scenario == p].T_stop_s.mean():.1f}" for p in ("pre60", "pre80", "pre90"))
        fa = " / ".join(str(int(x[x.scenario == n].false_escalation.sum())) for n in ("hard", "harder", "warm_bursts"))
        L(f"   {s:<24} stop {stops} s | false escalation {fa}")

    # --- statistics: Proposed vs each baseline, Hazard FNR, per hazard scenario; Holm across the family
    rows = []
    for sc_ in ["single", "multi", "lag5", "lag10", "lag20"]:
        for b in main_sys[1:]:
            a = ev[(ev.scenario == sc_) & (ev.system == "Proposed")].set_index("seed").Hazard_FNR
            c = ev[(ev.scenario == sc_) & (ev.system == b)].set_index("seed").Hazard_FNR
            j = pd.concat([a, c], axis=1).dropna()
            rows.append(dict(scenario=sc_, comparison=f"Proposed - {b}", **paired_stats(j.iloc[:, 0], j.iloc[:, 1])))
    st = pd.DataFrame(rows); st["p_holm"] = holm(st.p.values)
    st.to_csv(OUT / "stats.csv", index=False)
    L("\nPaired comparisons of Hazard FNR (Proposed minus baseline): mean diff [95% bootstrap CI], Cohen dz, Holm p")
    for r in st.itertuples():
        L(f"   {r.scenario:<7} {r.comparison:<34} {r.mean_diff:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] dz={r.dz:.2f} p_holm={r.p_holm:.3g}")
    json.dump({"tuned": extra}, open(OUT / "tuned.json", "w"), indent=1)
    open(OUT / "summary.txt", "w", encoding="utf-8").write("\n".join(log))
