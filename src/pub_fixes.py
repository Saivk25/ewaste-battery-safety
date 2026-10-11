"""
pub_fixes.py -- second round of journal fixes (software only, no hardware).

R1  literature-calibrated thermal model: onset landmarks moved to published values (SEI ~90 C,
    separator ~130 C, cathode / runaway ~200 C [Feng 2018], H2 venting ~115-120 C [Gardner 2025]),
    used ONLY for evaluation, with every setting frozen from the original tuning -> tests whether
    the method generalises to a thermal model it was not designed on.
R2  rate-of-rise corroboration: temperature corroborates a Critical-zone rise only if T >= T_CORR
    AND the temperature is still rising (least-squares slope over the last W s >= r C/s, computed
    from 1 s after the object's arrival). Aimed at the warm-item + gas-burst failure. W, r tuned on
    seeds 200-229 (admissible = no false Critical/E-stop in mild, hard and warm_bursts), evaluated
    once on held-out seeds 100-129.
R3  miss-rate vs false-escalation trade-off across each method's parameter range (sensor-only,
    held-out seeds), so the comparison does not hinge on one operating point per baseline.
R4  CFNC weight sensitivity: system ranking under other runaway / early-miss weights.
R5  learned baseline: logistic regression on per-step sensor features (T, H2, CO2, metal,
    temperature slope, 5 s max H2) trained on tuning seeds, gain tuned with the same rule.

Run from the repo root:  python src/pub_fixes.py  -> results/pub/fixes_*.csv, fixes_summary.txt
"""
import sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from pub_experiments import (scen, Camera, gated, fsm_run, b2_debounce, cusum, metrics, paired_stats, holm,
                             DT, HAZ, ALERT, ALPHA, TUNE, EVAL, OUT)

TUNED = json.load(open(OUT / "tuned.json"))["tuned"]
W_GRID, R_GRID = (3.0, 5.0, 10.0), (0.02, 0.05, 0.1, 0.2)


# ----------------------------------------------------------------------------- R1 calibrated model
def cal_sim(seed, duration_s=700.0, arrive=20.0, lag_s=0.0):
    import sensor_simulation as ss

    class CalThermal(ss.ThermalRunawayModel):
        PHASES = [(25, 90, 0.15, 1.2), (90, 130, 0.30, 1.5), (130, 200, 1.0, 1.5), (200, 500, 15.0, 2.0), (500, 850, 12.0, 1.8)]

        def step(self, battery_present):
            T, _ = super().step(battery_present)
            self.phase_str = ("Normal" if T < 90 else "Phase1_SEI" if T < 130 else "Phase2_Separator" if T < 200
                              else "Phase3_Runaway" if T < 500 else "Phase4_Fire")
            return T, self.phase_str

    class CalMQ2(ss.MQ2Sensor):
        def _temp_to_ppm(self, T):
            return 0.0 if T < 115.0 else min(60.0 * np.exp(0.042 * (T - 115.0)), 10000.0)

    class CalMQ135(ss.MQ135Sensor):
        def _temp_to_ppm(self, T):
            return 400.0 if T < 150.0 else 400.0 + 4600.0 / (1 + np.exp(-0.035 * (T - 230.0)))

    th, mq2, mq135 = CalThermal(seed), CalMQ2(seed + 1), CalMQ135(seed + 2)
    ind, us = ss.InductiveProximitySensor(seed + 3), ss.UltrasonicSensor(seed + 4)
    rows = []
    for t in np.arange(0.0, duration_s, DT):
        b = t >= arrive
        T, ph = th.step(b); p2, v2 = mq2.read(T); p135, v135 = mq135.read(T)
        o, sz = us.read(t)
        if b: o, sz = True, max(sz, 10.0)
        rows.append(dict(time=round(t, 2), temperature_C=T, mq2_ppm=p2, mq2_voltage=v2, mq135_ppm=p135, mq135_voltage=v135,
                         metal_present=int(ind.read(b)), object_present=int(o), object_size_cm=round(sz, 2), phase=ph, battery_present=int(b)))
    d = pd.DataFrame(rows)
    if lag_s:
        n = int(lag_s / DT); T = d.temperature_C.values.copy(); d["temperature_C"] = np.r_[np.full(n, T[0]), T[:-n]]
    return d, arrive


def scen2(name, seed):
    if name == "cal_single": return cal_sim(seed)
    if name == "cal_lag20": return cal_sim(seed, lag_s=20.0)
    if name == "warm75_bursts":                      # hotter benign item (about 75 C) + hard bursts
        import sensor_simulation as ss
        from gap_memory_redesign import NOISE
        d = ss.run_noise_stress_scenario(seed=seed + 1000, **NOISE["hard"]); d["temperature_C"] += 50.0
        return d, 0.0
    return scen(name, seed)


# ----------------------------------------------------------------------------- R2 rate-of-rise gate
def slope_series(T, obj, W):
    """Least-squares temperature slope (C/s) over the last W s, using only samples from 1 s after arrival."""
    n = len(T); out = np.full(n, np.nan); start = None; w = int(W / DT); skip = int(1.0 / DT)
    for i in range(n):
        if not obj[i]: start = None; continue
        if start is None: start = i
        a = max(start + skip, i - w + 1)
        if i - a + 1 >= 10:
            y = T[a:i + 1]; x = np.arange(len(y)) * DT; xm = x - x.mean()
            out[i] = float((xm * (y - y.mean())).sum() / (xm ** 2).sum())
    return out


def gated_rate(S, T, obj, W, r):
    from adaptive_risk import BETA_FAST, BETA_SLOW, C_GATE, M_PERSIST, T_CORR, RESET_TIME
    sl = slope_series(T, obj, W)
    R = np.zeros(len(S)); x = 0.0; run = 0; no = 0.0
    for i, s in enumerate(S):
        if not obj[i]:
            no += DT
            if no >= RESET_TIME: x = 0.0; run = 0
        else:
            no = 0.0
        if s <= C_GATE: run = 0; b = BETA_FAST
        else:
            run += 1
            corr = T[i] >= T_CORR and not np.isnan(sl[i]) and sl[i] >= r
            b = BETA_FAST if (corr or run >= M_PERSIST) else BETA_SLOW
        x = b * x + (1 - b) * s; R[i] = x
    return R


# ----------------------------------------------------------------------------- R3 / R5 helpers
def fixed(S, beta):
    R = np.zeros(len(S)); x = 0.0
    for i, s in enumerate(S): x = beta * x + (1 - beta) * s; R[i] = x
    return R


def expo(S, obj, lam, bmax=0.85):
    from adaptive_risk import RESET_TIME
    R = np.zeros(len(S)); x = 0.0; sp = 0.0; no = 0.0
    for i, s in enumerate(S):
        if not obj[i]:
            no += DT
            if no >= RESET_TIME: x = 0.0
        else: no = 0.0
        b = bmax * np.exp(-lam * abs(s - sp)); x = b * x + (1 - b) * s; sp = s; R[i] = x
    return R


def gated_m(S, T, obj, m):
    import adaptive_risk as ar
    saved = ar.M_PERSIST; ar.M_PERSIST = m
    try: return gated(S, T, obj)
    finally: ar.M_PERSIST = saved


def features(df):
    T = df.temperature_C.values; h = df.mq2_ppm.values; c = df.mq135_ppm.values; m = df.metal_present.values
    sl = np.nan_to_num(slope_series(T, df.object_present.values, 5.0))
    hmax = pd.Series(h).rolling(50, min_periods=1).max().values
    return np.c_[T / 100, np.log1p(h), np.log1p(c), m, sl, np.log1p(hmax)]


def cfnc_parts(df, st):
    ph = df.phase.values; miss = ~np.isin(st, ALERT)
    run = np.isin(ph, ["Phase3_Runaway", "Phase4_Fire"]); early = np.isin(ph, ["Phase1_SEI", "Phase2_Separator"])
    return dict(n_haz=int((run | early).sum()), miss_run=int((miss & run).sum()), miss_early=int((miss & early).sum()))


# ----------------------------------------------------------------------------- worker
def work(task):
    name, seed, P, cfg = task
    from gap_temporal_sweeps import fuzzy_series
    df, arrive = scen2(name, seed)
    S = fuzzy_series(df); T = df.temperature_C.values; obj = df.object_present.values
    fuse = lambda R: np.clip(R * (1 + ALPHA * P), 0, 1)
    sys_ = {}
    Rg = gated(S, T, obj)
    sys_["Proposed"] = fsm_run(df, fuse(Rg))
    sys_["Proposed_sensoronly"] = fsm_run(df, np.clip(Rg, 0, 1))
    for W in ([cfg["W"]] if cfg.get("W") else W_GRID):
        for r in ([cfg["r"]] if cfg.get("r") else R_GRID):
            Rr = gated_rate(S, T, obj, W, r)
            sys_[f"ProposedR_W{W:g}_r{r:g}"] = fsm_run(df, fuse(Rr))
            if cfg.get("W"): sys_["ProposedR_sensoronly"] = fsm_run(df, np.clip(Rr, 0, 1))
    if cfg.get("lr") is not None:                       # learned baseline, gain grid while tuning
        coef, icpt = np.array(cfg["lr"]["coef"]), cfg["lr"]["icpt"]
        p = 1 / (1 + np.exp(-(features(df) @ coef + icpt)))
        for g in ([cfg["lr_gain"]] if cfg.get("lr_gain") else (0.6, 0.8, 1.0)):
            sys_[f"B4_Logistic_g{g:g}"] = fsm_run(df, np.clip(g * p, 0, 1))
    if cfg.get("sweep"):                                # R3 trade-off curves (sensor-only)
        for b in (0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99): sys_[f"sw_Fixed_{b:g}"] = fsm_run(df, fixed(S, b))
        for lam in (0.05, 0.15, 0.5, 1.5, 5.0): sys_[f"sw_Exp_{lam:g}"] = fsm_run(df, expo(S, obj, lam))
        for k in (0.0, 0.5, 1, 2, 3, 4, 5): sys_[f"sw_B2deb_{k:g}"] = b2_debounce(df, k)
        for h in (1, 2, 4, 8, 16, 32): sys_[f"sw_CUSUM_{h:g}"] = cusum(S, h)
        for m in (5, 10, 20, 40, 80): sys_[f"sw_Gated_{m:g}"] = fsm_run(df, gated_m(S, T, obj, m))
        for W in W_GRID:
            for r in R_GRID: sys_[f"sw_GatedRate_{W:g}_{r:g}"] = fsm_run(df, gated_rate(S, T, obj, W, r))
    if cfg.get("full"):                                 # baselines on the calibrated model
        from baseline_comparison import baseline_rule_based, baseline_fixed_fuzzy
        sys_["B2_Rule"] = baseline_rule_based(df).state.values
        sys_["B3_FixedFuzzy"] = baseline_fixed_fuzzy(df).state.values
        sys_[f"B2_Debounce_{TUNED['b2_k']:g}s"] = b2_debounce(df, TUNED["b2_k"])
        sys_[f"CUSUM_h{TUNED['cusum_h']:g}"] = cusum(S, TUNED["cusum_h"])
    rows = []
    for k, st in sys_.items():
        rows.append(dict(scenario=name, seed=seed, system=k, **metrics(df, st, arrive), **cfnc_parts(df, st)))
    return rows


def lr_train_task(task):
    name, seed = task
    df, _ = scen2(name, seed)
    return features(df), np.isin(df.phase.values, HAZ).astype(int)


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    from multiprocessing import Pool
    from sklearn.linear_model import LogisticRegression
    log = []
    def L(s=""): print(s, flush=True); log.append(s)
    cam = Camera()
    def tasks(names, seeds, cfg):
        out = []
        for n in names:
            for s in seeds:
                df, _ = scen2(n, s); out.append((n, s, cam.series(df), cfg))
        return out

    with Pool(2) as pool:
        # R5: train the logistic baseline on tuning seeds
        parts = pool.map(lr_train_task, [(n, s) for n in ("single", "mild", "hard") for s in TUNE])
        X = np.vstack([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)
        lrp = {"coef": lr.coef_[0].tolist(), "icpt": float(lr.intercept_[0])}
        # tuning: rate gate (W, r) and logistic gain
        tune = pd.DataFrame(sum(pool.map(work, tasks(["single", "mild", "hard", "warm_bursts"], TUNE, {"lr": lrp})), []))
    tune.to_csv(OUT / "fixes_tuning.csv", index=False)

    def pick(prefix, noise):
        g = tune[tune.system.str.startswith(prefix)]
        bad = g[g.scenario.isin(noise)].groupby("system").false_escalation.sum()
        fnr = g[g.scenario == "single"].groupby("system").Hazard_FNR.mean()
        ok = [s for s in fnr.index if bad.get(s, 0) == 0]
        return (min(ok, key=lambda s: fnr[s]) if ok else None), fnr.round(4).to_dict(), bad.to_dict()
    rbest, rfnr, rbad = pick("ProposedR_", ["mild", "hard", "warm_bursts"])
    lbest, lfnr, lbad = pick("B4_Logistic", ["mild", "hard"])
    L(f"Rate gate tuning (admissible: no false escalation in mild/hard/warm_bursts): FNR {rfnr} | false {rbad} -> {rbest}")
    L(f"Logistic tuning (admissible: mild/hard): FNR {lfnr} | false {lbad} -> {lbest}")
    if rbest is None:
        L("No admissible rate-gate setting; using the least-false setting for reporting only.")
        rbest = min(rbad, key=lambda s: (rbad[s], rfnr.get(s, 1)))
    W, r = (float(v) for v in rbest.replace("ProposedR_W", "").split("_r"))
    gain = float(lbest.split("_g")[-1]) if lbest else 0.6
    cfg = {"W": W, "r": r, "lr": lrp, "lr_gain": gain, "full": True}

    names = ["single", "multi", "lag5", "lag20", "lag60", "pre60", "pre80", "pre90",
             "mild", "hard", "harder", "hot60", "hot90", "warm_bursts", "warm75_bursts"]
    with Pool(2) as pool:
        ev = pd.DataFrame(sum(pool.map(work, tasks(names, EVAL, cfg)), []))
        sw = pd.DataFrame(sum(pool.map(work, tasks(["single", "hard", "harder", "warm_bursts"], EVAL, {"sweep": True, "W": W, "r": r})), []))
        cal = pd.DataFrame(sum(pool.map(work, tasks(["cal_single", "cal_lag20"], EVAL, cfg)), []))
    ev.to_csv(OUT / "fixes_eval.csv", index=False); sw.to_csv(OUT / "fixes_sweep.csv", index=False); cal.to_csv(OUT / "fixes_calibrated.csv", index=False)
    json.dump({"W": W, "r": r, "lr_gain": gain, "lr": lrp}, open(OUT / "fixes_tuned.json", "w"), indent=1)

    RB = f"ProposedR_W{W:g}_r{r:g}"; LB = f"B4_Logistic_g{gain:g}"
    L(f"\nHELD-OUT: Proposed (original gate) vs ProposedR (rate gate W={W:g}s r={r:g}C/s) vs logistic baseline")
    for sc_ in names:
        x = ev[ev.scenario == sc_]
        def one(s):
            y = x[x.system == s]
            return (f"FNR {y.Hazard_FNR.mean():.4f} stop {y.T_stop_s.mean():6.1f}" if y.Hazard_FNR.notna().any()
                    else f"false {int(y.false_escalation.sum()):>2}/{int(y.false_estop.sum()):>2}")
        L(f"  {sc_:<14} Proposed {one('Proposed'):<26} | ProposedR {one(RB):<26} | Logistic {one(LB)}")
    rows = []
    for sc_ in ["single", "multi", "lag5", "lag20", "lag60"]:
        for b in (RB, LB):
            a = ev[(ev.scenario == sc_) & (ev.system == "Proposed")].set_index("seed").Hazard_FNR
            c = ev[(ev.scenario == sc_) & (ev.system == b)].set_index("seed").Hazard_FNR
            j = pd.concat([a, c], axis=1).dropna()
            rows.append(dict(scenario=sc_, comparison=f"Proposed - {b}", **paired_stats(j.iloc[:, 0], j.iloc[:, 1])))
    stt = pd.DataFrame(rows); stt["p_holm"] = holm(stt.p.values); stt.to_csv(OUT / "fixes_stats.csv", index=False)
    L("\nPaired Hazard FNR differences (Holm):")
    for q in stt.itertuples():
        L(f"  {q.scenario:<7} {q.comparison:<40} {q.mean_diff:+.4f} [{q.ci_lo:+.4f}, {q.ci_hi:+.4f}] p_holm={q.p_holm:.3g}")

    L("\nCALIBRATED THERMAL MODEL (frozen settings, held-out seeds): FNR | stop margin before runaway")
    for sc_ in ["cal_single", "cal_lag20"]:
        for s in ["Proposed", RB, "Proposed_sensoronly", "B2_Rule", f"B2_Debounce_{TUNED['b2_k']:g}s", "B3_FixedFuzzy", f"CUSUM_h{TUNED['cusum_h']:g}"]:
            y = cal[(cal.scenario == sc_) & (cal.system == s)]
            L(f"  {sc_:<11} {s:<26} FNR {y.Hazard_FNR.mean():.4f}  margin {y.stop_margin_s.mean():6.1f}  stopped {y.T_stop_s.notna().sum()}/30")

    L("\nTRADE-OFF (sensor-only, held-out): single-battery FNR vs fraction of hard/harder/warm_bursts runs with a false escalation")
    fam = sw.system.str.extract(r"sw_([A-Za-z0-9]+)_")[0]
    sw["family"] = fam
    tr = []
    for s, g in sw[sw.system.str.startswith("sw_")].groupby("system"):
        tr.append(dict(system=s, family=g.family.iloc[0], fnr=g[g.scenario == "single"].Hazard_FNR.mean(),
                       false_rate=g[g.scenario != "single"].false_escalation.mean()))
    tr = pd.DataFrame(tr).sort_values(["family", "false_rate"]); tr.to_csv(OUT / "fixes_tradeoff.csv", index=False)
    for q in tr.itertuples(): L(f"  {q.system:<24} FNR {q.fnr:.4f}  false {q.false_rate:.3f}")

    L("\nCFNC weight sensitivity (single, held-out; CFNC = (w_run*N_run + w_early*N_early)/N_haz): mean per system")
    base = ev[(ev.scenario == "single") & ~ev.system.str.startswith("ProposedR_W") | (ev.system == RB)]
    base = base[base.scenario == "single"]
    for wr, we in ((10, 2), (5, 1), (20, 1), (2, 1), (1, 1)):
        c = base.assign(cfnc=(wr * base.miss_run + we * base.miss_early) / base.n_haz.replace(0, np.nan)).groupby("system").cfnc.mean()
        L(f"  weights ({wr},{we}): " + ", ".join(f"{k} {v:.4f}" for k, v in c.items()))
    open(OUT / "fixes_summary.txt", "w", encoding="utf-8").write("\n".join(log))
