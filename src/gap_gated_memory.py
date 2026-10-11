"""
gap_gated_memory.py -- persistence-gated adaptive memory vs fixed memory.

Rule (adaptive temporal memory, beta(t) depends on how long a rise has lasted):
    base(t)  = b_slow * base(t-1) + (1 - b_slow) * S(t)        slow reference, always updated
    run(t)   = run(t-1) + 1 if S(t) > base(t-1) + delta else 0  how long S has stayed above it
    beta(t)  = b_fast if run(t) >= m else b_slow
    R(t)     = beta(t) * R(t-1) + (1 - beta(t)) * S(t)
A short burst never reaches m consecutive steps, so R moves slowly (noise is smoothed);
a sustained rise opens the gate after m steps and R follows S quickly.

Protocol: tune on seeds 200-229, evaluate on held-out seeds 100-129 (the paper's seeds).
ADMISSIBLE = no Critical or Emergency Stop in any mild or hard battery-free noise run on the
tuning seeds; among admissible settings, the lowest tuning Hazard FNR wins. The same rule
selects the best fixed beta. Also reported: the harder noise test and already-failing cells.

Run from the repo root:  python src/gap_gated_memory.py   -> results/gaps/gated_memory.*
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, itertools
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from scipy import stats
from sensor_simulation import run_simulation, run_noise_stress_scenario
from gap_temporal_sweeps import fuzzy_series, fixed_filter, adaptive_filter
from gap_memory_redesign import NOISE, fsm_states, hazard_metrics
from gap_prewarmed import sim as sim_prewarmed

OUT = Path("results/gaps"); DT = 0.1


def gated_filter(S, obj, b_slow, b_fast, delta, m, reset=3.0):
    R = np.zeros(len(S)); r = 0.0; base = 0.0; run = 0; noobj = 0.0
    for i, s in enumerate(S):
        if not obj[i]:
            noobj += DT
            if noobj >= reset: r = 0.0; base = 0.0; run = 0
        else:
            noobj = 0.0
        run = run + 1 if s > base + delta else 0
        b = b_fast if run >= m else b_slow
        r = b * r + (1 - b) * s
        base = b_slow * base + (1 - b_slow) * s
        R[i] = r
    return R


def level_gated_filter(S, obj, b_fast, b_slow, c, m, reset=3.0):
    """Fast memory while S is below the Critical zone (cheap early warning); slow memory for an
    excursion above c until it has lasted m consecutive steps, then fast again (sustained hazard)."""
    R = np.zeros(len(S)); r = 0.0; run = 0; noobj = 0.0
    for i, s in enumerate(S):
        if not obj[i]:
            noobj += DT
            if noobj >= reset: r = 0.0; run = 0
        else:
            noobj = 0.0
        if s <= c:
            run = 0; b = b_fast
        else:
            run += 1; b = b_fast if run >= m else b_slow
        r = b * r + (1 - b) * s; R[i] = r
    return R


def variants():
    v = [("fixed", (b,)) for b in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 0.97)]
    v += [("adaptive_paper", (0.85, 0.15)), ("adaptive_paper", (0.95, 0.15))]
    v += [("gated", p) for p in itertools.product((0.90, 0.95, 0.98), (0.0, 0.3), (0.05, 0.10, 0.15), (10, 20, 30))]
    v += [("level", p) for p in itertools.product((0.0, 0.2, 0.4), (0.95, 0.98, 0.99), (0.45, 0.50, 0.55), (10, 20, 30, 40))]
    return v


def apply(kind, p, S, obj):
    if kind == "fixed": return fixed_filter(S, p[0])
    if kind == "adaptive_paper": return adaptive_filter(S, obj, *p)
    if kind == "level": return level_gated_filter(S, obj, *p)
    return gated_filter(S, obj, *p)


def esc(dn, R):
    st = fsm_states(dn, R)
    return bool(np.isin(st, ["Critical", "Emergency_Stop"]).any()), bool((st == "Emergency_Stop").any())


def run_seed(seed):
    d = run_simulation(duration_s=400.0, battery_arrives_at=20.0, seed=seed); S = fuzzy_series(d); obj = d.object_present.values
    noise = {}
    for name, kw in NOISE.items():
        dn = run_noise_stress_scenario(seed=seed + 1000, **kw); noise[name] = (dn, fuzzy_series(dn), dn.object_present.values)
    rows = []
    for kind, p in variants():
        fnr, ts = hazard_metrics(d, apply(kind, p, S, obj))
        row = dict(seed=seed, kind=kind, params=json.dumps(p), Hazard_FNR=fnr, T_stop_s=ts)
        for name, (dn, Sn, on) in noise.items():
            c, e = esc(dn, apply(kind, p, Sn, on)); row[f"crit_{name}"] = c; row[f"estop_{name}"] = e
        rows.append(row)
    return rows


def prewarmed_seed(args):
    seed, chosen = args
    out = []
    for T0, sc in ((60, 3), (80, 6), (90, 10)):
        d = sim_prewarmed(seed, T0, sc); S = fuzzy_series(d); obj = d.object_present.values
        for label, prm in chosen.items():
            kind = "gated" if label.startswith("gated") else "level" if label.startswith("level") else ("adaptive_paper" if label == "paper_adaptive" else "fixed")
            st = fsm_states(d, apply(kind, tuple(json.loads(prm)), S, obj))
            t = d.time.values; idx = np.where((st == "Emergency_Stop") & (t >= 20.0))[0]
            out.append(dict(seed=seed, T0=T0, scale=sc, kind=label, T_stop_s=(t[idx[0]] - 20.0) if len(idx) else np.nan))
    return out


if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(6) as pool:
        tune = pd.DataFrame(sum(pool.map(run_seed, range(200, 230)), []))
        ev = pd.DataFrame(sum(pool.map(run_seed, range(100, 130)), []))
    tune.to_csv(OUT / "gated_memory_tuning.csv", index=False); ev.to_csv(OUT / "gated_memory_eval.csv", index=False)
    g = tune.groupby(["kind", "params"]).agg(fnr=("Hazard_FNR", "mean"), crit_mild=("crit_mild", "sum"), crit_hard=("crit_hard", "sum"),
                                             crit_harder=("crit_harder", "sum"), estop_harder=("estop_harder", "sum")).reset_index()
    g["admissible"] = (g.crit_mild == 0) & (g.crit_hard == 0)
    lines = []
    def P(s=""): print(s); lines.append(s)
    P("TUNING seeds 200-229 (admissible = no false Critical/E-stop in mild or hard noise)")
    for kind in ("fixed", "adaptive_paper", "gated", "level"):
        gg = g[g.kind == kind].sort_values(["admissible", "fnr"], ascending=[False, True])
        P(f"\n{kind}: {int(gg.admissible.sum())}/{len(gg)} admissible"); P(gg.head(10).to_string(index=False))
    chosen = {}
    for kind in ("fixed", "gated", "level"):
        ok = g[(g.kind == kind) & g.admissible].sort_values("fnr")
        chosen[f"{kind}_admissible"] = ok.iloc[0].params
    best_fixed_any = g[g.kind == "fixed"].sort_values("fnr").iloc[0].params
    chosen["fixed_best_unconstrained"] = best_fixed_any
    chosen["paper_adaptive"] = json.dumps((0.85, 0.15))
    P(f"\nchosen: {chosen}")
    P("\nEVALUATION held-out seeds 100-129")
    sel = {}
    for label, prm in chosen.items():
        kind = "gated" if label.startswith("gated") else "level" if label.startswith("level") else ("adaptive_paper" if label == "paper_adaptive" else "fixed")
        x = ev[(ev.kind == kind) & (ev.params == prm)].set_index("seed"); sel[label] = x
        P(f"{label:<26} {prm:<24} Hazard FNR {x.Hazard_FNR.mean():.4f} +/- {x.Hazard_FNR.std():.4f} | T_stop {x.T_stop_s.mean():.1f} | "
          f"false Critical mild/hard/harder {int(x.crit_mild.sum())}/{int(x.crit_hard.sum())}/{int(x.crit_harder.sum())} | false E-stop harder {int(x.estop_harder.sum())} (of 30)")
    res = {"chosen": chosen, "eval": {k: {"fnr": float(v.Hazard_FNR.mean()), "fnr_sd": float(v.Hazard_FNR.std()), "crit_hard": int(v.crit_hard.sum()),
                                          "crit_harder": int(v.crit_harder.sum()), "estop_harder": int(v.estop_harder.sum())} for k, v in sel.items()}}
    a = sel["level_admissible"].Hazard_FNR
    for other in ("fixed_admissible", "fixed_best_unconstrained", "paper_adaptive", "gated_admissible"):
        b = sel[other].Hazard_FNR; t, p = stats.ttest_rel(a, b)
        P(f"level minus {other}: {a.mean() - b.mean():+.4f}, paired t(29) = {t:.2f}, p = {p:.3g}; level lower in {int((a < b).sum())}, higher in {int((a > b).sum())}")
        res[f"level_minus_{other}"] = {"diff": float(a.mean() - b.mean()), "t": float(t), "p": float(p), "lower": int((a < b).sum()), "higher": int((a > b).sum())}
    with Pool(6) as pool:
        pw = pd.DataFrame(sum(pool.map(prewarmed_seed, [(s, chosen) for s in range(100, 130)]), []))
    pw.to_csv(OUT / "gated_memory_prewarmed.csv", index=False)
    P("\nAlready-failing cells: mean seconds from arrival to Emergency Stop")
    P(pw.groupby(["T0", "scale", "kind"]).T_stop_s.mean().unstack().round(1).to_string())
    res["prewarmed"] = {f"{k[0]}C_x{k[1]}": v for k, v in pw.groupby(["T0", "scale"]).apply(lambda q: q.groupby("kind").T_stop_s.mean().round(2).to_dict()).items()}
    json.dump(res, open(OUT / "gated_memory_summary.json", "w"), indent=1, default=str)
    open(OUT / "gated_memory_summary.txt", "w", encoding="utf-8").write("\n".join(lines))
