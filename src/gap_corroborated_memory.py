"""
gap_corroborated_memory.py -- corroboration-gated adaptive temporal memory.

beta(t) adapts to the evidence behind a rise in fuzzy risk S(t):
  * S(t) <= c (below the Critical zone)                    -> beta_fast   (cheap early warning)
  * S(t) >  c and temperature corroborates (T >= T_corr)   -> beta_fast   (hot cell: follow at once)
  * S(t) >  c, gas-only, lasted >= m consecutive steps     -> beta_fast   (sustained: follow)
  * S(t) >  c, gas-only, shorter than m steps              -> beta_slow   (likely a sensor burst)
    R(t) = beta(t) * R(t-1) + (1 - beta(t)) * S(t)
Gas-only rises are never ignored: they escalate once they persist for m steps.

Same protocol as gap_gated_memory.py: tune on seeds 200-229 (admissible = no false Critical /
E-stop in mild or hard noise, then lowest Hazard FNR, then fewest false E-stops in the harder
test, then smallest m), evaluate on held-out seeds 100-129, plus already-failing cells.
Run from the repo root:  python src/gap_corroborated_memory.py  -> results/gaps/corroborated_memory.*
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


def corroborated_filter(S, T, obj, b_fast, b_slow, c, m, t_corr, reset=3.0):
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
            run += 1
            b = b_fast if (T[i] >= t_corr or run >= m) else b_slow
        r = b * r + (1 - b) * s; R[i] = r
    return R


VARIANTS = [("fixed", (0.0,)), ("fixed", (0.95,)), ("adaptive_paper", (0.85, 0.15))]
VARIANTS += [("corroborated", p) for p in itertools.product((0.0,), (0.95, 0.98, 0.99), (0.5,), (20, 30, 40), (40.0, 45.0))]


def apply(kind, p, S, T, obj):
    if kind == "fixed": return fixed_filter(S, p[0])
    if kind == "adaptive_paper": return adaptive_filter(S, obj, *p)
    return corroborated_filter(S, T, obj, *p)


def esc(dn, R):
    st = fsm_states(dn, R)
    return bool(np.isin(st, ["Critical", "Emergency_Stop"]).any()), bool((st == "Emergency_Stop").any())


def run_seed(seed):
    d = run_simulation(duration_s=400.0, battery_arrives_at=20.0, seed=seed); S = fuzzy_series(d)
    T = d.temperature_C.values; obj = d.object_present.values
    noise = {}
    for name, kw in NOISE.items():
        dn = run_noise_stress_scenario(seed=seed + 1000, **kw)
        noise[name] = (dn, fuzzy_series(dn), dn.temperature_C.values, dn.object_present.values)
    rows = []
    for kind, p in VARIANTS:
        fnr, ts = hazard_metrics(d, apply(kind, p, S, T, obj))
        row = dict(seed=seed, kind=kind, params=json.dumps(p), Hazard_FNR=fnr, T_stop_s=ts)
        for name, (dn, Sn, Tn, on) in noise.items():
            cr, es = esc(dn, apply(kind, p, Sn, Tn, on)); row[f"crit_{name}"] = cr; row[f"estop_{name}"] = es
        rows.append(row)
    return rows


def prewarmed_seed(args):
    seed, chosen = args; out = []
    for T0, sc in ((60, 3), (80, 6), (90, 10)):
        d = sim_prewarmed(seed, T0, sc); S = fuzzy_series(d); T = d.temperature_C.values; obj = d.object_present.values
        for label, (kind, prm) in chosen.items():
            st = fsm_states(d, apply(kind, tuple(json.loads(prm)), S, T, obj))
            t = d.time.values; idx = np.where((st == "Emergency_Stop") & (t >= 20.0))[0]
            out.append(dict(seed=seed, T0=T0, scale=sc, kind=label, T_stop_s=(t[idx[0]] - 20.0) if len(idx) else np.nan))
    return out


if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(6) as pool:
        tune = pd.DataFrame(sum(pool.map(run_seed, range(200, 230)), []))
        ev = pd.DataFrame(sum(pool.map(run_seed, range(100, 130)), []))
    tune.to_csv(OUT / "corroborated_memory_tuning.csv", index=False); ev.to_csv(OUT / "corroborated_memory_eval.csv", index=False)
    g = tune.groupby(["kind", "params"]).agg(fnr=("Hazard_FNR", "mean"), crit_mild=("crit_mild", "sum"), crit_hard=("crit_hard", "sum"),
                                             estop_harder=("estop_harder", "sum")).reset_index()
    g["admissible"] = (g.crit_mild == 0) & (g.crit_hard == 0)
    g["m"] = g.params.apply(lambda s: json.loads(s)[3] if len(json.loads(s)) > 3 else 0)
    lines = []
    def P(s=""): print(s); lines.append(s)
    P("TUNING seeds 200-229"); P(g.sort_values(["kind", "admissible", "fnr", "estop_harder", "m"], ascending=[True, False, True, True, True]).to_string(index=False))
    pick = g[(g.kind == "corroborated") & g.admissible].sort_values(["fnr", "estop_harder", "m"]).iloc[0].params
    chosen = {"corroborated": ("corroborated", pick), "fixed_0": ("fixed", "[0.0]"), "fixed_0.95": ("fixed", "[0.95]"),
              "paper_adaptive": ("adaptive_paper", "[0.85, 0.15]")}
    P(f"\nchosen corroborated setting (tuning data only): {pick}")
    P("\nEVALUATION held-out seeds 100-129")
    sel = {}
    for label, (kind, prm) in chosen.items():
        x = ev[(ev.kind == kind) & (ev.params == prm)].set_index("seed"); sel[label] = x
        P(f"{label:<15} {prm:<32} Hazard FNR {x.Hazard_FNR.mean():.4f} +/- {x.Hazard_FNR.std():.4f} | T_stop {x.T_stop_s.mean():.1f} | false Critical mild/hard/harder "
          f"{int(x.crit_mild.sum())}/{int(x.crit_hard.sum())}/{int(x.crit_harder.sum())} | false E-stop hard/harder {int(x.estop_hard.sum())}/{int(x.estop_harder.sum())} (of 30)")
    res = {"chosen": pick, "eval": {k: {"fnr": float(v.Hazard_FNR.mean()), "fnr_sd": float(v.Hazard_FNR.std()), "t_stop": float(v.T_stop_s.mean()),
                                        "crit_hard": int(v.crit_hard.sum()), "crit_harder": int(v.crit_harder.sum()),
                                        "estop_harder": int(v.estop_harder.sum())} for k, v in sel.items()}}
    a = sel["corroborated"].Hazard_FNR
    for other in ("fixed_0", "fixed_0.95", "paper_adaptive"):
        b = sel[other].Hazard_FNR; t, p = stats.ttest_rel(a, b) if (a != b).any() else (np.nan, np.nan)
        P(f"corroborated minus {other}: {a.mean() - b.mean():+.4f}, paired t(29) = {t:.2f}, p = {p:.3g}; lower in {int((a < b).sum())}, higher in {int((a > b).sum())}")
        res[f"minus_{other}"] = {"diff": float(a.mean() - b.mean()), "t": float(t), "p": float(p), "lower": int((a < b).sum()), "higher": int((a > b).sum())}
    with Pool(6) as pool:
        pw = pd.DataFrame(sum(pool.map(prewarmed_seed, [(s, chosen) for s in range(100, 130)]), []))
    pw.to_csv(OUT / "corroborated_memory_prewarmed.csv", index=False)
    tab = pw.groupby(["T0", "scale", "kind"]).T_stop_s.agg(["mean", "max"]).unstack().round(1)
    P("\nAlready-failing cells: seconds from arrival to Emergency Stop (mean / max over 30 seeds)"); P(tab.to_string())
    res["prewarmed_mean"] = {f"{T0}C_x{sc}": grp.groupby("kind").T_stop_s.mean().round(2).to_dict() for (T0, sc), grp in pw.groupby(["T0", "scale"])}
    json.dump(res, open(OUT / "corroborated_memory_summary.json", "w"), indent=1)
    open(OUT / "corroborated_memory_summary.txt", "w", encoding="utf-8").write("\n".join(lines))
