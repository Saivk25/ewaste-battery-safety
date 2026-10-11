"""
pub_fixes2.py -- follow-up to pub_fixes.py.

(a) Joint retune of the gate on TUNING seeds 200-229 only: persistence m in {20, 40, 80} steps and the
    rate-of-rise corroboration (off, or W/r in {(10, 0.2), (10, 0.3), (20, 0.2)}). Selection: among
    settings with no false escalation in mild/hard and whose lag20 miss rate and pre90 stop time are
    within +0.01 and +1 s of the current gate, take the one with the fewest false escalations in
    harder + warm_bursts. Evaluated once on held-out seeds 100-129, with camera fusion and sensor-only.
(b) Logistic baseline re-run with gain 1.0: with the gain 0.6 chosen by pub_fixes.py its risk can never
    reach the 0.75 Emergency Stop threshold, so it warned but never stopped (T_stop undefined).

Run from the repo root:  python src/pub_fixes2.py -> results/pub/fixes2_*.csv, fixes2_summary.txt
"""
import sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from pub_experiments import Camera, fsm_run, b2_debounce, metrics, DT, ALPHA, TUNE, EVAL, OUT
from pub_fixes import scen2, gated_rate, gated_m, features, cfnc_parts

FX = json.load(open(OUT / "fixes_tuned.json")); TUNED = json.load(open(OUT / "tuned.json"))["tuned"]
GRID = [(m, wr) for m in (20, 40, 80) for wr in (None, (10.0, 0.2), (10.0, 0.3), (20.0, 0.2))]


def gate(S, T, obj, m, wr):
    import adaptive_risk as ar
    if wr is None: return gated_m(S, T, obj, m)
    saved = ar.M_PERSIST; ar.M_PERSIST = m
    try: return gated_rate(S, T, obj, *wr)
    finally: ar.M_PERSIST = saved


def tag(m, wr): return f"G_m{m}" + ("" if wr is None else f"_W{wr[0]:g}_r{wr[1]:g}")


def work(task):
    name, seed, P, grid, logistic = task
    from gap_temporal_sweeps import fuzzy_series
    df, arrive = scen2(name, seed)
    S = fuzzy_series(df); T = df.temperature_C.values; obj = df.object_present.values
    out = {}
    for m, wr in grid:
        R = gate(S, T, obj, m, wr)
        out[tag(m, wr)] = fsm_run(df, np.clip(R * (1 + ALPHA * P), 0, 1))
        out[tag(m, wr) + "_sensoronly"] = fsm_run(df, np.clip(R, 0, 1))
    if logistic:
        p = 1 / (1 + np.exp(-(features(df) @ np.array(FX["lr"]["coef"]) + FX["lr"]["icpt"])))
        out["B4_Logistic_g1"] = fsm_run(df, np.clip(p, 0, 1))
        out[f"B2_Debounce_{TUNED['b2_k']:g}s"] = b2_debounce(df, TUNED["b2_k"])
    return [dict(scenario=name, seed=seed, system=k, **metrics(df, st, arrive), **cfnc_parts(df, st)) for k, st in out.items()]


if __name__ == "__main__":
    from multiprocessing import Pool
    cam = Camera(); log = []
    def L(s=""): print(s, flush=True); log.append(s)
    def tasks(names, seeds, grid, logistic):
        out = []
        for n in names:
            for s in seeds:
                df, _ = scen2(n, s); out.append((n, s, cam.series(df), grid, logistic))
        return out

    if (OUT / "fixes2_tuning.csv").exists():           # tuning already done (seeds 200-229 only)
        tu = pd.read_csv(OUT / "fixes2_tuning.csv")
    else:
        with Pool(2) as pool:
            tu = pd.DataFrame(sum(pool.map(work, tasks(["single", "lag20", "pre90", "mild", "hard", "harder", "warm_bursts"], TUNE, GRID, False)), []))
        tu.to_csv(OUT / "fixes2_tuning.csv", index=False)
    tu = tu[~tu.system.str.endswith("_sensoronly")]
    g = tu.groupby(["system", "scenario"]).agg(fnr=("Hazard_FNR", "mean"), stop=("T_stop_s", "mean"), fa=("false_escalation", "sum")).unstack()
    ref = tag(20, None)
    ok = g[(g[("fa", "mild")] == 0) & (g[("fa", "hard")] == 0) & (g[("fnr", "lag20")] <= g.loc[ref, ("fnr", "lag20")] + 0.01)
           & (g[("stop", "pre90")] <= g.loc[ref, ("stop", "pre90")] + 1.0)]
    L("Tuning (seeds 200-229): system | single FNR | lag20 FNR | pre90 stop | false mild/hard/harder/warm_bursts")
    for s, r in g.iterrows():
        L(f"  {s:<22} {r[('fnr', 'single')]:.4f} {r[('fnr', 'lag20')]:.4f} {r[('stop', 'pre90')]:5.1f}  "
          f"{int(r[('fa', 'mild')])}/{int(r[('fa', 'hard')])}/{int(r[('fa', 'harder')])}/{int(r[('fa', 'warm_bursts')])}  {'admissible' if s in ok.index else ''}")
    best = (ok[("fa", "harder")] + ok[("fa", "warm_bursts")]).idxmin()
    m = int(best.split("_m")[1].split("_")[0]); wr = None if "_W" not in best else (float(best.split("_W")[1].split("_r")[0]), float(best.split("_r")[1]))
    L(f"-> chosen {best}")
    # also carried to held-out: the rate-gated setting with the fewest false escalations, ignoring the stop-time
    # tolerance (its hot-cell stop cost is reported, not hidden)
    ok2 = g[(g[("fa", "mild")] == 0) & (g[("fa", "hard")] == 0) & (g[("fnr", "lag20")] <= g.loc[ref, ("fnr", "lag20")] + 0.01)
            & g.index.str.contains("_W")]
    alt = (ok2[("fa", "harder")] + ok2[("fa", "warm_bursts")]).sort_values(kind="stable").index[0]
    am = int(alt.split("_m")[1].split("_")[0]); awr = (float(alt.split("_W")[1].split("_r")[0]), float(alt.split("_r")[1]))
    L(f"-> also evaluated {alt} (pre90 stop {g.loc[alt, ('stop', 'pre90')]:.1f} s vs {g.loc[ref, ('stop', 'pre90')]:.1f} s for the current gate)")

    names = ["single", "multi", "lag5", "lag20", "lag60", "pre60", "pre80", "pre90", "mild", "hard", "harder", "hot60", "hot90", "warm_bursts", "warm75_bursts", "cal_single"]
    grid = list(dict.fromkeys([(20, None), (m, wr), (am, awr)]))
    with Pool(2) as pool:
        ev = pd.DataFrame(sum(pool.map(work, tasks(names, EVAL, grid, True)), []))
    ev.to_csv(OUT / "fixes2_eval.csv", index=False)
    json.dump({"m": m, "W": wr[0] if wr else None, "r": wr[1] if wr else None, "chosen": best, "also_evaluated": alt},
              open(OUT / "fixes2_tuned.json", "w"), indent=1)
    L("\nHELD-OUT (seeds 100-129): FNR / mean stop (s) for hazard scenarios; false Critical / E-stop runs for battery-free")
    for s in sorted(ev.system.unique()):
        parts = []
        for n in names:
            y = ev[(ev.scenario == n) & (ev.system == s)]
            parts.append(f"{n} {y.Hazard_FNR.mean():.4f}/{y.T_stop_s.mean():.1f}" if y.Hazard_FNR.notna().any()
                         else f"{n} {int(y.false_escalation.sum())}/{int(y.false_estop.sum())}")
        L(f"  {s}: " + " | ".join(parts))
    open(OUT / "fixes2_summary.txt", "w", encoding="utf-8").write("\n".join(log))
