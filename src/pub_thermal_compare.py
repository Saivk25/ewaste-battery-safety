"""
pub_thermal_compare.py -- main thermal model vs the literature-calibrated one, side by side.

The main model's phase onsets (SEI, separator, runaway) sit 40-90 C below published values. The
literature-calibrated model (pub_fixes.cal_sim: SEI 90 C, separator 130 C, runaway 200 C, gas onsets
moved up with them) is run on the same seeds and IR-lag cases, so each result can be read on both.
If the main model's low onsets leave the controller less time, the main-model numbers are the
conservative ones.

Run: python src/pub_thermal_compare.py     Outputs: results/pub/thermal_compare.csv, thermal_compare_summary.txt
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pub_experiments import scen, Camera, OUT, metrics, b2_debounce
from pub_fixes import cal_sim
from pub_sensitivity import fuzzy, run_one, GATES

PAIRS = [("single", "cal_single", 0.0), ("lag20", "cal_lag20", 20.0), ("lag60", "cal_lag60", 60.0)]
SEEDS = list(range(100, 130))


def get(name, seed):
    for main, cal, lag in PAIRS:
        if name == cal: return cal_sim(seed, lag_s=lag)
    if name == "lag60":
        import sensor_simulation as ss
        d = ss.run_simulation(duration_s=400.0, battery_arrives_at=20.0, seed=seed); n = 600
        T = d.temperature_C.values.copy(); d["temperature_C"] = np.r_[np.full(n, T[0]), T[:-n]]
        return d, 20.0
    return scen(name, seed)


def work(task):
    name, seed, P = task
    from baseline_comparison import baseline_rule_based
    df, arrive = get(name, seed); S = fuzzy(df)
    rows = [dict(scenario=name, seed=seed, system=g, **run_one(df, arrive, S, P, g, "none", 1.0)) for g in GATES]
    rows.append(dict(scenario=name, seed=seed, system="B2_Rule", **metrics(df, baseline_rule_based(df).state.values, arrive)))
    rows.append(dict(scenario=name, seed=seed, system="B2_Debounce_3s", **metrics(df, b2_debounce(df, 3), arrive)))
    ph = df.phase.values; t = df.time.values
    on = lambda p: float(df.temperature_C.values[np.where(ph == p)[0][0]]) if (ph == p).any() else np.nan
    for r in rows: r.update(T_onset_SEI=on("Phase1_SEI"), T_onset_runaway=on("Phase3_Runaway"))
    return rows


def summarise(d):
    L = ["Main thermal model vs literature-calibrated model, held-out seeds 100-129.",
         "FNR = hazardous time with no alert | margin = mean seconds from stop to runaway (>0 = stopped first) | "
         "stopped = runs stopped before runaway / runs that reached runaway", ""]
    for main, cal, _ in PAIRS:
        for nm in (main, cal):
            x = d[d.scenario == nm]
            L.append(f"{nm:11s}" + (f" (IR reading at onset: SEI {x.T_onset_SEI.median():.0f} C, runaway "
                                    f"{x.T_onset_runaway.median():.0f} C)" if nm in ("single", "cal_single") else ""))
            for s in ["paper_gate", "recommended_gate", "B2_Rule", "B2_Debounce_3s"]:
                y = x[x.system == s]
                L.append(f"   {s:18s} FNR {y.Hazard_FNR.mean():.4f}  margin {y.stop_margin_s.mean():7.1f} s  "
                         f"stopped {int((y.stop_margin_s > 0).sum())}/{int(y.T_onset_runaway.notna().sum())}")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    from multiprocessing import Pool
    cam = Camera(); tasks = []
    for main, cal, _ in PAIRS:
        for nm in (main, cal):
            for seed in SEEDS:
                df, _ = get(nm, seed); tasks.append((nm, seed, cam.series(df)))
    with Pool(int(__import__("os").environ.get("NPROC", 14))) as pool:
        rows = [r for rs in pool.imap_unordered(work, tasks) for r in rs]
    d = pd.DataFrame(rows).sort_values(["scenario", "system", "seed"])
    d.to_csv(OUT / "thermal_compare.csv", index=False)
    txt = summarise(d); print(txt)
    (OUT / "thermal_compare_summary.txt").write_text(txt + "\n", encoding="utf-8")
