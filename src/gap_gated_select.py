"""
gap_gated_select.py -- final, pre-registered selection of the level-dependent adaptive memory.

Selection uses ONLY the tuning seeds (200-229):
  1. admissible: no false Critical/E-stop in mild or hard battery-free noise;
  2. lowest mean Hazard FNR;
  3. tie-break: fastest mean stop for already-failing cells (60 C x3, 80 C x6, 90 C x10),
     and the stop must be no slower than the paper's current setting (0.85, 0.15) on each.
Then the chosen setting is evaluated once on the held-out seeds 100-129.
Reads results/gaps/gated_memory_tuning.csv and _eval.csv (from gap_gated_memory.py).
Run from the repo root:  python src/gap_gated_select.py  -> results/gaps/gated_select.json
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from scipy import stats
from gap_gated_memory import prewarmed_seed

OUT = Path("results/gaps")

if __name__ == "__main__":
    from multiprocessing import Pool
    tune = pd.read_csv(OUT / "gated_memory_tuning.csv"); ev = pd.read_csv(OUT / "gated_memory_eval.csv")
    g = tune.groupby(["kind", "params"]).agg(fnr=("Hazard_FNR", "mean"), crit_mild=("crit_mild", "sum"), crit_hard=("crit_hard", "sum"),
                                             estop_harder=("estop_harder", "sum")).reset_index()
    lv = g[(g.kind == "level") & (g.crit_mild == 0) & (g.crit_hard == 0)]
    tied = lv[np.isclose(lv.fnr, lv.fnr.min())]
    print(f"admissible level settings: {len(lv)}; tied at lowest tuning FNR {lv.fnr.min():.4f}: {len(tied)}")
    cand = {f"level_{i}": p for i, p in enumerate(tied.params)}
    cand["paper_adaptive"] = json.dumps([0.85, 0.15])
    with Pool(6) as pool:
        pw = pd.DataFrame(sum(pool.map(prewarmed_seed, [(s, cand) for s in range(200, 230)]), []))
    stop = pw.groupby(["kind", "T0"]).T_stop_s.mean().unstack()
    paper = stop.loc["paper_adaptive"]
    ok = [k for k in cand if k != "paper_adaptive" and (stop.loc[k] <= paper + 1e-9).all()]
    pool_k = ok if ok else [k for k in cand if k != "paper_adaptive"]
    best = min(pool_k, key=lambda k: stop.loc[k].mean())
    chosen = cand[best]
    print("tuning-seed stop times (s):"); print(stop.round(2).to_string())
    print(f"settings no slower than the paper on every severity: {[cand[k] for k in ok]}")
    print(f"CHOSEN: {chosen}  (no-slower constraint met: {bool(ok)})")
    # held-out evaluation of the chosen setting
    x = ev[(ev.kind == "level") & (ev.params == chosen)].set_index("seed")
    ref = {"paper_adaptive": ev[(ev.kind == "adaptive_paper") & (ev.params == json.dumps([0.85, 0.15]))].set_index("seed"),
           "fixed_0.95": ev[(ev.kind == "fixed") & (ev.params == json.dumps([0.95]))].set_index("seed"),
           "fixed_0.0": ev[(ev.kind == "fixed") & (ev.params == json.dumps([0.0]))].set_index("seed")}
    res = {"chosen": json.loads(chosen), "heldout": {}}
    def summ(d): return {"fnr": round(float(d.Hazard_FNR.mean()), 4), "fnr_sd": round(float(d.Hazard_FNR.std()), 4), "T_stop": round(float(d.T_stop_s.mean()), 1),
                         "false_critical_hard": int(d.crit_hard.sum()), "false_critical_harder": int(d.crit_harder.sum()), "false_estop_harder": int(d.estop_harder.sum())}
    res["heldout"]["level_adaptive"] = summ(x)
    for k, d in ref.items():
        res["heldout"][k] = summ(d)
        t, p = stats.ttest_rel(x.Hazard_FNR, d.Hazard_FNR)
        res["heldout"][f"vs_{k}"] = {"diff": round(float(x.Hazard_FNR.mean() - d.Hazard_FNR.mean()), 4), "t": round(float(t), 2) if not np.isnan(t) else None,
                                     "p": float(p) if not np.isnan(p) else None, "lower": int((x.Hazard_FNR < d.Hazard_FNR).sum()), "higher": int((x.Hazard_FNR > d.Hazard_FNR).sum())}
    with Pool(6) as pool:
        pwe = pd.DataFrame(sum(pool.map(prewarmed_seed, [(s, {"level_adaptive": chosen, "paper_adaptive": json.dumps([0.85, 0.15]),
                                                               "fixed_0.95": json.dumps([0.95]), "fixed_0.0": json.dumps([0.0])}) for s in range(100, 130)]), []))
    st = pwe.groupby(["kind", "T0"]).T_stop_s.mean().unstack().round(2)
    res["heldout_prewarmed_stop_s"] = {k: {str(c): float(v) for c, v in row.items()} for k, row in st.iterrows()}
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT / "gated_select.json", "w"), indent=1)
