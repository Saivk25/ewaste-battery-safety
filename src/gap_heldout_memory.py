"""
gap_heldout_memory.py -- fair test of adaptive vs fixed temporal memory.

Tuning and evaluation use DIFFERENT seeds, so neither filter is tuned on the data it is
judged on:
  * tuning seeds 200-229: pick the best fixed beta and the best (beta_max, lambda) by
    Hazard FNR, subject to the battery-free spike test keeping peak risk below the
    Critical threshold (0.55) -- i.e. no false escalation.
  * evaluation seeds 100-129 (the paper's seeds): compare the two tuned filters, paired.
All filters use the same fuzzy series, the same FSM and no CNN fusion.

Run from the repo root:  python src/gap_heldout_memory.py   -> results/gaps/heldout_memory.*
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from scipy import stats
from sensor_simulation import run_simulation, run_noise_stress_scenario
from gap_temporal_sweeps import fuzzy_series, fixed_filter, adaptive_filter, fsm_frame, metrics

OUT = Path("results/gaps")
FIXED = [round(x, 2) for x in np.arange(0.30, 0.91, 0.05)]
ADAPT = [(bm, lam) for bm in (0.75, 0.85, 0.90, 0.95) for lam in (0.15, 0.5, 1, 2, 3, 5, 8)]


def run_seed(seed):
    d = run_simulation(duration_s=400.0, battery_arrives_at=20.0, seed=seed); S = fuzzy_series(d); obj = d.object_present.values
    n = run_noise_stress_scenario(seed=seed + 1000, n_spikes=8); Sn = fuzzy_series(n); objn = n.object_present.values
    rows = []
    for b in FIXED:
        rows.append(dict(seed=seed, kind="fixed", beta=b, bmax=np.nan, lam=np.nan,
                         **metrics(fsm_frame(d, fixed_filter(S, b)), 20.0), spike_peak=float(fixed_filter(Sn, b).max())))
    for bm, lam in ADAPT:
        rows.append(dict(seed=seed, kind="adaptive", beta=np.nan, bmax=bm, lam=lam,
                         **metrics(fsm_frame(d, adaptive_filter(S, obj, bm, lam)), 20.0), spike_peak=float(adaptive_filter(Sn, objn, bm, lam).max())))
    return rows


def best(df, kind):
    keys = ["beta"] if kind == "fixed" else ["bmax", "lam"]
    g = df[df.kind == kind].groupby(keys).agg(fnr=("Hazard_FNR", "mean"), spike_max=("spike_peak", "max")).reset_index()
    ok = g[g.spike_max < 0.55]
    return ok.sort_values("fnr").iloc[0], g


if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(5) as p:
        tune = pd.DataFrame(sum(p.map(run_seed, range(200, 230)), []))
        evl = pd.DataFrame(sum(p.map(run_seed, range(100, 130)), []))
    tune.to_csv(OUT / "heldout_memory_tuning.csv", index=False); evl.to_csv(OUT / "heldout_memory_eval.csv", index=False)
    bf, gf = best(tune, "fixed"); ba, ga = best(tune, "adaptive")
    print("TUNING (seeds 200-229), constraint: spike-test peak risk < 0.55 on every tuning seed")
    print(gf.to_string(index=False)); print(ga.sort_values("fnr").head(10).to_string(index=False))
    print(f"chosen fixed beta = {bf.beta}  (tuning FNR {bf.fnr:.4f}, spike max {bf.spike_max:.3f})")
    print(f"chosen adaptive = beta_max {ba.bmax}, lambda {ba.lam}  (tuning FNR {ba.fnr:.4f}, spike max {ba.spike_max:.3f})")
    f = evl[(evl.kind == "fixed") & (evl.beta == bf.beta)].set_index("seed")
    a = evl[(evl.kind == "adaptive") & (evl.bmax == ba.bmax) & (evl.lam == ba.lam)].set_index("seed")
    c = evl[(evl.kind == "adaptive") & (evl.bmax == 0.85) & (evl.lam == 0.15)].set_index("seed")
    f85 = evl[(evl.kind == "fixed") & (evl.beta == 0.85)].set_index("seed")
    res = {"chosen_fixed_beta": float(bf.beta), "chosen_adaptive": [float(ba.bmax), float(ba.lam)]}
    print("\nEVALUATION (held-out seeds 100-129), Hazard FNR mean +/- sd; spike peak max")
    for name, x in (("tuned fixed", f), ("tuned adaptive", a), ("paper adaptive (0.85, 0.15)", c), ("fixed 0.85", f85)):
        print(f"{name:<28} {x.Hazard_FNR.mean():.4f} +/- {x.Hazard_FNR.std():.4f} | T_stop {x.T_stop_s.mean():.1f} | spike peak max {x.spike_peak.max():.3f}")
        res[name] = {"fnr_mean": float(x.Hazard_FNR.mean()), "fnr_sd": float(x.Hazard_FNR.std()), "spike_peak_max": float(x.spike_peak.max())}
    t, pv = stats.ttest_rel(a.Hazard_FNR, f.Hazard_FNR)
    w = int((a.Hazard_FNR < f.Hazard_FNR).sum()); l = int((a.Hazard_FNR > f.Hazard_FNR).sum())
    print(f"\ntuned adaptive minus tuned fixed: {a.Hazard_FNR.mean() - f.Hazard_FNR.mean():+.4f}, paired t(29) = {t:.2f}, p = {pv:.3g}; adaptive lower in {w}, higher in {l} seeds")
    res["adaptive_minus_fixed"] = {"diff": float(a.Hazard_FNR.mean() - f.Hazard_FNR.mean()), "t": float(t), "p": float(pv), "adaptive_better_seeds": w, "fixed_better_seeds": l}
    json.dump(res, open(OUT / "heldout_memory_summary.json", "w"), indent=1)
