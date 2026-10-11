"""
run_multi_item_occlusion_sweep.py
=================================
30-seed sweeps for two robustness claims the paper makes but only tested on
the single-item hazardous scenario:

  (A) Multi-item conveyor: a benign metal object (Item 1) passes before the
      battery (Item 2). Checks that (i) the benign object causes no false
      alarm in any system, and (ii) the battery is still detected at the
      same speed as in the single-item sweep. Item 3 is not analysed: the
      proposed FSM latches in Emergency_Stop by design, so a third item after
      the battery cannot be judged without a manual reset.

  (B) Camera occlusion: the CNN cannot see the battery, so P_battery = 0.
      Because R_fused = R_t * (1 + alpha * P_battery), occlusion leaves
      R_fused = R_t and the temporal state is unaffected (same shortcut as the
      alpha = 0 ablation). B1 (vision-only) is blind by construction, so its
      occluded row is trivially FNR = 1.0 and is reported only for completeness.

Run from the git_repo root:  python src/run_multi_item_occlusion_sweep.py
Outputs: results/multi_item_sweep_per_seed.csv, results/occlusion_sweep_per_seed.csv,
         results/multi_item_occlusion_summary.csv
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU only: 8 workers would otherwise all grab the GPU
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy import stats

from sensor_simulation import run_simulation, run_multi_item_scenario
from adaptive_risk import run_adaptive_pipeline
from supervisory_control import run_supervisory
from baseline_comparison import (
    baseline_vision_only, baseline_rule_based, baseline_fixed_fuzzy, compute_metrics,
)

N_SEEDS, SEED_BASE = 30, 100
ALERT = ["Warning", "Critical", "Emergency_Stop"]
METRICS = ["Hazard_FNR", "T_warn_s", "T_stop_s", "CFNC", "FPR"]
RESULTS_DIR = Path(__file__).parent.parent / "results"


def num(v):
    return np.nan if isinstance(v, str) else float(v)


def proposed_frame(df_risk, occluded=False):
    d = df_risk.copy()
    if occluded:
        d["R_fused"] = d["R_temporal"]
    full, _ = run_supervisory(d)
    out = full[["time", "fsm_state", "R_fused", "battery_present", "phase"]].copy()
    out.columns = ["time", "state", "risk", "battery_present", "phase"]
    return out


def all_systems(df, df_risk, seed):
    return {
        "B1_Vision_Only": baseline_vision_only(df, seed=seed),
        "B2_Rule_Based": baseline_rule_based(df),
        "B3_Fixed_Fuzzy": baseline_fixed_fuzzy(df),
        "Proposed": proposed_frame(df_risk),
    }


def run_seed(seed):
    multi_rows, occl_rows = [], []
    # (A) multi-item: benign item at t=15-45, battery at t=60
    dm = run_multi_item_scenario(seed=seed, item3_gap_s=0.0)
    dm = dm[dm["time"] < 460.0].reset_index(drop=True)   # drop Item 3 / aftermath
    rm = run_adaptive_pipeline(dm, use_real_cnn=True)
    for name, d in all_systems(dm, rm, seed).items():
        m = compute_metrics(d, 60.0, name)
        item1 = d[(d["time"] >= 15.0) & (d["time"] < 60.0)]
        multi_rows.append({
            "seed": seed, "system": name,
            **{k: num(m[k]) for k in METRICS},
            "item1_alert_rate": float(item1["state"].isin(ALERT).mean()),
            "item1_estop_rate": float((item1["state"] == "Emergency_Stop").mean()),
        })

    # (B) occlusion on the canonical single-item scenario
    ds = run_simulation(duration_s=400.0, battery_arrives_at=20.0, seed=seed)
    rs = run_adaptive_pipeline(ds, use_real_cnn=True)
    occ = {
        "Proposed (camera ok)": proposed_frame(rs),
        "Proposed (occluded)": proposed_frame(rs, occluded=True),
        "B2_Rule_Based": baseline_rule_based(ds),
        "B3_Fixed_Fuzzy": baseline_fixed_fuzzy(ds),
    }
    blind = baseline_rule_based(ds)[["time", "battery_present", "phase"]].copy()
    blind["state"], blind["risk"] = "Normal", 0.0           # B1 sees nothing
    occ["B1_Vision_Only (occluded)"] = blind
    for name, d in occ.items():
        m = compute_metrics(d, 20.0, name)
        occl_rows.append({"seed": seed, "system": name, **{k: num(m[k]) for k in METRICS}})
    return multi_rows, occl_rows


def main():
    from multiprocessing import Pool
    seeds = [SEED_BASE + i for i in range(N_SEEDS)]
    multi_rows, occl_rows = [], []
    with Pool(2) as pool:   # 2 workers: each loads the CNN checkpoints (~1.5 GB RAM)
        for k, (mr, orr) in enumerate(pool.imap_unordered(run_seed, seeds), 1):
            multi_rows += mr
            occl_rows += orr
            print(f"seed batch {k}/{N_SEEDS} done", flush=True)
    multi_rows.sort(key=lambda r: r["seed"]); occl_rows.sort(key=lambda r: r["seed"])

    RESULTS_DIR.mkdir(exist_ok=True)
    dfm, dfo = pd.DataFrame(multi_rows), pd.DataFrame(occl_rows)
    dfm.to_csv(RESULTS_DIR / "multi_item_sweep_per_seed.csv", index=False)
    dfo.to_csv(RESULTS_DIR / "occlusion_sweep_per_seed.csv", index=False)

    summary = []
    for tag, df in [("multi_item", dfm), ("occlusion", dfo)]:
        cols = METRICS + (["item1_alert_rate", "item1_estop_rate"] if tag == "multi_item" else [])
        print(f"\n=== {tag}: mean +/- std over {N_SEEDS} seeds ===")
        for name, sub in df.groupby("system", sort=False):
            parts = []
            for c in cols:
                v = sub[c].dropna()
                mu = v.mean() if len(v) else float("nan")
                sd = v.std(ddof=1) if len(v) > 1 else 0.0
                parts.append(f"{c}={mu:.4f}+/-{sd:.4f} (n={len(v)})")
                summary.append({"scenario": tag, "system": name, "metric": c,
                                "mean": mu, "std": sd, "n": len(v)})
            print(f"{name:<28}", "  ".join(parts))

    def paired(df, a, b, col):
        x = df[df.system == a].set_index("seed")[col]
        y = df[df.system == b].set_index("seed")[col]
        j = pd.concat([x, y], axis=1).dropna()
        t, p = stats.ttest_rel(j.iloc[:, 0], j.iloc[:, 1])
        return (j.iloc[:, 0] - j.iloc[:, 1]).mean(), t, p, int((j.iloc[:, 0] < j.iloc[:, 1]).sum()), len(j)

    print("\n=== paired tests (first minus second; negative = first better) ===")
    for tag, df, a, b in [
        ("occlusion", dfo, "Proposed (occluded)", "B3_Fixed_Fuzzy"),
        ("occlusion", dfo, "Proposed (occluded)", "Proposed (camera ok)"),
        ("multi_item", dfm, "Proposed", "B3_Fixed_Fuzzy"),
        ("multi_item", dfm, "Proposed", "B1_Vision_Only"),
    ]:
        for col in ["Hazard_FNR", "CFNC"]:
            d, t, p, w, n = paired(df, a, b, col)
            print(f"{tag:<10} {col:<10} {a} vs {b}: diff={d:+.4f} t({n-1})={t:.2f} p={p:.4g} lower in {w}/{n}")
            summary.append({"scenario": tag, "system": f"{a} vs {b}", "metric": f"paired_{col}",
                            "mean": d, "std": t, "n": p})
    pd.DataFrame(summary).to_csv(RESULTS_DIR / "multi_item_occlusion_summary.csv", index=False)
    print("\nSaved results/*multi_item*/occlusion* CSVs")


if __name__ == "__main__":
    main()
