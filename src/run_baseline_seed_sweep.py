"""
run_baseline_seed_sweep.py
===========================
Runs ALL FOUR systems (B1/B2/B3/Proposed) across 30 different sensor-noise
realizations of the canonical hazardous scenario, to report mean +/- std for
each headline metric in the paper instead of a single arbitrary draw.

Why this differs from run_b1_seed_sweep.py (the earlier, narrower attempt):
that script fixed the sensor CSV and only varied B1's own internal
threshold-noise RNG. B2 and B3 have no RNG of their own and are pure
functions of the sensor data, so on a FIXED sensor draw they are IDENTICAL
every "seed" -- that sweep could never have shown B2/B3/Proposed variability
at all, only B1's. This version instead varies the seed passed to
sensor_simulation.run_simulation() itself (thermal/gas/metal sensor noise),
regenerating a fresh sensor draw each iteration, so all four systems are
actually exercised under different noise realizations of the same scenario.

The Proposed system's CNN component is intentionally NOT swept: run_
adaptive_pipeline() always uses RealCNN(seed=77) internally (unchanged, to
match the already-verified single-run methodology used elsewhere in this
project) -- so the same real battery photo/confidence is used every
iteration, and only the fuzzy/sensor-driven part of R_fused varies. This
isolates "how robust is each system to sensor noise" from "how robust is
the CNN to a different photo", which is a separate question already
answered by the coarse/fine test-set accuracy numbers.

Seeds used: run_simulation(seed=100+i) for i in 0..29, keeping well clear of
the canonical seed=42 used for the single verified headline run reported in
the paper's main text and tables.

Run from the git_repo root: python src/run_baseline_seed_sweep.py
(relies on checkpoints/ and data/ resolving relative to CWD, same as every
other script in this package -- do not run it from elsewhere.)

Outputs (into results/, alongside the existing B1-only sweep):
    results/baseline_seed_sweep_per_seed.csv   -- one row per (seed, system), all metrics
    results/baseline_seed_sweep_summary.csv    -- mean/std per (system, metric)
    Prints a formatted summary table.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from sensor_simulation import run_simulation
from adaptive_risk import run_adaptive_pipeline
from supervisory_control import run_supervisory
from baseline_comparison import (
    baseline_vision_only, baseline_rule_based, baseline_fixed_fuzzy,
    compute_metrics, BATTERY_ARRIVES,
)

N_SEEDS = 30
SEED_BASE = 100
METRICS = ["Hazard_FNR", "Critical_FNR", "FPR", "T_warn_s", "T_stop_s", "CFNC"]
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Seed-0-of-sweep sanity check: the canonical seed=42 single run's already-
# verified numbers (this session), to confirm compute_metrics/baseline logic
# hasn't silently drifted before trusting the sweep.
EXPECTED_CANONICAL = {
    "B1_Vision_Only": {"Hazard_FNR": 0.0688, "T_stop_s": 161.6},
    "Proposed":       {"Hazard_FNR": 0.0000, "T_stop_s": 191.1},   # v2 CNN + gated memory (Oct 2026)
}


def to_numeric(val):
    """compute_metrics() returns the string '>400' when a system never
    reaches that state. Keep that signal instead of coercing it -- drop it
    from mean/std (it isn't a real elapsed-time value) but don't hide it."""
    if isinstance(val, str):
        return np.nan
    return float(val)


def main():
    print("=" * 70)
    print("  run_baseline_seed_sweep.py")
    print(f"  {N_SEEDS} sensor-noise realizations x 4 systems (B1/B2/B3/Proposed)")
    print("=" * 70)

    # Sanity check against the canonical seed=42 single run first.
    df_canon = run_simulation(duration_s=400.0, battery_arrives_at=20.0, seed=42)
    df_b1_c = baseline_vision_only(df_canon, seed=0)
    m_b1_c = compute_metrics(df_b1_c, BATTERY_ARRIVES, "B1_Vision_Only")
    df_risk_c = run_adaptive_pipeline(df_canon, use_real_cnn=True)
    df_full_c, _ = run_supervisory(df_risk_c)
    df_prop_c = df_full_c[["time", "fsm_state", "R_fused", "battery_present", "phase"]].copy()
    df_prop_c.columns = ["time", "state", "risk", "battery_present", "phase"]
    m_prop_c = compute_metrics(df_prop_c, BATTERY_ARRIVES, "Proposed")

    print("\nSanity check vs. already-verified canonical seed=42 run:")
    ok_all = True
    for sysname, checks in EXPECTED_CANONICAL.items():
        actual = m_b1_c if sysname == "B1_Vision_Only" else m_prop_c
        for k, expected in checks.items():
            got = actual[k]
            ok = isinstance(got, (int, float)) and abs(got - expected) < 0.01
            ok_all = ok_all and ok
            print(f"  {sysname:<16} {k:<10} expected={expected:<8} actual={got:<8} {'OK' if ok else 'MISMATCH !!'}")
    if not ok_all:
        print("\n  MISMATCH detected -- stopping. Do not trust a sweep run on top of this.")
        sys.exit(1)
    print("  All sanity checks passed.\n")

    rows = []
    for i in range(N_SEEDS):
        seed = SEED_BASE + i
        df = run_simulation(duration_s=400.0, battery_arrives_at=20.0, seed=seed)

        df_b1 = baseline_vision_only(df, seed=seed)
        df_b2 = baseline_rule_based(df)
        df_b3 = baseline_fixed_fuzzy(df)

        df_risk = run_adaptive_pipeline(df, use_real_cnn=True)
        df_full, _ = run_supervisory(df_risk)
        df_prop = df_full[["time", "fsm_state", "R_fused", "battery_present", "phase"]].copy()
        df_prop.columns = ["time", "state", "risk", "battery_present", "phase"]

        for name, df_r in [("B1_Vision_Only", df_b1), ("B2_Rule_Based", df_b2),
                            ("B3_Fixed_Fuzzy", df_b3), ("Proposed", df_prop)]:
            m = compute_metrics(df_r, BATTERY_ARRIVES, name)
            rows.append({
                "seed": seed, "system": name,
                **{k: to_numeric(m[k]) for k in METRICS},
            })
        print(f"  seed {seed} done ({i+1}/{N_SEEDS})")

    df_sweep = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df_sweep.to_csv(RESULTS_DIR / "baseline_seed_sweep_per_seed.csv", index=False)
    print(f"\nSaved per-seed results: results/baseline_seed_sweep_per_seed.csv ({len(df_sweep)} rows)")

    print("\n" + "=" * 70)
    print("  Mean +/- Std across", N_SEEDS, "seeds")
    print("=" * 70)
    summary_rows = []
    for name in ["B1_Vision_Only", "B2_Rule_Based", "B3_Fixed_Fuzzy", "Proposed"]:
        sub = df_sweep[df_sweep["system"] == name]
        print(f"\n{name}:")
        for col in METRICS:
            vals = sub[col].dropna()
            n_dropped = len(sub) - len(vals)
            if len(vals) == 0:
                print(f"    {col:<14} all {len(sub)} seeds never reached this state")
                summary_rows.append({"system": name, "metric": col, "mean": np.nan, "std": np.nan, "n": 0})
                continue
            mean, std = vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0
            note = f" ({n_dropped} excluded as never-reached)" if n_dropped else ""
            print(f"    {col:<14} mean={mean:.4f}  std={std:.4f}  n={len(vals)}/{len(sub)}{note}")
            summary_rows.append({"system": name, "metric": col, "mean": mean, "std": std, "n": len(vals)})

    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "baseline_seed_sweep_summary.csv", index=False)
    print(f"\nSaved summary: results/baseline_seed_sweep_summary.csv")


if __name__ == "__main__":
    main()
