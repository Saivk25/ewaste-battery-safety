"""
run_b1_seed_sweep.py
=====================
Runs the B1_Vision_Only baseline across 30 seeds to get real mean±std for its
four headline metrics, instead of a single arbitrary draw. Reuses
baseline_vision_only() and compute_metrics() from baseline_comparison.py
directly (that function now takes an optional `seed` argument — added
specifically to support this sweep without changing its default, already-
verified behavior at seed=0).

IMPORTANT — provenance note:
Written by Claude Cowork but NOT executed there — the sandbox used for this
project's earlier work is out of disk and its shell is currently wedged
(repeated "no space left on device" RPC failures), so this could not be run
or verified before handing it to you. Run it yourself with:

    python run_b1_seed_sweep.py

and check the printed sanity block: seed 0's row should exactly match the
already-verified single-run numbers in COWORK_HANDOFF.md
(Hazard_FNR=0.1763, T_warn=224.5, T_stop=288.2, CFNC=0.3527). If it doesn't,
something about the data/code has changed since that was last verified and
the sweep numbers below should not be trusted until that's resolved.

Outputs:
    baseline_B1_seed_sweep.csv   — one row per seed (0-29), all 4 metrics
    Prints mean ± std for each metric across all 30 seeds.
"""

import sys
import numpy as np
import pandas as pd

from baseline_comparison import baseline_vision_only, compute_metrics, BATTERY_ARRIVES

N_SEEDS = 30
SENSOR_CSV = "outputs/sensor_data_hazardous.csv"
OUT_CSV = "baseline_B1_seed_sweep.csv"

# Single-run numbers already verified and recorded in COWORK_HANDOFF.md at
# seed=0 — used only as a sanity check on this sweep's seed-0 row.
EXPECTED_SEED0 = {"Hazard_FNR": 0.1763, "T_warn_s": 224.5, "T_stop_s": 288.2, "CFNC": 0.3527}


def to_numeric(val):
    """compute_metrics() returns the string '>400' for T_warn_s/T_stop_s when
    the system never reached that state (infinite time). Keep that signal
    instead of silently coercing it to a number — flag it and drop it from
    the mean/std (it would not be a real elapsed-time value)."""
    if isinstance(val, str):
        return np.nan
    return float(val)


def main():
    print("=" * 60)
    print("  run_b1_seed_sweep.py — UNVERIFIED SCRIPT")
    print("  Written by Claude Cowork but never executed there (sandbox")
    print("  disk full / shell wedged). This is the first real run.")
    print("=" * 60)

    try:
        df_sensor = pd.read_csv(SENSOR_CSV)
    except FileNotFoundError:
        print(f"✗ Could not find {SENSOR_CSV}. Run main_pipeline.py first to generate it.",
              file=sys.stderr)
        sys.exit(1)

    rows = []
    for seed in range(N_SEEDS):
        df_b1 = baseline_vision_only(df_sensor, seed=seed)
        m = compute_metrics(df_b1, BATTERY_ARRIVES, f"B1_seed{seed}")
        row = {
            "seed": seed,
            "Hazard_FNR": to_numeric(m["Hazard_FNR"]),
            "T_warn_s": to_numeric(m["T_warn_s"]),
            "T_stop_s": to_numeric(m["T_stop_s"]),
            "CFNC": to_numeric(m["CFNC"]),
        }
        rows.append(row)
        infinite_flag = " (never reached — recorded as NaN)" if (
            row["T_warn_s"] != row["T_warn_s"] or row["T_stop_s"] != row["T_stop_s"]) else ""
        print(f"  seed {seed:2d}: Hazard_FNR={row['Hazard_FNR']:.4f}  "
              f"T_warn_s={row['T_warn_s']}  T_stop_s={row['T_stop_s']}  "
              f"CFNC={row['CFNC']:.4f}{infinite_flag}")

    df_sweep = pd.DataFrame(rows)
    df_sweep.to_csv(OUT_CSV, index=False)
    print(f"\n✓ Saved per-seed results: {OUT_CSV}")

    # Sanity check against the already-verified seed=0 single run
    seed0 = df_sweep[df_sweep["seed"] == 0].iloc[0]
    print("\n" + "─" * 60)
    print("  Sanity check — seed 0 vs. previously verified single run:")
    mismatch = False
    for k, expected in EXPECTED_SEED0.items():
        actual = seed0[k]
        ok = abs(actual - expected) < 0.01
        mismatch = mismatch or not ok
        print(f"    {k:<12} expected={expected:<10} actual={actual:<10} {'OK' if ok else 'MISMATCH ⚠️'}")
    if mismatch:
        print("  ⚠️  Seed-0 mismatch — investigate before trusting the sweep below")
        print("     (different sensor CSV, different code version, etc.)")
    print("─" * 60)

    print("\n  Mean ± Std across all 30 seeds (NaN rows excluded per-column):")
    summary_rows = []
    for col in ["Hazard_FNR", "T_warn_s", "T_stop_s", "CFNC"]:
        vals = df_sweep[col].dropna()
        n_dropped = len(df_sweep) - len(vals)
        mean, std = vals.mean(), vals.std(ddof=1)
        print(f"    {col:<12} mean={mean:.4f}  std={std:.4f}  "
              f"(n={len(vals)}/{N_SEEDS}{f', {n_dropped} excluded as never-reached' if n_dropped else ''})")
        summary_rows.append({"metric": col, "mean": mean, "std": std, "n": len(vals)})

    pd.DataFrame(summary_rows).to_csv("baseline_B1_seed_sweep_summary.csv", index=False)
    print(f"\n✓ Saved summary: baseline_B1_seed_sweep_summary.csv")
    print("\n✅ Done. Please paste the full console output back for verification")
    print("   before these mean±std numbers go into the paper.")


if __name__ == "__main__":
    main()
