"""
run_alpha_ablation_sweep.py
=============================
Two things in one script, since they share the same expensive per-seed data:

1. CNN-fusion ablation: what happens with alpha=0 (no CNN confidence
   amplification at all -- R_fused = R_t, pure fuzzy+adaptive-beta, no
   visual confirmation) versus the paper's alpha=0.35?  This isolates the
   effect of Section 5.1's fusion term specifically, distinct from B1
   (which removes the WHOLE fuzzy/sensor system, not just the fusion term)
   and B3 (which removes adaptive beta, not CNN fusion).

2. Alpha-sensitivity curve: how do Hazard_FNR/T_stop/CFNC change as alpha
   varies across a wider range? Justifies why 0.35 specifically, rather
   than presenting it as an arbitrary empirically-tuned constant.

Why one run suffices for both: AdaptiveRiskModel.update() computes R_t
(the temporal risk BEFORE CNN fusion) and updates its own R_prev state
using R_t, never R_fused -- so R_t's entire trajectory is independent of
alpha. R_fused = clip(R_t * (1 + alpha * P_battery), 0, 1) is a cheap,
purely post-hoc transform. This script runs the real fis+cnn+temporal
pipeline ONCE per seed (the expensive part -- CNN checkpoint + inference),
captures R_t and P_battery per timestep, then re-derives R_fused and
re-runs the (cheap, pure-Python) FSM for every alpha value on top of that
same captured trajectory -- 30 real pipeline runs total, not
30 x len(ALPHAS).

Uses the same 30 seeds (100-129) as run_baseline_seed_sweep.py so the
alpha=0.35 row here is directly comparable to that sweep's "Proposed" row
(same seeds, same underlying noise realizations).

Run from the git_repo root: python src/run_alpha_ablation_sweep.py

Outputs (into results/):
    alpha_ablation_per_seed.csv   -- one row per (seed, alpha), all metrics
    alpha_ablation_summary.csv    -- mean/std per alpha, across 30 seeds
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy import stats

from sensor_simulation import run_simulation
from fuzzy_inference import MamdaniFIS
from adaptive_risk import AdaptiveRiskModel, RealCNN
from supervisory_control import SupervisoryFSM
from baseline_comparison import compute_metrics, BATTERY_ARRIVES

N_SEEDS = 30
SEED_BASE = 100
ALPHAS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5]
PAPER_ALPHA = 0.35
METRICS = ["Hazard_FNR", "Critical_FNR", "FPR", "T_warn_s", "T_stop_s", "CFNC"]
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Same canonical seed=42 sanity check used by run_baseline_seed_sweep.py,
# at alpha=0.35 (the paper's value) -- confirms this script's independent
# reimplementation of the fusion+FSM loop matches the tested pipeline
# exactly before trusting anything built on top of it.
EXPECTED_CANONICAL_ALPHA035 = {"Hazard_FNR": 0.0000, "T_stop_s": 191.1}   # v2 CNN + gated memory (Oct 2026)


def to_numeric(val):
    if isinstance(val, str):
        return np.nan
    return float(val)


def run_one_seed(seed: int, cnn: RealCNN):
    """Runs fis+cnn+temporal-update once, capturing per-timestep R_t and
    P_battery (alpha-independent) alongside what the FSM needs (temperature,
    h2, co2, battery_present, phase, time)."""
    df = run_simulation(duration_s=400.0, battery_arrives_at=20.0, seed=seed)
    fis = MamdaniFIS()
    arm = AdaptiveRiskModel()

    rows = []
    for _, r in df.iterrows():
        fuzzy_risk, _ = fis.infer(r["temperature_C"], r["mq2_ppm"], r["mq135_ppm"], bool(r["metal_present"]))
        cnn_out = cnn.predict(
            temperature=r["temperature_C"], mq2_ppm=r["mq2_ppm"],
            object_present=bool(r["object_present"]), metal_present=bool(r["metal_present"]),
            battery_present=bool(r["battery_present"]),
        )
        R_t, _beta, _ = arm.update(S_t=fuzzy_risk, P_battery=cnn_out["battery_prob"],
                                    object_present=bool(r["object_present"]), temperature=r["temperature_C"])
        rows.append((r["time"], R_t, cnn_out["battery_prob"], r["temperature_C"],
                     r["mq2_ppm"], r["mq135_ppm"], r["battery_present"], r["phase"]))
    return pd.DataFrame(rows, columns=["time", "R_t", "P_battery", "temperature_C",
                                        "mq2_ppm", "mq135_ppm", "battery_present", "phase"])


def metrics_for_alpha(captured: pd.DataFrame, alpha: float, system_name: str):
    R_fused = np.clip(captured["R_t"].values * (1.0 + alpha * captured["P_battery"].values), 0.0, 1.0)
    fsm = SupervisoryFSM()
    states = []
    for i, r in captured.iterrows():
        state, _speed = fsm.step(t=r["time"], R_fused=float(R_fused[i]), temperature=r["temperature_C"],
                                  h2_ppm=r["mq2_ppm"], co2_ppm=r["mq135_ppm"])
        states.append(state.value)
    df_result = pd.DataFrame({
        "time": captured["time"], "state": states, "risk": R_fused,
        "battery_present": captured["battery_present"], "phase": captured["phase"],
    })
    return compute_metrics(df_result, BATTERY_ARRIVES, system_name)


def main():
    print("=" * 70)
    print("  run_alpha_ablation_sweep.py")
    print(f"  {N_SEEDS} seeds x {len(ALPHAS)} alpha values ({ALPHAS})")
    print("=" * 70)

    # Sanity check: seed=42, alpha=0.35, against the already-verified
    # canonical single run reported in the paper.
    cnn_check = RealCNN(seed=77)
    captured42 = run_one_seed(42, cnn_check)
    m_check = metrics_for_alpha(captured42, PAPER_ALPHA, "sanity_check")
    print("\nSanity check vs. already-verified canonical seed=42, alpha=0.35 run:")
    ok_all = True
    for k, expected in EXPECTED_CANONICAL_ALPHA035.items():
        got = m_check[k]
        ok = isinstance(got, (int, float)) and abs(got - expected) < 0.01
        ok_all = ok_all and ok
        print(f"  {k:<10} expected={expected:<8} actual={got:<8} {'OK' if ok else 'MISMATCH !!'}")
    if not ok_all:
        print("\n  MISMATCH detected -- stopping. Do not trust this sweep.")
        sys.exit(1)
    print("  Sanity check passed.\n")

    rows = []
    for i in range(N_SEEDS):
        seed = SEED_BASE + i
        cnn = RealCNN(seed=77)  # fresh per seed, matching run_baseline_seed_sweep.py's convention
        captured = run_one_seed(seed, cnn)
        for alpha in ALPHAS:
            m = metrics_for_alpha(captured, alpha, f"alpha={alpha}")
            rows.append({"seed": seed, "alpha": alpha, **{k: to_numeric(m[k]) for k in METRICS}})
        print(f"  seed {seed} done ({i+1}/{N_SEEDS})")

    df_sweep = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df_sweep.to_csv(RESULTS_DIR / "alpha_ablation_per_seed.csv", index=False)
    print(f"\nSaved per-seed results: results/alpha_ablation_per_seed.csv ({len(df_sweep)} rows)")

    print("\n" + "=" * 70)
    print("  Mean +/- Std across", N_SEEDS, "seeds, per alpha")
    print("=" * 70)
    summary_rows = []
    for alpha in ALPHAS:
        sub = df_sweep[df_sweep["alpha"] == alpha]
        print(f"\nalpha={alpha}:")
        for col in METRICS:
            vals = sub[col].dropna()
            n_dropped = len(sub) - len(vals)
            if len(vals) == 0:
                print(f"    {col:<14} all {len(sub)} seeds never reached this state")
                summary_rows.append({"alpha": alpha, "metric": col, "mean": np.nan, "std": np.nan, "n": 0})
                continue
            mean, std = vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0
            note = f" ({n_dropped} excluded as never-reached)" if n_dropped else ""
            print(f"    {col:<14} mean={mean:.4f}  std={std:.4f}  n={len(vals)}/{len(sub)}{note}")
            summary_rows.append({"alpha": alpha, "metric": col, "mean": mean, "std": std, "n": len(vals)})
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "alpha_ablation_summary.csv", index=False)
    print(f"\nSaved summary: results/alpha_ablation_summary.csv")

    # Paired significance: alpha=0.35 (paper's choice) vs alpha=0.0 (CNN-fusion ablation)
    print("\n" + "=" * 70)
    print("  Ablation: alpha=0.35 (Proposed) vs alpha=0.0 (no CNN fusion)")
    print("=" * 70)
    piv = df_sweep.pivot(index="seed", columns="alpha", values="Hazard_FNR")
    diff = piv[PAPER_ALPHA] - piv[0.0]
    t, p = stats.ttest_rel(piv[PAPER_ALPHA], piv[0.0])
    print(f"  Hazard_FNR: mean_diff={diff.mean():.4f}  paired-t={t:.3f}  p={p:.4f}  "
          f"({(diff < 0).sum()}/{len(diff)} seeds where alpha=0.35 < alpha=0.0)")
    piv_cfnc = df_sweep.pivot(index="seed", columns="alpha", values="CFNC")
    diff_c = piv_cfnc[PAPER_ALPHA] - piv_cfnc[0.0]
    t_c, p_c = stats.ttest_rel(piv_cfnc[PAPER_ALPHA], piv_cfnc[0.0])
    print(f"  CFNC:       mean_diff={diff_c.mean():.4f}  paired-t={t_c:.3f}  p={p_c:.4f}  "
          f"({(diff_c < 0).sum()}/{len(diff_c)} seeds where alpha=0.35 < alpha=0.0)")


if __name__ == "__main__":
    main()
