"""
run_alpha_noise_stress_sweep.py
==================================
Second companion to run_alpha_ablation_sweep.py, and the more demanding
test of the two false-alarm checks. run_alpha_false_alarm_sweep.py (on
run_normal_scenario) found zero false Emergency_Stops even at alpha=1.5 --
but that scenario's fuzzy risk barely rises above zero at all (H2 never
leaves 0 without a battery), so a high alpha has almost nothing to
amplify there; it is a weak test of the tradeoff.

run_noise_stress_scenario() is the real test: it injects sharp, brief
sensor-noise spikes (H2/CO2, no battery, no thermal driver) large enough
to produce substantial fuzzy risk on their own (h2_medium(300ppm) = 0.69,
per that function's docstring) -- exactly the kind of transient,
non-battery risk elevation a large alpha could plausibly turn into a
false Emergency_Stop that alpha=0.35 would not. This is the scenario that
can actually falsify "raising alpha is free."

Same per-seed-capture-then-cheap-refold-per-alpha structure as the other
two alpha sweeps.

Run from the git_repo root: python src/run_alpha_noise_stress_sweep.py

Outputs (into results/):
    alpha_noise_stress_per_seed.csv   -- one row per (seed, alpha)
    alpha_noise_stress_summary.csv    -- mean/std FPR per alpha, across seeds
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from sensor_simulation import run_noise_stress_scenario
from fuzzy_inference import MamdaniFIS
from adaptive_risk import AdaptiveRiskModel, RealCNN
from supervisory_control import SupervisoryFSM
from baseline_comparison import compute_metrics

N_SEEDS = 30
SEED_BASE = 300  # distinct range from the other two alpha sweeps (100-129, 200-229)
ALPHAS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5]
RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_one_seed(seed: int, cnn: RealCNN):
    df = run_noise_stress_scenario(duration_s=300.0, seed=seed, n_spikes=8)
    fis = MamdaniFIS()
    arm = AdaptiveRiskModel()
    rows = []
    for _, r in df.iterrows():
        fuzzy_risk, _ = fis.infer(r["temperature_C"], r["mq2_ppm"], r["mq135_ppm"], bool(r["metal_present"]))
        cnn_out = cnn.predict(
            temperature=r["temperature_C"], mq2_ppm=r["mq2_ppm"],
            object_present=bool(r["object_present"]), metal_present=bool(r["metal_present"]),
            battery_present=False,
        )
        R_t, _beta, _ = arm.update(S_t=fuzzy_risk, P_battery=cnn_out["battery_prob"],
                                    object_present=bool(r["object_present"]), temperature=r["temperature_C"])
        rows.append((r["time"], R_t, cnn_out["battery_prob"], r["temperature_C"],
                     r["mq2_ppm"], r["mq135_ppm"], r["phase"], r["is_noise_spike"]))
    return pd.DataFrame(rows, columns=["time", "R_t", "P_battery", "temperature_C",
                                        "mq2_ppm", "mq135_ppm", "phase", "is_noise_spike"])


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
        "battery_present": 0, "phase": captured["phase"],
    })
    m = compute_metrics(df_result, battery_arrives=1e9, system_name=system_name)
    max_r_fused = float(R_fused.max())
    ever_critical = "Critical" in states or "Emergency_Stop" in states
    return m, max_r_fused, ever_critical


def main():
    print("=" * 70)
    print("  run_alpha_noise_stress_sweep.py")
    print(f"  {N_SEEDS} seeds x {len(ALPHAS)} alpha values, run_noise_stress_scenario (8 spikes, no battery)")
    print("=" * 70)

    rows = []
    for i in range(N_SEEDS):
        seed = SEED_BASE + i
        cnn = RealCNN(seed=77)
        captured = run_one_seed(seed, cnn)
        for alpha in ALPHAS:
            m, max_r, ever_crit = metrics_for_alpha(captured, alpha, f"alpha={alpha}")
            rows.append({"seed": seed, "alpha": alpha, "FPR": m["FPR"],
                         "max_R_fused": max_r, "ever_reached_critical_or_above": ever_crit})
        print(f"  seed {seed} done ({i+1}/{N_SEEDS})")

    df_sweep = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df_sweep.to_csv(RESULTS_DIR / "alpha_noise_stress_per_seed.csv", index=False)
    print(f"\nSaved per-seed results: results/alpha_noise_stress_per_seed.csv ({len(df_sweep)} rows)")

    print("\n" + "=" * 70)
    print("  FPR and max R_fused per alpha (battery never present, 8 noise spikes/seed)")
    print("=" * 70)
    summary_rows = []
    for alpha in ALPHAS:
        sub = df_sweep[df_sweep["alpha"] == alpha]
        fpr = sub["FPR"]
        maxr = sub["max_R_fused"]
        n_crit = sub["ever_reached_critical_or_above"].sum()
        print(f"  alpha={alpha:<5} FPR mean={fpr.mean():.5f} std={fpr.std(ddof=1):.5f}  "
              f"max_R_fused mean={maxr.mean():.4f} max={maxr.max():.4f}  "
              f"({n_crit}/{len(sub)} seeds ever reached Critical+)")
        summary_rows.append({"alpha": alpha, "FPR_mean": fpr.mean(), "FPR_std": fpr.std(ddof=1),
                              "max_R_fused_mean": maxr.mean(), "max_R_fused_max": maxr.max(),
                              "n_seeds_reached_critical_plus": int(n_crit), "n_seeds": len(sub)})
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "alpha_noise_stress_summary.csv", index=False)
    print(f"\nSaved summary: results/alpha_noise_stress_summary.csv")


if __name__ == "__main__":
    main()
