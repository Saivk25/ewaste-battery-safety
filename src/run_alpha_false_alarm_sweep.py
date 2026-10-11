"""
run_alpha_false_alarm_sweep.py
=================================
Companion to run_alpha_ablation_sweep.py. That script showed Hazard_FNR and
CFNC improving *monotonically* as alpha increases well past the paper's
0.35, on the hazardous (battery-present) scenario -- which taken alone
would seem to argue for a much higher alpha. But that sweep cannot show
the other side of the tradeoff: raising alpha also amplifies whatever
residual battery_prob the CNN assigns to a genuinely benign item, which
risks turning small, ordinary CNN uncertainty into false Emergency Stops
on items that were never dangerous.

This script tests exactly that, on run_normal_scenario() (no battery ever
present, only benign metal/non-metal items cycling), across the same
alpha values and same per-seed-capture-then-cheap-refold structure as
run_alpha_ablation_sweep.py. The headline number is FPR: the fraction of
time the system sits in Emergency_Stop while battery_present=0 --
compute_metrics() already defines this exactly, no new metric needed.

Run from the git_repo root: python src/run_alpha_false_alarm_sweep.py

Outputs (into results/):
    alpha_false_alarm_per_seed.csv   -- one row per (seed, alpha)
    alpha_false_alarm_summary.csv    -- mean/std FPR per alpha, across seeds
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from sensor_simulation import run_normal_scenario
from fuzzy_inference import MamdaniFIS
from adaptive_risk import AdaptiveRiskModel, RealCNN
from supervisory_control import SupervisoryFSM
from baseline_comparison import compute_metrics

N_SEEDS = 30
SEED_BASE = 200  # distinct range from the hazardous-scenario sweeps (100-129)
ALPHAS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5]
RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_one_seed(seed: int, cnn: RealCNN):
    df = run_normal_scenario(duration_s=200.0, seed=seed)
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
                     r["mq2_ppm"], r["mq135_ppm"], r["phase"]))
    return pd.DataFrame(rows, columns=["time", "R_t", "P_battery", "temperature_C",
                                        "mq2_ppm", "mq135_ppm", "phase"])


def fpr_for_alpha(captured: pd.DataFrame, alpha: float, system_name: str):
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
    return compute_metrics(df_result, battery_arrives=1e9, system_name=system_name)


def main():
    print("=" * 70)
    print("  run_alpha_false_alarm_sweep.py")
    print(f"  {N_SEEDS} seeds x {len(ALPHAS)} alpha values, run_normal_scenario (no battery ever)")
    print("=" * 70)

    rows = []
    max_p_battery = []
    for i in range(N_SEEDS):
        seed = SEED_BASE + i
        cnn = RealCNN(seed=77)
        captured = run_one_seed(seed, cnn)
        max_p_battery.append(captured["P_battery"].max())
        for alpha in ALPHAS:
            m = fpr_for_alpha(captured, alpha, f"alpha={alpha}")
            rows.append({"seed": seed, "alpha": alpha, "FPR": m["FPR"],
                         "max_state_reached": None})
        print(f"  seed {seed} done ({i+1}/{N_SEEDS}), max P_battery this seed = {max_p_battery[-1]:.3f}")

    df_sweep = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df_sweep.to_csv(RESULTS_DIR / "alpha_false_alarm_per_seed.csv", index=False)
    print(f"\nSaved per-seed results: results/alpha_false_alarm_per_seed.csv ({len(df_sweep)} rows)")
    print(f"\nOverall max CNN battery_prob seen on a genuinely benign item across all seeds: "
          f"{max(max_p_battery):.4f} (mean of per-seed max: {np.mean(max_p_battery):.4f})")

    print("\n" + "=" * 70)
    print("  FPR (false Emergency_Stop rate, battery never present) per alpha")
    print("=" * 70)
    summary_rows = []
    for alpha in ALPHAS:
        sub = df_sweep[df_sweep["alpha"] == alpha]["FPR"]
        mean, std = sub.mean(), sub.std(ddof=1)
        n_nonzero = (sub > 0).sum()
        print(f"  alpha={alpha:<5} FPR mean={mean:.5f}  std={std:.5f}  "
              f"({n_nonzero}/{len(sub)} seeds with at least one false Emergency_Stop)")
        summary_rows.append({"alpha": alpha, "FPR_mean": mean, "FPR_std": std,
                              "n_seeds_with_false_alarm": n_nonzero, "n_seeds": len(sub)})
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "alpha_false_alarm_summary.csv", index=False)
    print(f"\nSaved summary: results/alpha_false_alarm_summary.csv")


if __name__ == "__main__":
    main()
