"""
run_final_validation.py -- full-pipeline check of the final configuration (v2 coarse CNN with
battery threshold, corroboration-gated adaptive memory, fusion alpha = 0.35, FSM) against the
baselines on (a) battery-free noise tests and (b) batteries that arrive already failing.
Held-out seeds 100-129. Run from the repo root:  python src/run_final_validation.py
Outputs: results/final_validation_noise.csv, results/final_validation_prewarmed.csv
"""
import os, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from sensor_simulation import run_noise_stress_scenario
from adaptive_risk import run_adaptive_pipeline
from supervisory_control import run_supervisory
from baseline_comparison import baseline_vision_only, baseline_rule_based, baseline_fixed_fuzzy
from gap_memory_redesign import NOISE
from gap_prewarmed import sim as sim_prewarmed

RES = Path("results")


def systems(df, seed):
    r = run_adaptive_pipeline(df, use_real_cnn=True)
    full, _ = run_supervisory(r)
    return {"Proposed": full.fsm_state.values, "B1_Vision_Proxy": baseline_vision_only(df, seed=seed).state.values,
            "B2_Rule_Based": baseline_rule_based(df).state.values, "B3_Fixed_Fuzzy": baseline_fixed_fuzzy(df).state.values}, r


def main():
    noise_rows, pw_rows = [], []
    for k, seed in enumerate(range(100, 130)):
        for name in ("mild", "hard", "harder"):
            dn = run_noise_stress_scenario(seed=seed + 1000, **NOISE[name])
            st, r = systems(dn, seed)
            for s, v in st.items():
                noise_rows.append(dict(seed=seed, noise=name, system=s, critical=bool((v == "Critical").any()),
                                       estop=bool((v == "Emergency_Stop").any()), P_battery=float(r.cnn_battery_prob.max())))
        for T0, sc in ((60, 3), (80, 6), (90, 10)):
            d = sim_prewarmed(seed, T0, sc)
            st, r = systems(d, seed)
            t = d.time.values
            for s, v in st.items():
                idx = np.where((v == "Emergency_Stop") & (t >= 20.0))[0]
                pw_rows.append(dict(seed=seed, T0=T0, scale=sc, system=s, T_stop_s=(t[idx[0]] - 20.0) if len(idx) else np.nan))
        print(f"seed {seed} done ({k + 1}/30)", flush=True)
    nz = pd.DataFrame(noise_rows); pw = pd.DataFrame(pw_rows)
    nz.to_csv(RES / "final_validation_noise.csv", index=False); pw.to_csv(RES / "final_validation_prewarmed.csv", index=False)
    print(nz.groupby(["noise", "system"]).agg(false_critical=("critical", "sum"), false_estop=("estop", "sum")).to_string())
    print(pw.groupby(["T0", "scale", "system"]).T_stop_s.agg(["mean", "max", "count"]).round(1).to_string())


if __name__ == "__main__":
    main()
