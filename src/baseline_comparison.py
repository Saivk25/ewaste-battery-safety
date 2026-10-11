"""
baseline_comparison.py — Meaningful baseline comparison for paper
Focus metrics on: early warning detection, emergency stop speed, false alarms
"""
from typing import Dict, List
import numpy as np
import pandas as pd
import sys; sys.path.insert(0, ".")
from fuzzy_inference import MamdaniFIS

DT = 0.1
BATTERY_ARRIVES = 20.0


def baseline_vision_only(df, seed: int = 0):
    """B1: vision PROXY. Does NOT run the trained CNN: emits a noisy battery
    probability driven only by the true temperature band (0.85 / 0.45 / 0.05)
    and applies fixed thresholds -- no gas, metal or temporal information.
    A real camera cannot see temperature, so this is a rough stand-in; its
    behaviour under occlusion is modelled in run_multi_item_occlusion_sweep.py
    by blinding it completely.
    seed defaults to 0 to preserve the exact reproducible run already
    verified and reported in COWORK_HANDOFF.md — pass a different seed
    (e.g. for a multi-seed sweep) without changing that default behavior."""
    # Fixed local RNG: previously used the unseeded global np.random state,
    # which made B1's numbers (esp. T_stop_s) different on every run.
    rng = np.random.RandomState(seed)
    results = []
    for _, row in df.iterrows():
        T = row["temperature_C"]
        if T > 90:   prob = np.clip(0.85 + rng.normal(0,0.06), 0, 1)
        elif T > 50: prob = np.clip(0.45 + rng.normal(0,0.08), 0, 1)
        else:        prob = np.clip(0.05 + rng.normal(0,0.03), 0, 1)
        state = "Emergency_Stop" if prob > 0.65 else "Warning" if prob > 0.35 else "Normal"
        results.append({"time": row["time"], "state": state,
                         "risk": prob, "battery_present": row["battery_present"],
                         "phase": row["phase"]})
    return pd.DataFrame(results)


def baseline_rule_based(df):
    """B2: Hard thresholds on sensors. No probabilistic reasoning."""
    results = []
    for _, row in df.iterrows():
        T, h2, co2, metal = (row["temperature_C"], row["mq2_ppm"],
                              row["mq135_ppm"], row["metal_present"])
        if T > 90 or h2 > 500 or co2 > 2500:
            state = "Emergency_Stop"; risk = 0.90
        elif T > 50 or h2 > 80 or (metal and T > 35):
            state = "Warning"; risk = 0.40
        else:
            state = "Normal"; risk = 0.05
        results.append({"time": row["time"], "state": state,
                         "risk": risk, "battery_present": row["battery_present"],
                         "phase": row["phase"]})
    return pd.DataFrame(results)


def baseline_fixed_fuzzy(df):
    """B3: Fuzzy inference, fixed beta=0.85 (no adaptive temporal update)."""
    fis = MamdaniFIS(); beta = 0.85; R_prev = 0.0
    results = []
    for _, row in df.iterrows():
        fuzzy_risk, _ = fis.infer(row["temperature_C"], row["mq2_ppm"],
                                   row["mq135_ppm"], bool(row["metal_present"]))
        R_t = beta * R_prev + (1 - beta) * fuzzy_risk; R_prev = R_t
        state = ("Emergency_Stop" if R_t > 0.75 else
                 "Critical" if R_t > 0.55 else
                 "Warning" if R_t > 0.30 else "Normal")
        results.append({"time": row["time"], "state": state,
                         "risk": R_t, "battery_present": row["battery_present"],
                         "phase": row["phase"]})
    return pd.DataFrame(results)


def compute_metrics(df_result, battery_arrives, system_name):
    """
    Metrics focused on critical-phase detection (when battery is dangerous):
    - Hazard_FNR: fraction of hazardous-phase time where system was in Normal state
    - Early_Warning_Time: time from H2 onset (t=289s) to first Warning/higher
    - T_to_Emergency: from battery arrival to Emergency Stop
    - FPR: false alarms in non-battery periods
    - CFNC: catastrophic false negative cost = (10*N_runaway_missed + 2*N_missed) / N_hazard_steps.
      Equals 2 x Hazard_FNR whenever no miss occurs in Phase 3/4 (true for every reported
      single-item run), so it only adds information when runaway-phase misses occur.
    """
    # Critical phase = Phase2+ (T>45°C, actual danger developing)
    critical_phases = ["Phase1_SEI","Phase2_Separator","Phase3_Runaway","Phase4_Fire"]
    hazard_rows = df_result[df_result["phase"].isin(critical_phases)]
    nobatt_rows = df_result[df_result["battery_present"] == 0]

    # Hazard FNR: not detected during any dangerous phase
    if len(hazard_rows) > 0:
        detected = hazard_rows["state"].isin(["Emergency_Stop","Critical","Warning"])
        hazard_FNR = 1.0 - detected.mean()
        # Critical FNR (Phase3/4 only — most dangerous)
        crit_rows = df_result[df_result["phase"].isin(["Phase3_Runaway","Phase4_Fire"])]
        if len(crit_rows) > 0:
            crit_detected = crit_rows["state"].isin(["Emergency_Stop","Critical"])
            crit_FNR = 1.0 - crit_detected.mean()
        else:
            crit_FNR = 0.0
    else:
        hazard_FNR = 0.0; crit_FNR = 0.0

    # FPR — false Emergency Stop when no battery
    FPR = (nobatt_rows["state"] == "Emergency_Stop").mean() if len(nobatt_rows) > 0 else 0.0

    # Time to Emergency Stop from battery arrival
    emg = df_result[(df_result["state"]=="Emergency_Stop") &
                    (df_result["time"] >= battery_arrives)]
    T_stop = emg["time"].iloc[0] - battery_arrives if len(emg) > 0 else float("inf")

    # Time to first Warning from battery arrival
    warn = df_result[(df_result["state"].isin(["Warning","Critical","Emergency_Stop"])) &
                     (df_result["time"] >= battery_arrives)]
    T_warn = warn["time"].iloc[0] - battery_arrives if len(warn) > 0 else float("inf")

    # CFNC: missed during runaway phase * severity weight
    if len(hazard_rows) > 0:
        missed = hazard_rows[~hazard_rows["state"].isin(["Emergency_Stop","Critical","Warning"])]
        phase3_missed = missed[missed["phase"].isin(["Phase3_Runaway","Phase4_Fire"])]
        CFNC = (len(phase3_missed) * 10.0 + len(missed) * 2.0) / (len(hazard_rows) + 1e-6)
    else:
        CFNC = 0.0

    return {
        "System":         system_name,
        "Hazard_FNR":     round(hazard_FNR, 4),
        "Critical_FNR":   round(crit_FNR, 4),
        "FPR":            round(FPR, 4),
        "T_warn_s":       round(T_warn, 1) if T_warn != float("inf") else ">400",
        "T_stop_s":       round(T_stop, 1) if T_stop != float("inf") else ">400",
        "CFNC":           round(CFNC, 4),
    }


def run_all_baselines(df_sensor, df_proposed):
    df_b1 = baseline_vision_only(df_sensor)
    df_b2 = baseline_rule_based(df_sensor)
    df_b3 = baseline_fixed_fuzzy(df_sensor)

    df_prop = df_proposed[["time","fsm_state","R_fused","battery_present","phase"]].copy()
    df_prop.columns = ["time","state","risk","battery_present","phase"]

    results = []
    for name, df_r in [("B1_Vision_Only", df_b1), ("B2_Rule_Based", df_b2),
                        ("B3_Fixed_Fuzzy", df_b3), ("Proposed", df_prop)]:
        results.append(compute_metrics(df_r, BATTERY_ARRIVES, name))
    return pd.DataFrame(results)


if __name__ == "__main__":
    # NOTE: these paths were previously broken (pointed at outputs/ewaste_system/
    # sensor_data_hazardous.csv, which is a stale re-run, and at a hardcoded
    # Windows path "proposed_results.csv" that main_pipeline.py never creates).
    # The actual "proposed system" output is outputs/full_pipeline_output.csv,
    # produced by supervisory_control.run_supervisory() in main_pipeline.py,
    # paired with the sensor data from the SAME run: outputs/sensor_data_hazardous.csv.
    df_sensor   = pd.read_csv("outputs/sensor_data_hazardous.csv")
    df_proposed = pd.read_csv("outputs/full_pipeline_output.csv")
    df_metrics  = run_all_baselines(df_sensor, df_proposed)
    df_metrics.to_csv("outputs/baseline_comparison.csv", index=False)
    print("\n" + "="*80)
    print("BASELINE COMPARISON — E-Waste Safety System")
    print("="*80)
    print(df_metrics.to_string(index=False))
    print("="*80)
    print("\nKey: Hazard_FNR=miss rate during dangerous phases | T_warn_s=time to first warning")
    print("     T_stop_s=time to Emergency Stop | CFNC=Catastrophic False Negative Cost")
