"""
main_pipeline.py — End-to-end simulation runner
Run this single file to regenerate everything from scratch.
"""
import sys, os, time
import os
from pathlib import Path

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

os.makedirs("outputs", exist_ok=True)
sys.path.insert(0, "outputs")

print("="*65)
print("  E-WASTE SAFETY SYSTEM — Full Simulation Pipeline")
print("  Hierarchical CNN + Adaptive Fuzzy Temporal Supervisory Control")
print("="*65)

steps = [
    ("1. Physics-based sensor simulation",  "sensor_simulation",  "run_simulation"),
    ("3. Adaptive risk + fuzzy inference",   "adaptive_risk",      "run_adaptive_pipeline"),
    ("4. Supervisory FSM",                   "supervisory_control","run_supervisory"),
    ("5. Baseline comparison",               "baseline_comparison","run_all_baselines"),
]

import pandas as pd

# Step 1
print(f"\n{'─'*55}\nStep 1: Physics-based sensor simulation\n{'─'*55}")
from sensor_simulation import run_simulation, run_normal_scenario
t0 = time.time()
df_hazard = run_simulation(duration_s=400.0, battery_arrives_at=20.0, seed=42)
df_normal = run_normal_scenario(duration_s=200.0, seed=99)
df_hazard.to_csv("outputs/sensor_data_hazardous.csv", index=False)
df_normal.to_csv("outputs/sensor_data_normal.csv", index=False)
print(f"  ✓ Hazardous: {len(df_hazard)} steps | Normal: {len(df_normal)} steps")
print(f"  ✓ Peak temp: {df_hazard['temperature_C'].max():.1f}°C | H₂: {df_hazard['mq2_ppm'].max():.0f}ppm | CO₂: {df_hazard['mq135_ppm'].max():.0f}ppm")
print(f"  ✓ Done in {time.time()-t0:.1f}s")

# Step 2 (removed): the old cnn_model.py backbone "comparison" was simulated (it never loads a
# trained checkpoint), so it is no longer run. Real CNN results: train_cnn.py, evaluate_model.py,
# gap_cnn_checks.py.
import json

# Step 3
print(f"\n{'─'*55}\nStep 3: Fuzzy inference + Adaptive temporal risk\n{'─'*55}")
from adaptive_risk import run_adaptive_pipeline
t0 = time.time()
df_risk = run_adaptive_pipeline(df_hazard)
df_risk.to_csv("outputs/adaptive_risk_output.csv", index=False)
print(f"  ✓ Peak R_fused: {df_risk['R_fused'].max():.4f} | Peak fuzzy: {df_risk['fuzzy_risk'].max():.4f}")
print(f"  ✓ Done in {time.time()-t0:.1f}s")

# Step 4
print(f"\n{'─'*55}\nStep 4: Supervisory FSM\n{'─'*55}")
from supervisory_control import run_supervisory
t0 = time.time()
df_full, df_trans = run_supervisory(df_risk)
df_full.to_csv("outputs/full_pipeline_output.csv", index=False)
df_trans.to_csv("outputs/state_transitions.csv", index=False)
print(f"  ✓ State distribution:\n    {dict(df_full['fsm_state'].value_counts())}")
if "Emergency_Stop" in df_full["fsm_state"].values:
    t_emg = df_full[df_full["fsm_state"]=="Emergency_Stop"]["time"].iloc[0]
    print(f"  ✓ Emergency Stop triggered at t={t_emg}s ({t_emg-20:.1f}s after battery arrival)")
print(f"  ✓ Done in {time.time()-t0:.1f}s")

# Step 5
print(f"\n{'─'*55}\nStep 5: Baseline comparison\n{'─'*55}")
from baseline_comparison import run_all_baselines
t0 = time.time()
df_metrics = run_all_baselines(df_hazard, df_full)
df_metrics.to_csv("outputs/baseline_comparison.csv", index=False)
print(f"  ✓ Comparison table:")
print(df_metrics[["System","Hazard_FNR","T_warn_s","T_stop_s","CFNC"]].to_string(index=False))
print(f"  ✓ Done in {time.time()-t0:.1f}s")

# Step 6 — All figures
print(f"\n{'─'*55}\nStep 6: Generating all figures\n{'─'*55}")
from visualization import (plot_fuzzy_memberships, plot_sensor_timeseries,
    plot_risk_and_fsm, plot_emergency_stop_detail,
    plot_baseline_comparison, plot_scenario_overlay)
import architecture_diagram
t0 = time.time()
plot_fuzzy_memberships()
plot_sensor_timeseries(df_full)
plot_risk_and_fsm(df_full)
plot_emergency_stop_detail(df_full, df_trans)
plot_baseline_comparison()
plot_scenario_overlay(df_full, df_normal)
print(f"  ✓ Done in {time.time()-t0:.1f}s")

print(f"\n{'='*65}")
print("  ✅ ALL STEPS COMPLETE")
print(f"{'='*65}")
print(f"\n  Output files in: outputs/")
print(f"  Figures in:      outputs/figures/")
print(f"\n  Key results summary:")
# Prefer the real, checkpoint-evaluated numbers (results/results_EfficientNetB0_fine.json,
# written by train_cnn.py/evaluate_model.py) over cnn_model.py's simulated backbone
# comparison (df_cnn), which never loads a trained checkpoint and only exists to produce
# an illustrative multi-backbone table/figure.
real_fine_path = Path("results/results_EfficientNetB0_fine.json")
if real_fine_path.exists():
    with open(real_fine_path) as f:
        real_fine = json.load(f)
    fine_acc = real_fine["test_acc"]
    battery_fnr = real_fine["test_metrics"]["per_class"]["LiIon_18650"]["fnr"]
    print(f"  • Recommended CNN:  EfficientNet-B0 (Fine Acc: {fine_acc:.4f}; measured coarse latency about 15 ms GPU / 29 ms CPU)")
    print(f"  • Battery FNR:      {battery_fnr:.4f}")
if "Emergency_Stop" in df_full["fsm_state"].values:
    t_emg = df_full[df_full["fsm_state"]=="Emergency_Stop"]["time"].iloc[0]
    print(f"  • Emergency Stop:   t={t_emg}s ({t_emg-20:.1f}s after battery arrival)")
print(f"  • Peak fused risk:  {df_risk['R_fused'].max():.4f}")
print(f"  • Transitions:      {len(df_trans)} FSM state changes logged")
