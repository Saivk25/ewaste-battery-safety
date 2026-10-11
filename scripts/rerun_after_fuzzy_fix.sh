#!/usr/bin/env bash
# Re-run every simulation result quoted in the paper after the open-shoulder fuzzy fix (commit a2552fc).
# CNN and detector results do not depend on the fuzzy system and are not re-run. Run from the repo root.
set -e
export PYTHONPATH=src
for s in main_pipeline run_baseline_seed_sweep run_multi_item_occlusion_sweep run_final_validation \
         run_alpha_ablation_sweep run_alpha_false_alarm_sweep run_alpha_noise_stress_sweep \
         gap_heldout_memory gap_corroborated_memory final_stats \
         pub_experiments pub_lag_extra pub_fixes pub_realtrace pub_fixes2 pub_faults \
         conveyor_sim pub_sensitivity pub_gas_sensitivity pub_thermal_compare; do
  echo "=== $s $(date +%T)"
  python src/$s.py > results/rerun_logs/$s.log 2>&1
done
echo "=== all done $(date +%T)"
