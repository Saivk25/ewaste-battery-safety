#!/usr/bin/env bash
# GPU part of the v3 rerun: CNN seed/backbone repeats on data_v3, then every CNN result. Run from the repo root.
set -e
export PYTHONPATH=src PYTHONIOENCODING=utf-8
mkdir -p results/rerun_logs
done_run() { ls "runs_cnn/$1/results/"*.json >/dev/null 2>&1; }   # a finished run leaves its results json
for s in 1 2; do done_run EfficientNet-B0_s$s || timeout 5400 python src/train_cnn.py --stage coarse --backbone EfficientNet-B0 --data-dir data_v3 --seed $s --out-dir runs_cnn/EfficientNet-B0_s$s > results/rerun_logs/train_B0_s$s.log 2>&1; echo "B0 s$s $(date +%T)"; done
for s in 1 2 42; do done_run MobileNetV2_s$s || timeout 5400 python src/train_cnn.py --stage coarse --backbone MobileNetV2 --data-dir data_v3 --seed $s --out-dir runs_cnn/MobileNetV2_s$s > results/rerun_logs/train_MNV2_s$s.log 2>&1; echo "MNV2 s$s $(date +%T)"; done
done_run EfficientNet-B3_s42 || timeout 5400 python src/train_cnn.py --stage coarse --backbone EfficientNet-B3 --data-dir data_v3 --seed 42 --out-dir runs_cnn/EfficientNet-B3_s42 > results/rerun_logs/train_B3.log 2>&1; echo "B3 $(date +%T)"
for s in cnn_repeat_eval plot_final_cms generate_roc_pr generate_gradcam; do python src/$s.py > results/rerun_logs/$s.log 2>&1; echo "$s $(date +%T)"; done
python src/gap_cnn_checks.py --tag v3 > results/rerun_logs/gap_cnn_checks.log 2>&1; echo "gap_cnn_checks $(date +%T)"
python src/detector_yolo.py clutter > results/rerun_logs/detector_clutter.log 2>&1; echo "detector clutter $(date +%T)"
echo "GPU part done $(date +%T)"
