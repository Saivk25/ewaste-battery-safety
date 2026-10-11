#!/bin/bash
# Reproduces every result in results/pub/ and the paper tables, in order. Run from the repo root.
# Data: see docs/DATA.md (data_v2 built by src/build_coarse_v2.py; RecyBat24 in data/raw/recybat24_aug;
# Kwak et al. 2026 traces unzipped into data/raw/kwak_tr/). GPU recommended; total several hours.
set -e
export PYTHONIOENCODING=utf-8
python -m pytest tests -q --ignore=tests/verify_dashboard_interactive.py   # 1. unit + dashboard smoke tests
python src/pub_experiments.py      # 2. baselines, stress tests, dwell, statistics -> eval.csv, stats.csv, summary.txt
python src/pub_lag_extra.py        # 3. 60 s / 120 s temperature lag
python src/pub_fixes.py            # 4. calibrated thermal model, rate gate, trade-off sweep, CFNC weights, logistic
python src/pub_realtrace.py        # 5. measured cell temperature curves
python src/pub_fixes2.py           # 6. joint gate retune on tuning seeds, held-out check
for s in 1 2; do                    # 7. CNN seed repeats and backbones (production model: checkpoints/, seed 42)
  python src/train_cnn.py --stage coarse --backbone EfficientNet-B0 --data-dir data_v3 --seed $s --out-dir runs_cnn/EfficientNet-B0_s$s
done
for s in 1 2 42; do
  python src/train_cnn.py --stage coarse --backbone MobileNetV2 --data-dir data_v3 --seed $s --out-dir runs_cnn/MobileNetV2_s$s
done
python src/train_cnn.py --stage coarse --backbone EfficientNet-B3 --data-dir data_v3 --seed 42 --out-dir runs_cnn/EfficientNet-B3_s42
python src/cnn_repeat_eval.py      # 8. -> cnn_repeat_eval.csv, cnn_backbones_summary.csv
python src/detector_yolo.py all    # 9. YOLOv8n detector + cluttered frames -> detector_clutter.csv, detector_summary.json
