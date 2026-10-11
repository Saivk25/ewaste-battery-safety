#!/usr/bin/env bash
# CPU part of the v3 rerun: outside-image tests, moving belt, then every simulation (same chain as after the fuzzy fix).
set -e
export PYTHONPATH=src PYTHONIOENCODING=utf-8
mkdir -p results/rerun_logs
python src/external_photo_test.py --batsort C:/ewaste_ext/batsort/repo --xbat C:/ewaste_ext/xbat > results/rerun_logs/external.log 2>&1; echo "external $(date +%T)"
python src/demo_photo_test.py --photos C:/ewaste_ext/demo_photos > results/rerun_logs/demo_photos.log 2>&1; echo "demo photos $(date +%T)"
bash scripts/rerun_after_fuzzy_fix.sh
echo "CPU part done $(date +%T)"
