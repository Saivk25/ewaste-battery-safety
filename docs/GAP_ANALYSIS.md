# Review-gap analysis

## Resolution (10 Oct 2026, second round) -- reviewer attacks other than hardware

| Attack | Fix | Result |
|---|---|---|
| Only 3 Li-ion types; no Ni-MH / Ni-Cd photos | `src/external_photo_test.py` on the BatSort / Singapore Battery Dataset (Zhao et al. 2024, 10 battery types, web photos never used here) | Coarse CNN at the divert threshold: Ni-MH 21/21, Ni-Cd 24/24, all types 355/360 (97.2 % after removing near-duplicates of our data). The divert uses the coarse "Battery" decision, which is not Li-ion specific |
| Camera never saw a real device | XBAT+ v1.0 RGB (Rukundo et al. 2026, 421 webcam photos of real battery-containing WEEE from Irish collection drums) | Camera alone diverts 301/420 battery-containing devices; the rest depend on the sensor hood |
| Camera never saw a real belt | `src/detector_realbelt.py` on ZeroWaste-w frames (Bashkirova et al. 2022, a working recycling-plant conveyor, no battery class), split by video: 264 frames from 9 videos for fine-tuning, 136 from 7 other videos for testing | Detector v2 drew a false battery box (white paper read as a pouch cell) in 17.6 % of real frames. Fine-tuned v3: 0.7 % (1/136), pasted held-out batteries found 99-100 % up to 75 % covered, original val mAP50 unchanged (0.995). Batteries are still pasted photos |
| Rules, thresholds and alpha are hand-set | `src/pub_sensitivity.py`: every constant x0.8 / 0.9 / 1.1 / 1.2, one at a time, held-out seeds | No effect (recommended gate): alpha, T_CORR, both dwell times, M_PERSIST. Small: temperature sets (miss rate +0.023), H2 sets (1 stop lost). Sensitive: Warning threshold (+20 %: lag20 misses 0.029 -> 0.158), Critical threshold (-10/-20 %: 30-61 of 120 benign runs falsely Critical), C_GATE (+20 %: 37/120), CO2 sets (-20 %: 15/120), E-stop threshold (+10/+20 %: 5-6 fewer stops before runaway). These five need commissioning on site |
| Gas comes from a model | `src/pub_gas_sensitivity.py`: H2 or CO2 x0.25-2, delayed 30/60 s, removed, or rescaled to the literature H2 : CO2 ratio (0.28, Mathieu et al. 2022; the model gives 5.3) | Miss rates unchanged except the literature ratio (0.042 / 0.064 instead of 0.006 / 0.029); stops before runaway kept. Without gas (or gas delayed 60 s) a cell arriving at 90 C still stops in 4.1 s |
| A temperature above 500 C fell out of every fuzzy set (found 10 Oct while proof-reading) | Open-shoulder membership functions (`src/fuzzy_inference.py`, regression tests); every simulation result re-run (`scripts/rerun_after_fuzzy_fix.sh`) | With no gas, a cell arriving at 90 C now stops in 4.1 s (before: never stopped). Main tables moved by at most 0.1 s; belt result unchanged (0/330) |
| Main thermal model's onsets 40-90 C too low | `src/pub_thermal_compare.py`: same seeds on the literature-calibrated model | Stop margin 8-22 s on the main model vs 113-174 s calibrated; every calibrated run stopped before runaway. The main-model numbers are the conservative ones. On the main model with IR lag, the plain rule stops 27/27 before runaway, ours 23/27 |
| Camera missed most swollen or damaged batteries (outside photos) | Coarse model v3 (`src/build_coarse_v3.py`): 35 hand-vetted CC-licensed photos of swollen, corroded and leaking batteries, split by photographer/battery 22/6/7, outside test photos excluded; threshold re-chosen on validation (0.1113) | Damaged test batteries 5/7 (was 3/7), outside swollen photos 3/5 (was 2/5), intact 259/259 kept; cost: 21 vs 18 false alarms of 426, look-alikes correct 4/12 vs 7/12 |
| Intact batteries with P < 0.5 shredded on the belt | Divert threshold = validated detection threshold 0.1411 | 0/30 shredded (was 3/30), 1/30 benign items to the bin |

## Resolution (10 Oct 2026) -- moving-belt line test

`src/conveyor_sim.py`, results in `results/conveyor/summary.txt` (held-out seeds 100-129, 30 items per scenario,
17 scenarios x 6 systems), paper v5 Section 9.12 / Table 18, dashboard tab **Conveyor Line**.

| Gap | Fix | Result |
|---|---|---|
| Belt speed, sensor placement and distance to the shredder not modelled | Belt at 0.25 m/s: camera 0 m, sensor hood 0.25-1.0 m, diverter 1.5 m, shredder 2.5 m; outcomes counted per item | Failing cells that entered the shredder while self-heating: camera-only 210/330, plain rule 60/330, **ours 0/330** (47/330 without the diverter) |
| Gas is diluted on an open belt | Dilution 0.2 under the hood, 0.5 in the bin, 0.03 open; MQ sensor lag; 0.5 s averaging, scaled back by the commissioning dilution | Without compensation diluted CO2 capped the fuzzy risk at about 0.69 and E-stop was unreachable (bug, fixed). Outcomes unchanged, but belt time lost on fast hidden cells 2.2 s with compensation vs 6.1 s without |
| Ambient temperature | Temperature fed as rise above measured ambient | At 45 C ambient no outcome changes; it only reduces belt time lost (stated in the paper) |
| Only one response (stop the shredder) | Tiered response: divert when P_battery >= 0.1411 or the hood reaches Warning; diverted items go to a bin with its own IR + gas sensors; E-stop only for a confirmed hazard; FSM speed drives the belt (100 / 60 / 10 / 0 %) | Warm item + gas burst and 45 C ambient + burst: rule stops the belt 30/30, ours diverts the one item 30/30. Benign: 1/30 false diverts (to the bin), 0 false stops |
| Battery hidden inside a device | Sensor hood triggers the divert even with no camera detection | Hidden failing cell at 60 C: rule and camera-only shred 30/30; ours diverts 30/30, bin alarm 7.0 s before runaway |
| Intact batteries the camera scored below 0.5 were shredded (3/30) | Divert at the detection threshold already chosen on validation, P_battery >= 0.1411 (was 0.5) | Test: 100 % of batteries diverted, 4.2 % of non-batteries (validation 99.91 % / 2.9 %). Intact batteries shredded 0/30 (was 3/30); benign items falsely diverted 1/30 (was 0) |

Still open: about 4 % of non-batteries go to the reject bin (a cheap error: no belt stop); cells that
arrive at 80-90 C are in runaway within seconds and can only be stopped, not diverted; the belt model is a simulation.

## Resolution (9-10 Oct 2026) -- journal-readiness and Shark Tank rounds

| Gap | Fix | Result |
|---|---|---|
| Weak baselines | B0 camera-only stop, B2 + debounce (tuned), CUSUM (tuned), logistic regression (`src/pub_experiments.py`, `src/pub_fixes.py`) | Proposed 0.0064 vs B2-D 0.0435 (Holm p = 0.001); logistic perfect in-distribution, 0.25 missed with 60 s temperature lag |
| Assumption-breaking cases untested | temperature lag 5-120 s, hot benign objects, warm item + gas bursts, already-failing cells | warm item -> Critical 30/30 with the original gate; fixed by the retuned gate (0/30, held-out) |
| One operating point per baseline | parameter sweeps of every method (`results/pub/fixes_tradeoff.csv`, paper Fig. 4) | original gate = tuned debounce without camera; retuned gate better (0 false escalations) |
| Simulator built by us | literature-calibrated thermal model; 7 measured cell curves (Kwak et al. 2026) | all stops before runaway in both |
| Statistics | paired bootstrap CIs, Cohen dz, Holm correction | `results/pub/stats.csv` |
| One CNN run, no backbone comparison | 3 seeds x EfficientNet-B0, MobileNetV2, EfficientNet-B3 | B0 the only backbone with 259/259 in every run |
| Camera sees clean photos | YOLOv8n detector + cluttered composite frames; v1 (no negatives) boxed 99 % of empty frames, v2 retrained | `results/pub/detector_summary*.json` |
| Sensor failures | `src/sensor_health.py` (stuck, dead, dropout, camera loss, drift + auto-zero); FSM bug fixed (Sensor_Fault ignored the risk) | `results/pub/faults_summary.txt`, `tests/test_sensor_health.py` |
| Cost, speed, placement, payback | `src/business_case.py`, `src/bench_speed.py`, dashboard Business Case tab | `docs/SHARK_TANK_REVIEW.md` |
| Wrong software versions in paper | PyTorch 2.13.0 / CUDA 13.2 | paper Section 8.1 |

Still open: hardware validation, real belt images, real gas data, NiMH/NiCd photos, drift compensation on a
continuously full belt, a controller watchdog.

## Resolution (8 Oct 2026) -- what was fixed

| Gap | Fix | Result |
|---|---|---|
| Coarse CNN recognised only 4.4% of the RecyBat24 Li-ion photos the pipeline feeds it | Coarse dataset v2 (`src/build_coarse_v2.py`): one RecyBat24 image per source photo added to Battery, splits reused, MD5 check (0 duplicates); retrained | 91.68% on 685 test images; 100% of the 2,208 RecyBat24 test images labelled Battery |
| Three Garbage-12 batteries missed after retraining | Battery decision threshold `P_battery >= 0.1411`, chosen on validation (false-alarm rate kept at the v1 level) | 259 / 259 test batteries, 18 / 426 false alarms |
| Adaptive memory was no better than a tuned fixed memory | Held-out protocol (tune 200-229, evaluate 100-129) with battery-free noise constraints; new corroboration-gated adaptive memory (`src/gap_corroborated_memory.py`) | Hazard FNR 0.0412 (= best possible) with 0/30 false Critical in hard noise; old rule 0.1012 and 17/30 |
| Stale lambda claim | Old rule replaced; documented as `mode="exp"` | -- |
| Time on the belt | Already-failing-cell scenario in the final validation | Proposed stops in 10.2 / 3.1 / 3.1 s; B2 faster for the two hottest cases (dwell-time floor ~3 s) |
| Simulated backbone comparison still run by `main_pipeline.py` | Removed | -- |
| Literature 2022-2025 | Three checked sources added (Liu et al. 2025, Giannikos et al. 2025, Ullah et al. 2026) | -- |
| Confidence intervals, calibration, per-source fine accuracy | `src/gap_cnn_checks.py`, `src/gap_fine_sources.py` | Wilson / Clopper-Pearson intervals; ECE ~0.125; 164/164 fine source photos correct |

Final numbers for every scenario: `results/final_stats.json`. Still open: hardware validation, real recycling-line images,
NiMH/NiCd photos, backbone comparison on corrected data, flatline/drift/camera-dropout fault detection, sensor realism.

---

## Original findings (7 Oct 2026)

Experiments run on the current, bug-fixed code to test weaknesses a reviewer could raise.
Scripts: `src/gap_cnn_checks.py`, `src/gap_temporal_sweeps.py`, `src/gap_prewarmed.py`.
Outputs: `results/gaps/`. The deck, paper and guide were **not** changed by this work.

## 1. The coarse CNN does not recognise the Li-ion photos the simulation feeds it (CRITICAL)
- All 95 coarse-test batteries are Garbage-12 images (`g12_*`: AA/9 V/car batteries). Recall 95/95; false-negative rate 95% Clopper-Pearson upper bound 3.8%.
- The same coarse model on the 2,208 RecyBat Li-ion test photos (18650 / pouch / LiPo): argmax = "Battery" for only **4.4%** (mean P_battery 0.108, 0% above 0.5, 89% below 0.2). Typical prediction: Electronics.
- The simulation's `RealCNN` feeds RecyBat photos to the coarse model and *forced the label "Battery"* whatever the model said. Fixed in `src/adaptive_risk.py` (label now the real argmax; probability unchanged, 72 tests pass).
- Consequence: "0 of 95 missed" holds for consumer-battery photos only; the hierarchical gate (coarse then fine) is not validated for the three Li-ion types.
- Calibration (coarse test set): ECE 0.128, mean confidence 0.76 vs accuracy 0.877 (under-confident). Accuracy 87.72% has Wilson 95% interval 84.6 to 90.3%.

## 2. Adaptive memory vs a tuned fixed memory (the "fixed beta" confound) (CRITICAL)
Same fuzzy series S(t), same FSM (dwell times), no CNN fusion, 30 seeds, Hazard FNR:

| Filter | Hazard FNR | Adaptive(0.85, 0.15) minus this | paired p |
|---|---|---|---|
| fixed 0.50 | 0.0541 | +0.047 | <0.0001 |
| fixed 0.70 | 0.0762 | +0.025 | <0.0001 |
| fixed 0.80 | 0.0954 | +0.006 | 0.023 |
| fixed 0.85 | 0.1044 | -0.003 | 0.113 (n.s.) |
| **adaptive (0.85, 0.15)** | **0.1012** | | |
| adaptive lambda 0.5 | 0.0954 | | (equals fixed 0.80) |
| adaptive lambda 2.5 | 0.0712 | | |
| adaptive lambda 5 | 0.0669 | | (equals fixed 0.60) |

- Against a fixed beta of 0.85 (same FSM, no fusion) the adaptive filter is **not** significantly better (-0.003, p = 0.11). Lower fixed beta values beat it clearly.
- Decomposition of the headline 0.1194 to 0.0882 (Proposed vs B3): B3 as coded (own thresholds, no dwell) 0.1194; fixed 0.85 with the FSM 0.1044; adaptive 0.1012; plus CNN fusion with the seed-77 camera draw 0.0882. Most of the gap is FSM logic and fusion, not adaptivity.
- Lambda sensitivity (fresh): larger lambda lowers Hazard FNR (0.101 at 0.15 to 0.071 at 2.5) and raises the peak risk in the battery-free spike test (0.32 to 0.42); no false Emergency Stop in any variant (0/30). The earlier "lambda changes little" claim does not hold on the current code.
- Spike test (battery-free): peak R mean 0.308 (fixed 0.85), 0.318 (adaptive 0.15), 0.461 (fixed 0.50). The "alert share" is dominated by a ~15 s start-up transient and is not informative.
- Honest reading: in this simulator the adaptive coefficient behaves like a lower fixed memory; it does not show a benefit that a tuned fixed beta cannot give. The headline p < 0.0001 versus B3 stays true but should not be attributed to adaptivity.

## 3. Stalled seeds
Seeds 122 and 129 never leave ambient (no hazardous steps, FNR = 0 for every system); seed 107 peaks at 68 C. The Proposed-vs-B3 result survives excluding them: n = 27, -0.0289, t = -5.89, p = 3e-6.

## 4. Camera probability in the simulation
Per-seed P_battery drawn from the coarse model's real outputs on RecyBat photos (mean 0.13): Hazard FNR 0.0707 (P = 0: 0.1012). With a confident camera (P = 0.90): 0.0000. The camera helps a lot when confident, but the trained coarse model is rarely confident on Li-ion photos.

## 5. Time on the belt
Batteries that arrive already failing (T0 at arrival, faster runaway), 30 seeds, seconds from arrival:

| T0 / rate | Proposed stop | fixed 0.85 stop | B2 stop |
|---|---|---|---|
| 25 C / x1 (main scenario) | 230.9 | 231.0 | 239.5 |
| 60 C / x3 | 11.3 | 11.4 | 13.9 |
| 80 C / x6 | 3.9 | 3.3 | 1.8 |
| 90 C / x10 | 5.2 | 5.2 | 0.0 |

- The FSM adds a minimum of about 3 s by design (2 s Warning + 1 s Critical dwell). For violently failing cells B2 stops faster (immediate thresholds).
- Belt speed, sensor placement and the distance to the shredder are not modelled. *(Addressed 10 Oct 2026: see the moving-belt section at the top.)*

## 6. Not tested / not fixed
FSM threshold and membership sensitivity; third item after a latched stop; other chemistries and heating rates; sensor realism (IR surface vs core temperature, MQ cross-sensitivity); safety rating; literature coverage (6 papers); roles; project report.
