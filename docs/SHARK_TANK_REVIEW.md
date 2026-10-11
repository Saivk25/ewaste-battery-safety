# Shark Tank review: how the project could fail, and what was done about it

A hostile-panel walk-through (investor, plant manager, safety engineer, examiner). Each row gives the attack,
the honest answer, the evidence in this repo, and whether it was fixed in software. Hardware validation is out of
scope by decision: everything below is simulation, public data or published figures.

Numbers are held-out results (tuning seeds 200-229, evaluation seeds 100-129) unless stated otherwise.

## 1. "Is the problem real and big enough?"

| Attack | Answer | Evidence |
|---|---|---|
| Battery fires are rare, why pay for this? | They are frequent and expensive. Fire Rover's industry report counts 448 publicly reported waste/recycling facility fires in the US and Canada in 2025 with about $2.5 billion in damage; EuRIC puts the average thermal event at a European facility at about EUR 1.3 million; the UK Environmental Services Association attributes about 70 % of recycling-centre fires to batteries, about GBP 1 billion a year. Insurance premiums have risen sharply, and some insurers have withdrawn. | [Resource Recycling 2026](https://resource-recycling.com/e-scrap/2026/03/27/report-pegs-fire-losses-at-2-5b-in-us-and-canada-recycling-industry/), [Waste360](https://www.waste360.com/industry-insights/waste-industry-still-battling-battery-fires-as-losses-estimate-2-5-billion), [Batteries News (UK)](https://batteriesnews.com/battery-fires-at-recycling-centres-are-costing-uk-1bn-a-year-figures-show/), Terazono et al. 2024 (paper ref. 26) |
| Who already sells this? | Thermal-camera / IR fire detection (FLIR, PYROsmart, about 500 plants) and suppression (Fire Rover) react once something is hot or burning; X-ray / AI sorters remove batteries upstream but are major capital purchases. Our niche is a low-cost supervisory layer at the shredder infeed that combines a camera with gas and temperature and slows the shredder before ignition. It complements suppression; it does not replace it. | [FLIR waste & recycling](https://www.flir.com/instruments/fire-prevention/waste-and-recycling/), [BLW Visser PYROsmart](https://blwvisser.com/en/service/fire-detection/applications-uses/), [Fire Rover](https://firerover.com/industries/wasteandrecycling/) |

## 2. Cost

| Attack | Answer | Evidence / fix |
|---|---|---|
| What does a line cost? | Prototype-grade sensors and a Jetson Orin Nano Super: about **$280-495** per line. Industrial-grade (fixed H2 detector, Optris IR pyrometer, industrial camera): about **$2,500-4,400**. A professional thermal camera alone is quoted at up to $2,500 per unit plus monitoring. | `src/business_case.py` (prices with sources in its comments); dashboard **Business Case** tab |
| Hobby MQ sensors in a factory? | Not for deployment: MQ sensors are indicators, drift, need recalibration every few months and react to many gases. The design is sensor-agnostic (fuzzy inputs are ppm and C), so the industrial tier swaps them for certified detectors. | MQ datasheets; Section "Maintenance" below |
| Does it pay back? | It depends on two plant numbers we cannot measure: fires per year and how many the system would prevent. The calculator makes the trade-off explicit; with one fire in five years, EUR 1.3 M per fire and 30 % prevented, even the industrial tier pays back in under a year, unless false stops are frequent. | Business Case tab (assumptions labelled) |
| What do false stops cost? | More than the hardware. One false 10-minute stop per shift at $2,000/h is about $200,000 a year at 600 shifts. That is why false escalations are measured in every test. | Business Case tab; `results/pub/eval.csv` |

## 3. Speed and placement

| Attack | Answer | Evidence / fix |
|---|---|---|
| Can it keep up with the belt? | Yes, with margin. Measured on an idle laptop (`results/pub/speed.json`): EfficientNet-B0 14 ms per item on the RTX 5060 GPU, 28-32 ms on the laptop CPU (about 31-35 items/s); MobileNetV2 7 ms GPU / 18-20 ms CPU; YOLOv8n 13 ms GPU / 44 ms CPU per 640 px frame (23 frames/s); fuzzy + memory + FSM 1.2 ms per 0.1 s control step (1 % of the budget). A 0.7 m/s belt with items every 0.3 m needs about 2-3 classifications per second. An edge board (Jetson / Raspberry Pi) was not measured and will be slower than this laptop. | `src/bench_speed.py` |
| Where must the sensors sit? | Far enough upstream that the shredder has stopped before the item arrives: belt speed x (detection-to-stop time + shredder rundown). At 0.7 m/s, a 4.2 s stop and a 5 s rundown that is **6.4 m**. The controller's own 3-4 s dwell is a deliberate false-alarm guard; shorter dwell halves the stop time but multiplies false escalations (dwell table, paper Table 13). | Business Case tab; paper Section 9.10 |
| What happens on a moving belt? | Simulated end to end (`src/conveyor_sim.py`): camera, sensor hood, diverter and shredder at fixed positions on a 0.25 m/s belt, diluted gas, sensor lag, ambient temperature. Suspect items are diverted to a monitored bin; the belt stops only for a confirmed hazard. Failing cells that reached the shredder while self-heating: camera-only 210/330, plain rule 60/330, ours 0/330. | `results/conveyor/summary.txt`, paper Table 18, dashboard **Conveyor Line** |
| A punctured, already-hot cell? | Stopped in 3.1 s (gate of Tables 9-13) or 4.2 s (recommended gate) from arrival; the plain rule is faster (0 to 1.8 s) because it stops on one threshold crossing and pays for it with false stops. | paper Tables 10, 13, 17 |

## 4. Does it actually work, and against what?

| Attack | Answer | Evidence |
|---|---|---|
| "You tuned on your test data." | No: every tunable setting was chosen on seeds 200-229 and reported once on 100-129. | `src/pub_experiments.py`, `src/pub_fixes*.py` |
| A simple debounced rule would do. | With the camera, the full system misses 0.0064 of hazardous time against 0.0435 for a tuned 3 s debounce (Holm p = 0.001). Without the camera, the original gate was **not** better than the debounce; the retuned gate (8 s persistence + rate of rise) reaches 0.0441 misses with **zero** false escalations in the hard, harder and warm-item tests, which no debounce setting achieves. | paper Fig. 4, Table 17 |
| Why not just train an ML model? | A logistic model trained on the simulator is perfect in-distribution, but with the temperature 60 s late it misses 0.25 of hazardous time (ours 0.056) and it falsely stops on every 60 C benign item. | paper Section 9.11 |
| Your simulator is made up. | Checked two ways: a second thermal model with published onset temperatures (all systems stop before runaway, ours 174 s early) and 7 measured cell temperature curves from Kwak et al. 2026 (21/21 runs stopped before runaway, about 20 min early). Gas still comes from a model: the open dataset has no gas channel. | `src/pub_fixes.py`, `src/pub_realtrace.py`, dashboard **Real Measured Cell** |
| Your camera saw clean photos. | True. A YOLOv8n detector was trained for cluttered frames: v1 (battery photos only) boxed 99 % of empty frames; v2 (negatives + clutter from training-split photos) finds the battery in 99.7 % of frames even at 75 % occlusion with 1 % false alarms, against 0.7-3 % recall for the whole-frame classifier. Train and test frames come from the same composite generator, so real belt images are still needed. | `src/detector_yolo.py`, paper Table 15 |
| You hand-tuned the rules. | Every hand-set constant was moved +/-20 % on held-out seeds. Alpha, T_CORR, M_PERSIST and the dwell times do not matter, the temperature and H2 sets matter little; the Warning, Critical and Emergency thresholds, C_GATE and the CO2 sets do, so they are commissioning parameters, not universal constants. | `results/pub/sensitivity_summary.txt` |
| Your gas is invented. | Rescaling either gas 4x down or 2x up, delaying it 60 s or removing it leaves the miss rate unchanged; temperature carries detection. With the literature gas composition misses rise from 0.006 to 0.042, with every stop still before runaway. | `results/pub/gas_sensitivity_summary.txt` |
| Your thermal model is wrong. | Its onsets are too low, which makes it the hard case: margins 8-22 s, against 113-174 s on a model with published onsets. | `results/pub/thermal_compare_summary.txt` |
| Never tested on a real belt. | Frames from a working recycling plant (ZeroWaste): the detector first boxed white paper as a pouch cell in 17.6 % of frames; after fine-tuning on other videos of the plant, 0.7 %, still finding 99-100 % of pasted batteries. No real battery on a working belt was available. | `results/pub/detector_realbelt_summary.json` |
| Only Li-ion? | The divert uses the coarse Battery class: 21/21 Ni-MH and 24/24 Ni-Cd photos from an outside dataset are detected. | `results/external/summary.txt` |
| One lucky training run? | EfficientNet-B0 over 3 seeds: 91.4 % +/- 0.2, all 259 test batteries found in every run; MobileNetV2 and EfficientNet-B3 are 1-2 points more accurate but each missed one battery. | paper Table 14 |

## 5. Failure modes (what breaks it) and what was fixed

| Failure | Before | Fix (software) | Result |
|---|---|---|---|
| Warm benign item + gas bursts | false Critical in 30/30 runs | gate retuned on tuning seeds: 8 s gas persistence + rate-of-rise corroboration | 0/30 on held-out seeds; cost about 1 s on already-hot cells, up to 8 s on a gas-only rise |
| IR sensor frozen | missed 0.064 of hazard time, stopped 0.6 s before runaway | `src/sensor_health.py`: stuck-sensor alarm, fail-safe corroboration | alarm after about 10 s, plant slowed for the whole hazard (0 missed counting the fault state), stop 2.1 s before runaway; no false fault on healthy runs |
| Gas sensor dead | temperature path still stops it (0.0064 missed) | physics cross-check (hot cell, no H2) | alarm once the cell passes 140 C (about 250 s after arrival in this model) |
| Camera dropout | sensors alone (0.044 missed) | camera-loss alarm, Sensor_Fault (50 % speed) | operator alerted at once |
| Missing readings (3 s every minute) | n/a | last value held, fault flagged | no crash; 0.0021 missed counting the fault state |
| Gas sensor drift | over-sensitive or false escalations | auto-zero on empty belt + recalibration request | fixed when the belt has gaps; **a continuously full belt needs scheduled zero checks** |
| Sensor fault during a real hazard | **bug: Sensor_Fault ignored the risk and kept the shredder at 50 %** | fault never raises speed; remaining sensors can still reach Critical / E-stop | regression tests in `tests/test_sensor_health.py` |
| Battery hidden behind another item | camera-only stop missed 5/28 runs | sensors + fusion still catch it | 0.0064 missed with fusion |
| Gas diluted on an open belt | diluted CO2 capped the fuzzy risk at about 0.69: **E-stop unreachable** (bug) | gas averaged 0.5 s and scaled back by the commissioning dilution | E-stop reachable; belt time lost on fast hidden cells 2.2 s instead of 6.1 s |
| Warm item + gas burst on the belt | plain rule stops the whole belt 30/30 | divert the one item to a monitored bin instead | 30/30 diverted, 0 belt stops; 1/30 benign items falsely diverted to the bin |
| Battery hidden inside a device | camera-only and rule shred it (30/30 at 60 C) | sensor hood triggers the divert | 30/30 diverted, bin alarm 7.0 s before runaway |
| Objects at 90 C or hotter | stop | none by design: a 90 C object at a shredder infeed is itself an ignition risk | documented |

## 6. Maintenance, safety and liability

| Attack | Answer |
|---|---|
| Who recalibrates the gas sensors? | MQ-class sensors drift and need recalibration every few months; the monitor flags drift and zeroes it on an empty belt. Industrial detectors have their own calibration schedule. |
| Is it safety-certified? | No. It is a supervisory layer: it slows or stops the shredder through the plant's existing, certified emergency-stop circuit (functional-safety standards such as IEC 61508 / ISO 13849 apply to that circuit, not to this software). The latch can only be cleared manually, and a sensor fault can no longer unlatch or delay a stop (both tested). |
| What if the software crashes? | The plant's hardware interlocks and suppression remain the last line; a watchdog that stops the line when the controller stops reporting is a deployment requirement (not built). |

## 7. What still cannot be claimed

- No hardware, no real conveyor images, no real gas data (the measured cells have temperature only). The moving-belt test is a simulation.
- About 5 % of non-battery items are sent to the reject bin by mistake (divert threshold P_battery >= 0.1113); cell-shaped objects such as capacitors and cans are the usual cause.
- The fraction of real fires the system would prevent is unknown; the payback calculator takes it as an assumption.
- Only three Li-ion types in the fine classifier. Ni-MH / Ni-Cd were checked only on 45 outside web photos (all detected as batteries), not trained on.
- The clutter test uses composite images.

## 8. Thirty-second pitch (honest version)

Battery fires cost the recycling industry billions a year, and most start at the shredder. We put a camera, a gas
sensor and an IR thermometer at the infeed for a few hundred dollars per line in prototype form (a few thousand
with industrial sensors), and a supervisory controller that slows the shredder on early evidence and stops it
before runaway, while ignoring the gas bursts and warm items that would make a simple threshold rule stop the plant
all day. Tested on held-out simulations, a literature-calibrated model and measured cell curves; not yet on
hardware, which is the next step.
