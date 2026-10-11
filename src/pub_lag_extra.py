"""pub_lag_extra.py -- longer IR-temperature lags (60 s, 120 s) on held-out seeds, motivated by reports
that cells vent hydrogen minutes before the case heats (Gardner et al. 2025). Uses the tuned baselines
from results/pub/tuned.json. Run after pub_experiments.py:  python src/pub_lag_extra.py -> results/pub/eval_lag_extra.csv"""
import sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8"); sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import pub_experiments as pe

if __name__ == "__main__":
    from multiprocessing import Pool
    extra = json.load(open("results/pub/tuned.json"))["tuned"]; extra["dwell"] = False
    cam = pe.Camera()
    tasks = pe.camera_tasks(cam, ["lag60", "lag120"], pe.EVAL, extra)
    with Pool(2) as pool:
        ev = pd.DataFrame(sum(pool.map(pe.work, tasks), []))
    ev.to_csv("results/pub/eval_lag_extra.csv", index=False)
    print(ev.groupby(["scenario", "system"])[["Hazard_FNR", "T_stop_s", "stop_margin_s"]].mean().round(3).to_string())
