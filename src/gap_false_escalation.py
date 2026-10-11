"""gap_false_escalation.py -- split false escalations in the battery-free hard/harder noise tests into
Critical (10% speed) vs Emergency Stop, for the main filter settings. Held-out seeds 100-129.
Run from the repo root: python src/gap_false_escalation.py -> results/gaps/false_escalation_split.csv"""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from sensor_simulation import run_noise_stress_scenario
import gap_memory_redesign as g
V = [("fixed", (0.30,)), ("fixed", (0.50,)), ("fixed", (0.70,)), ("fixed", (0.85,)), ("fixed", (0.95,)),
     ("adaptive", (0.85, 0.15)), ("adaptive", (0.95, 0.15)), ("adaptive", (0.75, 8))]
def one(seed):
    out = []
    for name in ("hard", "harder"):
        dn = run_noise_stress_scenario(seed=seed + 1000, **g.NOISE[name]); Sn = g.fuzzy_series(dn); on = dn.object_present.values
        for kind, p in V:
            st = g.fsm_states(dn, g.apply(kind, p, Sn, on))
            out.append(dict(seed=seed, noise=name, kind=kind, params=str(p), critical=bool((st == "Critical").any()),
                            estop=bool((st == "Emergency_Stop").any()), time_crit_s=float((st == "Critical").sum() * 0.1)))
    return out
if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(6) as p:
        df = pd.DataFrame(sum(p.map(one, range(100, 130)), []))
    df.to_csv("results/gaps/false_escalation_split.csv", index=False)
    print(df.groupby(["noise", "kind", "params"]).agg(runs_critical=("critical", "sum"), runs_estop=("estop", "sum"), mean_s_in_critical=("time_crit_s", "mean")).to_string())
