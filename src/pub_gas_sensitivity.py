"""
pub_gas_sensitivity.py -- does the result depend on the made-up gas model?

No open dataset has a gas channel for the measured cells, so the vent gas is a model
(sensor_simulation: H2 from 70 C, CO2 from 120 C). Its composition is not the literature's: at
150 C it gives H2 : CO2-excess of about 2.5, while the mean of 40 published vent-gas analyses is
14.4 % H2 : 51.6 % CO2 = 0.28 (Mathieu et al., Energy & Fuels 2022; H2 ranges 0-36 % between studies).

So the vent gas of every hazard run is transformed after simulation and the system is re-run:
  h2 x f, co2-excess x f      one gas weaker / stronger (f = 0.25, 0.5, 2)
  lit_ratio                   H2 scaled so H2 : CO2-excess matches 0.28 at runaway onset
  delay 30 / 60 s             gas reaches the sensor late
  no_gas                      H2 = 0, CO2 = 400 ppm (gas sensors see nothing)
Benign runs are not transformed (their gas is sensor bursts, not vent gas). Camera probability is
taken from the untransformed run. Held-out seeds 100-129, both gates.

Run: python src/pub_gas_sensitivity.py      Outputs: results/pub/gas_sensitivity.csv, gas_sensitivity_summary.txt
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pub_experiments import scen, Camera, OUT, DT
from pub_sensitivity import fuzzy, run_one, GATES

SCEN = ["single", "lag20", "pre60", "pre80", "pre90"]
SEEDS = list(range(100, 130))
LIT_RATIO = 14.4 / 51.6
VARIANTS = ["base", "h2x0.25", "h2x0.5", "h2x2", "co2x0.25", "co2x0.5", "co2x2", "lit_ratio", "delay30", "delay60", "no_gas"]


def model_ratio(df):
    """H2 : CO2-excess in the model at the first runaway step (where the vent is strongest)."""
    i = np.where(df.phase.isin(["Phase3_Runaway", "Phase4_Fire"]).values)[0]
    i = i[0] if len(i) else len(df) - 1
    return df.mq2_ppm.values[i] / max(df.mq135_ppm.values[i] - 400.0, 1.0)


def transform(df, v):
    d = df.copy(); h, c = d.mq2_ppm.values.astype(float), d.mq135_ppm.values.astype(float)
    ex = c - 400.0
    if v.startswith("h2x"): h = h * float(v[3:])
    elif v.startswith("co2x"): c = 400.0 + ex * float(v[4:])
    elif v == "lit_ratio": h = h * LIT_RATIO / model_ratio(df)
    elif v.startswith("delay"):
        n = int(int(v[5:]) / DT); h = np.r_[np.zeros(n), h[:-n]]; c = np.r_[np.full(n, 400.0), c[:-n]]
    elif v == "no_gas": h = np.zeros_like(h); c = np.full_like(c, 400.0)
    d["mq2_ppm"], d["mq135_ppm"] = h, c
    return d


def work(task):
    name, seed, P = task
    df0, arrive = scen(name, seed)
    rows = []
    for v in VARIANTS:
        df = transform(df0, v) if v != "base" else df0
        S = fuzzy(df)
        for gate in GATES:
            rows.append(dict(scenario=name, seed=seed, variant=v, gate=gate, model_ratio=model_ratio(df0),
                             **run_one(df, arrive, S, P, gate, "none", 1.0)))
    return rows


def summarise(d):
    L = [f"Gas-model sensitivity, held-out seeds 100-129. Model H2 : CO2-excess at runaway onset: median "
         f"{d[d.variant == 'base'].model_ratio.median():.2f} (literature mean {LIT_RATIO:.2f}).",
         "FNR = hazardous time with no alert; stop<runaway of 30 runs; stop = mean seconds from arrival.", ""]
    for gate in GATES:
        L.append(f"== {gate}")
        L.append(f"   {'variant':10s} " + " ".join(f"{s:>26s}" for s in SCEN))
        for v in VARIANTS:
            cells = []
            for s in SCEN:
                x = d[(d.gate == gate) & (d.variant == v) & (d.scenario == s)]
                cells.append(f"{x.Hazard_FNR.mean():.4f} {int((x.stop_margin_s > 0).sum()):2d}/30 {x.T_stop_s.mean():6.1f}s")
            L.append(f"   {v:10s} " + " ".join(f"{c:>26s}" for c in cells))
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    from multiprocessing import Pool
    cam = Camera(); tasks = []
    for name in SCEN:
        for seed in SEEDS:
            df, _ = scen(name, seed); tasks.append((name, seed, cam.series(df)))
    with Pool(int(__import__("os").environ.get("NPROC", 14))) as pool:
        rows = [r for rs in pool.imap_unordered(work, tasks) for r in rs]
    d = pd.DataFrame(rows).sort_values(["gate", "variant", "scenario", "seed"])
    d.to_csv(OUT / "gas_sensitivity.csv", index=False)
    txt = summarise(d); print(txt)
    (OUT / "gas_sensitivity_summary.txt").write_text(txt + "\n", encoding="utf-8")
