"""
pub_realtrace.py -- replay MEASURED thermal-runaway temperature curves through the pipeline (fix R6).

Data: Kwak et al., "Multi-modal thermal runaway dataset of fresh and aged lithium-ion battery cells
and modules", Scientific Data (2026), OSF doi:10.17605/OSF.IO/C2HNQ (Dataset_TR.zip, cell-level
surface temperature, heater-driven abuse at 5.5 C/min). Download it into data/raw/kwak_tr/ (git-ignored,
not redistributed here).

The measured surface temperature drives everything that depends on temperature; H2 and CO2 come from
the calibrated gas model of pub_fixes.cal_sim (H2 from 115 C, CO2 from 150 C) because the dataset has
no gas channel. Phases use the literature landmarks (90 / 130 / 200 / 500 C). All settings are frozen
from the tuning seeds. Three noise seeds per trace (sensor noise and camera draw).

Run from the repo root:  python src/pub_realtrace.py  -> results/pub/realtrace_eval.csv, realtrace_summary.txt
"""
import sys, glob, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from pub_experiments import Camera, gated, fsm_run, b2_debounce, cusum, metrics, DT, ALPHA, OUT
from pub_fixes import gated_rate, cfnc_parts

RAW = Path("data/raw/kwak_tr/Dataset_TR/Cell")
TUNED = json.load(open(OUT / "tuned.json"))["tuned"]
FX = json.load(open(OUT / "fixes_tuned.json")) if (OUT / "fixes_tuned.json").exists() else None
SEEDS = (100, 101, 102)


def load(f):
    x = pd.read_excel(f, header=None).apply(pd.to_numeric, errors="coerce").dropna()
    t, T = x[0].values.astype(float), x[1].values.astype(float)
    t = t - t[0]
    end = t[np.argmax(T)] + 120.0                    # keep up to 2 min after the peak (drop the long cool-down)
    tt = np.arange(0.0, min(t[-1], end), DT)
    return tt, np.interp(tt, t, T)


def trace_df(tt, Treal, seed, arrive=20.0):
    import sensor_simulation as ss
    from pub_fixes import cal_sim  # noqa: F401  (same gas onsets as the calibrated model)
    class MQ2(ss.MQ2Sensor):
        def _temp_to_ppm(self, T): return 0.0 if T < 115.0 else min(60.0 * np.exp(0.042 * (T - 115.0)), 10000.0)
    class MQ135(ss.MQ135Sensor):
        def _temp_to_ppm(self, T): return 400.0 if T < 150.0 else 400.0 + 4600.0 / (1 + np.exp(-0.035 * (T - 230.0)))
    rng = np.random.RandomState(seed)
    mq2, mq135, ind = MQ2(seed + 1), MQ135(seed + 2), ss.InductiveProximitySensor(seed + 3)
    n0 = int(arrive / DT); rows = []
    T_all = np.r_[np.full(n0, ss.T_AMBIENT), Treal]
    for i, Tm in enumerate(T_all):
        b = i >= n0
        T = Tm + rng.normal(0, 0.5)                    # IR sensor noise on top of the measured surface temperature
        p2, v2 = mq2.read(Tm); p135, v135 = mq135.read(Tm)
        ph = ("Normal" if (not b or Tm < 90) else "Phase1_SEI" if Tm < 130 else "Phase2_Separator" if Tm < 200
              else "Phase3_Runaway" if Tm < 500 else "Phase4_Fire")
        rows.append(dict(time=round(i * DT, 2), temperature_C=round(T, 2), mq2_ppm=p2, mq2_voltage=v2, mq135_ppm=p135,
                         mq135_voltage=v135, metal_present=int(ind.read(b)), object_present=int(b), object_size_cm=10.0 if b else 0.0,
                         phase=ph, battery_present=int(b)))
    return pd.DataFrame(rows), arrive


def work(task):
    name, seed, df, arrive, P = task
    from gap_temporal_sweeps import fuzzy_series
    from baseline_comparison import baseline_rule_based, baseline_fixed_fuzzy
    S = fuzzy_series(df); T = df.temperature_C.values; obj = df.object_present.values
    fuse = lambda R: np.clip(R * (1 + ALPHA * P), 0, 1)
    sys_ = {"Proposed": fsm_run(df, fuse(gated(S, T, obj))),
            "Proposed_occluded": fsm_run(df, np.clip(gated(S, T, obj), 0, 1)),
            "B2_Rule": baseline_rule_based(df).state.values,
            f"B2_Debounce_{TUNED['b2_k']:g}s": b2_debounce(df, TUNED["b2_k"]),
            "B3_FixedFuzzy": baseline_fixed_fuzzy(df).state.values,
            f"CUSUM_h{TUNED['cusum_h']:g}": cusum(S, TUNED["cusum_h"])}
    if FX:
        sys_["ProposedR"] = fsm_run(df, fuse(gated_rate(S, T, obj, FX["W"], FX["r"])))
    return [dict(trace=name, seed=seed, system=k, **metrics(df, st, arrive), **cfnc_parts(df, st)) for k, st in sys_.items()]


if __name__ == "__main__":
    from multiprocessing import Pool
    files = sorted(glob.glob(str(RAW / "**" / "*Temperature.xlsx"), recursive=True))
    assert files, "download Dataset_TR.zip into data/raw/kwak_tr/ and unzip it"
    cam = Camera(); tasks = []; info = []
    for f in files:
        name = f"{Path(f).parent.parent.name}/{Path(f).parent.name}/{Path(f).stem.split('_')[0]}"
        tt, T = load(f)
        info.append(f"{name}: {len(T) * DT:.0f} s, start {T[0]:.1f} C, peak {T.max():.1f} C, reaches 200 C at {tt[np.argmax(T >= 200)]:.0f} s")
        for s in SEEDS:
            df, arrive = trace_df(tt, T, s)
            tasks.append((name, s, df, arrive, cam.series(df)))
    with Pool(2) as pool:
        ev = pd.DataFrame(sum(pool.map(work, tasks), []))
    ev.to_csv(OUT / "realtrace_eval.csv", index=False)
    lines = ["Measured traces (Kwak et al. 2026):"] + ["  " + i for i in info]
    lines.append("\nPer system over 7 traces x 3 noise seeds: Hazard FNR | stop margin before 200 C (s) | runs stopped before 200 C")
    for s, g in ev.groupby("system"):
        lines.append(f"  {s:<22} FNR {g.Hazard_FNR.mean():.4f}  margin {g.stop_margin_s.mean():7.1f}  stopped before runaway {int((g.stop_margin_s > 0).sum())}/{len(g)}")
    print("\n".join(lines)); open(OUT / "realtrace_summary.txt", "w", encoding="utf-8").write("\n".join(lines))
