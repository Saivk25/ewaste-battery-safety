"""
pub_sensitivity.py -- how much do the hand-set numbers matter?

One-at-a-time sensitivity of every hand-set constant of the proposed system: each is multiplied by
0.8, 0.9, 1.1 and 1.2 while all others stay at their paper values, on held-out seeds 100-129.

  fuzzy sets     every breakpoint of the temperature / H2 / CO2 membership functions (one input at a
                 time; done by feeding the input divided by the factor, which is identical)
  FSM            normal->warning, warning->critical, critical->emergency thresholds; Warning and
                 Critical dwell times
  memory gate    C_GATE, T_CORR, M_PERSIST
  fusion         alpha (CNN weight)

Both gates are run: the gate of the paper tables (G_m20) and the recommended gate (G_m80_W10_r0.2).
Nothing here is tuned: the point is whether the conclusions survive +/-20 % on every setting.

Run: python src/pub_sensitivity.py   (about 30 min on 16 processes)
Outputs: results/pub/sensitivity.csv, sensitivity_summary.txt
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pub_experiments import scen, Camera, fsm_run, metrics, ALPHA, DT, OUT
from pub_fixes import gated_rate

SCEN = ["single", "lag20", "pre60", "pre90", "hard", "harder", "hot60", "warm_bursts"]
HAZARD = ["single", "lag20", "pre60", "pre90"]
BENIGN = ["hard", "harder", "hot60", "warm_bursts"]
SEEDS = list(range(100, 130))
FACTORS = [0.8, 0.9, 1.0, 1.1, 1.2]
GATES = {"paper_gate": (20, None), "recommended_gate": (80, (10.0, 0.2))}
FSM_KEYS = ["normal_to_warning", "warning_to_critical", "critical_to_emergency"]
PARAMS = ["mf_temperature", "mf_h2", "mf_co2", *FSM_KEYS, "dwell_warning", "dwell_critical",
          "C_GATE", "T_CORR", "M_PERSIST", "alpha"]


def fuzzy(df, fT=1.0, fh=1.0, fc=1.0):
    from fuzzy_inference import MamdaniFIS
    fis = MamdaniFIS()
    return np.array([fis.infer(r.temperature_C / fT, r.mq2_ppm / fh, r.mq135_ppm / fc, bool(r.metal_present))[0]
                     for r in df.itertuples()])


def run_one(df, arrive, S, P, gate, param, f):
    import adaptive_risk as ar, supervisory_control as sc
    m, wr = GATES[gate]
    saved_ar = {k: getattr(ar, k) for k in ("C_GATE", "T_CORR", "M_PERSIST")}
    saved_th = dict(sc.THRESHOLDS)
    try:
        ar.M_PERSIST = m
        if param in ("C_GATE", "T_CORR"): setattr(ar, param, saved_ar[param] * f)
        if param == "M_PERSIST": ar.M_PERSIST = int(round(m * f))
        if param in FSM_KEYS: sc.THRESHOLDS[param] = saved_th[param] * f
        T, obj = df.temperature_C.values, df.object_present.values
        if wr is None:
            from pub_experiments import gated
            R = gated(S, T, obj)
        else:
            R = gated_rate(S, T, obj, *wr)
        alpha = ALPHA * f if param == "alpha" else ALPHA
        Rf = np.clip(R * (1 + alpha * P), 0, 1)
        dwell = None
        if param.startswith("dwell"):
            w, c = sc.MIN_DWELL[sc.State.WARNING], sc.MIN_DWELL[sc.State.CRITICAL]
            dwell = (w * f, c) if param == "dwell_warning" else (w, c * f)
        return metrics(df, fsm_run(df, Rf, dwell=dwell), arrive)
    finally:
        for k, v in saved_ar.items(): setattr(ar, k, v)
        sc.THRESHOLDS.clear(); sc.THRESHOLDS.update(saved_th)


def work(task):
    name, seed, P = task
    df, arrive = scen(name, seed)
    base = fuzzy(df)
    rows = []
    for param in PARAMS:
        for f in FACTORS:
            if f == 1.0 and param != PARAMS[0]: continue          # base case once
            if param.startswith("mf_") and f != 1.0:
                S = fuzzy(df, **{{"mf_temperature": "fT", "mf_h2": "fh", "mf_co2": "fc"}[param]: f})
            else:
                S = base
            for gate in GATES:
                p = "base" if f == 1.0 else param
                rows.append(dict(scenario=name, seed=seed, gate=gate, param=p, factor=f,
                                 **run_one(df, arrive, S, P, gate, param, f)))
    return rows


def summarise(d):
    L = ["One-at-a-time sensitivity, held-out seeds 100-129 (30 runs per scenario). Each hand-set constant x0.8 / 0.9 / 1.1 / 1.2.",
         "FNR = share of hazardous time with no alert (mean over runs). stop<runaway = runs that stopped before runaway",
         "(single, lag20, pre60, pre90 pooled, of 120). false Crit = benign runs with a false Critical or E-stop",
         "(hard, harder, hot60, warm_bursts pooled, of 120).", ""]
    for gate in GATES:
        g = d[d.gate == gate]
        base = g[g.param == "base"]
        def stats(x):
            hz = x[x.scenario.isin(HAZARD)]; bn = x[x.scenario.isin(BENIGN)]
            return (x[x.scenario == "single"].Hazard_FNR.mean(), x[x.scenario == "lag20"].Hazard_FNR.mean(),
                    int((hz.stop_margin_s > 0).sum()), int(bn.false_escalation.sum()),
                    x[x.scenario == "pre90"].T_stop_s.mean())
        b = stats(base)
        L.append(f"== {gate}   base: single FNR {b[0]:.4f} | lag20 FNR {b[1]:.4f} | stop<runaway {b[2]}/120 | "
                 f"false Crit {b[3]}/120 | pre90 stop {b[4]:.1f} s")
        L.append(f"   {'param':22s} {'factor':>6s}  single FNR  lag20 FNR  stop<runaway  false Crit  pre90 stop")
        worst = {}
        for param in PARAMS:
            for f in FACTORS:
                if f == 1.0: continue
                s = stats(g[(g.param == param) & (g.factor == f)])
                L.append(f"   {param:22s} {f:6.1f}  {s[0]:10.4f} {s[1]:10.4f} {s[2]:9d}/120 {s[3]:7d}/120 {s[4]:9.1f}")
                w = worst.setdefault(param, [0.0, 0, 0])
                w[0] = max(w[0], s[0] - b[0], s[1] - b[1]); w[1] = max(w[1], b[2] - s[2]); w[2] = max(w[2], s[3] - b[3])
        L.append("   worst change over +/-20 % (FNR increase | stops lost | false Crit added):")
        for param, (fnr, lost, fc) in worst.items():
            L.append(f"     {param:22s} {fnr:+.4f} | {lost:+d} | {fc:+d}")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    from multiprocessing import Pool
    OUT.mkdir(parents=True, exist_ok=True)
    cam = Camera()
    tasks = []
    for name in SCEN:
        for seed in SEEDS:
            df, _ = scen(name, seed)
            tasks.append((name, seed, cam.series(df)))
    with Pool(int(__import__("os").environ.get("NPROC", 14))) as pool:
        rows = [r for rs in pool.imap_unordered(work, tasks) for r in rs]
    d = pd.DataFrame(rows).sort_values(["gate", "param", "factor", "scenario", "seed"])
    d.to_csv(OUT / "sensitivity.csv", index=False)
    txt = summarise(d); print(txt)
    (OUT / "sensitivity_summary.txt").write_text(txt + "\n", encoding="utf-8")
