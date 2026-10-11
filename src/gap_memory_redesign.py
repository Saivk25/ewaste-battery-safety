"""
gap_memory_redesign.py -- can a redesigned adaptive memory beat a tuned fixed memory
once noise actually punishes over-reaction?

Filters (same fuzzy series, same FSM, no CNN fusion):
  fixed       R = b*R + (1-b)*S
  adaptive    the paper's rule: beta = beta_max * exp(-lambda*|dS|)
  persistence NEW rule: slow memory (beta_slow) by default; switch to fast memory (beta_fast)
              only after S has exceeded R by more than delta for m consecutive steps.
              Short spikes never satisfy the persistence test, so they are smoothed;
              a sustained rise is followed quickly.

Noise tests (battery-free, ground truth "never escalate"):
  mild   the existing noise-stress scenario: 8 spikes x 0.3 s, H2 350 ppm, CO2 1800 ppm
  hard   10 spikes x 1.5 s, H2 900 ppm, CO2 3000 ppm
  harder 10 spikes x 2.5 s, H2 1200 ppm, CO2 3500 ppm
A filter is ADMISSIBLE if it never reaches Critical or Emergency Stop in any mild or hard
noise run on the tuning seeds. "harder" (2.5 s bursts at 1200 ppm H2) is reported only: such a
burst is indistinguishable from a real hazard to the rule base, so every filter escalates. Among admissible settings, the lowest tuning Hazard FNR wins.
Tuning seeds 200-229; evaluation (held-out) seeds 100-129.

Run from the repo root:  python src/gap_memory_redesign.py   -> results/gaps/memory_redesign.*
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, itertools
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from scipy import stats
from sensor_simulation import run_simulation, run_noise_stress_scenario
from supervisory_control import SupervisoryFSM
from baseline_comparison import compute_metrics
from gap_temporal_sweeps import fuzzy_series, fixed_filter, adaptive_filter

OUT = Path("results/gaps"); DT = 0.1
NOISE = {"mild": dict(n_spikes=8, spike_mq2_ppm=350.0, spike_mq135_ppm=1800.0, spike_duration_steps=3),
         "hard": dict(n_spikes=10, spike_mq2_ppm=900.0, spike_mq135_ppm=3000.0, spike_duration_steps=15),
         "harder": dict(n_spikes=10, spike_mq2_ppm=1200.0, spike_mq135_ppm=3500.0, spike_duration_steps=25)}


def persistence_filter(S, obj, b_slow, b_fast, delta, m, reset=3.0):
    R = np.zeros(len(S)); r = 0.0; run = 0; noobj = 0.0
    for i, s in enumerate(S):
        if not obj[i]:
            noobj += DT
            if noobj >= reset: r = 0.0; run = 0
        else:
            noobj = 0.0
        run = run + 1 if s - r > delta else 0
        b = b_fast if run >= m else b_slow
        r = b * r + (1 - b) * s; R[i] = r
    return R


def fsm_states(df, R):
    f = SupervisoryFSM(); T = df.temperature_C.values; h = df.mq2_ppm.values; c = df.mq135_ppm.values; t = df.time.values
    return np.array([f.step(t[i], float(R[i]), T[i], h[i], c[i])[0].value for i in range(len(R))])


def hazard_metrics(df, R):
    st = fsm_states(df, R)
    fr = pd.DataFrame({"time": df.time, "state": st, "risk": R, "battery_present": df.battery_present, "phase": df.phase})
    m = compute_metrics(fr, 20.0, "x")
    num = lambda v: np.nan if isinstance(v, str) else float(v)
    return num(m["Hazard_FNR"]), num(m["T_stop_s"])


def escalated(df, R):
    st = fsm_states(df, R)
    return bool(np.isin(st, ["Critical", "Emergency_Stop"]).any())


def variants():
    v = [("fixed", (round(b, 2),)) for b in np.arange(0.30, 0.96, 0.05)]
    v += [("adaptive", (bm, lam)) for bm in (0.75, 0.85, 0.95) for lam in (0.15, 1, 3, 8)]
    v += [("persistence", p) for p in itertools.product((0.85, 0.90, 0.95), (0.2, 0.3, 0.5), (0.05, 0.10, 0.20), (5, 10, 20))]
    return v


def apply(kind, p, S, obj):
    if kind == "fixed": return fixed_filter(S, p[0])
    if kind == "adaptive": return adaptive_filter(S, obj, *p)
    return persistence_filter(S, obj, *p)


def run_seed(seed):
    d = run_simulation(duration_s=400.0, battery_arrives_at=20.0, seed=seed); S = fuzzy_series(d); obj = d.object_present.values
    noise = {}
    for name, kw in NOISE.items():
        dn = run_noise_stress_scenario(seed=seed + 1000, **kw); noise[name] = (dn, fuzzy_series(dn), dn.object_present.values)
    rows = []
    for kind, p in variants():
        fnr, tstop = hazard_metrics(d, apply(kind, p, S, obj))
        row = dict(seed=seed, kind=kind, params=json.dumps(p), Hazard_FNR=fnr, T_stop_s=tstop)
        for name, (dn, Sn, on) in noise.items():
            Rn = apply(kind, p, Sn, on); row[f"esc_{name}"] = escalated(dn, Rn); row[f"peak_{name}"] = float(Rn.max())
        rows.append(row)
    return rows


if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(6) as pool:
        tune = pd.DataFrame(sum(pool.map(run_seed, range(200, 230)), []))
        ev = pd.DataFrame(sum(pool.map(run_seed, range(100, 130)), []))
    tune.to_csv(OUT / "memory_redesign_tuning.csv", index=False); ev.to_csv(OUT / "memory_redesign_eval.csv", index=False)
    g = tune.groupby(["kind", "params"]).agg(fnr=("Hazard_FNR", "mean"), esc_mild=("esc_mild", "sum"), esc_hard=("esc_hard", "sum"),
                                             esc_harder=("esc_harder", "sum")).reset_index()
    g["admissible"] = (g.esc_hard == 0) & (g.esc_mild == 0)
    lines = []
    def P(s=""): print(s); lines.append(s)
    P("TUNING seeds 200-229. admissible = no Critical/E-stop in any mild or hard battery-free noise run")
    for kind in ("fixed", "adaptive", "persistence"):
        gg = g[g.kind == kind].sort_values("fnr")
        P(f"\n{kind}: {int(gg.admissible.sum())}/{len(gg)} admissible")
        P(gg.head(8 if kind != "fixed" else 14).to_string(index=False))
    chosen = {}
    for kind in ("fixed", "adaptive", "persistence"):
        ok = g[(g.kind == kind) & g.admissible].sort_values("fnr")
        chosen[kind] = ok.iloc[0].params if len(ok) else None
    P(f"\nchosen (lowest tuning FNR among admissible): {chosen}")
    P("\nEVALUATION held-out seeds 100-129")
    sel = {}
    for kind, prm in chosen.items():
        if prm is None: P(f"{kind}: no admissible setting"); continue
        x = ev[(ev.kind == kind) & (ev.params == prm)].set_index("seed"); sel[kind] = x
        P(f"{kind:<12} {prm:<22} Hazard FNR {x.Hazard_FNR.mean():.4f} +/- {x.Hazard_FNR.std():.4f} | T_stop {x.T_stop_s.mean():.1f} | "
          f"false escalations mild/hard/harder {int(x.esc_mild.sum())}/{int(x.esc_hard.sum())}/{int(x.esc_harder.sum())} of 30")
    ref = ev[(ev.kind == "adaptive") & (ev.params == json.dumps((0.85, 0.15)))].set_index("seed")
    P(f"{'paper adaptive':<12} {'[0.85, 0.15]':<22} Hazard FNR {ref.Hazard_FNR.mean():.4f} | false escalations mild/hard/harder {int(ref.esc_mild.sum())}/{int(ref.esc_hard.sum())}/{int(ref.esc_harder.sum())}")
    res = {"chosen": chosen}
    if "fixed" in sel:
        for kind in ("persistence", "adaptive"):
            if kind in sel:
                a, f = sel[kind].Hazard_FNR, sel["fixed"].Hazard_FNR
                t, p = stats.ttest_rel(a, f)
                P(f"{kind} minus fixed: {a.mean() - f.mean():+.4f}, paired t(29) = {t:.2f}, p = {p:.3g}; {kind} lower in {int((a < f).sum())}, higher in {int((a > f).sum())} seeds")
                res[f"{kind}_minus_fixed"] = {"diff": float(a.mean() - f.mean()), "t": float(t), "p": float(p)}
    json.dump(res, open(OUT / "memory_redesign_summary.json", "w"), indent=1)
    open(OUT / "memory_redesign_summary.txt", "w", encoding="utf-8").write("\n".join(lines))
