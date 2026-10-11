"""
gap_temporal_sweeps.py -- review-gap experiments on the temporal filter (no CNN retraining).

A. Fixed-beta control: is the adaptive memory better than ANY fixed beta, or only than 0.85?
   (temporal filter compared WITHOUT CNN fusion, same S(t), same FSM, 30 seeds)
B. Fresh lambda / beta_max sensitivity on the CURRENT (bug-fixed) code.
C. Trade-off: the same filters on the battery-free noise-stress scenario (false alarms).
D. Fusion with realistic camera probabilities: P_battery drawn per seed from the coarse
   model's actual outputs on RecyBat battery photos (results/gaps/coarse_P_on_fine_test_<tag>.npy).
E. Time-to-stop for batteries that arrive already failing (pre-warmed, faster runaway).

Run from the repo root after gap_cnn_checks.py:  python src/gap_temporal_sweeps.py
Outputs in results/gaps/.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from scipy import stats

import sensor_simulation as ss
from sensor_simulation import run_simulation, run_noise_stress_scenario, ThermalRunawayModel, MQ2Sensor, MQ135Sensor, InductiveProximitySensor
from fuzzy_inference import MamdaniFIS
from supervisory_control import run_supervisory
from baseline_comparison import compute_metrics
from adaptive_risk import AdaptiveRiskModel

OUT = Path("results/gaps"); OUT.mkdir(parents=True, exist_ok=True)
N, SEED0, DT, RESET = 30, 100, 0.1, 3.0
FIXED = [0.50, 0.60, 0.70, 0.73, 0.75, 0.80, 0.85]
ADAPT = [(0.85, 0.15), (0.85, 0.5), (0.85, 1.0), (0.85, 2.5), (0.85, 5.0), (0.95, 0.15), (0.95, 2.5), (0.75, 0.15)]
ALERT = ["Warning", "Critical", "Emergency_Stop"]
PB_FILE = OUT / os.environ.get("PB_FILE", "coarse_P_on_fine_test_v1.npy")   # camera probabilities from gap_cnn_checks.py
PB = np.load(PB_FILE)


def fuzzy_series(df):
    fis = MamdaniFIS()
    return np.array([fis.infer(r.temperature_C, r.mq2_ppm, r.mq135_ppm, bool(r.metal_present))[0] for r in df.itertuples()])


def fixed_filter(S, beta):
    R = np.zeros(len(S)); r = 0.0
    for i, s in enumerate(S):
        r = beta * r + (1 - beta) * s; R[i] = r
    return R


def adaptive_filter(S, obj, bmax, lam):
    R = np.zeros(len(S)); r = 0.0; sp = 0.0; noobj = 0.0
    for i, s in enumerate(S):
        if not obj[i]:
            noobj += DT
            if noobj >= RESET: r = 0.0
        else:
            noobj = 0.0
        b = bmax * np.exp(-lam * abs(s - sp))
        r = b * r + (1 - b) * s; sp = s; R[i] = r
    return R


def fsm_frame(df, R):
    d = df.copy(); d["R_fused"] = R
    full, _ = run_supervisory(d)
    out = full[["time", "fsm_state", "R_fused", "battery_present", "phase"]].copy()
    out.columns = ["time", "state", "risk", "battery_present", "phase"]
    return out


def num(v): return np.nan if isinstance(v, str) else float(v)


def metrics(frame, arrive):
    m = compute_metrics(frame, arrive, "x")
    return {"Hazard_FNR": num(m["Hazard_FNR"]), "T_warn_s": num(m["T_warn_s"]), "T_stop_s": num(m["T_stop_s"])}


class Prewarmed(ThermalRunawayModel):
    def __init__(self, seed, T0, scale):
        super().__init__(seed); self.T = T0; self.scale = scale

    def _rate(self, T):
        return super()._rate(T) * self.scale


def sim_prewarmed(seed, T0, scale, dur=200.0, arrive=20.0):
    th = Prewarmed(seed, T0, scale); mq2 = MQ2Sensor(seed + 1); mq135 = MQ135Sensor(seed + 2); ind = InductiveProximitySensor(seed + 3)
    rows = []
    for t in np.arange(0.0, dur, DT):
        batt = t >= arrive
        T, ph = th.step(batt)
        p2, v2 = mq2.read(T); p135, v135 = mq135.read(T); metal = ind.read(batt)
        rows.append(dict(time=round(t, 2), temperature_C=T, mq2_ppm=p2, mq2_voltage=v2, mq135_ppm=p135, mq135_voltage=v135,
                         metal_present=int(metal), object_present=int(batt), object_size_cm=10.0 if batt else 0.0, phase=ph, battery_present=int(batt)))
    return pd.DataFrame(rows)


def one_seed(k):
    seed = SEED0 + k
    rng = np.random.RandomState(1000 + k)
    out = []
    # ---- hazardous scenario
    df = run_simulation(duration_s=400.0, battery_arrives_at=20.0, seed=seed)
    S = fuzzy_series(df); obj = df.object_present.values
    peak_T = float(df.temperature_C.max()); hazard_rows = int(df.phase.isin(["Phase1_SEI", "Phase2_Separator", "Phase3_Runaway", "Phase4_Fire"]).sum())
    variants = {f"fixed_{b:.2f}": fixed_filter(S, b) for b in FIXED}
    variants.update({f"adapt_b{bm:.2f}_l{lam:g}": adaptive_filter(S, obj, bm, lam) for bm, lam in ADAPT})
    for name, R in variants.items():
        m = metrics(fsm_frame(df, R), 20.0)
        out.append(dict(kind="hazard", seed=seed, variant=name, peak_T=peak_T, hazard_rows=hazard_rows, **m))
    # D. fusion with realistic camera probability (drawn per seed) on the current adaptive filter
    Rcur = variants["adapt_b0.85_l0.15"]
    p_draw = float(rng.choice(PB))
    for label, p in (("fusion_P0", 0.0), ("fusion_Pdraw", p_draw), ("fusion_P0.90", 0.90)):
        Rf = np.clip(Rcur * (1 + 0.35 * p), 0, 1)
        m = metrics(fsm_frame(df, Rf), 20.0)
        out.append(dict(kind="hazard", seed=seed, variant=label, peak_T=peak_T, hazard_rows=hazard_rows, P_used=p, **m))
    # ---- C. noise stress (battery-free)
    dn = run_noise_stress_scenario(seed=123 + k, n_spikes=8)
    Sn = fuzzy_series(dn); objn = dn.object_present.values
    nv = {f"fixed_{b:.2f}": fixed_filter(Sn, b) for b in FIXED}
    nv.update({f"adapt_b{bm:.2f}_l{lam:g}": adaptive_filter(Sn, objn, bm, lam) for bm, lam in ADAPT})
    for name, R in nv.items():
        fr = fsm_frame(dn, R)
        out.append(dict(kind="noise", seed=seed, variant=name, peak_R=float(R.max()),
                        alert_rate=float(fr.state.isin(ALERT).mean()), estop=float((fr.state == "Emergency_Stop").any())))
    # ---- E. pre-warmed batteries
    for T0, sc in ((25, 1), (60, 3), (80, 6), (90, 10)):
        dp = sim_prewarmed(seed, T0, sc); Sp = fuzzy_series(dp); objp = dp.object_present.values
        for name, R in (("adaptive", adaptive_filter(Sp, objp, 0.85, 0.15)), ("fixed_0.85", fixed_filter(Sp, 0.85))):
            fr = fsm_frame(dp, R)
            w = fr[(fr.state.isin(ALERT)) & (fr.time >= 20.0)]; s = fr[(fr.state == "Emergency_Stop") & (fr.time >= 20.0)]
            b2 = ((dp.temperature_C > 90) | (dp.mq2_ppm > 500) | (dp.mq135_ppm > 2500))
            tb2 = dp[b2 & (dp.time >= 20.0)]
            out.append(dict(kind="prewarmed", seed=seed, variant=f"{name}", T0=T0, scale=sc,
                            T_warn_s=(w.time.iloc[0] - 20.0) if len(w) else np.nan, T_stop_s=(s.time.iloc[0] - 20.0) if len(s) else np.nan,
                            T_b2_stop_s=(tb2.time.iloc[0] - 20.0) if len(tb2) else np.nan, peak_T=float(dp.temperature_C.max())))
    return out


if __name__ == "__main__":
    from multiprocessing import Pool
    rows = []
    with Pool(8) as pool:
        for i, r in enumerate(pool.imap_unordered(one_seed, range(N)), 1):
            rows += r; print(f"seed {i}/{N} done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "temporal_sweeps_per_seed.csv", index=False)

    # sanity: my adaptive filter must equal AdaptiveRiskModel at (0.85, 0.15)
    d0 = run_simulation(duration_s=60.0, battery_arrives_at=5.0, seed=100); S0 = fuzzy_series(d0)
    arm = AdaptiveRiskModel(); ref = np.array([arm.update(float(s), 0.0, bool(o))[0] for s, o in zip(S0, d0.object_present.values)])
    assert np.allclose(ref, adaptive_filter(S0, d0.object_present.values, 0.85, 0.15), atol=1e-9), "filter mismatch"
    print("sanity: adaptive filter == AdaptiveRiskModel")

    h = df[df.kind == "hazard"]
    lines = []
    def P(s=""): print(s); lines.append(s)
    P("=== A/B. Hazard FNR by temporal filter (no CNN fusion), mean +/- sd over 30 seeds ===")
    ref_adapt = h[h.variant == "adapt_b0.85_l0.15"].set_index("seed")["Hazard_FNR"]
    for v in [f"fixed_{b:.2f}" for b in FIXED] + [f"adapt_b{bm:.2f}_l{lam:g}" for bm, lam in ADAPT]:
        x = h[h.variant == v].set_index("seed")["Hazard_FNR"]
        j = pd.concat([ref_adapt, x], axis=1).dropna()
        t, p = stats.ttest_rel(j.iloc[:, 0], j.iloc[:, 1]) if v != "adapt_b0.85_l0.15" else (np.nan, np.nan)
        ts = h[h.variant == v].T_stop_s.dropna()
        P(f"{v:<22} FNR {x.mean():.4f} +/- {x.std():.4f} | T_stop {ts.mean():.1f} (n={len(ts)}) | vs adaptive(0.85,0.15): diff(adaptive-this) {j.iloc[:,0].mean()-j.iloc[:,1].mean():+.4f} p={p:.3g}")
    P("\n=== Excluding stalled seeds (no hazardous steps or never ran away): adaptive vs fixed_0.85 ===")
    stalled = sorted(set(h[(h.hazard_rows == 0)].seed) | set(h[h.peak_T < 90].seed)); P(f"stalled seeds: {stalled}")
    for excl in (sorted(set(h[h.hazard_rows == 0].seed)), stalled):
        a = h[(h.variant == 'adapt_b0.85_l0.15') & ~h.seed.isin(excl)].set_index('seed').Hazard_FNR
        f = h[(h.variant == 'fixed_0.85') & ~h.seed.isin(excl)].set_index('seed').Hazard_FNR
        t, p = stats.ttest_rel(a, f); P(f"excluding {excl}: n={len(a)} adaptive {a.mean():.4f} fixed {f.mean():.4f} diff {a.mean()-f.mean():+.4f} t={t:.2f} p={p:.3g}")
    P("\n=== C. Noise-stress (battery-free): peak R, share of time in an alert state, seeds with false Emergency Stop ===")
    n = df[df.kind == "noise"]
    for v in [f"fixed_{b:.2f}" for b in FIXED] + [f"adapt_b{bm:.2f}_l{lam:g}" for bm, lam in ADAPT]:
        x = n[n.variant == v]; P(f"{v:<22} peak R mean {x.peak_R.mean():.3f} max {x.peak_R.max():.3f} | alert share {x.alert_rate.mean():.4f} | false E-stop seeds {int(x.estop.sum())}/30")
    P("\n=== D. Fusion with realistic camera probability ===")
    print("P_battery pool: n=%d mean %.3f" % (len(PB), PB.mean()))
    for v in ("fusion_P0", "fusion_Pdraw", "fusion_P0.90"):
        x = h[h.variant == v]; P(f"{v:<14} FNR {x.Hazard_FNR.mean():.4f} +/- {x.Hazard_FNR.std():.4f}  mean P used {x.get('P_used', pd.Series([np.nan])).mean():.3f}  T_stop {x.T_stop_s.mean():.1f}")
    P("\n=== E. Pre-warmed / faster batteries: time from arrival to first alert and to Emergency Stop (s) ===")
    pw = df[df.kind == "prewarmed"]
    for (T0, sc), g in pw.groupby(["T0", "scale"]):
        for v in ("adaptive", "fixed_0.85"):
            x = g[g.variant == v]
            P(f"T0={T0:>2}C rate x{sc:<2} {v:<10} T_warn {x.T_warn_s.mean():6.1f} (n={x.T_warn_s.notna().sum()}) | T_stop {x.T_stop_s.mean():6.1f} (n={x.T_stop_s.notna().sum()}) | B2 stop {x.T_b2_stop_s.mean():6.1f} | peak T {x.peak_T.mean():.0f}")
    open(OUT / "temporal_sweeps_summary.txt", "w", encoding="utf-8").write("\n".join(lines))
    print("saved results/gaps/temporal_sweeps_*.csv/txt")
