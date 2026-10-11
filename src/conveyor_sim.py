"""
conveyor_sim.py -- moving-belt simulation of the pre-shred line (Oct 2026 failure-condition fixes).

The earlier scenarios hold one battery under the sensors for minutes. A real feed belt carries an
item from the camera to the shredder in seconds, so this module simulates the line itself:

    camera (x=0) -> sensor hood (0.25-1.0 m) -> diverter (1.5 m) -> shredder mouth (2.5 m)
    belt 0.25 m/s: camera to shredder 10 s, 3 s under the hood at full speed

Fixes tested here (numbers in the paper's failure-condition list):
 1  moving belt     the item moves; the FSM state sets the belt speed (Normal 100 %, Warning 60 %,
                    Critical 10 %, Emergency_Stop 0 %, Sensor_Fault 50 %), so a suspicious item is
                    slowed down and stays under the sensors longer
 2  intact battery  tiered response: the diverter pushes an item into a reject bin when the camera saw a
                    battery (P_battery >= BATTERY_THRESHOLD, 0.1113) or the hood flagged it (Warning or worse while it was under
                    the sensors); the bin has its own IR + gas sensors running the same pipeline.
                    Emergency_Stop on the belt stays the action for a confirmed hazard
 3  slow gas        MQ sensors are first-order lags (rise 5-8 s, fall 45-60 s, x4 in a stress case)
                    and the vent gas is diluted before it reaches the sensor (hood 0.2, open belt 0.03,
                    closed bin 0.5 of the source concentration). The gas readings are scaled back by
                    the dilution measured at commissioning (hood 0.2, bin 0.5) before fuzzy inference,
                    because the fuzzy sets were written for a sensor at the source (after a 0.5 s average)
 4  hot climate     the fuzzy sets were written for 25 C ambient; the system feeds them the rise above
                    the measured ambient (IR on the empty belt, 10 s average) instead of the raw reading
 9  per battery     outcomes are counted per item: did a battery (hazardous or not) enter the shredder

Geometry, belt speed, dilution and sensor lags are assumptions, not measurements; dilution and lag are
swept. Camera probabilities are real outputs of the trained coarse checkpoint on held-out test photos
(battery photos for visible batteries, Cardboard_Trash photos for batteries hidden inside a device).

Run from the repo root:  python src/conveyor_sim.py   -> results/conveyor/
"""
import os, sys, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
import supervisory_control as sc
from fuzzy_inference import MamdaniFIS
from sensor_simulation import ThermalRunawayModel, MQ2Sensor, MQ135Sensor
from adaptive_risk import BETA_FAST, BETA_SLOW, C_GATE, M_PERSIST, T_CORR, ALPHA_CNN, RESET_TIME

OUT = Path(__file__).resolve().parent.parent / "results" / "conveyor"
POOL = OUT / "camera_pool.npz"
DT = 0.1
V_BELT = 0.25                                      # m/s
X_HOOD0, X_HOOD1, X_DIV, X_SHRED = 0.25, 1.0, 1.5, 2.5
P_DIVERT = 0.1113                                  # = evaluate_model.BATTERY_THRESHOLD (chosen on validation)
T_REF = 25.0                                       # ambient the fuzzy sets were written for
DIL = {"hood": 0.2, "open": 0.03, "bin": 0.5}
IR_NOISE = 0.2
WARMUP = 30.0                                      # s of empty belt before the item reaches the camera
T_END = 300.0                                      # s after the camera
BIN_T = 120.0                                      # s the bin is watched after a divert
HAZ = ("Phase1_SEI", "Phase2_Separator", "Phase3_Runaway", "Phase4_Fire")
RUNAWAY = ("Phase3_Runaway", "Phase4_Fire")
EVAL = list(range(100, 130))


# ----------------------------------------------------------------------------- item physics
class Cell(ThermalRunawayModel):
    """Failing cell that reaches the camera already at T0 and self-heats `scale` times faster than the
    bench model (same idea as gap_prewarmed.Prewarmed). Self-heating is computed at the 25 C reference;
    the surface sits (ambient - 25) higher, and the phase label follows the self-heating temperature."""
    def __init__(self, seed, T0, scale):
        super().__init__(seed); self.T = T0; self.scale = scale

    def _rate(self, T):
        return super()._rate(T) * self.scale


def item_track(kind, rng, amb, n):
    """Surface temperature, phase, H2 and CO2 source concentration for n steps after the camera."""
    k = kind.split("_")
    T = np.full(n, amb); ph = np.array(["Normal"] * n, dtype=object); h2 = np.zeros(n); co2 = np.full(n, 400.0)
    fail = next((p for p in k if p.startswith("fail")), None)
    if fail:
        T0, scale = {"fail60": (60, 3), "fail80": (80, 6), "fail90": (90, 10)}[fail]
        c = Cell(int(rng.randint(1 << 30)), T0, scale)
        for i in range(n):
            Ts, ph[i] = c.step(True)
            T[i] = Ts + (amb - T_REF)
            h2[i] = MQ2Sensor._temp_to_ppm(None, T[i]); co2[i] = MQ135Sensor._temp_to_ppm(None, T[i])
    elif "warm" in k:
        T[:] = 50.0
    return T, ph, h2, co2


def camera_p(kind, rng, pool):
    k = kind.split("_")
    if "hidden" in k:                                       # battery inside a device: the camera sees the device
        return float(rng.choice(pool["test_cardboard_trash"]))
    if "intact" in k or any(p.startswith("fail") for p in k):
        src = pool["test_battery_coarse"] if rng.rand() < 0.5 else pool["test_battery_recybat"]
        return float(rng.choice(src))
    return float(rng.choice(pool["test_nonbattery"]))


class Gas:
    """MQ sensor: first-order lag toward the concentration at the sensor, multiplicative noise."""
    def __init__(self, rng, base, rise, fall, noise):
        self.rng, self.base, self.rise, self.fall, self.noise = rng, base, rise, fall, noise; self.y = base

    def read(self, c):
        self.y += (c - self.y) * DT / (self.rise if c > self.y else self.fall)
        return max(self.y * self.rng.normal(1.0, self.noise), 0.0)


# ----------------------------------------------------------------------------- controllers
def undilute(h2, co2, dil):
    """Source-equivalent gas concentration from a reading taken at commissioning dilution `dil`."""
    return h2 / dil, 400.0 + (co2 - 400.0) / dil


class Ours:
    """Fuzzy inference + corroboration-gated memory + CNN fusion + supervisory FSM (paper pipeline),
    fed the temperature rise above the measured ambient (ambient_fix) and dilution-compensated gas (dil)."""
    def __init__(self, ambient_fix=True, dil=DIL["hood"]):
        self.fis = MamdaniFIS(); self.fsm = sc.SupervisoryFSM(); self.fix = ambient_fix; self.dil = dil
        self.r = 0.0; self.run = 0; self.no = 0.0; self.amb = None; self.g = None

    def step(self, t, T_ir, h2, co2, present, P):
        if self.dil:   # 0.5 s average first, so the 1/dil gain does not amplify sample noise
            self.g = (h2, co2) if self.g is None else tuple(g + (v - g) * DT / 0.5 for g, v in zip(self.g, (h2, co2)))
            h2, co2 = undilute(*self.g, self.dil)
        if self.amb is None: self.amb = T_ir
        if not present: self.amb += (T_ir - self.amb) * DT / 10.0
        T_in = T_REF + (T_ir - self.amb) if self.fix else T_ir
        S, _ = self.fis.infer(T_in, h2, co2, present)
        if not present:
            self.no += DT
            if self.no >= RESET_TIME: self.r = 0.0; self.run = 0
        else:
            self.no = 0.0
        if S <= C_GATE: self.run = 0; b = BETA_FAST
        else:
            self.run += 1; b = BETA_FAST if (T_in >= T_CORR or self.run >= M_PERSIST) else BETA_SLOW
        self.r = b * self.r + (1 - b) * S
        Rf = float(np.clip(self.r * (1 + ALPHA_CNN * (P if present else 0.0)), 0, 1))
        st, speed = self.fsm.step(t, Rf, T_ir, h2, co2)
        return st.value, speed / 100.0


class Rule:
    """Plain fixed-threshold rule (baseline B2) on the same dilution-compensated gas: stop latches,
    warning slows the belt to 60 %."""
    def __init__(self): self.latched = False

    def step(self, t, T_ir, h2, co2, present, P):
        h2, co2 = undilute(h2, co2, DIL["hood"])
        if self.latched or T_ir > 90 or h2 > 500 or co2 > 2500:
            self.latched = True; return "Emergency_Stop", 0.0
        if T_ir > 50 or h2 > 80 or (present and T_ir > 35):
            return "Warning", 0.6
        return "Normal", 1.0


class Free:
    """Camera-only line: no sensors act on the belt."""
    def step(self, *a): return "Normal", 1.0


# divert: None = no diverter, "camera" = camera only, "tiered" = camera or hood flag
SYSTEMS = {
    "camera_only":    dict(ctrl=Free,                                divert="camera", bin=False),
    "rule":           dict(ctrl=Rule,                                divert=None,     bin=False),
    "ours":           dict(ctrl=Ours,                                divert="tiered", bin=True),
    "ours_no_divert": dict(ctrl=Ours,                                divert=None,     bin=False),
    "ours_fixed_T":   dict(ctrl=lambda: Ours(ambient_fix=False),     divert="tiered", bin=True),
    "ours_raw_gas":   dict(ctrl=lambda: Ours(dil=None),              divert="tiered", bin=True),
}
FLAG = ("Warning", "Critical", "Emergency_Stop", "Sensor_Fault")


# ----------------------------------------------------------------------------- one item
def scenario_params(kind):
    k = kind.split("_")
    return dict(amb=45.0 if "hot45" in k else 25.0, dil="open" if "open" in k else "hood",
                lag=4.0 if "slowgas" in k else 1.0, gas_dead="gasdead" in k, burst="burst" in k)


def trial(kind, seed, system, pool, trace=False):
    p = scenario_params(kind); spec = SYSTEMS[system]
    world = np.random.RandomState(seed)                     # same item for every system (paired)
    n = int(round((T_END + BIN_T) / DT))
    T_it, ph, h2s, co2s = item_track(kind, world, p["amb"], n)
    P = camera_p(kind, world, pool)
    battery = any(w in kind for w in ("intact", "fail"))
    noise = np.random.RandomState(seed + 1)
    mq2 = Gas(noise, 0.0, 5.0 * p["lag"], 45.0, 0.03); mq135 = Gas(noise, 400.0, 8.0 * p["lag"], 60.0, 0.02)
    ctrl = spec["ctrl"](); cam_hit = spec["divert"] is not None and P >= P_DIVERT; flagged = False
    burst_at = None
    x = -WARMUP * V_BELT; speed = 1.0; i_item = -1; loss = 0.0
    res = dict(scenario=kind, seed=seed, system=system, battery=battery, P_battery=P, diverted=False,
               entered_shredder=False, hazardous_at_entry=False, runaway_at_entry=False, belt_stopped=False,
               divert_reason="", t_entry=np.nan, belt_loss_s=0.0, bin_alarm=np.nan, bin_margin_s=np.nan, bin_unwatched_fire=False,
               t_divert=np.nan, t_bin_alarm=np.nan)
    rows = []
    t = -WARMUP; step = 0
    while t < T_END:
        on_belt = not res["diverted"] and x < X_SHRED
        if x >= 0 and i_item < 0: i_item = 0
        idx = min(max(i_item, 0), n - 1)
        under = on_belt and X_HOOD0 <= x < X_HOOD1
        if p["burst"] and under and burst_at is None: burst_at = step
        T_ir = (T_it[idx] if under else p["amb"]) + noise.normal(0, IR_NOISE)
        dil = DIL[p["dil"]]
        c2 = dil * h2s[idx] if under else 0.0
        c135 = 400.0 + dil * (co2s[idx] - 400.0) if under else 400.0
        r2, r135 = mq2.read(c2), mq135.read(c135)
        if p["gas_dead"]: r2, r135 = 0.0, 400.0
        if burst_at is not None and step - burst_at < 15:   # 1.5 s "hard" burst of the paper, at source-equivalent scale
            r2, r135 = dil * (900.0 + noise.normal(0, 20)), 400.0 + dil * (2600.0 + noise.normal(0, 80))
        state, speed = ctrl.step(t, T_ir, r2, r135, under, P)
        if under and state in FLAG: flagged = True
        if trace:
            rows.append(dict(t=round(t, 2), x=x, under=under, T_ir=T_ir, h2=r2, co2=r135, state=state, speed=speed,
                             phase=ph[idx] if i_item >= 0 else "Normal", T_item=T_it[idx] if i_item >= 0 else p["amb"],
                             h2_src=h2s[idx] if i_item >= 0 else 0.0))
        if on_belt and x >= 0: loss += (1 - speed) * DT
        if speed == 0.0 and on_belt and x >= 0: res["belt_stopped"] = True
        x += V_BELT * speed * DT
        if i_item >= 0: i_item += 1
        if (cam_hit or (flagged and spec["divert"] == "tiered")) and on_belt and x >= X_DIV:
            res.update(diverted=True, divert_reason="camera" if cam_hit else "hood", t_divert=round(t, 2)); t_div = i_item
            break
        if on_belt and x >= X_SHRED:
            idx = min(i_item, n - 1)
            res.update(entered_shredder=True, t_entry=round(t, 2), hazardous_at_entry=bool(ph[idx] in HAZ),
                       runaway_at_entry=bool(ph[idx] in RUNAWAY))
            break
        if isinstance(ctrl, (Ours, Rule)) and state == "Emergency_Stop": break   # latched: operator clears the belt
        t += DT; step += 1
    res["belt_loss_s"] = round(loss, 2)

    if res["diverted"]:
        onset = next((j for j in range(t_div, n) if ph[j] in RUNAWAY), None)
        if spec["bin"]:
            b = Ours(ambient_fix=system != "ours_fixed_T", dil=None if system == "ours_raw_gas" else DIL["bin"])
            bn = np.random.RandomState(seed + 2)
            g2 = Gas(bn, 0.0, 5.0 * p["lag"], 45.0, 0.03); g135 = Gas(bn, 400.0, 8.0 * p["lag"], 60.0, 0.02)
            for _ in range(int(WARMUP / DT)):                 # bin sensors already running, bin empty
                b.step(0.0, p["amb"] + bn.normal(0, IR_NOISE), g2.read(0.0), g135.read(400.0), False, 0.0)
            alarm = None
            for j in range(t_div, min(t_div + int(BIN_T / DT), n)):
                T_ir = T_it[j] + bn.normal(0, IR_NOISE)
                r2 = g2.read(DIL["bin"] * h2s[j]); r135 = g135.read(400.0 + DIL["bin"] * (co2s[j] - 400.0))
                if p["gas_dead"]: r2, r135 = 0.0, 400.0
                st, _ = b.step(j * DT, T_ir, r2, r135, True, P)
                if st in ("Critical", "Emergency_Stop"): alarm = j; break
            res["bin_alarm"] = alarm is not None
            if alarm is not None: res["t_bin_alarm"] = round(alarm * DT, 2)
            if alarm is not None and onset is not None: res["bin_margin_s"] = round((onset - alarm) * DT, 2)
        else:
            res["bin_unwatched_fire"] = onset is not None
    return (res, pd.DataFrame(rows)) if trace else res


# ----------------------------------------------------------------------------- camera pool (real checkpoint)
def build_camera_pool():
    import torch
    from torchvision import datasets
    from evaluate_model import load_checkpoint, get_test_transform, DATA_DIR
    from gap_cnn_checks import softmax_all
    model, classes, _ = load_checkpoint("EfficientNet-B0", "coarse")
    bi, ei = classes.index("Battery"), classes.index("Cardboard_Trash")
    out = {}
    for split in ("val", "test"):
        ds = datasets.ImageFolder(DATA_DIR / "coarse" / split, get_test_transform())
        Pr, Y = softmax_all(model, ds); pb = Pr[:, bi]
        out[f"{split}_battery_coarse"] = pb[Y == bi]; out[f"{split}_nonbattery"] = pb[Y != bi]
        out[f"{split}_electronics"] = pb[Y == ei]
    for split in ("val", "test"):
        ds = datasets.ImageFolder(DATA_DIR / "fine" / split, get_test_transform())
        out[f"{split}_battery_recybat"] = softmax_all(model, ds)[0][:, bi]
    OUT.mkdir(parents=True, exist_ok=True); np.savez(POOL, **out)
    return out


def load_pool():
    return dict(np.load(POOL)) if POOL.exists() else build_camera_pool()


# ----------------------------------------------------------------------------- experiment
SCENARIOS = ["benign", "warm_burst", "hot45_burst", "intact", "fail60", "fail80", "fail90",
             "fail60_hidden", "fail80_hidden", "fail90_hidden",
             "hot45_benign", "hot45_intact", "hot45_fail80_hidden",
             "open_fail80_hidden", "slowgas_fail80_hidden", "gasdead_fail80_hidden", "gasdead_fail80"]


def work(task):
    kind, seed, pool = task
    return [trial(kind, seed, s, pool) for s in SYSTEMS]


def summarise(df, pool, log):
    def L(s=""): print(s, flush=True); log.append(s)
    v = pool
    vb = np.r_[v["val_battery_coarse"], v["val_battery_recybat"]]; tb = np.r_[v["test_battery_coarse"], v["test_battery_recybat"]]
    L(f"Divert threshold P_battery >= {P_DIVERT}")
    L(f"  validation: batteries diverted {np.mean(vb >= P_DIVERT):.4f} (n={len(vb)}; coarse-set photos {np.mean(v['val_battery_coarse'] >= P_DIVERT):.4f}, "
      f"RecyBat {np.mean(v['val_battery_recybat'] >= P_DIVERT):.4f}) | non-battery diverted {np.mean(v['val_nonbattery'] >= P_DIVERT):.4f} (n={len(v['val_nonbattery'])})")
    L(f"  test:       batteries diverted {np.mean(tb >= P_DIVERT):.4f} (n={len(tb)}; coarse-set photos {np.mean(v['test_battery_coarse'] >= P_DIVERT):.4f}, "
      f"RecyBat {np.mean(v['test_battery_recybat'] >= P_DIVERT):.4f}) | non-battery diverted {np.mean(v['test_nonbattery'] >= P_DIVERT):.4f} (n={len(v['test_nonbattery'])})")
    L(f"  battery hidden in a device (Cardboard_Trash photos, test): diverted {np.mean(v['test_cardboard_trash'] >= P_DIVERT):.4f}")
    L("")
    L(f"Moving belt, held-out seeds {EVAL[0]}-{EVAL[-1]} (30 items per scenario). Per item:")
    L("  battery scenarios: shred = battery entered the shredder | haz = entered while self-heating (>= 45 C) | div = diverted | stop = belt E-stop")
    L("                     bin = diverted failing cell caught by the bin sensors before runaway | unwatched = failing cell in an unmonitored bin")
    L("  benign scenarios:  div = false divert | stop = false E-stop | loss = belt seconds lost per item")
    for kind in SCENARIOS:
        L(f"-- {kind}")
        for s in SYSTEMS:
            x = df[(df.scenario == kind) & (df.system == s)]
            if x.battery.iloc[0]:
                fail = "fail" in kind
                binok = x[x.diverted & x.bin_alarm.notna()]
                bin_s = (f"bin {int((binok.bin_alarm.astype(bool) & (binok.bin_margin_s > 0)).sum())}/{int(x.diverted.sum())} "
                         f"margin {binok.bin_margin_s.mean():5.1f} s" if fail and len(binok) else "")
                L(f"   {s:<15} shred {x.entered_shredder.mean():.3f} | haz {x.hazardous_at_entry.mean():.3f} | div {x.diverted.mean():.3f} | "
                  f"stop {x.belt_stopped.mean():.3f} | loss {x.belt_loss_s.mean():5.1f} s {bin_s}"
                  + (f" | unwatched {int(x.bin_unwatched_fire.sum())}" if x.bin_unwatched_fire.any() else ""))
            else:
                L(f"   {s:<15} div {x.diverted.mean():.3f} (hood {int((x.divert_reason == 'hood').sum())}) | stop {x.belt_stopped.mean():.3f} | loss {x.belt_loss_s.mean():5.1f} s")
    bat = df[df.battery & df.scenario.str.contains("fail")]
    L("")
    L("Pooled over all failing-cell scenarios (per battery): hazardous batteries that entered the shredder")
    for s in SYSTEMS:
        x = bat[bat.system == s]
        k = int(x.hazardous_at_entry.sum()); m = len(x)
        L(f"   {s:<15} {k}/{m} = {k / m:.3f}")


if __name__ == "__main__":
    from multiprocessing import Pool
    OUT.mkdir(parents=True, exist_ok=True)
    pool = load_pool()
    tasks = [(k, s, pool) for k in SCENARIOS for s in EVAL]
    rows = []
    with Pool(12) as pp:
        for r in pp.imap_unordered(work, tasks, chunksize=4):
            rows += r
    df = pd.DataFrame(rows).sort_values(["scenario", "seed", "system"])
    df.to_csv(OUT / "eval.csv", index=False)
    log = []
    summarise(df, pool, log)
    open(OUT / "summary.txt", "w", encoding="utf-8").write("\n".join(log))
