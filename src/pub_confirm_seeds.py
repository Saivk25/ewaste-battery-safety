"""
pub_confirm_seeds.py -- independent confirmation of the retuned gate on fresh seeds 300-329.

The gate settings were fixed by pub_fixes2.py (tuning seeds 200-229) and first evaluated on seeds 100-129,
which were also the stress-test seeds that motivated the retune. Seeds 300-329 were never used for any
design or tuning choice, so this run is a blind check. Same scenarios, systems and metrics as pub_fixes2.py.

Run from the repo root:  python src/pub_confirm_seeds.py -> results/pub/confirm_eval.csv, confirm_summary.txt
"""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
from pub_experiments import Camera, OUT
from pub_fixes import scen2
from pub_fixes2 import work

SEEDS = range(300, 330)
GRID = [(20, None), (80, None), (80, (10.0, 0.2))]
NAMES = ["single", "multi", "lag5", "lag20", "lag60", "pre60", "pre80", "pre90", "mild", "hard", "harder",
         "hot60", "hot90", "warm_bursts", "warm75_bursts", "cal_single"]

if __name__ == "__main__":
    from multiprocessing import Pool
    cam = Camera()
    tasks = []
    for n in NAMES:
        for s in SEEDS:
            df, _ = scen2(n, s); tasks.append((n, s, cam.series(df), GRID, True))
    with Pool(2) as pool:
        ev = pd.DataFrame(sum(pool.map(work, tasks), []))
    ev.to_csv(OUT / "confirm_eval.csv", index=False)
    log = ["CONFIRMATION (fresh seeds 300-329): FNR / mean stop (s) for hazard scenarios; false Critical / E-stop runs for battery-free"]
    for s in sorted(ev.system.unique()):
        parts = []
        for n in NAMES:
            y = ev[(ev.scenario == n) & (ev.system == s)]
            parts.append(f"{n} {y.Hazard_FNR.mean():.4f}/{y.T_stop_s.mean():.1f}" if y.Hazard_FNR.notna().any()
                         else f"{n} {int(y.false_escalation.sum())}/{int(y.false_estop.sum())}")
        log.append(f"  {s}: " + " | ".join(parts))
    print("\n".join(log))
    open(OUT / "confirm_summary.txt", "w", encoding="utf-8").write("\n".join(log))
