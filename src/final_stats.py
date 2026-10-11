"""final_stats.py -- collect every number the paper, deck and guide quote for the final
configuration into results/final_stats.json (paired tests on the shared 30 seeds)."""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from scipy import stats

R = Path("results"); out = {}


def paired(df, a, b, col="Hazard_FNR", key="system"):
    x = df[df[key] == a].set_index("seed")[col]; y = df[df[key] == b].set_index("seed")[col]
    j = pd.concat([x, y], axis=1).dropna(); d = j.iloc[:, 0] - j.iloc[:, 1]
    t, p = stats.ttest_rel(j.iloc[:, 0], j.iloc[:, 1]) if d.abs().sum() > 0 else (np.nan, np.nan)
    return {"a": a, "b": b, "mean_a": float(j.iloc[:, 0].mean()), "mean_b": float(j.iloc[:, 1].mean()), "diff": float(d.mean()),
            "t": float(t), "p": float(p), "a_lower": int((d < 0).sum()), "a_higher": int((d > 0).sum()), "n": len(j)}


b = pd.read_csv(R / "baseline_seed_sweep_per_seed.csv")
out["single_30seed"] = {s: {m: [float(g[m].mean()), float(g[m].std()), int(g[m].notna().sum())] for m in ("Hazard_FNR", "T_warn_s", "T_stop_s", "CFNC")}
                        for s, g in b.groupby("system")}
out["single_tests"] = [paired(b, "Proposed", o) for o in ("B3_Fixed_Fuzzy", "B1_Vision_Only", "B2_Rule_Based")]
excl = [107, 122, 129]
out["single_tests_excl_stalled"] = [paired(b[~b.seed.isin(excl)], "Proposed", "B3_Fixed_Fuzzy")]
m = pd.read_csv(R / "multi_item_sweep_per_seed.csv"); o = pd.read_csv(R / "occlusion_sweep_per_seed.csv")
out["multi_tests"] = [paired(m, "Proposed", x) for x in ("B3_Fixed_Fuzzy", "B1_Vision_Only", "B2_Rule_Based")]
out["occl_tests"] = [paired(o, "Proposed (occluded)", x) for x in ("B3_Fixed_Fuzzy", "B2_Rule_Based", "Proposed (camera ok)")]
out["multi"] = {s: {k: [float(g[k].mean()), float(g[k].std())] for k in ("Hazard_FNR", "T_stop_s", "item1_alert_rate")} for s, g in m.groupby("system")}
out["occl"] = {s: {k: [float(g[k].mean()), float(g[k].std()), int(g[k].notna().sum())] for k in ("Hazard_FNR", "T_stop_s")} for s, g in o.groupby("system")}
a = pd.read_csv(R / "alpha_ablation_per_seed.csv")
out["alpha_test"] = paired(a.assign(system=a.alpha.astype(str)), "0.35", "0.0")
s = pd.read_csv(R / "alpha_ablation_summary.csv")
out["alpha_table"] = {str(al): {mt: float(g[g.metric == mt]["mean"].iloc[0]) for mt in ("Hazard_FNR", "T_warn_s", "T_stop_s")} for al, g in s.groupby("alpha")}
out["alpha_false_alarm"] = pd.read_csv(R / "alpha_false_alarm_summary.csv").to_dict("records")
out["alpha_noise_stress"] = pd.read_csv(R / "alpha_noise_stress_summary.csv").to_dict("records")
nz = pd.read_csv(R / "final_validation_noise.csv")
out["noise_validation"] = {f"{n}|{s}": {"false_critical": int(g.critical.sum()), "false_estop": int(g.estop.sum())} for (n, s), g in nz.groupby(["noise", "system"])}
pw = pd.read_csv(R / "final_validation_prewarmed.csv")
out["prewarmed"] = {f"{t}C_x{sc}|{s}": [float(g.T_stop_s.mean()), float(g.T_stop_s.max())] for (t, sc, s), g in pw.groupby(["T0", "scale", "system"])}
out["canonical_transitions"] = pd.read_csv("outputs/state_transitions.csv").to_dict("records")
out["canonical_baselines"] = pd.read_csv("outputs/baseline_comparison.csv").to_dict("records")
for f in ("corroborated_memory_summary.json", "heldout_memory_summary.json", "coarse_v2_threshold.json", "cnn_v2_test_split.json",
          "cnn_checks_v1.json", "cnn_checks_v2.json", "fine_sources.json"):
    out["gaps/" + f] = json.load(open(R / "gaps" / f))
json.dump(out, open(R / "final_stats.json", "w"), indent=1, default=float)
for k in ("single_tests", "single_tests_excl_stalled", "multi_tests", "occl_tests"):
    for r in out[k]: print(k, r["a"], "vs", r["b"], f"{r['mean_a']:.4f} vs {r['mean_b']:.4f} diff {r['diff']:+.4f} t={r['t']:.2f} p={r['p']:.3g} lower {r['a_lower']} higher {r['a_higher']}")
print("alpha", out["alpha_test"]); print("canonical", out["canonical_transitions"])
