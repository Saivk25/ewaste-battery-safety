"""
cnn_repeat_eval.py -- training-seed repeats and backbone comparison on the corrected data (fix 5).

Every model is trained on data_v3 with src/train_cnn.py --out-dir runs_cnn/<backbone>_s<seed>
(the production model is EfficientNet-B0 seed 42 in checkpoints/). Each one is evaluated on the
same data_v3 test set:
  * accuracy and macro F1 with the most likely class;
  * battery threshold chosen on that model's own validation set (smallest threshold whose
    validation false-alarm rate is <= 3.05%, the rule used for the production model), then test
    accuracy, battery recall (consumer / Li-ion / damaged, intact batteries missed) and false alarms with that threshold;
  * single-image latency (batch 1, GPU and CPU).
Run from the repo root:  python src/cnn_repeat_eval.py  -> results/pub/cnn_repeat_eval.csv
"""
import sys, time, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd, torch
from torchvision import datasets
from sklearn.metrics import f1_score
from evaluate_model import load_checkpoint, get_test_transform, device
from gap_cnn_checks import softmax_all

D = Path("data_v3/coarse")


def runs():
    out = [("EfficientNet-B0", 42, Path("checkpoints/best_EfficientNetB0_coarse.pth"))]
    for d in sorted(Path("runs_cnn").glob("*_s*")):
        bb, seed = d.name.rsplit("_s", 1)
        tag = bb.replace("-", "").replace(" ", "_")
        p = d / "checkpoints" / f"best_{tag}_coarse.pth"
        if p.exists():
            out.append((bb, int(seed), p))
    return out


def latency(model, dev):
    m = model.to(dev).eval(); x = torch.randn(1, 3, 224, 224, device=dev)
    with torch.no_grad():
        for _ in range(20): m(x)
        if dev.type == "cuda": torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(100): m(x)
        if dev.type == "cuda": torch.cuda.synchronize()
    return (time.perf_counter() - t) / 100 * 1000


if __name__ == "__main__":
    tf = get_test_transform()
    val = datasets.ImageFolder(D / "val", tf); test = datasets.ImageFolder(D / "test", tf)
    names = [Path(s).name for s, _ in test.samples]
    g12 = np.array([n.startswith("g12_") for n in names])
    dmg = np.array([n.startswith("damaged_") for n in names])      # data_v3: held-out swollen / corroded / leaking batteries
    rows = []
    for bb, seed, path in runs():
        model, classes, _ = load_checkpoint(bb, "coarse", path); bi = classes.index("Battery")
        Pv, Yv = softmax_all(model, val); Pt, Yt = softmax_all(model, test)
        pv, yv = Pv[:, bi], Yv == bi
        tau = next(t for t in sorted(np.unique(np.r_[pv, 0.5])) if (pv[~yv] >= t).mean() <= 0.0305)
        am = Pt.argmax(1); th = am.copy(); th[Pt[:, bi] >= tau] = bi
        yb = Yt == bi
        rows.append(dict(backbone=bb, seed=seed, acc_argmax=float((am == Yt).mean()), f1_argmax=float(f1_score(Yt, am, average="macro")),
                         tau=float(tau), acc_tau=float((th == Yt).mean()), f1_tau=float(f1_score(Yt, th, average="macro")),
                         battery_recall_tau=float((th[yb] == bi).mean()), consumer_recall_tau=float((th[yb & g12] == bi).mean()),
                         liion_recall_tau=float((th[yb & ~g12 & ~dmg] == bi).mean()), intact_missed=int(((th != bi) & yb & ~dmg).sum()),
                         damaged_detected=int(((th == bi) & dmg).sum()), n_damaged=int(dmg.sum()), false_alarms_tau=int(((th == bi) & ~yb).sum()),
                         params_M=sum(p.numel() for p in model.parameters()) / 1e6,
                         latency_gpu_ms=latency(model, device) if device.type == "cuda" else np.nan,
                         latency_cpu_ms=latency(model, torch.device("cpu"))))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows); Path("results/pub").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/pub/cnn_repeat_eval.csv", index=False)
    agg = df.groupby("backbone").agg(n=("seed", "count"), acc_tau_mean=("acc_tau", "mean"), acc_tau_sd=("acc_tau", "std"),
                                     f1_tau_mean=("f1_tau", "mean"), battery_recall_min=("battery_recall_tau", "min"),
                                     false_alarms_mean=("false_alarms_tau", "mean"), params_M=("params_M", "first"),
                                     gpu_ms=("latency_gpu_ms", "mean"), cpu_ms=("latency_cpu_ms", "mean"))
    print(agg.round(4).to_string()); agg.to_csv("results/pub/cnn_backbones_summary.csv")
