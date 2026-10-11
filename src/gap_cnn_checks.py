"""
gap_cnn_checks.py -- review-gap checks on the trained CNNs (no retraining).

1. Calibration of the coarse model on its own test set (ECE, reliability bins).
2. Confidence intervals: Wilson 95% for accuracy, Clopper-Pearson for battery recall.
3. Domain check: the coarse model's battery probability on the RecyBat fine-stage
   test photos (18650 / pouch / LiPo). The simulation feeds exactly these photos to
   the coarse model, but the coarse test set's batteries come from TrashNet/Garbage-12.

Run from the repo root:  python src/gap_cnn_checks.py
Usage: python src/gap_cnn_checks.py [--checkpoint PATH] [--tag NAME]
Outputs: results/gaps/cnn_checks_<tag>.json, results/gaps/coarse_P_on_fine_test_<tag>.npy
"""
import sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, torch
from torch.utils.data import DataLoader
from torchvision import datasets
from scipy import stats
from evaluate_model import load_checkpoint, get_test_transform, DATA_DIR, device

OUT = Path("results/gaps"); OUT.mkdir(parents=True, exist_ok=True)


def softmax_all(model, ds):
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    P, Y = [], []
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            P.append(torch.softmax(model(x.to(device)), 1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1); tot = 0; rows = []
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            gap = abs(correct[m].mean() - conf[m].mean()); tot += m.sum() / len(conf) * gap
            rows.append([float(edges[i]), float(edges[i + 1]), int(m.sum()), float(conf[m].mean()), float(correct[m].mean())])
    return float(tot), rows


def main(ckpt_path=None, tag="v1"):
    res = {}
    model, classes, _ = load_checkpoint("EfficientNet-B0", "coarse", ckpt_path)
    bi = classes.index("Battery")
    ds = datasets.ImageFolder(DATA_DIR / "coarse" / "test", get_test_transform())
    assert ds.classes == classes
    P, Y = softmax_all(model, ds)
    pred = P.argmax(1); conf = P.max(1); correct = (pred == Y).astype(float)
    n = len(Y); k = int(correct.sum())
    e, rel = ece(conf, correct)
    res["coarse_test"] = {"n": n, "accuracy": k / n, "accuracy_wilson95": wilson(k, n), "ece_10bin": e, "mean_confidence": float(conf.mean()), "reliability_bins": rel}
    nb = int((Y == bi).sum()); tp = int(((pred == bi) & (Y == bi)).sum())
    res["coarse_test"]["battery"] = {"n": nb, "recall": tp / nb, "fn": nb - tp, "fnr_clopper_pearson95": cp(nb - tp, nb),
                                     "rule_of_three_upper": 3 / nb if nb - tp == 0 else None}
    nonb = n - nb; fp = int(((pred == bi) & (Y != bi)).sum())
    res["coarse_test"]["non_battery"] = {"n": nonb, "false_alarms": fp, "fpr": fp / nonb, "fpr_clopper_pearson95": cp(fp, nonb)}
    print("coarse test: acc %.4f Wilson95 %.4f-%.4f | ECE %.4f | battery recall %d/%d, FNR CP95 %.4f-%.4f" %
          (k / n, *res["coarse_test"]["accuracy_wilson95"], e, tp, nb, *res["coarse_test"]["battery"]["fnr_clopper_pearson95"]))

    # domain check on the fine-stage test photos (all batteries by construction)
    fds = datasets.ImageFolder(DATA_DIR / "fine" / "test", get_test_transform())
    Pf, Yf = softmax_all(model, fds)
    pb = Pf[:, bi]
    np.save(OUT / f"coarse_P_on_fine_test_{tag}.npy", pb)
    res["coarse_on_fine_test"] = {"n": len(pb), "recall_argmax": float((Pf.argmax(1) == bi).mean()), "mean_P_battery": float(pb.mean()),
                                  "median_P_battery": float(np.median(pb)), "p10": float(np.percentile(pb, 10)), "p90": float(np.percentile(pb, 90)),
                                  "share_P_above_0.5": float((pb > 0.5).mean()), "share_P_below_0.2": float((pb < 0.2).mean()), "by_subtype": {}}
    for ci, cname in enumerate(fds.classes):
        m = Yf == ci
        res["coarse_on_fine_test"]["by_subtype"][cname] = {"n": int(m.sum()), "recall_argmax": float((Pf[m].argmax(1) == bi).mean()),
                                                          "mean_P_battery": float(pb[m].mean())}
        top = np.bincount(Pf[m].argmax(1), minlength=len(classes))
        res["coarse_on_fine_test"]["by_subtype"][cname]["predicted_as"] = {classes[j]: int(top[j]) for j in range(len(classes)) if top[j]}
    print("coarse model on RecyBat battery photos:", json.dumps(res["coarse_on_fine_test"], indent=1)[:1400])
    json.dump(res, open(OUT / f"cnn_checks_{tag}.json", "w"), indent=1)
    print(f"saved results/gaps/cnn_checks_{tag}.json")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", default=None); ap.add_argument("--tag", default="v1")
    a = ap.parse_args(); main(a.checkpoint, a.tag)
