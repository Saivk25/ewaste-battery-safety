"""
eval_coarse_v3.py -- the v2 coarse model (production until now) against v3 (trained with damaged-battery photos).

Both models get their own battery threshold, chosen on the v3 validation set with the production rule (smallest
threshold with validation false-alarm rate <= 3.05%). Then, on photos neither model was trained on:
  * data_v3 test split: accuracy with the threshold, battery recall (consumer, Li-ion, damaged), false alarms;
  * the 44 Wikimedia Commons demo photos (results/external/demo_photos.csv labels), including 5 swollen batteries;
  * BatSort (other chemistries) and XBAT+ (real devices), near-duplicates of our data removed.

Run: python src/eval_coarse_v3.py --v3 runs_cnn/v3_B0_s42/checkpoints/best_EfficientNetB0_coarse.pth
Outputs: results/gaps/coarse_v3_eval.json, coarse_v3_eval.txt
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_model import load_checkpoint, get_test_transform, device
from gap_cnn_checks import softmax_all

FA_CAP = 0.0305


def probs(model, paths, bi):
    tf = get_test_transform(); out = []
    with torch.no_grad():
        for i in range(0, len(paths), 32):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + 32]]).to(device)
            out.append(torch.softmax(model(x), 1)[:, bi].cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--v3", required=True)
    ap.add_argument("--v2", default="checkpoints/best_EfficientNetB0_coarse.pth"); a = ap.parse_args()
    demo = pd.read_csv("results/external/demo_photos.csv")
    bs = pd.read_csv("results/external/batsort.csv"); bs = bs[~bs.dup]
    xb = pd.read_csv("results/external/xbat.csv"); xb = xb[~xb.dup]
    res, L = {}, []
    for tag, path in (("v2", a.v2), ("v3", a.v3)):
        model, classes, _ = load_checkpoint("EfficientNet-B0", "coarse", path=path); bi = classes.index("Battery")
        val = datasets.ImageFolder("data_v3/coarse/val", get_test_transform())
        Pv, Yv = softmax_all(model, val); pv, yv = Pv[:, bi], Yv == bi
        tau = float(next(t for t in sorted(np.unique(np.r_[pv, 0.5])) if (pv[~yv] >= t).mean() <= FA_CAP))
        test = datasets.ImageFolder("data_v3/coarse/test", get_test_transform())
        Pt, Yt = softmax_all(model, test); pt = Pt[:, bi]; yb = Yt == bi
        pred = Pt.argmax(1); pred[pt >= tau] = bi
        names = np.array([Path(p).name for p, _ in test.samples])
        dmg = yb & np.char.startswith(names, "damaged_")
        r = dict(threshold=tau, test_accuracy=float((pred == Yt).mean()), n_test=int(len(Yt)),
                 battery_recall=float((pt[yb] >= tau).mean()), n_battery=int(yb.sum()),
                 damaged_test_detected=[int((pt[dmg] >= tau).sum()), int(dmg.sum())],
                 false_alarms=[int((pt[~yb] >= tau).sum()), int((~yb).sum())])
        pd_ = probs(model, [Path("C:/ewaste_ext/demo_photos") / f for f in demo.file], bi); dd = pd_ >= tau
        sw = demo.category.str.startswith("Swollen").values
        r["commons"] = dict(batteries=[int(dd[(demo.expected == "battery").values].sum()), int((demo.expected == "battery").sum())],
                            swollen=[int(dd[sw].sum()), int(sw.sum())],
                            devices=[int(dd[(demo.expected == "hidden").values].sum()), int((demo.expected == "hidden").sum())],
                            non_battery_correct=[int((~dd[(demo.expected == "not battery").values]).sum()), int((demo.expected == "not battery").sum())],
                            per_photo=dict(zip(demo.file, np.round(pd_, 3).tolist())))
        pb = probs(model, list(bs.path), bi); px = probs(model, list(xb.path), bi)
        r["batsort_detected"] = [int((pb >= tau).sum()), len(pb)]
        r["batsort_nimh_nicd"] = [int((pb[bs.chemistry.isin(["Ni-MH", "Ni-Cd"]).values] >= tau).sum()), int(bs.chemistry.isin(["Ni-MH", "Ni-Cd"]).sum())]
        r["xbat_diverted"] = [int((px >= tau).sum()), len(px)]
        res[tag] = r
        L.append(f"== {tag}  threshold {tau:.4f}")
        L.append(f"   data_v3 test: accuracy {r['test_accuracy']:.4f} (n={r['n_test']}), battery recall {r['battery_recall']:.4f} "
                 f"(n={r['n_battery']}), damaged batteries {r['damaged_test_detected'][0]}/{r['damaged_test_detected'][1]}, "
                 f"false alarms {r['false_alarms'][0]}/{r['false_alarms'][1]}")
        c = r["commons"]
        L.append(f"   Commons photos: batteries {c['batteries'][0]}/{c['batteries'][1]}, swollen {c['swollen'][0]}/{c['swollen'][1]}, "
                 f"devices {c['devices'][0]}/{c['devices'][1]}, non-batteries correct {c['non_battery_correct'][0]}/{c['non_battery_correct'][1]}")
        L.append(f"   BatSort {r['batsort_detected'][0]}/{r['batsort_detected'][1]} (Ni-MH + Ni-Cd {r['batsort_nimh_nicd'][0]}/{r['batsort_nimh_nicd'][1]}), "
                 f"XBAT+ devices diverted {r['xbat_diverted'][0]}/{r['xbat_diverted'][1]}")
    txt = "\n".join(L); print(txt)
    json.dump(res, open("results/gaps/coarse_v3_eval.json", "w"), indent=1)
    Path("results/gaps/coarse_v3_eval.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
