"""
external_photo_test.py -- coarse CNN on photos it has never seen, from two outside datasets.

1. BatSort / Singapore Battery Dataset (Zhao et al. 2024, arXiv 2404.05802,
   github.com/FriedrichZhao/Singapore_Battery_Dataset): ~360 web photos of 10 battery types,
   including Ni-MH (Exell_Green) and Ni-Cd (Exell_Yellow), alkaline, lithium primary and Li-ion
   (Klarus 18650). Our datasets have no Ni-MH / Ni-Cd photos, so this checks that the coarse
   "Battery" decision (the one that triggers the divert) is not specific to Li-ion.
2. XBAT+ v1.0 RGB (Rukundo et al. 2026, Scientific Data, doi:10.5281/zenodo.18022530, CC BY 4.0):
   421 webcam photos of real small WEEE devices from 15 battery-containing categories, sorted from
   Irish collection drums. The battery is inside the device, so this measures how often the camera
   alone would divert a battery-containing device -- the case the sensor hood exists for.

Near-duplicates: web photos may also be in Garbage-12. Every external image is compared with all
coarse train/val/test images by MD5 and by a 64-bit difference hash; images within Hamming distance
DUP_HAMMING of any of ours are reported and excluded from the headline numbers.

Run: python src/external_photo_test.py --batsort C:/ewaste_ext/batsort/repo --xbat C:/ewaste_ext/xbat
Outputs: results/external/{summary.txt, batsort.csv, xbat.csv}
"""
import argparse, hashlib, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image

from evaluate_model import load_checkpoint, get_test_transform, DATA_DIR, BATTERY_THRESHOLD, device

OUT = Path("results/external")
DUP_HAMMING = 4
IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CHEM = {"Duracell_Alk": "alkaline", "Energizer_Alk": "alkaline", "GP_Alk": "alkaline", "IKEA_Alk": "alkaline",
        "Energizer_Ind": "alkaline (industrial)", "Energizer_Li": "lithium primary",
        "Exell_Green": "Ni-MH", "Exell_Yellow": "Ni-Cd", "Klarus": "Li-ion 18650", "Other": "other"}
XBAT_CATS = ["AlarmClock", "ComputerMouse", "E-Razor", "E-Toothbrush", "FireAlarm", "FlashLight", "GameController",
             "iPad", "KidToy", "MobilePhone", "NewsRadio", "NightTorch", "PhotoCamera", "RemoteControl", "Thermometer"]


def images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG)


def dhash(img, n=8):
    g = np.asarray(img.convert("L").resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    return int("".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).ravel()), 2)


def fingerprints(paths):
    md5, dh = [], []
    for p in paths:
        md5.append(hashlib.md5(p.read_bytes()).hexdigest())
        with Image.open(p) as im: dh.append(dhash(im))
    return md5, np.array(dh, dtype=np.uint64)


def nearest_hamming(q, ref):
    x = np.bitwise_xor(ref, np.uint64(q))
    return int(min(bin(int(v)).count("1") for v in x)) if len(ref) else 64


def classify(model, paths, bi, ei):
    tf = get_test_transform(); pb, pe, top = [], [], []
    with torch.no_grad():
        for i in range(0, len(paths), 32):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + 32]]).to(device)
            P = torch.softmax(model(x), 1).cpu().numpy()
            pb += list(P[:, bi]); pe += list(P[:, ei]); top += list(P.argmax(1))
    return np.array(pb), np.array(pe), np.array(top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batsort", required=True); ap.add_argument("--xbat", required=True)
    a = ap.parse_args()
    model, classes, _ = load_checkpoint("EfficientNet-B0", "coarse")
    bi, ei = classes.index("Battery"), classes.index("Electronics")
    ours = images(DATA_DIR / "coarse")
    o_md5, o_dh = fingerprints(ours); o_md5 = set(o_md5)

    def run(paths):
        md5, dh = fingerprints(paths)
        ham = [nearest_hamming(h, o_dh) for h in dh]
        pb, pe, top = classify(model, paths, bi, ei)
        return pd.DataFrame(dict(path=[str(p) for p in paths], P_battery=pb, P_electronics=pe,
                                 top=[classes[t] for t in top], md5_dup=[m in o_md5 for m in md5],
                                 hamming=ham, dup=[m in o_md5 or h <= DUP_HAMMING for m, h in zip(md5, ham)]))

    bs = images(a.batsort); b = run(bs)
    b["folder"] = [Path(p).relative_to(a.batsort).parts[0] for p in b.path]; b["chemistry"] = b.folder.map(CHEM)
    xs = [p for p in images(a.xbat) if "RGB" in str(p)]; x = run(xs)
    lab = [p.parent.parent / "labels" / (p.stem + ".txt") for p in xs]
    x["category"] = [XBAT_CATS[int(l.read_text().split()[0])] if l.exists() else "?" for l in lab]
    x["split"] = ["test" if "RGB_test" in str(p) else "train" for p in xs]
    OUT.mkdir(parents=True, exist_ok=True); b.to_csv(OUT / "batsort.csv", index=False); x.to_csv(OUT / "xbat.csv", index=False)

    t = BATTERY_THRESHOLD; L = []
    L.append(f"Coarse EfficientNet-B0, Battery if P_battery >= {t} (the divert threshold). Near-duplicate = same MD5 or "
             f"dHash Hamming <= {DUP_HAMMING} to any of our {len(ours)} coarse images; excluded from the counts.")
    L.append(f"\nBatSort (web photos, {len(b)} images, {int(b.dup.sum())} near-duplicates of ours, {int(b.md5_dup.sum())} exact)")
    for chem, g in b[~b.dup].groupby("chemistry"):
        L.append(f"  {chem:22s} n={len(g):3d}  detected as battery {np.mean(g.P_battery >= t):.3f}  "
                 f"({int((g.P_battery >= t).sum())}/{len(g)}), median P {g.P_battery.median():.3f}")
    nb = b[~b.dup]
    L.append(f"  {'ALL':22s} n={len(nb):3d}  detected as battery {np.mean(nb.P_battery >= t):.3f}")
    for chem in ("Ni-MH", "Ni-Cd"):
        g = b[b.chemistry == chem]
        L.append(f"  {chem} including near-duplicates: {int((g.P_battery >= t).sum())}/{len(g)}")
    L.append(f"  ALL including near-duplicates: {int((b.P_battery >= t).sum())}/{len(b)} = {np.mean(b.P_battery >= t):.3f}")
    L.append(f"\nXBAT+ RGB (real WEEE devices that hold a battery, {len(x)} images, {int(x.dup.sum())} near-duplicates)")
    nx = x[~x.dup]
    L.append(f"  camera alone diverts (P_battery >= {t}): {np.mean(nx.P_battery >= t):.3f} ({int((nx.P_battery >= t).sum())}/{len(nx)})")
    L.append(f"  top class: " + ", ".join(f"{k} {v}" for k, v in nx.top.value_counts().items()))
    for cat, g in nx.groupby("category"):
        L.append(f"  {cat:15s} n={len(g):3d}  diverted by camera {np.mean(g.P_battery >= t):.3f}  top={g.top.mode()[0]}")
    txt = "\n".join(L); print(txt); (OUT / "summary.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
