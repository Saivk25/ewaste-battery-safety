"""
demo_photo_test.py -- the trained coarse + fine models on freely licensed Wikimedia Commons photos that are not in
any of our datasets, to know before the review which photos pass, fail or are borderline.

Expected outcome per photo: "battery" (should reach P_battery >= BATTERY_THRESHOLD), "not battery" (should stay below it), or
"hidden" (a device with a battery inside: the camera may or may not see it; the sensor hood exists for this case).
Verdict: PASS / FAIL, and BORDERLINE when BATTERY_THRESHOLD <= P_battery < 0.5 (detected only thanks to the low threshold).
Near-duplicates of our coarse images (dHash Hamming <= 4) are flagged.

Run: python src/demo_photo_test.py --photos C:/ewaste_ext/demo_photos
     (the folder holds the photos and sources.csv written by the download script; licences and authors are kept)
Outputs: results/external/demo_photos.csv, demo_photos_summary.txt
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_model import load_checkpoint, get_test_transform, DATA_DIR, BATTERY_THRESHOLD, device
from external_photo_test import images, fingerprints, nearest_hamming, DUP_HAMMING

FIX = {  # search hits that were not what their search term said (checked by eye)
    "12_Phone_battery.jpg": ("Phone-charging kiosk", "not battery"),
    "13_Laptop_battery_pack.jpg": ("Coin cell", "battery"),
    "19_Power_bank.jpg": ("Tool battery pack", "battery"),
    "20_Power_bank.jpg": ("Metal mesh grille", "not battery"),
    "33_Lipstick.jpg": ("Person (no battery)", "not battery"),
    "34_Lipstick.jpg": ("Person (no battery)", "not battery"),
}
DROP = {"24_Lead-acid_battery.jpg", "37_Aluminium_can.jpg"}   # diagrams, not photos


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--photos", required=True); a = ap.parse_args()
    src = pd.read_csv(Path(a.photos) / "sources.csv")
    src = src[~src.file.isin(DROP)].copy()
    for f, (cat, exp) in FIX.items():
        src.loc[src.file == f, ["category", "expected"]] = cat, exp
    paths = [Path(a.photos) / f for f in src.file]

    coarse, cc, _ = load_checkpoint("EfficientNet-B0", "coarse"); fine, fc, _ = load_checkpoint("EfficientNet-B0", "fine")
    bi = cc.index("Battery"); tf = get_test_transform()
    with torch.no_grad():
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths]).to(device)
        Pc = torch.softmax(coarse(x), 1).cpu().numpy(); Pf = torch.softmax(fine(x), 1).cpu().numpy()
    ours = images(DATA_DIR / "coarse"); _, o_dh = fingerprints(ours)
    _, dh = fingerprints(paths)

    src["P_battery"] = Pc[:, bi].round(3)
    src["top_class"] = [cc[i] for i in Pc.argmax(1)]
    src["top_P"] = Pc.max(1).round(3)
    src["subtype_if_battery"] = [fc[j] if p >= BATTERY_THRESHOLD else "" for j, p in zip(Pf.argmax(1), Pc[:, bi])]
    src["near_duplicate"] = [nearest_hamming(h, o_dh) <= DUP_HAMMING for h in dh]
    det = src.P_battery >= BATTERY_THRESHOLD

    def verdict(r, d):
        if r.expected == "hidden": return "SEEN" if d else "SENSORS NEEDED"
        ok = d if r.expected == "battery" else not d
        if not ok: return "FAIL"
        return "BORDERLINE" if d and r.P_battery < 0.5 else "PASS"
    src["verdict"] = [verdict(r, d) for r, d in zip(src.itertuples(), det)]
    out = Path("results/external"); out.mkdir(parents=True, exist_ok=True)
    src.to_csv(out / "demo_photos.csv", index=False)

    L = [f"Coarse model on {len(src)} Wikimedia Commons photos not in our data (Battery if P_battery >= {BATTERY_THRESHOLD}; "
         f"BORDERLINE = detected with P below 0.5; {int(src.near_duplicate.sum())} near-duplicates of our images).", ""]
    for exp, g in src.groupby("expected", sort=False):
        L.append(f"{exp}: " + ", ".join(f"{v} {n}" for v, n in g.verdict.value_counts().items()))
    L.append("")
    for r in src.itertuples():
        L.append(f"  {r.verdict:<14} {r.file:<36} {r.category:<34} P_battery {r.P_battery:.3f}  top {r.top_class} {r.top_P:.2f}"
                 + (f"  subtype {r.subtype_if_battery}" if r.subtype_if_battery else "") + ("  [near-duplicate]" if r.near_duplicate else ""))
    txt = "\n".join(L); print(txt); (out / "demo_photos_summary.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
