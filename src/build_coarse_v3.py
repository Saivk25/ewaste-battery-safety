"""
build_coarse_v3.py -- coarse dataset v3: add photos of swollen, bulging, leaking and corroded batteries to the Battery
class, so the camera also recognises the damaged batteries that are the most dangerous on a line.

Why: on 44 outside Wikimedia Commons photos the v2 model missed 3 of 5 swollen or damaged batteries
(results/external/demo_photos_summary.txt); the training photos were clean product shots.

Rules (leak-safe):
  * data_v2 is copied unchanged; only the Battery class of each split gains damaged-battery photos.
  * Sources: Openverse (CC-licensed Flickr / Wikimedia photos) and, if present, the Roboflow "swollen-battery"
    set (CC BY 4.0). Every photo was checked by eye; the accepted list and the reason for each rejection are in
    data_ext/damaged_vetting.csv (one row per candidate).
  * Several shots of one battery by one photographer are one group; groups (not photos) are split 60/20/20, so no
    battery appears in two splits.
  * The 44 Commons demo photos are the untouched outside test: any candidate whose MD5 or dHash (Hamming <= 4)
    matches one of them, or whose source page is the same, is dropped.
  * Afterwards every file is fingerprinted (MD5) and no content may repeat across splits.

Run from the repo root:  python src/build_coarse_v3.py --vetting data_ext/damaged_vetting.csv --raw C:/ewaste_ext/damaged_raw
                         [--roboflow C:/ewaste_ext/roboflow_swollen]
Outputs: data_v3/coarse/{train,val,test}/<class>/, results/gaps/coarse_v3_build.json
"""
import argparse, collections, hashlib, json, random, re, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from external_photo_test import dhash, nearest_hamming, images

SRC, DST = Path("data_v2"), Path("data_v3")
OVERSAMPLE = 8
DEMO = Path("C:/ewaste_ext/demo_photos")


def group_key(row):
    """Same photographer + same title stem (numbers stripped) = same battery."""
    stem = re.sub(r"[\d_\-#()]+", " ", str(row.title)).lower()
    stem = re.sub(r"\b(img|dsc|jpg|photo|pic)\b", " ", stem)
    return f"{row.provider}|{row.creator}|{' '.join(stem.split()[:4])}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vetting", required=True); ap.add_argument("--raw", required=True); ap.add_argument("--roboflow")
    a = ap.parse_args()
    random.seed(42)
    v = pd.read_csv(a.vetting)
    src = pd.read_csv(Path(a.raw) / "sources.csv").merge(v, on="file")
    keep = src[src.accept == 1].copy()
    keep["path"] = [Path(a.raw) / f for f in keep.file]
    keep["group"] = [group_key(r) for r in keep.itertuples()]
    if a.roboflow:
        rf = [p for p in images(a.roboflow) if "__MACOSX" not in str(p)]
        # Roboflow exports augment each source image; the source id is the part before '.rf.' / '_jpg'
        ids = [re.sub(r"(_jpg|_png|_jpeg)?\.rf\..*$", "", p.stem) for p in rf]
        keep = pd.concat([keep, pd.DataFrame(dict(file=[p.name for p in rf], path=rf, group=[f"roboflow|{i}" for i in ids],
                                                  license="cc-by 4.0", source="https://universe.roboflow.com/datasettocode/swollen-battery",
                                                  query="roboflow swollen-battery", accept=1))], ignore_index=True)

    # drop anything that is (or shows the same battery as) an outside test photo
    demo = pd.read_csv(DEMO / "sources.csv")
    demo_md5 = {hashlib.md5((DEMO / f).read_bytes()).hexdigest() for f in demo.file}
    demo_dh = np.array([dhash(Image.open(DEMO / f)) for f in demo.file], dtype=np.uint64)
    md5 = [hashlib.md5(Path(p).read_bytes()).hexdigest() for p in keep.path]
    ham = [nearest_hamming(dhash(Image.open(p)), demo_dh) for p in keep.path]
    clash = [(m in demo_md5) or (h <= 4) or (str(s) in set(demo.source)) for m, h, s in zip(md5, ham, keep.source)]
    dropped_clash = int(np.sum(clash))
    keep = keep[~np.array(clash)].copy(); keep["md5"] = [m for m, c in zip(md5, clash) if not c]
    keep = keep.drop_duplicates("md5")

    groups = sorted(keep.group.unique()); random.shuffle(groups)
    n = len(groups); cut1, cut2 = int(round(0.6 * n)), int(round(0.8 * n))
    split_of = {g: ("train" if i < cut1 else "val" if i < cut2 else "test") for i, g in enumerate(groups)}
    keep["split"] = keep.group.map(split_of)

    if DST.exists(): shutil.rmtree(DST)
    shutil.copytree(SRC, DST)
    for r in keep.itertuples():
        out = DST / "coarse" / r.split / "Battery" / f"damaged_{Path(r.path).stem}.jpg"
        Image.open(r.path).convert("RGB").save(out, quality=95)

    # 22-ish damaged photos among ~2,000 battery photos would barely register; repeat them in TRAIN only
    # (random augmentation makes each copy different; val/test keep one copy, so nothing leaks)
    for r in keep[keep.split == "train"].itertuples():
        f = DST / "coarse" / "train" / "Battery" / f"damaged_{Path(r.path).stem}.jpg"
        for k in range(1, OVERSAMPLE):
            shutil.copy2(f, f.with_name(f"{f.stem}_rep{k}.jpg"))

    seen = collections.defaultdict(set)
    for p in (DST / "coarse").rglob("*.*"):
        seen[hashlib.md5(p.read_bytes()).hexdigest()].add(p.parts[-3])   # p.parts[-3] is the split
    cross = sum(1 for s in seen.values() if len(s) > 1)
    assert cross == 0, f"{cross} files repeat across splits"
    counts = keep.split.value_counts().to_dict()
    info = dict(added_per_split=counts, train_repeats=OVERSAMPLE, groups=n, dropped_as_outside_test_clash=dropped_clash, cross_split_duplicates=cross,
                per_query=keep.groupby(["split", "query"]).size().reset_index().values.tolist())
    Path("results/gaps").mkdir(parents=True, exist_ok=True)
    json.dump(info, open("results/gaps/coarse_v3_build.json", "w"), indent=1, default=str)
    keep[["file", "split", "group", "query", "license", "source"]].to_csv("data_ext/damaged_used.csv", index=False)
    print(json.dumps(info, indent=1, default=str))


if __name__ == "__main__":
    main()
