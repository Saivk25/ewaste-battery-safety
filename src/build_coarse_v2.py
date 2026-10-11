"""
build_coarse_v2.py -- coarse dataset v2: add RecyBat Li-ion photos to the Battery class.

Why: the v1 coarse Battery class held only Garbage-12 batteries (AA/9 V/car), and the v1
model recognised just 4.4% of the RecyBat Li-ion test photos the pipeline actually feeds it
(see results/gaps/cnn_checks.json).

Rules (leak-safe):
  * RecyBat splits are reused as they are (already grouped by source photo), so a source
    photo never appears in two splits: fine/train -> coarse train, fine/val -> val, fine/test -> test.
  * ONE randomly chosen augmentation per source photo (seed 42), so test/val images are
    independent photos, not 12-16 near copies of the same one.
  * The non-battery classes and the existing Garbage-12 batteries are copied unchanged.
  * Afterwards every file is fingerprinted (MD5) and no content may repeat across splits.

Run from the repo root:  python src/build_coarse_v2.py   -> data_v2/coarse/{train,val,test}/<class>/
"""
import re, shutil, hashlib, json, random, collections
from pathlib import Path

SRC = Path("data"); DST = Path("data_v2")
random.seed(42)
if DST.exists():
    shutil.rmtree(DST)
for sp in ("train", "val", "test"):
    for cls_dir in (SRC / "coarse" / sp).iterdir():
        out = DST / "coarse" / sp / cls_dir.name; out.mkdir(parents=True)
        for f in cls_dir.iterdir():
            shutil.copy2(f, out / f.name)

added = collections.Counter()
for sp in ("train", "val", "test"):
    for sub in ("LiIon_18650", "LiIon_Pouch", "LiPo"):
        by_src = collections.defaultdict(list)
        for f in (SRC / "fine" / sp / sub).iterdir():
            by_src[re.sub(r"_(rotated|bright|contrast|flip|crop|blur|occl).*$", "", f.name)].append(f)
        for src_id in sorted(by_src):
            f = random.choice(sorted(by_src[src_id]))
            shutil.copy2(f, DST / "coarse" / sp / "Battery" / f.name)
            added[(sp, sub)] += 1

# content fingerprints across splits
seen, dups = {}, []
for sp in ("train", "val", "test"):
    for f in (DST / "coarse" / sp).rglob("*"):
        if f.is_file():
            h = hashlib.md5(f.read_bytes()).hexdigest()
            if h in seen and seen[h][0] != sp:
                dups.append((str(f), seen[h]))
            seen.setdefault(h, (sp, str(f)))
counts = {sp: {c.name: len(list(c.iterdir())) for c in (DST / "coarse" / sp).iterdir()} for sp in ("train", "val", "test")}
print(json.dumps(counts, indent=1)); print("RecyBat photos added:", dict(added)); print("cross-split duplicate files:", len(dups))
json.dump({"counts": counts, "added": {f"{k[0]}/{k[1]}": v for k, v in added.items()}, "cross_split_duplicates": len(dups)},
          open(DST / "manifest.json", "w"), indent=1)
assert not dups, dups[:3]
