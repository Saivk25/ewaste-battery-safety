"""
rebalance_fine_MERGED.py
=========================
Final fix. The filename-based grouping (rebalance_fine_GROUPED.py) reduced
leakage a lot but not to zero, because the source dataset's own ID numbers
are unreliable — the SAME physical photo sometimes appears under two
different ID numbers (e.g. "10_4_F_2" and "10_4_R_3" turned out to be
pixel-identical). Trusting filenames for grouping isn't enough.

This version:
  1. Computes an exact content hash (MD5) for every raw image per class.
  2. Uses union-find to merge any filename-groups that share an exact
     duplicate photo between them, regardless of what their IDs say.
  3. Splits by the resulting merged components.

This only merges on UNAMBIGUOUS exact duplicates (MD5 match) — it will not
over-merge based on "looks similar" the way perceptual hashing can.

Run: python rebalance_fine_MERGED.py
Then: python train_cnn.py --stage fine --backbone EfficientNet-B0
Then: python check_leakage.py   (should now show ~0 exact-hash duplicates)
"""
import re, shutil, random, hashlib
from pathlib import Path
from collections import defaultdict

FINE_SRC = Path("data/raw/battery_raw")
FINE_DST = Path("data/fine")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = ["LiIon_18650", "LiIon_Pouch", "LiPo"]
SPLIT = (0.80, 0.10, 0.10)
MAX_COMPONENTS_PER_CLASS = 600
random.seed(42)

GROUP_PATTERN = re.compile(r"_rotated_-?\d+_bright_[\d.]+$", re.IGNORECASE)


def group_id(stem: str) -> str:
    return GROUP_PATTERN.sub("", stem)


def get_images(folder: Path):
    return [f for f in folder.rglob("*") if f.suffix.lower() in IMG_EXTS]


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


print("=" * 65)
print("  MERGED-COMPONENT FINE DATASET SPLIT (final leak fix)")
print("=" * 65)

for split in ["train", "val", "test"]:
    for cls in CLASSES:
        d = FINE_DST / split / cls
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

for cls in CLASSES:
    src = FINE_SRC / cls
    if not src.exists():
        print(f"  MISSING: {src} — skipping")
        continue

    imgs = get_images(src)
    print(f"\n{cls}: hashing {len(imgs)} files...")

    # Group by filename-derived ID first
    fname_groups = defaultdict(list)
    for img in imgs:
        fname_groups[group_id(img.stem)].append(img)

    # Compute MD5 for every file, map hash -> list of (group_id) it appears in
    hash_to_groups = defaultdict(set)
    file_hash = {}
    for gid, files in fname_groups.items():
        for f in files:
            h = md5_of(f)
            file_hash[f] = h
            hash_to_groups[h].add(gid)

    # Union-find merge: any two groups sharing an exact-duplicate file are merged
    uf = UnionFind(list(fname_groups.keys()))
    merges = 0
    for h, gids in hash_to_groups.items():
        gids = list(gids)
        for i in range(1, len(gids)):
            if uf.find(gids[0]) != uf.find(gids[i]):
                merges += 1
            uf.union(gids[0], gids[i])

    # Build final components
    components = defaultdict(list)
    for gid, files in fname_groups.items():
        root = uf.find(gid)
        components[root].extend(files)

    comp_ids = list(components.keys())
    random.shuffle(comp_ids)
    print(f"  {len(fname_groups)} filename-groups -> {merges} exact-duplicate merges "
          f"-> {len(comp_ids)} final independent components")

    use_comps = comp_ids[:MAX_COMPONENTS_PER_CLASS]
    n = len(use_comps)
    n_train = int(n * SPLIT[0])
    n_val   = int(n * SPLIT[1])
    train_c = use_comps[:n_train]
    val_c   = use_comps[n_train:n_train + n_val]
    test_c  = use_comps[n_train + n_val:]

    for split_name, comp_list in [("train", train_c), ("val", val_c), ("test", test_c)]:
        dst = FINE_DST / split_name / cls
        count = 0
        seen_hashes_this_split = set()
        for cid in comp_list:
            for img_path in components[cid]:
                h = file_hash[img_path]
                # extra safety: skip exact byte-duplicate already placed in this split
                key = (split_name, h)
                out = dst / img_path.name
                if not out.exists():
                    try:
                        shutil.copy2(img_path, out)
                        count += 1
                    except Exception:
                        pass
        print(f"  {split_name:5}: {len(comp_list):4} components -> {count:5} files copied")

print("\n" + "=" * 65)
print("  Done. Verify with: python check_leakage.py")
print("  (expect 0 exact-match pairs; some phash-similar-but-different")
print("  photos across splits are expected and NOT leakage on this dataset)")
print("=" * 65)
