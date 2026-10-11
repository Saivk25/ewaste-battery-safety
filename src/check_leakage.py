"""
check_leakage.py — Detects real train/val/test leakage: exact byte-identical
images (MD5) appearing in more than one split.

Covers both the fine (3-class Li-ion subtype) dataset — the one that had a
confirmed leak, fixed via rebalance_fine_MERGED.py's MD5-based union-find
component splitting — and the coarse (6-class) dataset, whose split (built
by dataset_preparation.py) was never independently checked before.

Uses MD5, not perceptual hashing (phash). This dataset is low-detail,
uniform product photography with a small fixed set of augmentation presets
reused across every source photo; verified directly (mean pixel difference
~65/255, clearly different images) that phash produces massive false
positives even at hamming distance 0 — two genuinely different source
photos, put through the same rotation+brightness preset, can collide to
the exact same 64-bit phash. rebalance_fine_MERGED.py's own docstring
already documents this ("some phash-similar-but-different photos across
splits are expected and NOT leakage on this dataset") — MD5 exact-byte
match is the unambiguous signal, which is what the actual fix is built on.

Run from your project root:
  python check_leakage.py            # both datasets
  python check_leakage.py fine       # one dataset only
  python check_leakage.py coarse
"""
import hashlib
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DATASETS = {
    "fine":   (Path("data/fine"),   ["LiIon_18650", "LiIon_Pouch", "LiPo"]),
    "coarse": (Path("data/coarse"), ["Battery", "Cardboard_Trash", "Glass", "Metal", "Paper", "Plastic"]),
}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def get_hashes(folder: Path):
    out = {}
    if not folder.exists():
        return out
    for f in folder.rglob("*"):
        if f.suffix.lower() in IMG_EXTS:
            try:
                out[f] = md5_of(f)
            except Exception:
                pass
    return out


def check_dataset(name: str, base_dir: Path, classes: list) -> int:
    print("=" * 70)
    print(f"  EXACT-DUPLICATE (MD5) LEAKAGE CHECK: {name}")
    print("=" * 70)

    total_leaks = 0
    for cls in classes:
        train_h = get_hashes(base_dir / "train" / cls)
        val_h   = get_hashes(base_dir / "val" / cls)
        test_h  = get_hashes(base_dir / "test" / cls)

        print(f"\n{cls}: train={len(train_h)} val={len(val_h)} test={len(test_h)} images hashed")

        train_hashes = set(train_h.values())
        for split_name, split_h in [("val", val_h), ("test", test_h)]:
            leaks = [(f, h) for f, h in split_h.items() if h in train_hashes]
            if leaks:
                print(f"  ⚠ {len(leaks)} exact byte-identical file(s) also present in train")
                for f, h in leaks[:5]:
                    train_match = next(tf.name for tf, th in train_h.items() if th == h)
                    print(f"      train/{train_match}  ==  {split_name}/{f.name}  (MD5 {h[:12]}...)")
                if len(leaks) > 5:
                    print(f"      ... and {len(leaks)-5} more")
                total_leaks += len(leaks)
            else:
                print(f"  ✓ No exact-duplicate files found between train and {split_name}")

    print("\n" + "=" * 70)
    if total_leaks > 0:
        print(f"  RESULT ({name}): LEAKAGE CONFIRMED — {total_leaks} exact-duplicate file(s) across splits.")
    else:
        print(f"  RESULT ({name}): clean — no exact-duplicate files span train and val/test.")
    print("=" * 70)
    return total_leaks


if __name__ == "__main__":
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None
    results = {}
    for name, (base_dir, classes) in DATASETS.items():
        if only and name != only:
            continue
        results[name] = check_dataset(name, base_dir, classes)
        print()

    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for name, count in results.items():
        status = f"LEAKAGE CONFIRMED ({count} exact duplicates)" if count > 0 else "clean"
        print(f"  {name:8}: {status}")
