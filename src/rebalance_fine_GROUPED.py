"""
rebalance_fine_GROUPED.py
==========================
Real fix for the leak confirmed by check_leakage.py.

battery_raw/{cls} contains already-augmented images (rotation/brightness
baked into pixels, e.g. "rb24_train_10_1_1_cyl_rotated_15_bright_0.5.jpg").
Multiple such siblings share one true source photo. Splitting at the file
level (even with content-hash dedup) scatters siblings across train/val/test
-> leakage. This script groups by source photo ID (everything before
"_rotated_..._bright_...") and splits at the GROUP level instead.

No further augmentation is applied — the data is already augmented enough.

Run: python rebalance_fine_GROUPED.py
Then: python train_cnn.py --stage fine --backbone EfficientNet-B0
"""
import re, shutil, random
from pathlib import Path
from collections import defaultdict

FINE_SRC = Path("data/raw/battery_raw")
FINE_DST = Path("data/fine")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = ["LiIon_18650", "LiIon_Pouch", "LiPo"]
SPLIT = (0.80, 0.10, 0.10)
MAX_GROUPS_PER_CLASS = 600   # cap for balance + reasonable train time
random.seed(42)

GROUP_PATTERN = re.compile(r"_rotated_-?\d+_bright_[\d.]+$", re.IGNORECASE)


def group_id(stem: str) -> str:
    """Strip the augmentation suffix to recover the true source-photo ID."""
    return GROUP_PATTERN.sub("", stem)


def get_images(folder: Path):
    return [f for f in folder.rglob("*") if f.suffix.lower() in IMG_EXTS]


print("=" * 65)
print("  GROUP-AWARE FINE DATASET SPLIT (fixes augmentation-sibling leak)")
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
    groups = defaultdict(list)
    for img in imgs:
        gid = group_id(img.stem)
        groups[gid].append(img)

    group_ids = list(groups.keys())
    random.shuffle(group_ids)

    n_groups_total = len(group_ids)
    print(f"\n{cls}: {len(imgs)} files -> {n_groups_total} unique source photos "
          f"(avg {len(imgs)/max(n_groups_total,1):.1f} augmented siblings each)")

    # Cap number of groups used (keeps train time reasonable, balances classes)
    use_groups = group_ids[:MAX_GROUPS_PER_CLASS]
    n = len(use_groups)
    n_train = int(n * SPLIT[0])
    n_val   = int(n * SPLIT[1])
    train_g = use_groups[:n_train]
    val_g   = use_groups[n_train:n_train + n_val]
    test_g  = use_groups[n_train + n_val:]

    for split_name, group_list in [("train", train_g), ("val", val_g), ("test", test_g)]:
        dst = FINE_DST / split_name / cls
        count = 0
        for gid in group_list:
            for img_path in groups[gid]:
                out = dst / img_path.name
                if not out.exists():
                    try:
                        shutil.copy2(img_path, out)
                        count += 1
                    except Exception:
                        pass
        print(f"  {split_name:5}: {len(group_list):4} source photos -> {count:5} files copied")

print("\n" + "=" * 65)
print("  Done. Every file in val/test comes from a source photo whose")
print("  siblings do NOT appear anywhere in train. Verify with:")
print("  python check_leakage.py")
print("=" * 65)
