"""
rebalance_coarse_MERGED.py
===========================
Fixes a confirmed train/val/test leak in data/coarse: 750 exact
byte-identical files present in both train and val/test, found by
check_leakage.py.

Root cause: run_once_sort_datasets.py pools data/raw/trashnet (native
TrashNet class dirs) and data/raw/garbage12 (Kaggle's "garbage-
classification" dataset, itself built by extending TrashNet) into the
same class folders, deduplicating only by destination FILENAME
(`if not out.exists()`). Garbage-12 already contains copies of TrashNet's
own images internally under different filenames (confirmed: g12_Battery_
battery1.jpg == byte-identical to garbage12/Battery/battery1.jpg, both
945-file folders). Filename-based dedup can't catch same-content-
different-name duplicates, so the same photo enters the pool twice, and
whatever process built data/coarse's split had no way to know those two
pool entries were the same image.

This script rebuilds data/coarse from the true raw sources directly
(data/raw/trashnet's native lowercase TrashNet class dirs, and
data/raw/garbage12's native class dirs) - explicitly NOT reading
data/raw/trashnet/Battery or data/raw/trashnet/Electronics, since those
are confirmed derived copies of garbage12 content, not an independent
source - dedupes the pool by MD5 before splitting (same principle as
rebalance_fine_MERGED.py's fix for the fine dataset), then re-applies the
same per-class augmentation targets dataset_preparation.py used.

Run: python rebalance_coarse_MERGED.py
Then: python train_cnn.py --stage coarse --backbone EfficientNet-B0
Then: python check_leakage.py coarse   (should show 0 exact-duplicate files)
"""
import random, shutil, hashlib
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

RAW_DIR    = Path("data/raw")
COARSE_DST = Path("data/coarse")
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
SPLIT      = (0.80, 0.10, 0.10)
SEED       = 42
random.seed(SEED)
np.random.seed(SEED)

# Class name -> list of (source_root, prefix) to pool from. Deliberately
# excludes data/raw/trashnet/Battery and data/raw/trashnet/Electronics,
# which are confirmed derived copies of garbage12 content (see docstring).
SOURCES = {
    "Battery":     [(RAW_DIR / "garbage12" / "Battery", "g12")],
    "Cardboard_Trash": [(RAW_DIR / "trashnet" / "cardboard", "tn"),
                     (RAW_DIR / "trashnet" / "trash", "tn")],
    "Glass":       [(RAW_DIR / "trashnet" / "glass", "tn"),
                     (RAW_DIR / "garbage12" / "White-glass", "g12")],
    "Metal":       [(RAW_DIR / "trashnet" / "metal", "tn"),
                     (RAW_DIR / "garbage12" / "Metal", "g12")],
    "Paper":       [(RAW_DIR / "trashnet" / "paper", "tn"),
                     (RAW_DIR / "garbage12" / "Paper", "g12")],
    "Plastic":     [(RAW_DIR / "trashnet" / "plastic", "tn"),
                     (RAW_DIR / "garbage12" / "Plastic", "g12")],
}

AUG_TARGETS = {
    "Battery": 600, "Cardboard_Trash": 400, "Glass": 300,
    "Metal": 350, "Paper": 300, "Plastic": 350,
}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def augment_image(img: Image.Image, seed: int = 0) -> Image.Image:
    """Same augmentation ops as dataset_preparation.py, for consistency."""
    rng = random.Random(seed)
    img = img.convert("RGB")
    ops = rng.sample(range(7), k=rng.randint(2, 4))
    for op in ops:
        if op == 0:
            img = img.filter(ImageFilter.BoxBlur(rng.randint(1, 3)))
        elif op == 1:
            img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.70, 1.30))
        elif op == 2:
            img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.75, 1.35))
        elif op == 3:
            img = img.rotate(rng.uniform(-30, 30), expand=False, fillcolor=(128, 128, 128))
        elif op == 4:
            if rng.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif op == 5:
            w, h = img.size
            margin = int(min(w, h) * 0.15)
            left, top = rng.randint(0, margin), rng.randint(0, margin)
            right, bottom = w - rng.randint(0, margin), h - rng.randint(0, margin)
            img = img.crop((left, top, right, bottom)).resize((w, h), Image.BILINEAR)
        elif op == 6:
            arr = np.array(img)
            h, w = arr.shape[:2]
            cx, cy = rng.randint(w//4, 3*w//4), rng.randint(h//4, 3*h//4)
            size = rng.randint(w//8, w//4)
            x1, x2 = max(0, cx-size//2), min(w, cx+size//2)
            y1, y2 = max(0, cy-size//2), min(h, cy+size//2)
            arr[y1:y2, x1:x2] = 128
            img = Image.fromarray(arr)
    return img


print("=" * 65)
print("  COARSE DATASET REBUILD (MD5-deduped pool, clean split)")
print("=" * 65)

for split in ["train", "val", "test"]:
    for cls in SOURCES:
        d = COARSE_DST / split / cls
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

for cls, sources in SOURCES.items():
    print(f"\n{cls}:")
    pool = []  # (path, prefix)
    for src_dir, prefix in sources:
        if not src_dir.exists():
            print(f"  MISSING: {src_dir} — skipping")
            continue
        files = [f for f in src_dir.iterdir() if f.is_file() and f.suffix in IMG_EXTS]
        print(f"  {src_dir}: {len(files)} files")
        pool.extend((f, prefix) for f in files)

    # Dedupe by MD5 across the whole pool -- this is the actual fix.
    seen_hashes = {}
    deduped = []
    for f, prefix in pool:
        h = md5_of(f)
        if h in seen_hashes:
            continue
        seen_hashes[h] = f
        deduped.append((f, prefix, h))
    print(f"  Pool: {len(pool)} files -> {len(deduped)} after MD5 dedup "
          f"({len(pool)-len(deduped)} exact duplicates removed)")

    random.shuffle(deduped)
    n = len(deduped)
    n_train = int(n * SPLIT[0])
    n_val   = int(n * SPLIT[1])
    train_f = deduped[:n_train]
    val_f   = deduped[n_train:n_train+n_val]
    test_f  = deduped[n_train+n_val:]

    for split_name, split_list in [("train", train_f), ("val", val_f), ("test", test_f)]:
        dst = COARSE_DST / split_name / cls
        if split_name == "train" and AUG_TARGETS.get(cls, 0) > len(split_list):
            target = AUG_TARGETS[cls]
            created = 0
            for f, prefix, h in split_list:
                img = Image.open(f).convert("RGB").resize((256, 256))
                img.save(dst / f"orig_{prefix}_{f.stem}.jpg", quality=92)
                created += 1
            aug_idx = 0
            while created < target:
                f, prefix, h = split_list[aug_idx % len(split_list)]
                img = Image.open(f).convert("RGB").resize((256, 256))
                img = augment_image(img, seed=aug_idx + 1000)
                img.save(dst / f"aug_{cls}_{aug_idx:05d}.jpg", quality=88)
                created += 1
                aug_idx += 1
            print(f"    train: {len(split_list)} orig -> {created} (augmented)")
        else:
            for f, prefix, h in split_list:
                img = Image.open(f).convert("RGB").resize((256, 256))
                img.save(dst / f"{prefix}_{f.stem}.jpg", quality=92)
            print(f"    {split_name}: {len(split_list)} images")

print("\n" + "=" * 65)
print("  Done. Verify with: python check_leakage.py coarse")
print("  (expect 0 exact-duplicate files)")
print("=" * 65)
