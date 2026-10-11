"""
rebalance_fine_FIXED.py
========================
Fixes the data leakage in the original rebalance_fine.py.

ROOT CAUSE of the leak:
  The original script pooled images from data/raw/battery_raw/{cls} together
  WITH images already sitting in data/fine/{train,val,test}/{cls} (which by
  that point already contain augmented derivatives), deduplicated only by
  filename, then reshuffled and re-split the combined pool. Because augmented
  derivatives have different filenames from their source image (e.g.
  "orig_IMG001.jpg" vs "aug_00042.jpg"), a derivative of a held-out test
  image could easily end up back in train, and vice versa. That is why
  EfficientNet-B0 fine hit an exact 100.0000% test accuracy.

FIX:
  1. Only ever read from data/raw/battery_raw/{cls} (untouched originals).
     Never read from existing data/fine/{split}/{cls} folders.
  2. Deduplicate by file content hash, not filename (extra safety).
  3. Split the ORIGINAL files into train/val/test ONCE.
  4. Only THEN augment — and only the train split.
  5. Scope: the 3 real battery chemistries only (LiIon_18650, LiIon_Pouch,
     LiPo), matching what RecyBat24-aug actually provides and what your
     paper's own Limitations section already states (NiMH/NiCd proxy data
     is not real and should not be trained/reported on until real images
     are collected).

Run: python rebalance_fine_FIXED.py
Then: python train_cnn.py --stage fine --backbone EfficientNet-B0
"""
import shutil, random, hashlib
from pathlib import Path

FINE_SRC  = Path("data/raw/battery_raw")
FINE_DST  = Path("data/fine")
IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
random.seed(42)

MAX_PER_CLASS = 500
SPLIT = (0.80, 0.10, 0.10)

# Only the 3 REAL chemistries. NiMH/NiCd excluded — see docstring above.
CLASSES = ["LiIon_18650", "LiIon_Pouch", "LiPo"]


def file_hash(path: Path) -> str:
    """Content hash so visually-identical files can't leak across splits
    even if they have different filenames."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def get_images(folder: Path):
    return [f for f in folder.rglob("*") if f.suffix.lower() in IMG_EXTS]


def augment_image(img, seed=0):
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
    rng = random.Random(seed)
    img = img.convert("RGB")
    ops = rng.sample(range(6), k=rng.randint(2, 4))
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
            arr = np.array(img)
            h, w = arr.shape[:2]
            cx, cy = rng.randint(w // 4, 3 * w // 4), rng.randint(h // 4, 3 * h // 4)
            size = rng.randint(w // 8, w // 4)
            x1, x2 = max(0, cx - size // 2), min(w, cx + size // 2)
            y1, y2 = max(0, cy - size // 2), min(h, cy + size // 2)
            arr[y1:y2, x1:x2] = 128
            img = Image.fromarray(arr)
    return img


print("=" * 60)
print("  Fine Dataset Rebalancing (LEAK-FIXED, 3-class scope)")
print("=" * 60)

# Wipe and rebuild fine/ for these 3 classes ONLY
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        d = FINE_DST / split / cls
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

from PIL import Image
summary = {}

for cls in CLASSES:
    src = FINE_SRC / cls
    if not src.exists():
        print(f"  MISSING: {src} — skipping")
        continue

    # Dedup by content hash from RAW ONLY — no pooling from old splits
    all_imgs = get_images(src)
    seen_hashes, unique = set(), []
    for img in all_imgs:
        try:
            h = file_hash(img)
        except Exception:
            continue
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(img)

    random.shuffle(unique)
    n = len(unique)
    n_train = int(n * SPLIT[0])
    n_val = int(n * SPLIT[1])
    train_f = unique[:n_train]
    val_f   = unique[n_train:n_train + n_val]
    test_f  = unique[n_train + n_val:]

    # Val/test: originals only, resized, NEVER augmented
    for split_name, files in [("val", val_f), ("test", test_f)]:
        dst = FINE_DST / split_name / cls
        for img_path in files:
            try:
                im = Image.open(img_path).convert("RGB").resize((256, 256))
                im.save(dst / f"orig_{img_path.stem}.jpg", quality=92)
            except Exception:
                pass

    # Train: originals copied, then augmented up to MAX_PER_CLASS
    dst_train = FINE_DST / "train" / cls
    created = 0
    for img_path in train_f:
        try:
            im = Image.open(img_path).convert("RGB").resize((256, 256))
            im.save(dst_train / f"orig_{img_path.stem}.jpg", quality=92)
            created += 1
        except Exception:
            pass
    aug_idx = 0
    while created < MAX_PER_CLASS and train_f:
        src_img = train_f[aug_idx % len(train_f)]
        try:
            im = Image.open(src_img).convert("RGB").resize((256, 256))
            im = augment_image(im, seed=aug_idx + 1000)
            im.save(dst_train / f"aug_{cls}_{aug_idx:05d}.jpg", quality=88)
            created += 1
        except Exception:
            pass
        aug_idx += 1

    summary[cls] = dict(raw=n, train_orig=len(train_f), train_final=created,
                         val=len(val_f), test=len(test_f))
    print(f"  {cls:15}: {n} raw (deduped) -> train {len(train_f)} orig "
          f"({created} after aug) | val {len(val_f)} | test {len(test_f)}")

print("\n" + "=" * 60)
print("  Done. Val/test contain ONLY untouched originals — zero overlap")
print("  with train (verified by content hash, not filename).")
print("=" * 60)
print("\n  Now run:")
print("  python train_cnn.py --stage fine --backbone EfficientNet-B0")
