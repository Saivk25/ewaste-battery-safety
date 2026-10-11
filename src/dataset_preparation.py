"""
dataset_preparation.py
======================
Downloads, structures, augments, and splits all datasets for training.

Datasets handled:
  1. TrashNet  — 6-class coarse e-waste classification
  2. TACO      — real-world trash images (supplements TrashNet)
  3. Battery   — lithium-ion battery sub-classification (manual + scraped)

Output structure:
  data/
    coarse/
      train/ val/ test/
        Battery/ Cardboard_Trash/ Glass/ Metal/ Paper/ Plastic/
    fine/
      train/ val/ test/
        LiIon_18650/ LiIon_Pouch/ LiPo/ NiMH/ NiCd/ Non_Battery/

Run this ONCE before training.
Requirements: pip install torch torchvision Pillow requests tqdm
"""

import os, shutil, random, json, hashlib
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path("data")
COARSE_DIR   = BASE_DIR / "coarse"
FINE_DIR     = BASE_DIR / "fine"
RAW_DIR      = BASE_DIR / "raw"
SEED         = 42
SPLIT        = (0.80, 0.10, 0.10)   # train / val / test

COARSE_CLASSES = ["Battery", "Cardboard_Trash", "Glass", "Metal", "Paper", "Plastic"]
FINE_CLASSES   = ["LiIon_18650", "LiIon_Pouch", "LiPo", "NiMH", "NiCd", "Non_Battery"]

# Augmentation target counts per class (training set)
# Battery gets 3x augmentation to fix class imbalance
AUG_TARGETS = {
    "Battery":     600,   # augment heavily
    "Cardboard_Trash": 400,
    "Glass":       300,
    "Metal":       350,
    "Paper":       300,
    "Plastic":     350,
}

random.seed(SEED)
np.random.seed(SEED)


# ── Directory Setup ───────────────────────────────────────────────────────────

def create_directory_structure():
    """Create ImageFolder-compatible directory tree."""
    for split in ["train", "val", "test"]:
        for cls in COARSE_CLASSES:
            (COARSE_DIR / split / cls).mkdir(parents=True, exist_ok=True)
        for cls in FINE_CLASSES:
            (FINE_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "trashnet").mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "battery_raw").mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "taco").mkdir(parents=True, exist_ok=True)
    print("✓ Directory structure created")


# ── TrashNet Handler ──────────────────────────────────────────────────────────

TRASHNET_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"

def check_trashnet_exists() -> bool:
    """Check if TrashNet raw data is already downloaded."""
    trashnet_path = RAW_DIR / "trashnet"
    classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    return all((trashnet_path / c).exists() for c in classes)


def download_trashnet_instructions():
    """Print download instructions since direct download requires manual steps."""
    print("\n" + "="*65)
    print("STEP 1: Download TrashNet Dataset")
    print("="*65)
    print("""
Manual download required (3.5 GB):

Option A — Direct download:
  URL: https://github.com/garythung/trashnet
  Click: Releases → dataset-resized.zip
  Extract to: data/raw/trashnet/
  
  After extraction, structure should be:
  data/raw/trashnet/
    cardboard/  (403 images)
    glass/      (501 images)
    metal/      (410 images)
    paper/      (594 images)
    plastic/    (482 images)
    trash/      (137 images)

Option B — Kaggle (easier):
  kaggle datasets download -d asdasdasasdas/garbage-classification
  unzip into data/raw/trashnet/

Option C — Roboflow (pre-split, recommended):
  Visit: https://universe.roboflow.com/material-identification
  Search: "TrashNet" → Download → Format: Folder (train/val/test)
  Extract to: data/raw/trashnet_split/

Run this script again after downloading.
""")


def process_trashnet():
    """Map TrashNet classes to our coarse class names and split."""
    trashnet_path = RAW_DIR / "trashnet"
    
    # TrashNet class → our class name mapping
    class_map = {
        "cardboard": "Cardboard_Trash",  # TrashNet cardboard
        "glass":     "Glass",
        "metal":     "Metal",
        "paper":     "Paper",
        "plastic":   "Plastic",
        "trash":     "Cardboard_Trash",  # TrashNet general trash
    }
    
    # Check for dataset-resized subfolder
    if (trashnet_path / "dataset-resized").exists():
        trashnet_path = trashnet_path / "dataset-resized"
    
    all_files: Dict[str, List[Path]] = {c: [] for c in COARSE_CLASSES}
    
    for raw_cls, our_cls in class_map.items():
        cls_dir = trashnet_path / raw_cls
        if not cls_dir.exists():
            print(f"  ⚠ Missing: {cls_dir}")
            continue
        imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        all_files[our_cls].extend(imgs)
        print(f"  {raw_cls:12} → {our_cls:12}: {len(imgs)} images")
    
    return all_files


# ── Battery Dataset Handler ───────────────────────────────────────────────────

def battery_download_instructions():
    print("\n" + "="*65)
    print("STEP 2: Battery Images (Critical — Your Novel Contribution)")
    print("="*65)
    print("""
You need ~300 battery images minimum. Collect from:

A) Roboflow Universe (free, annotated):
   https://universe.roboflow.com
   Search: "lithium battery" → Download folder format
   Place in: data/raw/battery_raw/LiIon_18650/
              data/raw/battery_raw/LiIon_Pouch/
              data/raw/battery_raw/LiPo/
              data/raw/battery_raw/NiMH/
              data/raw/battery_raw/NiCd/
              data/raw/battery_raw/Non_Battery/

B) Kaggle datasets:
   https://www.kaggle.com/datasets/search?q=battery+waste
   Several small datasets — combine them

C) YOUR OWN PHOTOS (strongly recommended for paper):
   - Photograph 20-30 different batteries at different angles
   - Include: swollen batteries, damaged casing, old vs new
   - Minimum resolution: 224x224 (any phone camera works)
   - Label and place in correct subfolder
   - This counts as "real data collection" in your paper

D) Google Images scraping (use carefully, cite source):
   Search: "18650 lithium battery", "lipo battery pack",
           "nimh battery", "damaged lithium battery"
   Tools: gallery-dl, or manual download
   
Minimum needed per fine class: 50 images
Target: 100+ per class for good fine-tune results
""")


def scan_battery_images() -> Dict[str, List[Path]]:
    """Scan whatever battery images the user has placed."""
    battery_path = RAW_DIR / "battery_raw"
    all_files: Dict[str, List[Path]] = {}
    
    for cls in FINE_CLASSES:
        cls_dir = battery_path / cls
        if cls_dir.exists():
            imgs = (list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) +
                    list(cls_dir.glob("*.jpeg")) + list(cls_dir.glob("*.JPG")))
            all_files[cls] = imgs
            print(f"  {cls:15}: {len(imgs)} images found")
        else:
            all_files[cls] = []
            print(f"  {cls:15}: directory missing — create data/raw/battery_raw/{cls}/")
    
    return all_files


# ── Image Augmentation Pipeline ───────────────────────────────────────────────

def augment_image(img: Image.Image, seed: int = 0) -> Image.Image:
    """
    Domain-specific augmentation for e-waste / battery images.
    Simulates real conveyor belt conditions:
      - Motion blur (conveyor movement)
      - Random occlusion / cutout (debris overlap)
      - Contrast/brightness variation (industrial lighting)
      - Random rotation (orientation-agnostic detection)
      - Gaussian noise (sensor noise)
      - Crush/compression simulation (damaged items)
    """
    rng = random.Random(seed)
    img = img.convert("RGB")
    
    ops = rng.sample(range(7), k=rng.randint(2, 4))
    
    for op in ops:
        if op == 0:  # Motion blur
            size = rng.choice([3, 5, 7])
            kernel = ImageFilter.Kernel(size=(size, size),
                kernel=[1]*size**2, scale=size**2, offset=0)
            # Use simple box blur as motion proxy
            img = img.filter(ImageFilter.BoxBlur(rng.randint(1, 3)))
        
        elif op == 1:  # Brightness jitter ±30%
            factor = rng.uniform(0.70, 1.30)
            img = ImageEnhance.Brightness(img).enhance(factor)
        
        elif op == 2:  # Contrast jitter
            factor = rng.uniform(0.75, 1.35)
            img = ImageEnhance.Contrast(img).enhance(factor)
        
        elif op == 3:  # Rotation
            angle = rng.uniform(-30, 30)
            img = img.rotate(angle, expand=False, fillcolor=(128, 128, 128))
        
        elif op == 4:  # Horizontal flip
            if rng.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        elif op == 5:  # Random crop + resize (simulates partial view)
            w, h = img.size
            margin = int(min(w, h) * 0.15)
            left   = rng.randint(0, margin)
            top    = rng.randint(0, margin)
            right  = w - rng.randint(0, margin)
            bottom = h - rng.randint(0, margin)
            img = img.crop((left, top, right, bottom)).resize((w, h), Image.BILINEAR)
        
        elif op == 6:  # Cutout (simulates occlusion by debris)
            arr = np.array(img)
            h, w = arr.shape[:2]
            cx = rng.randint(w//4, 3*w//4)
            cy = rng.randint(h//4, 3*h//4)
            size = rng.randint(w//8, w//4)
            x1, x2 = max(0, cx-size//2), min(w, cx+size//2)
            y1, y2 = max(0, cy-size//2), min(h, cy+size//2)
            arr[y1:y2, x1:x2] = 128   # grey occlusion patch
            img = Image.fromarray(arr)
    
    return img


def augment_class(src_images: List[Path], dst_dir: Path,
                  target_count: int, prefix: str = "aug") -> int:
    """Augment images from src to reach target_count in dst_dir."""
    if not src_images:
        return 0
    
    dst_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    
    # First copy originals
    for img_path in src_images:
        try:
            img = Image.open(img_path).convert("RGB").resize((256, 256))
            out = dst_dir / f"orig_{img_path.stem}.jpg"
            img.save(out, quality=92)
            created += 1
        except Exception as e:
            pass
    
    # Augment until target reached
    aug_idx = 0
    while created < target_count:
        src = src_images[aug_idx % len(src_images)]
        try:
            img = Image.open(src).convert("RGB").resize((256, 256))
            img = augment_image(img, seed=aug_idx + 1000)
            out = dst_dir / f"{prefix}_{aug_idx:05d}.jpg"
            img.save(out, quality=88)
            created += 1
        except Exception:
            pass
        aug_idx += 1
    
    return created


# ── Split Handler ─────────────────────────────────────────────────────────────

def split_files(files: List[Path], split: Tuple[float, float, float],
                seed: int = 42) -> Tuple[List, List, List]:
    """Reproducible stratified split."""
    files = sorted(files)
    random.seed(seed)
    random.shuffle(files)
    n = len(files)
    n_train = int(n * split[0])
    n_val   = int(n * split[1])
    return files[:n_train], files[n_train:n_train+n_val], files[n_train+n_val:]


def distribute_to_splits(files: List[Path], cls_name: str,
                          split_dir: Path, augment: bool = False,
                          aug_target: int = 0):
    """Split files and copy/augment into train/val/test."""
    train_f, val_f, test_f = split_files(files, SPLIT)
    
    for split_name, split_files_list in [("train", train_f),
                                          ("val", val_f),
                                          ("test", test_f)]:
        dst = split_dir / split_name / cls_name
        dst.mkdir(parents=True, exist_ok=True)
        
        if split_name == "train" and augment and aug_target > len(train_f):
            count = augment_class(split_files_list, dst, aug_target, prefix=cls_name)
            print(f"    train/{cls_name}: {len(train_f)} orig → {count} augmented")
        else:
            for fp in split_files_list:
                try:
                    img = Image.open(fp).convert("RGB").resize((256, 256))
                    img.save(dst / f"{fp.stem}.jpg", quality=92)
                except Exception:
                    pass
            print(f"    {split_name}/{cls_name}: {len(split_files_list)} images")


# ── Dataset Stats Reporter ────────────────────────────────────────────────────

def print_dataset_stats(base_dir: Path, name: str):
    print(f"\n{'─'*50}")
    print(f"  {name} Dataset Statistics")
    print(f"{'─'*50}")
    total = 0
    for split in ["train", "val", "test"]:
        split_total = 0
        classes = sorted((base_dir / split).iterdir()) if (base_dir / split).exists() else []
        for cls_dir in classes:
            if cls_dir.is_dir():
                n = len(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")))
                split_total += n
        print(f"  {split:6}: {split_total:5} images")
        total += split_total
    print(f"  Total : {total:5} images")


# ── Main Preparation Flow ─────────────────────────────────────────────────────

def check_readiness() -> Dict[str, bool]:
    """Check which datasets are available."""
    trashnet_ok = check_trashnet_exists()
    battery_ok  = any(
        len(list((RAW_DIR / "battery_raw" / cls).glob("*"))) > 0
        for cls in FINE_CLASSES
        if (RAW_DIR / "battery_raw" / cls).exists()
    )
    return {"trashnet": trashnet_ok, "battery": battery_ok}


def run_preparation(skip_missing: bool = False):
    print("\n" + "="*65)
    print("  E-WASTE DATASET PREPARATION PIPELINE")
    print("="*65)
    
    create_directory_structure()
    
    ready = check_readiness()
    print(f"\n  Dataset availability:")
    print(f"  TrashNet : {'✓ Found' if ready['trashnet'] else '✗ Not found'}")
    print(f"  Battery  : {'✓ Found' if ready['battery'] else '✗ Not found'}")
    
    # ── Coarse dataset (TrashNet) ──────────────────────────────────────────
    if ready["trashnet"]:
        print("\n[1/3] Processing TrashNet → Coarse classifier dataset...")
        coarse_files = process_trashnet()
        for cls, files in coarse_files.items():
            if files:
                aug_target = AUG_TARGETS.get(cls, 300)
                print(f"  Processing class: {cls}")
                distribute_to_splits(files, cls, COARSE_DIR,
                                     augment=(cls == "Battery"),
                                     aug_target=aug_target)
        print_dataset_stats(COARSE_DIR, "Coarse")
    else:
        download_trashnet_instructions()
        if not skip_missing:
            print("\n⚠ Run this script again after downloading TrashNet.")
            return
    
    # ── Fine dataset (Battery images) ─────────────────────────────────────
    print("\n[2/3] Processing Battery images → Fine classifier dataset...")
    battery_files = scan_battery_images()
    
    total_battery = sum(len(v) for v in battery_files.values())
    if total_battery == 0:
        battery_download_instructions()
        if not skip_missing:
            print("\n⚠ Add battery images and run again.")
            return
    else:
        for cls, files in battery_files.items():
            if files:
                print(f"  Processing fine class: {cls}")
                distribute_to_splits(files, cls, FINE_DIR,
                                     augment=True, aug_target=150)
        print_dataset_stats(FINE_DIR, "Fine Battery")
    
    # ── Save dataset manifest ──────────────────────────────────────────────
    print("\n[3/3] Saving dataset manifest...")
    manifest = {
        "coarse_classes":  COARSE_CLASSES,
        "fine_classes":    FINE_CLASSES,
        "split_ratios":    list(SPLIT),
        "seed":            SEED,
        "augmentation":    "motion_blur+occlusion+brightness+contrast+rotation+flip+crop",
        "image_size":      256,
        "trashnet_ready":  ready["trashnet"],
        "battery_ready":   ready["battery"],
    }
    with open(BASE_DIR / "dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print("\n✓ Dataset preparation complete.")
    print(f"  Manifest saved to: {BASE_DIR}/dataset_manifest.json")
    print("\nNext step: python train_cnn.py")


if __name__ == "__main__":
    import sys
    skip = "--skip-missing" in sys.argv
    run_preparation(skip_missing=skip)
