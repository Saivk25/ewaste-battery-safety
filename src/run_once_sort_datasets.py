"""
run_once_sort_datasets.py
=========================
Run this ONCE after downloading all datasets.
Sorts downloaded images into the correct folder structure
that dataset_preparation.py expects.

Run BEFORE dataset_preparation.py.
"""
import os, shutil
from pathlib import Path

BASE = Path("data/raw")

COARSE_MAP = {
    # TrashNet original classes
    "cardboard":   "Electronics",
    "glass":       "Glass",
    "metal":       "Metal",
    "paper":       "Paper",
    "plastic":     "Plastic",
    "trash":       "Electronics",
    # Garbage-12 classes
    "Battery":     "Battery",
    "Cardboard":   "Electronics",
    "Glass":       "Glass",
    "Metal":       "Metal",
    "Paper":       "Paper",
    "Plastic":     "Plastic",
    "Trash":       "Electronics",
    "Biological":  "Electronics",
    "White-glass": "Glass",
    "Clothes":     "Electronics",
    "Shoes":       "Electronics",
    # Misc aliases
    "battery":     "Battery",
    "electronics": "Electronics",
}

FINE_MAP = {
    # RecyBat24 classes
    "Cylindrical":  "LiIon_18650",
    "Pouch":        "LiIon_Pouch",
    "Prismatic":    "LiPo",
    "cylindrical":  "LiIon_18650",
    "pouch":        "LiIon_Pouch",
    "prismatic":    "LiPo",
    # Roboflow generic
    "battery":      "LiIon_18650",
    "Battery":      "LiIon_18650",
    "li-ion":       "LiIon_18650",
    "lithium":      "LiIon_18650",
    "lipo":         "LiPo",
    "LiPo":         "LiPo",
    "nimh":         "NiMH",
    "NiMH":         "NiMH",
    "nicd":         "NiCd",
    "NiCd":         "NiCd",
    "non_battery":  "Non_Battery",
    "other":        "Non_Battery",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPG", ".JPEG", ".PNG"}


def copy_class_folder(src: Path, dst: Path, prefix: str = "") -> int:
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in src.iterdir():
        if img.is_file() and img.suffix in IMG_EXTS:
            name = f"{prefix}_{img.name}" if prefix else img.name
            out = dst / name
            if not out.exists():
                shutil.copy2(img, out)
                count += 1
    return count


def sort_source(src_root: Path, dst_root: Path, class_map: dict,
                label: str, recurse_splits: bool = False):
    """Walk src_root looking for class subdirectories."""
    total = 0
    if recurse_splits:
        # Handle train/val/test splits inside source
        for split_dir in src_root.iterdir():
            if split_dir.is_dir() and split_dir.name in ["train", "val", "test", "valid"]:
                total += sort_source(split_dir, dst_root, class_map, label)
        return total

    for cls_dir in src_root.iterdir():
        if not cls_dir.is_dir():
            continue
        our_cls = class_map.get(cls_dir.name)
        if not our_cls:
            # Try case-insensitive
            for k, v in class_map.items():
                if k.lower() == cls_dir.name.lower():
                    our_cls = v; break
        if not our_cls:
            print(f"    Unmapped class: {cls_dir.name} (skipping)")
            continue
        dst = dst_root / our_cls
        n = copy_class_folder(cls_dir, dst, prefix=f"{label}_{cls_dir.name}")
        if n > 0:
            print(f"    {cls_dir.name:20} → {our_cls:15} : {n} images")
        total += n
    return total


def main():
    print("="*60)
    print("  Dataset Sorter — E-Waste Safety System")
    print("="*60)

    coarse_dst = BASE / "trashnet"
    fine_dst   = BASE / "battery_raw"
    coarse_dst.mkdir(parents=True, exist_ok=True)
    fine_dst.mkdir(parents=True, exist_ok=True)

    # ── COARSE SOURCES ────────────────────────────────────────────
    print("\n[COARSE] Sorting for coarse classifier...")

    # Garbage-12
    g12 = BASE / "garbage12"
    if g12.exists():
        # Find the actual image directory (may be nested)
        for candidate in [g12, g12/"Garbage classification"/"Garbage classification",
                           g12/"Garbage classification"]:
            if candidate.exists() and any(candidate.iterdir()):
                n = sort_source(candidate, coarse_dst, COARSE_MAP, "g12")
                print(f"  Garbage-12: {n} images sorted"); break
    else:
        print("  Garbage-12: not found (download from kaggle datasets download -d mostafaabla/garbage-classification)")

    # TrashNet (if not already structured)
    tn = BASE / "trashnet_raw"
    if tn.exists():
        n = sort_source(tn, coarse_dst, COARSE_MAP, "trashnet")
        print(f"  TrashNet raw: {n} images sorted")

    # HuggingFace TrashNet
    tn_hf = BASE / "trashnet_hf"
    if tn_hf.exists():
        # HF dataset saves differently — check for split folders
        n = sort_source(tn_hf, coarse_dst, COARSE_MAP, "tn_hf", recurse_splits=True)
        print(f"  TrashNet HF: {n} images sorted")

    print(f"\n  Coarse class summary:")
    for cls in ["Battery","Electronics","Glass","Metal","Paper","Plastic"]:
        p = coarse_dst / cls
        if p.exists():
            n = len([f for f in p.iterdir() if f.suffix in IMG_EXTS])
            status = "✓" if n >= 100 else "⚠ LOW"
            print(f"    {cls:12}: {n:5} images {status}")
        else:
            print(f"    {cls:12}: MISSING")

    # ── FINE SOURCES ──────────────────────────────────────────────
    print("\n[FINE] Sorting for fine (battery) classifier...")

    # RecyBat24 base
    rb24 = BASE / "recybat24"
    if rb24.exists():
        n = sort_source(rb24, fine_dst, FINE_MAP, "rb24", recurse_splits=True)
        print(f"  RecyBat24: {n} images sorted")
    else:
        print("  RecyBat24: not found (download from https://zenodo.org/records/13126689)")

    # RecyBat24-aug
    rb24a = BASE / "recybat24_aug"
    if rb24a.exists():
        n = sort_source(rb24a, fine_dst, FINE_MAP, "rb24aug", recurse_splits=True)
        print(f"  RecyBat24-aug: {n} images sorted")
    else:
        print("  RecyBat24-aug: not found (optional — https://zenodo.org/records/15226091)")

    # Roboflow downloads (each goes in its own subfolder under roboflow/)
    rf_dir = BASE / "roboflow"
    if rf_dir.exists():
        for sub in rf_dir.iterdir():
            if sub.is_dir():
                n = sort_source(sub, fine_dst, FINE_MAP, f"rf_{sub.name}",
                                recurse_splits=True)
                if n > 0:
                    print(f"  Roboflow/{sub.name}: {n} images sorted")

    print(f"\n  Fine class summary:")
    for cls in ["LiIon_18650","LiIon_Pouch","LiPo","NiMH","NiCd","Non_Battery"]:
        p = fine_dst / cls
        if p.exists():
            n = len([f for f in p.iterdir() if f.suffix in IMG_EXTS])
            status = "✓" if n >= 80 else "⚠ LOW (<80)"
            print(f"    {cls:15}: {n:5} images {status}")
        else:
            print(f"    {cls:15}: MISSING")

    print("\n" + "="*60)
    print("  Sorting complete. Next step:")
    print("  python dataset_preparation.py")
    print("="*60)


if __name__ == "__main__":
    main()
