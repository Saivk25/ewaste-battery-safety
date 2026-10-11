"""
train_cnn.py
============
Trains hierarchical CNN for e-waste safety classification.
Optimised for RTX 5060 8GB VRAM / 16GB RAM.

Stage 1: Coarse classifier — 6-class e-waste type (TrashNet + TACO)
Stage 2: Fine classifier   — battery sub-type (collected battery images)

Backbones compared: MobileNetV2, EfficientNet-B0, EfficientNet-B3
Selected model: EfficientNet-B0 (best accuracy/latency tradeoff)

Training strategy:
  Phase A — Feature extraction (freeze backbone, train head only): 5 epochs
  Phase B — Full fine-tuning (unfreeze all, low LR): 20 epochs
  Total: 25 epochs per model

GPU memory management for RTX 5060 8GB:
  Batch size: 32 (safe for B0/B3 at 224x224)
  Mixed precision (torch.cuda.amp) — halves VRAM usage
  Gradient checkpointing for B3 if needed

Run: python train_cnn.py --stage coarse
     python train_cnn.py --stage fine
     python train_cnn.py --stage both  (default)
"""

import os, json, argparse, time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
from torchvision import datasets, transforms, models
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, precision_score, recall_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
SAVE_DIR    = Path("checkpoints"); SAVE_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results");     RESULTS_DIR.mkdir(exist_ok=True)
LOG_DIR     = Path("logs");        LOG_DIR.mkdir(exist_ok=True)

IMG_SIZE    = 224     # EfficientNet standard input
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE", 32))  # 32 is safe for 8GB VRAM; lower it if other apps share the GPU
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4))  # set NUM_WORKERS=0 when RAM is short
PIN_MEMORY  = True
SEED        = 42

# Training schedule
EPOCHS_FREEZE = 5     # Phase A: head-only training
EPOCHS_FINETUNE = 20  # Phase B: full fine-tuning
TOTAL_EPOCHS  = EPOCHS_FREEZE + EPOCHS_FINETUNE

LR_HEAD       = 1e-3  # Phase A learning rate (head only)
LR_FINETUNE   = 2e-5  # Phase B learning rate (full model, very low)
WEIGHT_DECAY  = 1e-4
LABEL_SMOOTHING = 0.1  # helps generalisation

# Battery class weight multiplier (fix class imbalance)
BATTERY_WEIGHT = 3.0

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(split: str) -> transforms.Compose:
    """
    Training: heavy augmentation matching real conveyor-belt conditions.
    Val/Test: centre crop only (deterministic evaluation).
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet stats (pretrained models)
        std =[0.229, 0.224, 0.225]
    )
    
    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
            transforms.RandomRotation(degrees=30),
            transforms.RandomGrayscale(p=0.05),
            # GaussianBlur simulates motion blur on conveyor
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            normalize,
            # RandomErasing simulates debris occlusion (critical for battery detection)
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15),
                                     ratio=(0.3, 3.3), value='random'),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            normalize,
        ])


# ── Model Builder ─────────────────────────────────────────────────────────────

def build_model(backbone: str, num_classes: int,
                pretrained: bool = True) -> nn.Module:
    """
    Build model with ImageNet pretrained backbone + custom classification head.
    Head: GlobalAvgPool → Dropout(0.3) → BatchNorm → Linear(num_classes)
    """
    weights_map = {
        "MobileNetV2":    models.MobileNet_V2_Weights.IMAGENET1K_V1,
        "EfficientNet-B0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "EfficientNet-B3": models.EfficientNet_B3_Weights.IMAGENET1K_V1,
    }
    
    if backbone == "MobileNetV2":
        model = models.mobilenet_v2(weights=weights_map[backbone] if pretrained else None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
        
    elif backbone == "EfficientNet-B0":
        model = models.efficientnet_b0(weights=weights_map[backbone] if pretrained else None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
        
    elif backbone == "EfficientNet-B3":
        model = models.efficientnet_b3(weights=weights_map[backbone] if pretrained else None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    
    return model.to(device)


def freeze_backbone(model: nn.Module, backbone: str):
    """Freeze all layers except the classifier head."""
    if backbone == "MobileNetV2":
        for param in model.features.parameters():
            param.requires_grad = False
    else:  # EfficientNet variants
        for param in model.features.parameters():
            param.requires_grad = False


def unfreeze_all(model: nn.Module):
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


# ── Class Weight Calculator ───────────────────────────────────────────────────

def compute_class_weights(dataset: datasets.ImageFolder,
                           battery_boost: float = BATTERY_WEIGHT) -> torch.Tensor:
    """
    Compute inverse-frequency class weights.
    Battery class gets additional boost to minimise False Negative Rate.
    """
    class_counts = np.bincount([label for _, label in dataset.samples])
    class_counts = np.where(class_counts == 0, 1, class_counts)  # avoid div/0
    weights = 1.0 / class_counts
    
    # Boost battery class
    battery_idx = dataset.class_to_idx.get("Battery", -1)

# If Battery not present, use first available battery-type class
    if battery_idx == -1:
        for cls in ["LiIon_18650", "LiIon_Pouch", "LiPo"]:
            if cls in dataset.class_to_idx:
                battery_idx = dataset.class_to_idx[cls]
                break
    if battery_idx >= 0:
            weights[battery_idx] *= battery_boost
            
    weights = weights / weights.sum() * len(weights)   # normalise
    return torch.FloatTensor(weights).to(device)


def get_weighted_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    """WeightedRandomSampler ensures balanced mini-batches during training."""
    class_counts = np.bincount([label for _, label in dataset.samples])
    class_counts = np.where(class_counts == 0, 1, class_counts)
    sample_weights = [1.0 / class_counts[label] for _, label in dataset.samples]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scaler) -> Tuple[float, float]:
    """One training epoch with mixed precision (AMP)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():   # Mixed precision — halves VRAM usage on RTX 5060
            logits = model(imgs)
            loss   = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion,
             class_names: List[str]) -> Tuple[float, float, Dict]:
    """Evaluation with full per-class metrics."""
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels)
        total_loss  += loss.item() * imgs.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    n          = len(all_labels)
    
    acc = (all_preds == all_labels).mean()
    metrics = {
        "accuracy":  float(acc),
        "f1_macro":  float(f1_score(all_labels, all_preds, average='macro', zero_division=0)),
        "precision": float(precision_score(all_labels, all_preds, average='macro', zero_division=0)),
        "recall":    float(recall_score(all_labels, all_preds, average='macro', zero_division=0)),
        "loss":      total_loss / n,
        "per_class": {},
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }
    
    # Per-class FNR (critical for safety evaluation)
    cm = confusion_matrix(all_labels, all_preds)
    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        fnr  = fn / (fn + tp + 1e-9)
        metrics["per_class"][cls] = {
            "precision": round(float(prec), 4),
            "recall":    round(float(rec),  4),
            "f1":        round(float(2*prec*rec/(prec+rec+1e-9)), 4),
            "fnr":       round(float(fnr),  4),
            "tp": int(tp), "fn": int(fn), "fp": int(fp),
        }
    
    return total_loss / n, float(acc), metrics


# ── Training Orchestrator ─────────────────────────────────────────────────────

def train_model(backbone: str, stage: str,
                data_dir: Path) -> Dict:
    """
    Full training pipeline for one backbone and stage.
    Returns dict of training history and best metrics.
    """
    print(f"\n{'='*60}")
    print(f"  Training: {backbone} | Stage: {stage}")
    print(f"  Device: {device} | Batch: {BATCH_SIZE} | AMP: True")
    print(f"{'='*60}")
    
    stage_dir = data_dir / ("coarse" if stage == "coarse" else "fine")
    if not stage_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {stage_dir}\nRun dataset_preparation.py first.")
    
    # Datasets & Loaders
        # Remove empty classes automatically
    train_dir = stage_dir / "train"
    val_dir   = stage_dir / "val"
    test_dir  = stage_dir / "test"

    # Remove empty class folders BEFORE loading dataset
    valid_classes = []
    for cls in os.listdir(train_dir):
        class_path = train_dir / cls
        if class_path.is_dir() and len(list(class_path.glob("*"))) > 0:
            valid_classes.append(cls)

    print("Valid classes:", valid_classes)

    # Now load dataset normally (only valid folders exist ideally)
    train_ds = datasets.ImageFolder(train_dir, get_transforms("train"))
    val_ds   = datasets.ImageFolder(val_dir,   get_transforms("val"))
    test_ds  = datasets.ImageFolder(test_dir,  get_transforms("test"))

    class_names = train_ds.classes
    
    print(f"  Classes ({len(class_names)}): {class_names}")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    
    sampler    = get_weighted_sampler(train_ds)
    train_ld   = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_ld     = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_ld    = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    # Model, loss, optimizer, scheduler
    model      = build_model(backbone, len(class_names))
    cls_weights = compute_class_weights(train_ds)
    criterion  = nn.CrossEntropyLoss(weight=cls_weights,
                                     label_smoothing=LABEL_SMOOTHING)
    scaler     = GradScaler()
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
               "lr": [], "epoch_time": []}
    best_val_acc = 0.0
    best_metrics = {}
    best_path    = SAVE_DIR / f"best_{backbone.replace('-','').replace(' ','_')}_{stage}.pth"
    
    for epoch in range(1, TOTAL_EPOCHS + 1):
        t0 = time.time()
        
        # Phase A → B transition
        if epoch == 1:
            freeze_backbone(model, backbone)
            optimizer  = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
            scheduler  = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=LR_HEAD,
                steps_per_epoch=len(train_ld), epochs=EPOCHS_FREEZE)
            print(f"\n  Phase A: Head-only training (epochs 1–{EPOCHS_FREEZE})")
        
        if epoch == EPOCHS_FREEZE + 1:
            unfreeze_all(model)
            optimizer  = optim.AdamW(model.parameters(),
                                     lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
            scheduler  = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=EPOCHS_FINETUNE, eta_min=1e-7)
            print(f"\n  Phase B: Full fine-tuning (epochs {EPOCHS_FREEZE+1}–{TOTAL_EPOCHS})")
        
        train_loss, train_acc = train_epoch(model, train_ld, criterion, optimizer, scaler)
        val_loss, val_acc, val_metrics = evaluate(model, val_ld, criterion, class_names)
        
        if epoch <= EPOCHS_FREEZE:
            scheduler.step()
        else:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - t0
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["epoch_time"].append(epoch_time)
        
        print(f"  Ep {epoch:2d}/{TOTAL_EPOCHS} | "
              f"T_loss: {train_loss:.4f} T_acc: {train_acc:.4f} | "
              f"V_loss: {val_loss:.4f} V_acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e} | {epoch_time:.1f}s")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = val_metrics
            torch.save({
                "epoch":       epoch,
                "backbone":    backbone,
                "stage":       stage,
                "class_names": class_names,
                "model_state": model.state_dict(),
                "val_acc":     val_acc,
                "val_metrics": val_metrics,
            }, best_path)
            print(f"    ✓ New best saved: val_acc={val_acc:.4f}")
    
    # Final test evaluation
    print(f"\n  Loading best checkpoint for test evaluation...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc, test_metrics = evaluate(model, test_ld, criterion, class_names)
    
    print(f"\n  ── Final Test Results: {backbone} / {stage} ──")
    print(f"  Test Accuracy : {test_acc:.4f}")
    print(f"  Test F1 Macro : {test_metrics['f1_macro']:.4f}")
    print(f"  Test Precision: {test_metrics['precision']:.4f}")
    print(f"  Test Recall   : {test_metrics['recall']:.4f}")
    print(f"\n  Per-class FNR (lower = safer):")
    for cls, m in test_metrics["per_class"].items():
        print(f"    {cls:20}: FNR={m['fnr']:.4f}  F1={m['f1']:.4f}  "
              f"Prec={m['precision']:.4f}  Rec={m['recall']:.4f}")
    
    result = {
        "backbone":     backbone,
        "stage":        stage,
        "class_names":  class_names,
        "history":      history,
        "best_val_acc": best_val_acc,
        "test_acc":     test_acc,
        "test_metrics": test_metrics,
        "checkpoint":   str(best_path),
    }
    
    # Save results JSON
    tag = backbone.replace("-","").replace(" ","_")
    results_path = RESULTS_DIR / f"results_{tag}_{stage}.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")
    
    return result


# ── Plot Training Curves ──────────────────────────────────────────────────────

def plot_training_history(results: List[Dict], stage: str):
    """Publication-ready training curves for all backbones."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"MobileNetV2": "#e74c3c",
               "EfficientNet-B0": "#2980b9",
               "EfficientNet-B3": "#27ae60"}
    
    for r in results:
        bb  = r["backbone"]
        col = colors.get(bb, "gray")
        h   = r["history"]
        ep  = range(1, len(h["val_acc"]) + 1)
        axes[0].plot(ep, h["train_acc"], color=col, lw=1.5, ls='--', alpha=0.6)
        axes[0].plot(ep, h["val_acc"],   color=col, lw=2.5, label=f'{bb} val')
        axes[1].plot(ep, h["train_loss"],color=col, lw=1.5, ls='--', alpha=0.6)
        axes[1].plot(ep, h["val_loss"],  color=col, lw=2.5, label=f'{bb} val')
    
    # Phase boundary
    for ax in axes:
        ax.axvline(EPOCHS_FREEZE, color='black', ls=':', lw=1.5, alpha=0.5)
        ax.text(EPOCHS_FREEZE+0.3, ax.get_ylim()[0], 'Unfreeze →', fontsize=8, alpha=0.6)
    
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"Validation Accuracy — {stage.capitalize()} Stage")
    axes[0].legend(); axes[0].set_ylim(0, 1.05)
    
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].set_title(f"Training Loss — {stage.capitalize()} Stage")
    axes[1].legend()
    
    plt.tight_layout()
    path = RESULTS_DIR / f"training_curves_{stage}.png"
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Training curves saved: {path}")


# ── Comparison Table Generator ────────────────────────────────────────────────

def build_results_table(results: List[Dict]) -> None:
    """Print and save the final comparison table for the paper."""
    print(f"\n{'='*90}")
    print(f"  FINAL CNN COMPARISON TABLE")
    print(f"{'='*90}")
    print(f"  {'Backbone':<20} {'Stage':<8} {'Test_Acc':<10} {'F1_Mac':<8} "
          f"{'Prec':<8} {'Recall':<8} {'Batt_FNR':<10}")
    print(f"  {'─'*80}")
    
    table_rows = []
    for r in results:
        bb = r["backbone"]
        st = r["stage"]
        m  = r["test_metrics"]
        
        # Battery FNR
        batt_key = "Battery" if st == "coarse" else "LiIon_18650"
        batt_fnr = m["per_class"].get(batt_key, {}).get("fnr", "N/A")
        
        print(f"  {bb:<20} {st:<8} {r['test_acc']:<10.4f} {m['f1_macro']:<8.4f} "
              f"{m['precision']:<8.4f} {m['recall']:<8.4f} {str(batt_fnr):<10}")
        table_rows.append({
            "Backbone": bb, "Stage": st,
            "Test_Acc": r["test_acc"], "F1_Macro": m["f1_macro"],
            "Precision": m["precision"], "Recall": m["recall"],
            "Battery_FNR": batt_fnr,
        })
    
    print(f"{'='*90}")
    
    import pandas as pd
    df = pd.DataFrame(table_rows)
    df.to_csv(RESULTS_DIR / "final_comparison_table.csv", index=False)
    print(f"  Table saved: {RESULTS_DIR}/final_comparison_table.csv")


# ── VRAM Monitor ─────────────────────────────────────────────────────────────

def print_gpu_info():
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"\n  GPU: {gpu.name}")
        print(f"  VRAM: {gpu.total_memory / 1e9:.1f} GB")
        print(f"  CUDA: {torch.version.cuda}")
        alloc = torch.cuda.memory_allocated(0) / 1e9
        print(f"  Allocated: {alloc:.2f} GB")
    else:
        print("  ⚠ CUDA not available — training on CPU (very slow)")


# ── Entry Point ───────────────────────────────────────────────────────────────

BACKBONES = ["MobileNetV2", "EfficientNet-B0", "EfficientNet-B3"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hierarchical CNN for e-waste safety")
    parser.add_argument("--stage",    choices=["coarse","fine","both"], default="both")
    parser.add_argument("--backbone", choices=BACKBONES + ["all"],     default="all")
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--epochs",   default=TOTAL_EPOCHS, type=int)
    parser.add_argument("--seed",     default=SEED, type=int, help="training seed (default 42)")
    parser.add_argument("--out-dir",  default=None, type=Path,
                        help="write checkpoints/results here instead of checkpoints/ and results/ "
                             "(use for repeat runs so the production checkpoint is not overwritten)")
    args = parser.parse_args()
    if args.seed != SEED:
        SEED = args.seed
        torch.manual_seed(SEED); np.random.seed(SEED)
    if args.out_dir is not None:
        SAVE_DIR = args.out_dir / "checkpoints"; RESULTS_DIR = args.out_dir / "results"
        SAVE_DIR.mkdir(parents=True, exist_ok=True); RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print_gpu_info()
    
    stages    = ["coarse","fine"] if args.stage == "both" else [args.stage]
    backbones = BACKBONES if args.backbone == "all" else [args.backbone]
    
    all_results = []
    for stage in stages:
        stage_results = []
        for bb in backbones:
            try:
                r = train_model(bb, stage, args.data_dir)
                all_results.append(r)
                stage_results.append(r)
            except FileNotFoundError as e:
                print(f"\n  ✗ {e}")
                print(f"  Run: python dataset_preparation.py first\n")
                break
        
        if stage_results:
            plot_training_history(stage_results, stage)
     
    if all_results:
        build_results_table(all_results)
        print(f"\n✅ Training complete. Checkpoints in: {SAVE_DIR}/")
        print(f"   Results in: {RESULTS_DIR}/")
        print(f"\nNext step: python evaluate_model.py --backbone EfficientNet-B0")
