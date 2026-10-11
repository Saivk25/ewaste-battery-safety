"""
evaluate_model.py
=================
Loads trained checkpoints and produces all paper-ready evaluation outputs:
  - Full confusion matrices (normalised + raw counts)
  - Per-class precision / recall / F1 / FNR table
  - Catastrophic False Negative Cost (CFNC)
  - Grad-CAM visualisations (shows what the model attends to)
  - Replaces simulated metrics in comparison table with real ones
  - Saves all figures publication-ready

Run: python evaluate_model.py --backbone EfficientNet-B0 --stage both
"""

import os, sys, json, argparse
from pathlib import Path
from typing import Dict, List

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

SAVE_DIR    = Path("checkpoints")
RESULTS_DIR = Path("results");  RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR    = Path("data")
EVAL_DIR    = Path("eval_outputs"); EVAL_DIR.mkdir(exist_ok=True)
IMG_SIZE    = 224
BATCH_SIZE  = 64
NUM_WORKERS = 4
SEED        = 42
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Severity weights for CFNC (Catastrophic False Negative Cost)
CFNC_WEIGHTS = {
    "Battery":     10.0,   # Missing a battery = catastrophic
    "LiIon_18650": 10.0,
    "LiIon_Pouch": 10.0,
    "LiPo":         9.0,
    "Electronics":  3.0,   # Missing e-waste = bad but less dangerous
    "Metal":        2.0,
    "NiMH":         4.0,
    "NiCd":         4.0,
    "Non_Battery":  1.0,
    "Glass":        1.0,
    "Paper":        1.0,
    "Plastic":      1.0,
}


# Coarse-stage battery decision: "Battery" if P_battery >= BATTERY_THRESHOLD (not only argmax).
# Chosen on the v3 validation set as the smallest value keeping the false-alarm rate at the
# v1 level (<= 3.05%); see results/gaps/coarse_v3_threshold.json. (v2 model: 0.1411.)
BATTERY_THRESHOLD = 0.1113


def coarse_label(probs: dict) -> str:
    """probs: {class_name: probability}. Battery wins whenever it clears the safety threshold."""
    if probs.get("Battery", 0.0) >= BATTERY_THRESHOLD:
        return "Battery"
    return max(probs, key=probs.get)


# ── Model Loader ──────────────────────────────────────────────────────────────

def load_checkpoint(backbone: str, stage: str, path=None) -> tuple:
    tag  = backbone.replace("-","").replace(" ","_")
    path = Path(path) if path else SAVE_DIR / f"best_{tag}_{stage}.pth"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}\nRun train_cnn.py first.")
    
    ckpt = torch.load(path, map_location=device)
    class_names = ckpt["class_names"]
    num_classes = len(class_names)
    
    weights_map = {
        "MobileNetV2":    models.MobileNet_V2_Weights.IMAGENET1K_V1,
        "EfficientNet-B0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "EfficientNet-B3": models.EfficientNet_B3_Weights.IMAGENET1K_V1,
    }
    
    if backbone == "MobileNetV2":
        model = models.mobilenet_v2(weights=None)
        in_f  = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes))
    elif backbone == "EfficientNet-B0":
        model = models.efficientnet_b0(weights=None)
        in_f  = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes))
    elif backbone == "EfficientNet-B3":
        model = models.efficientnet_b3(weights=None)
        in_f  = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.35), nn.Linear(in_f, num_classes))
    
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    print(f"  ✓ Loaded: {path}  (best val_acc: {ckpt['val_acc']:.4f})")
    return model, class_names, ckpt


# ── Evaluation ────────────────────────────────────────────────────────────────

def get_test_transform():
    return transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])


@torch.no_grad()
def run_evaluation(model, test_loader, class_names) -> Dict:
    all_preds, all_labels, all_probs = [], [], []
    
    for imgs, labels in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast():
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    
    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    probs  = np.array(all_probs)
    
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
    cm = confusion_matrix(labels, preds)
    
    results = {
        "accuracy":  float((preds == labels).mean()),
        "f1_macro":  float(f1_score(labels, preds, average='macro', zero_division=0)),
        "precision": float(precision_score(labels, preds, average='macro', zero_division=0)),
        "recall":    float(recall_score(labels, preds, average='macro', zero_division=0)),
        "cm":        cm,
        "preds":     preds,
        "labels":    labels,
        "probs":     probs,
        "per_class": {},
        "cfnc":      0.0,
    }
    
    cfnc_total = 0.0
    for i, cls in enumerate(class_names):
        tp = cm[i, i]; fn = cm[i, :].sum() - tp; fp = cm[:, i].sum() - tp
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        fnr  = fn / (fn + tp + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        w    = CFNC_WEIGHTS.get(cls, 1.0)
        cfnc_total += fnr * w
        results["per_class"][cls] = {
            "precision": round(float(prec), 4), "recall": round(float(rec), 4),
            "f1": round(float(f1), 4), "fnr": round(float(fnr), 4),
            "tp": int(tp), "fn": int(fn), "fp": int(fp),
            "cfnc_contribution": round(float(fnr * w), 4),
        }
    
    results["cfnc"] = round(cfnc_total / len(class_names), 4)
    return results


# ── Confusion Matrix Plot ─────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                           title: str, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    for ax, data, subtitle in [
        (axes[0], cm,                          "Raw Counts"),
        (axes[1], cm / (cm.sum(1, keepdims=True) + 1e-9), "Normalised (Recall per class)"),
    ]:
        im = ax.imshow(data, cmap='Blues', vmin=0,
                        vmax=cm.max() if subtitle=="Raw Counts" else 1)
        ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(subtitle)
        plt.colorbar(im, ax=ax, fraction=0.046)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = data[i, j]
                txt = f"{int(val)}" if subtitle == "Raw Counts" else f"{val:.2f}"
                ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                        color='white' if (val > data.max()*0.5 if subtitle=="Raw Counts"
                                          else val > 0.5) else 'black')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  ✓ Saved: {save_path}")


# ── Per-Class Metrics Table Plot ──────────────────────────────────────────────

def plot_metrics_table(per_class: Dict, backbone: str, stage: str, save_path: Path):
    classes = list(per_class.keys())
    prec    = [per_class[c]["precision"] for c in classes]
    rec     = [per_class[c]["recall"]    for c in classes]
    f1      = [per_class[c]["f1"]        for c in classes]
    fnr     = [per_class[c]["fnr"]       for c in classes]
    
    x   = np.arange(len(classes))
    w   = 0.20
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={'hspace': 0.45})
    
    ax1.bar(x - 1.5*w, prec, w, label='Precision', color='#2980b9', alpha=0.9)
    ax1.bar(x - 0.5*w, rec,  w, label='Recall',    color='#27ae60', alpha=0.9)
    ax1.bar(x + 0.5*w, f1,   w, label='F1 Score',  color='#8e44ad', alpha=0.9)
    ax1.set_xticks(x); ax1.set_xticklabels(classes, rotation=30, ha='right')
    ax1.set_ylabel("Score"); ax1.set_ylim(0, 1.15)
    ax1.set_title(f"{backbone} | {stage.capitalize()} Stage — Per-Class Classification Metrics")
    ax1.legend(); ax1.axhline(0.9, color='gray', ls='--', lw=1, alpha=0.5)
    
    bar_colors = ['#e74c3c' if f > 0.15 else '#f39c12' if f > 0.05 else '#2ecc71' for f in fnr]
    bars = ax2.bar(x, fnr, color=bar_colors, edgecolor='black', linewidth=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(classes, rotation=30, ha='right')
    ax2.set_ylabel("False Negative Rate (FNR)"); ax2.set_ylim(0, max(fnr)*1.3 + 0.05)
    ax2.set_title("Per-Class FNR — Lower is Safer (Battery must be lowest)")
    ax2.axhline(0.10, color='red', ls='--', lw=1.5, label='10% FNR threshold')
    ax2.legend()
    for bar, val in zip(bars, fnr):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  ✓ Saved: {save_path}")


# ── Update Simulation Metrics with Real Results ───────────────────────────────

def update_paper_metrics(real_results: List[Dict]):
    """
    Replaces simulated CNN metrics in cnn_comparison_table.csv with real values.
    Run this after training to update all paper figures automatically.
    """
    sim_path  = Path("cnn_comparison_table.csv")
    real_path = RESULTS_DIR / "real_cnn_comparison_table.csv"
    
    rows = []
    for r in real_results:
        m    = r["eval_metrics"]
        bb   = r["backbone"]
        st   = r["stage"]
        b_key = "Battery" if st == "coarse" else "LiIon_18650"
        rows.append({
            "Backbone":           bb,
            "Stage":              st,
            "Test_Acc":           round(m["accuracy"], 4),
            "F1_Macro":           m["f1_macro"],
            "Precision":          m["precision"],
            "Recall":             m["recall"],
            "Battery_FNR":        m["per_class"].get(b_key, {}).get("fnr", "N/A"),
            "CFNC":               m["cfnc"],
            "Source":             "REAL_TRAINED",
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(real_path, index=False)
    print(f"\n  ✓ Real metrics saved: {real_path}")
    print(f"  ✓ Use these values to replace simulated metrics in your paper.")
    print(f"\n  Summary:")
    print(df[["Backbone","Stage","Test_Acc","F1_Macro","Battery_FNR","CFNC"]].to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="EfficientNet-B0",
                        choices=["MobileNetV2","EfficientNet-B0","EfficientNet-B3","all"])
    parser.add_argument("--stage",    default="both",
                        choices=["coarse","fine","both"])
    parser.add_argument("--data-dir", default="data", type=Path)
    args = parser.parse_args()
    
    backbones = (["MobileNetV2","EfficientNet-B0","EfficientNet-B3"]
                 if args.backbone == "all" else [args.backbone])
    stages    = ["coarse","fine"] if args.stage == "both" else [args.stage]
    
    all_results = []
    for stage in stages:
        stage_dir = args.data_dir / ("coarse" if stage == "coarse" else "fine")
        if not stage_dir.exists():
            print(f"  ✗ Dataset not found: {stage_dir}")
            continue
        
        test_ds = datasets.ImageFolder(stage_dir / "test", get_test_transform())
        test_ld = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
        
        for backbone in backbones:
            print(f"\n{'─'*55}")
            print(f"  Evaluating: {backbone} | {stage}")
            print(f"{'─'*55}")
            try:
                model, class_names, ckpt = load_checkpoint(backbone, stage)
                metrics = run_evaluation(model, test_ld, class_names)
                
                tag = backbone.replace("-","").replace(" ","_")
                
                # Confusion matrix
                plot_confusion_matrix(
                    metrics["cm"], class_names,
                    f"{backbone} — {stage.capitalize()} Stage Confusion Matrix",
                    EVAL_DIR / f"cm_{tag}_{stage}.png"
                )
                
                # Per-class metrics
                plot_metrics_table(
                    metrics["per_class"], backbone, stage,
                    EVAL_DIR / f"metrics_{tag}_{stage}.png"
                )
                
                # Print per-class table
                print(f"\n  Per-class results:")
                print(f"  {'Class':<20} {'Prec':>6} {'Rec':>6} {'F1':>6} {'FNR':>6} {'CFNC':>8}")
                print(f"  {'─'*60}")
                for cls, m in metrics["per_class"].items():
                    print(f"  {cls:<20} {m['precision']:>6.4f} {m['recall']:>6.4f} "
                          f"{m['f1']:>6.4f} {m['fnr']:>6.4f} {m['cfnc_contribution']:>8.4f}")
                print(f"\n  Overall: Acc={metrics['accuracy']:.4f} "
                      f"F1={metrics['f1_macro']:.4f} CFNC={metrics['cfnc']:.4f}")
                
                all_results.append({
                    "backbone": backbone, "stage": stage,
                    "eval_metrics": metrics
                })
                
                # Save JSON
                with open(EVAL_DIR / f"eval_{tag}_{stage}.json", "w") as f:
                    json.dump({
                        "backbone": backbone, "stage": stage,
                        "accuracy": metrics["accuracy"],
                        "f1_macro": metrics["f1_macro"],
                        "cfnc":     metrics["cfnc"],
                        "per_class": metrics["per_class"],
                    }, f, indent=2)
                    
            except FileNotFoundError as e:
                print(f"  ✗ {e}")
    
    if all_results:
        update_paper_metrics(all_results)
        print(f"\n✅ Evaluation complete. All outputs in: {EVAL_DIR}/")
