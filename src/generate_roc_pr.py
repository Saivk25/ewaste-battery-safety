"""
generate_roc_pr.py
===================
Generates ROC and Precision-Recall curves for both CNN stages (coarse 6-class
sort, fine 3-class battery subtype) using the already-trained EfficientNet-B0
checkpoints. Built for the paper resubmission (explainability / robustness
section).

Reuses evaluate_model.py's checkpoint loading and test-transform logic
directly (imports load_checkpoint / get_test_transform / DATA_DIR / etc. from
it) instead of reimplementing them, per project convention of not duplicating
verified pipeline code.

IMPORTANT — provenance note:
This script was written by Claude Cowork but could NOT be executed or
verified in that session (the sandbox has no working PyTorch/CUDA stack and
ran out of disk trying to build one). It has NOT produced any real numbers
yet. Run it yourself with:

    python generate_roc_pr.py

and check the printed sanity-check block against the known-good numbers in
COWORK_HANDOFF_1.md (coarse test acc ~92.6%, fine test acc ~99.64%) before
trusting the output PNGs. If anything errors or the sanity check fails,
that's real signal — don't paper over it.

Outputs:
    results/roc_pr_coarse.png
    results/roc_pr_fine.png
"""

import sys
from pathlib import Path

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Reuse the project's own, already-verified loading logic instead of rebuilding it.
from evaluate_model import load_checkpoint, get_test_transform, DATA_DIR, RESULTS_DIR, BATCH_SIZE, device

# num_workers=0 deliberately, not evaluate_model.py's NUM_WORKERS=4: on Windows,
# DataLoader workers are separate processes that each re-import torch (incl. its
# CUDA DLLs) from scratch. With 4 workers doing that simultaneously the page file
# can't back the memory and torch's DLL loader dies with WinError 1455. This
# script only runs the test set once for a plot, not training — no need for
# parallel loading, so num_workers=0 avoids spawning any subprocesses at all.
NUM_WORKERS = 0

# Known-good numbers from COWORK_HANDOFF_1.md — used only as a sanity check,
# not hardcoded into any output. If your real run drifts far from these,
# something about the checkpoint/data pairing has changed and the new
# ROC/PR numbers should be treated with suspicion until that's understood.
# Coarse updated 2026-08-20: rebalance_coarse_MERGED.py fixed a confirmed
# train/val/test leak (Garbage-12 silently re-packaged ~half of TrashNet's
# images under new filenames; see rebalance_coarse_MERGED.py docstring).
# 0.926 was the leaked number; 0.8772 is the real, retrained, clean one.
EXPECTED_ACC = {"coarse": 0.8772, "fine": 0.9964}

# Battery-relevant class per stage (mirrors the exact convention already
# used in evaluate_model.py's update_paper_metrics()).
BATTERY_CLASS = {"coarse": "Battery", "fine": "LiIon_18650"}

BACKBONE = "EfficientNet-B0"


@torch.no_grad()
def collect_probs(model, loader):
    """Run the full test set through the model once, returning integer
    labels and the full softmax probability matrix (N, C). No autocast —
    this is a one-off figure-generation pass, not a perf-critical loop, and
    autocast's CUDA-only assumptions in evaluate_model.py aren't worth
    dragging in here."""
    all_labels, all_probs = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)
        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def plot_roc_pr(labels, probs, class_names, stage, backbone, save_path):
    n_classes = len(class_names)
    battery_cls = BATTERY_CLASS[stage]
    battery_idx = class_names.index(battery_cls) if battery_cls in class_names else None

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle(f"{backbone} — {stage.capitalize()} Stage — ROC / Precision-Recall (one-vs-rest)",
                 fontsize=13, fontweight="bold")

    cmap = plt.get_cmap("tab10")
    roc_aucs, pr_aps = {}, {}

    for i, cls in enumerate(class_names):
        y_true = (labels == i).astype(int)
        y_score = probs[:, i]

        is_battery = (i == battery_idx)
        color = "#e74c3c" if is_battery else cmap(i % 10)
        lw = 3.0 if is_battery else 1.4
        alpha = 1.0 if is_battery else 0.75
        z = 10 if is_battery else 1

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        roc_aucs[cls] = roc_auc
        label = f"{cls} (AUC={roc_auc:.3f})" + ("  ← battery" if is_battery else "")
        ax_roc.plot(fpr, tpr, color=color, lw=lw, alpha=alpha, zorder=z, label=label)

        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        pr_aps[cls] = ap
        label2 = f"{cls} (AP={ap:.3f})" + ("  ← battery" if is_battery else "")
        ax_pr.plot(rec, prec, color=color, lw=lw, alpha=alpha, zorder=z, label=label2)

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Chance")
    ax_roc.set_xlim([0, 1]); ax_roc.set_ylim([0, 1.02])
    ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves")
    ax_roc.legend(fontsize=8, loc="lower right")
    ax_roc.grid(alpha=0.3)

    baseline = float(np.mean(labels == battery_idx)) if battery_idx is not None else None
    if baseline is not None:
        ax_pr.axhline(baseline, color="gray", ls="--", lw=1, alpha=0.5,
                       label=f"Battery baseline ({baseline:.3f})")
    ax_pr.set_xlim([0, 1]); ax_pr.set_ylim([0, 1.02])
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curves")
    ax_pr.legend(fontsize=8, loc="lower left")
    ax_pr.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

    return roc_aucs, pr_aps


def run_stage(stage: str):
    print(f"\n{'─'*60}\n  ROC/PR — {stage} stage\n{'─'*60}")
    stage_dir = DATA_DIR / stage
    test_dir = stage_dir / "test"
    if not test_dir.exists():
        print(f"  ✗ Test set not found: {test_dir} — skipping {stage}")
        return None

    test_ds = datasets.ImageFolder(test_dir, get_test_transform())
    test_ld = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
    print(f"  Test set: {len(test_ds)} images, classes: {test_ds.classes}")

    model, class_names, ckpt = load_checkpoint(BACKBONE, stage)

    # class_names comes from the checkpoint (what the model was trained on).
    # ImageFolder infers classes from the test directory's subfolders.
    # These MUST match, in the same order, or every downstream number here
    # (and in evaluate_model.py, which has this exact same assumption) is
    # silently wrong. Fail loudly instead of producing a mislabeled plot.
    if class_names != test_ds.classes:
        print(f"  ✗ MISMATCH: checkpoint classes {class_names} != test-dir classes {test_ds.classes}")
        print(f"    Not generating a plot for {stage} — fix the class ordering first.")
        return None

    labels, probs = collect_probs(model, test_ld)
    acc = float((probs.argmax(1) == labels).mean())
    print(f"  Real test accuracy this run: {acc:.4f}  (handoff doc reports {EXPECTED_ACC[stage]:.4f})")
    if abs(acc - EXPECTED_ACC[stage]) > 0.02:
        print(f"  ⚠️  Accuracy drifted more than 2 points from the documented number.")
        print(f"     Could be a different checkpoint, different test split, or a real regression.")
        print(f"     Investigate before trusting the curves below.")

    save_path = RESULTS_DIR / f"roc_pr_{stage}.png"
    roc_aucs, pr_aps = plot_roc_pr(labels, probs, class_names, stage, BACKBONE, save_path)

    battery_cls = BATTERY_CLASS[stage]
    print(f"  Per-class AUC / AP:")
    for cls in class_names:
        marker = "  <- battery-relevant class" if cls == battery_cls else ""
        print(f"    {cls:<15} AUC={roc_aucs[cls]:.4f}  AP={pr_aps[cls]:.4f}{marker}")
    print(f"  ✓ Saved: {save_path}")
    return {"accuracy": acc, "roc_auc": roc_aucs, "pr_ap": pr_aps}


if __name__ == "__main__":
    print("="*60)
    print("  generate_roc_pr.py — UNVERIFIED SCRIPT")
    print("  Written by Claude Cowork but never executed there (no working")
    print("  PyTorch in that sandbox). This is the first real run.")
    print("  Read the sanity-check output below before trusting the PNGs.")
    print("="*60)
    print(f"  Device: {device}")

    results = {}
    for stage in ["coarse", "fine"]:
        r = run_stage(stage)
        if r:
            results[stage] = r

    if len(results) < 2:
        print("\n✗ One or both stages failed to produce output — see errors above.", file=sys.stderr)
        sys.exit(1)

    print("\n✅ Done. Both PNGs written to results/. Please sanity-check the printed")
    print("   accuracy numbers against COWORK_HANDOFF_1.md before using these figures.")
