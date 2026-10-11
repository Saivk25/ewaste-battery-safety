"""
generate_gradcam.py
====================
Generates Grad-CAM visualisations for both CNN stages using the trained
EfficientNet-B0 checkpoints, to show what the model is actually attending to
when it makes a Battery / battery-subtype call. Built for the paper
resubmission (explainability section).

Reuses evaluate_model.py's checkpoint loading and test-transform logic
directly rather than reimplementing it.

IMPORTANT — provenance note:
This script was written by Claude Cowork but could NOT be executed or
verified in that session (the sandbox has no working PyTorch/CUDA stack and
ran out of disk trying to build one). It has NOT produced any real images
yet. Run it yourself with:

    python generate_gradcam.py

Read the printed selection summary (how many correct/misclassified examples
were found and why) before trusting the grid PNGs — if it prints a warning
about falling back to a different selection than requested, that's real
signal about the test set, not a bug to silently ignore.

Outputs:
    results/gradcam_coarse.png
    results/gradcam_fine.png

Grad-CAM target layer: model.features[-1], the final conv-bn-activation
block of EfficientNet-B0 (output 1280 channels, spatial map before global
pooling) — the standard Grad-CAM target for this architecture.
"""

import sys
from pathlib import Path

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from evaluate_model import load_checkpoint, get_test_transform, DATA_DIR, RESULTS_DIR, BATCH_SIZE, IMG_SIZE, device

# num_workers=0 deliberately, not evaluate_model.py's NUM_WORKERS=4: on Windows,
# DataLoader workers are separate processes that each re-import torch (incl. its
# CUDA DLLs) from scratch. With 4 workers doing that simultaneously the page file
# can't back the memory and torch's DLL loader dies with WinError 1455. This
# script only runs the test set once to pick examples, not training — no need
# for parallel loading, so num_workers=0 avoids spawning any subprocesses at all.
NUM_WORKERS = 0

BACKBONE = "EfficientNet-B0"
BATTERY_CLASS = {"coarse": "Battery", "fine": "LiIon_18650"}
N_CORRECT = 8   # aim for 6-8 correct examples
N_WRONG = 3     # aim for 2-3 misclassified examples


# ── Grad-CAM core ────────────────────────────────────────────────────────

class GradCAM:
    """Minimal, dependency-free Grad-CAM via forward/backward hooks.
    Hand-rolled instead of pulling in an external grad-cam package so there's
    nothing extra to install and the whole computation is auditable here."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, img_tensor, target_class):
        """img_tensor: (1,C,H,W), already on the model's device.
        Returns a (H_in, W_in) numpy heatmap in [0,1]."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(img_tensor)
        score = logits[0, target_class]
        score.backward()

        # (1,C,h,w) -> channel weights via global-average-pooled gradient
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1,1,h,w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=img_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()

        cam_max = cam.max()
        if cam_max > 1e-8:
            cam = cam / cam_max
        else:
            cam = np.zeros_like(cam)  # dead/uninformative gradient — show as blank, don't fake it
        return cam, float(torch.softmax(logits, dim=1)[0, target_class].item())

    def close(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def _get_jet_cmap():
    # matplotlib deprecated cm.get_cmap() in favor of colormaps[] around 3.7 —
    # support both so this doesn't break depending on the user's installed version.
    try:
        return matplotlib.colormaps["jet"]
    except AttributeError:
        return cm.get_cmap("jet")


_JET = _get_jet_cmap()


def overlay_heatmap(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = np.asarray(pil_img.convert("RGB")).astype(np.float32) / 255.0
    heat = _JET(cam)[..., :3]  # drop alpha channel from colormap
    blended = (1 - alpha) * base + alpha * heat
    return np.clip(blended, 0, 1)


# ── Selection logic ─────────────────────────────────────────────────────

@torch.no_grad()
def run_predictions(model, loader):
    preds, labels = [], []
    for imgs, lab in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(lab.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def select_examples(stage, preds, labels, class_names, samples):
    """Returns (correct_list, wrong_list), each a list of dicts with
    index/path/true/pred. Prioritises the battery-relevant class for
    'correct' examples; for 'wrong' examples, prioritises misclassifications
    that involve the battery-relevant class, falling back to any
    misclassification (and explaining why) if none exist."""
    battery_cls = BATTERY_CLASS[stage]
    battery_idx = class_names.index(battery_cls) if battery_cls in class_names else None

    correct_mask = preds == labels
    wrong_mask = ~correct_mask

    if battery_idx is not None:
        correct_battery = np.where(correct_mask & (labels == battery_idx))[0]
    else:
        correct_battery = np.array([], dtype=int)

    if stage == "coarse":
        # Coarse has real non-battery classes -> spread mostly on Battery.
        chosen_correct = list(correct_battery[:N_CORRECT])
        if len(chosen_correct) < N_CORRECT:
            other_correct = np.where(correct_mask)[0]
            other_correct = [i for i in other_correct if i not in chosen_correct]
            chosen_correct += other_correct[: N_CORRECT - len(chosen_correct)]
    else:
        # Fine stage: all 3 classes ARE battery subtypes (no non-battery
        # class in this checkpoint — see COWORK_HANDOFF_1.md gotcha #2).
        # Spread correct examples across all subtypes instead.
        chosen_correct = []
        per_class = max(1, N_CORRECT // len(class_names))
        for ci in range(len(class_names)):
            idx = np.where(correct_mask & (labels == ci))[0]
            chosen_correct += list(idx[:per_class])
        chosen_correct = chosen_correct[:N_CORRECT]

    # Misclassified selection
    battery_wrong = np.where(wrong_mask & ((labels == battery_idx) | (preds == battery_idx)))[0] \
        if battery_idx is not None else np.array([], dtype=int)
    fallback_note = None
    if len(battery_wrong) >= 1:
        chosen_wrong = list(battery_wrong[:N_WRONG])
        if len(chosen_wrong) < N_WRONG:
            rest = [i for i in np.where(wrong_mask)[0] if i not in chosen_wrong]
            chosen_wrong += rest[: N_WRONG - len(chosen_wrong)]
    else:
        chosen_wrong = list(np.where(wrong_mask)[0][:N_WRONG])
        fallback_note = (f"No misclassifications involving '{battery_cls}' exist in this test set "
                          f"(consistent with the documented 0 FNR for Battery) — showing the "
                          f"{len(chosen_wrong)} general misclassifications found instead.")

    def to_records(idx_list):
        out = []
        for i in idx_list:
            path, _ = samples[i]
            out.append({"idx": int(i), "path": path,
                        "true": class_names[labels[i]], "pred": class_names[preds[i]]})
        return out

    return to_records(chosen_correct), to_records(chosen_wrong), fallback_note


# ── Grid rendering ───────────────────────────────────────────────────────

def render_grid(gradcam, correct_records, wrong_records, stage, backbone, save_path):
    from torchvision import transforms
    tensor_tf = get_test_transform()

    def load_tensor_and_display(path):
        img = Image.open(path).convert("RGB")
        tensor = tensor_tf(img).unsqueeze(0).to(device)
        # Recreate what the model sees, spatially, for the overlay background
        # (resize+crop only, no normalize) so heatmap and image align.
        display_pil = transforms.Compose([
            transforms.Resize((256, 256)), transforms.CenterCrop(IMG_SIZE),
        ])(img)
        return tensor, display_pil

    all_records = [(r, True) for r in correct_records] + [(r, False) for r in wrong_records]
    n = len(all_records)
    if n == 0:
        print(f"  ✗ No examples selected for {stage} — nothing to render.")
        return

    n_cols = max(len(correct_records), 1)
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 6.2),
                              gridspec_kw={"hspace": 0.5})
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)
    fig.suptitle(f"{backbone} — {stage.capitalize()} Stage — Grad-CAM (target layer: features[-1])",
                 fontsize=13, fontweight="bold")

    for ax in axes.flat:
        ax.axis("off")

    for col, rec in enumerate(correct_records):
        tensor, disp = load_tensor_and_display(rec["path"])
        cam, conf = gradcam(tensor, target_class=rec["_pred_idx"])
        overlay = overlay_heatmap(disp, cam)
        ax = axes[0, col]
        ax.imshow(overlay)
        ax.set_title(f"✓ {rec['true']}\nconf={conf:.2f}", fontsize=9, color="#2E7D32")

    for col in range(n_cols):
        ax = axes[1, col]
        if col < len(wrong_records):
            rec = wrong_records[col]
            tensor, disp = load_tensor_and_display(rec["path"])
            cam, conf = gradcam(tensor, target_class=rec["_pred_idx"])
            overlay = overlay_heatmap(disp, cam)
            ax.imshow(overlay)
            ax.set_title(f"✗ true={rec['true']}\npred={rec['pred']} ({conf:.2f})",
                         fontsize=9, color="#B3261E")
        else:
            ax.axis("off")

    fig.text(0.01, 0.72, "Correctly\nclassified", fontsize=10, rotation=90, va="center")
    fig.text(0.01, 0.28, "Mis-\nclassified", fontsize=10, rotation=90, va="center")

    plt.tight_layout(rect=[0.03, 0, 1, 0.95])
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def run_stage(stage: str):
    print(f"\n{'─'*60}\n  Grad-CAM — {stage} stage\n{'─'*60}")
    test_dir = DATA_DIR / stage / "test"
    if not test_dir.exists():
        print(f"  ✗ Test set not found: {test_dir} — skipping {stage}")
        return

    test_ds = datasets.ImageFolder(test_dir, get_test_transform())
    test_ld = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    model, class_names, ckpt = load_checkpoint(BACKBONE, stage)
    if class_names != test_ds.classes:
        print(f"  ✗ MISMATCH: checkpoint classes {class_names} != test-dir classes {test_ds.classes}")
        print(f"    Not generating Grad-CAM for {stage} — fix class ordering first.")
        return

    if stage == "fine":
        print(f"  Note: fine-stage checkpoint has 3 classes, all battery subtypes")
        print(f"  ({class_names}) — there is no 'non-battery' class at this stage")
        print(f"  (NiMH/NiCd/Non_Battery are excluded proxy classes, see handoff gotcha #2).")
        print(f"  Showing subtype discrimination instead of Battery-vs-non.")

    preds, labels = run_predictions(model, test_ld)
    acc = float((preds == labels).mean())
    print(f"  Real test accuracy this run: {acc:.4f}")

    correct_records, wrong_records, fallback_note = select_examples(
        stage, preds, labels, class_names, test_ds.samples)
    if fallback_note:
        print(f"  ⚠️  {fallback_note}")
    print(f"  Selected {len(correct_records)} correct / {len(wrong_records)} misclassified examples")

    # Attach the model's predicted index to each record for Grad-CAM targeting
    # (Grad-CAM explains the model's actual decision, right or wrong.)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    for r in correct_records + wrong_records:
        r["_pred_idx"] = class_to_idx[r["pred"]]

    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)
    try:
        save_path = RESULTS_DIR / f"gradcam_{stage}.png"
        render_grid(gradcam, correct_records, wrong_records, stage, BACKBONE, save_path)
        print(f"  ✓ Saved: {save_path}")
    finally:
        gradcam.close()


if __name__ == "__main__":
    print("="*60)
    print("  generate_gradcam.py — UNVERIFIED SCRIPT")
    print("  Written by Claude Cowork but never executed there (no working")
    print("  PyTorch in that sandbox). This is the first real run.")
    print("  Read the selection summary below before trusting the PNGs.")
    print("="*60)
    print(f"  Device: {device}")

    for stage in ["coarse", "fine"]:
        run_stage(stage)

    print("\n✅ Done (if no errors above). Please open both PNGs and confirm the")
    print("   heatmaps actually vary between images and land on the object, not")
    print("   uniformly on background — that's the real check that this worked.")
