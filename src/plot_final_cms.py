"""plot_final_cms.py -- confusion matrices for the final CNNs: coarse v2 with the battery
threshold rule (data_v3 test set) and the fine model. -> docs/figures/fig_cm_final.png"""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8"); sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets
from evaluate_model import load_checkpoint, get_test_transform, BATTERY_THRESHOLD
from gap_cnn_checks import softmax_all

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
cm_m, cc, _ = load_checkpoint("EfficientNet-B0", "coarse")
ds = datasets.ImageFolder(Path("data_v3/coarse/test"), get_test_transform()); P, Y = softmax_all(cm_m, ds)
bi = cc.index("Battery"); pred = P.argmax(1); pred[P[:, bi] >= BATTERY_THRESHOLD] = bi
fm, fc, _ = load_checkpoint("EfficientNet-B0", "fine")
fds = datasets.ImageFolder(Path("data/fine/test"), get_test_transform()); Pf, Yf = softmax_all(fm, fds)
for a, y, p, cls, title in ((ax[0], Y, pred, cc, f"Coarse (6 classes), {len(Y)} test images, Battery if P >= {BATTERY_THRESHOLD}"),
                            (ax[1], Yf, Pf.argmax(1), fc, f"Fine (3 Li-ion types), {len(Yf)} test images")):
    cm = np.zeros((len(cls), len(cls)), int)
    for t_, p_ in zip(y, p): cm[t_, p_] += 1
    norm = cm / cm.sum(1, keepdims=True)
    a.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cls)):
        for j in range(len(cls)):
            a.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=8, color="white" if norm[i, j] > 0.6 else "black")
    a.set_xticks(range(len(cls))); a.set_xticklabels(cls, rotation=30, fontsize=8); a.set_yticks(range(len(cls))); a.set_yticklabels(cls, fontsize=8)
    a.set_xlabel("Predicted"); a.set_ylabel("True"); a.set_title(title, fontsize=9)
    print(title, "accuracy", round(float((np.array(y) == np.array(p)).mean()), 4))
fig.tight_layout(); Path("docs/figures").mkdir(exist_ok=True); fig.savefig("docs/figures/fig_cm_final.png", dpi=200)
