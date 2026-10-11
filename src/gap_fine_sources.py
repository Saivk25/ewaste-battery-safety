"""
gap_fine_sources.py -- fine-model accuracy counted per independent source photo.

The fine test set holds 2,208 images, but they are rotation/brightness augmentations of only
164 source photos (12-16 each). This reports image-level accuracy, source-level accuracy
(majority vote and "all augmentations correct"), and a Wilson interval on the 164 sources.
Run from the repo root:  python src/gap_fine_sources.py   -> results/gaps/fine_sources.json
"""
import os, sys, re, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8"); sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, torch, collections
from torch.utils.data import DataLoader
from torchvision import datasets
from evaluate_model import load_checkpoint, get_test_transform, DATA_DIR, device
from gap_cnn_checks import wilson

m, classes, _ = load_checkpoint("EfficientNet-B0", "fine"); m.eval()
ds = datasets.ImageFolder(DATA_DIR / "fine" / "test", get_test_transform())
preds = []
with torch.no_grad():
    for x, _ in DataLoader(ds, batch_size=64, num_workers=0):
        preds.append(m(x.to(device)).argmax(1).cpu().numpy())
preds = np.concatenate(preds)
src = collections.defaultdict(list)
for (path, y), p in zip(ds.samples, preds):
    sid = re.sub(r"_(rotated|bright|contrast|flip|crop|blur|occl).*$", "", Path(path).name)
    src[(sid, y)].append(p == y)
img_acc = float((preds == np.array([y for _, y in ds.samples])).mean())
maj = [np.mean(v) > 0.5 for v in src.values()]; allc = [all(v) for v in src.values()]
n = len(src); k = int(sum(maj))
res = {"images": len(preds), "image_accuracy": img_acc, "sources": n, "source_accuracy_majority": k / n,
       "source_accuracy_majority_wilson95": wilson(k, n), "sources_all_augmentations_correct": int(sum(allc)),
       "sources_with_any_error": [s for (s, y), v in src.items() if not all(v)]}
print(json.dumps(res, indent=1)); json.dump(res, open("results/gaps/fine_sources.json", "w"), indent=1, default=float)
