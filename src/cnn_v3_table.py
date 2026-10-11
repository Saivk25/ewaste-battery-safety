"""
cnn_v3_table.py -- Table 2 row for the production coarse model (v3) on the data_v3 test split, with the battery
threshold rule: accuracy (Wilson 95%), macro F1, battery precision, battery recall split by source (Garbage-12,
RecyBat24, damaged), Clopper-Pearson upper bound on the miss rate, false alarms. Fine model unchanged.

Run: python src/cnn_v3_table.py      Output: results/gaps/cnn_v3_table.json
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats
from torchvision import datasets
from sklearn.metrics import f1_score, precision_score

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_model import load_checkpoint, get_test_transform, BATTERY_THRESHOLD
from gap_cnn_checks import softmax_all, wilson

model, classes, _ = load_checkpoint("EfficientNet-B0", "coarse")
bi = classes.index("Battery")
ds = datasets.ImageFolder("data_v3/coarse/test", get_test_transform())
P, Y = softmax_all(model, ds)
pred = P.argmax(1); pred[P[:, bi] >= BATTERY_THRESHOLD] = bi
names = np.array([Path(p).name for p, _ in ds.samples])
yb = Y == bi
dmg = yb & np.char.startswith(names, "damaged_")
rb = yb & np.char.startswith(names, "rb")          # RecyBat24 Li-ion photos
g12 = yb & np.char.startswith(names, "g12_")       # Garbage-12 consumer batteries
det = pred == bi
k, n = int((pred == Y).sum()), len(Y)
miss = int((yb & ~det).sum())
out = dict(threshold=BATTERY_THRESHOLD, n_test=n, accuracy=k / n, accuracy_wilson95=list(wilson(k, n)),
           macro_f1=float(f1_score(Y, pred, average="macro")), battery_precision=float(precision_score(yb, det)),
           battery_recall=float(det[yb].mean()), n_battery=int(yb.sum()), battery_missed=miss,
           miss_rate_cp95_upper=float(stats.beta.ppf(0.975, miss + 1, int(yb.sum()) - miss)),
           garbage12=[int(det[g12].sum()), int(g12.sum())], recybat=[int(det[rb].sum()), int(rb.sum())],
           damaged=[int(det[dmg].sum()), int(dmg.sum())],
           false_alarms=[int((det & ~yb).sum()), int((~yb).sum())], n_train=sum(1 for _ in Path("data_v3/coarse/train").rglob("*.*")),
           n_val=sum(1 for _ in Path("data_v3/coarse/val").rglob("*.*")))
json.dump(out, open("results/gaps/cnn_v3_table.json", "w"), indent=1)
print(json.dumps(out, indent=1))
