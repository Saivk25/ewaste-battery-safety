"""
detector_yolo.py -- battery DETECTION in cluttered conveyor frames (journal fix 4).

The two-stage CNN classifies one clean photo per object. A real belt shows many items per
frame, partly hidden. This script
  1. converts RecyBat24-aug bounding boxes (COCO) to YOLO format, keeping RecyBat24's own
     train/val split (no source photo appears in both) and 3 augmentations per source photo;
  2. trains YOLOv8n (ultralytics) to detect Li-ion cells (one class: battery). Version 1 trained on
     battery photos only and boxed almost every item in a cluttered frame (99% of battery-free frames
     had a box), so the training set now also holds battery-free photos (empty labels) and synthetic
     cluttered frames built ONLY from training-split photos (coarse train items + RecyBat24 train crops);
     YOLO's own validation frames use half of the RecyBat24 val source photos and the cluttered test
     below uses the other half, so no test battery is seen in training or model selection;
  3. builds synthetic cluttered frames: a grey belt with 3-8 non-battery e-waste photos
     (coarse v2 test set) plus one battery crop from RecyBat24 val photos, with the battery
     covered 0 / 25 / 50 / 75 % by another item, and battery-free frames;
  4. compares the detector (hit = predicted box centre inside the true box, conf >= 0.25)
     with the whole-frame coarse classifier (battery if P_battery >= BATTERY_THRESHOLD).

Run from the repo root:  python src/detector_yolo.py [prepare|train|clutter|all]
Outputs: data_det/ (YOLO data, git-ignored), runs_det/ (training), results/pub/detector_*.csv/json
"""
import os, sys, json, re, random, shutil, collections
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from PIL import Image

RAW = Path("data/raw/recybat24_aug"); DET = Path("data_det"); OUT = Path("results/pub")
PER_SOURCE_TRAIN, PER_SOURCE_VAL = 3, 1
N_FRAMES, OCCLUSIONS, SEED = 300, (0.0, 0.25, 0.5, 0.75), 42
N_NEG, N_COMP, N_COMP_FREE = 1500, 1500, 500      # extra training images: plain negatives, cluttered, cluttered battery-free


def val_half(name, test):
    """Split RecyBat24 val source photos: one half for YOLO validation frames, the other for the clutter test."""
    import hashlib
    return (int(hashlib.md5(src_id(name).encode()).hexdigest(), 16) % 2 == 1) == test


def src_id(n): return re.sub(r"_(rotated|bright|contrast|flip|crop|blur|occl).*$", "", n)


def prepare():
    random.seed(SEED)
    if DET.exists(): shutil.rmtree(DET)
    for split, per in (("train", PER_SOURCE_TRAIN), ("val", PER_SOURCE_VAL)):
        ann = json.load(open(RAW / split / "annotations.json"))
        boxes = collections.defaultdict(list)
        for a in ann["annotations"]:
            boxes[a["image_id"]].append(a["bbox"])
        by_src = collections.defaultdict(list)
        for im in ann["images"]:
            if split == "val" and not val_half(im["file_name"], test=False):
                continue
            if boxes.get(im["id"]) and (RAW / split / im["file_name"]).exists():   # a few annotated files are missing on disk
                by_src[src_id(im["file_name"])].append(im)
        (DET / "images" / split).mkdir(parents=True); (DET / "labels" / split).mkdir(parents=True)
        n = 0
        for s in sorted(by_src):
            for im in random.sample(by_src[s], min(per, len(by_src[s]))):
                shutil.copy2(RAW / split / im["file_name"], DET / "images" / split / im["file_name"])
                W, H = im["width"], im["height"]
                lines = [f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}" for x, y, w, h in boxes[im["id"]]]
                (DET / "labels" / split / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines))
                n += 1
        print(f"{split}: {n} images from {len(by_src)} source photos")
    add_extra_training_frames()
    (DET / "data.yaml").write_text(f"path: {DET.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: battery\n")


def crops(split, k, rng, test=None):
    ann = json.load(open(RAW / split / "annotations.json")); boxes = {a["image_id"]: a["bbox"] for a in ann["annotations"]}
    ims = [im for im in ann["images"] if im["id"] in boxes and (RAW / split / im["file_name"]).exists()
           and (test is None or val_half(im["file_name"], test))]
    rng.shuffle(ims); out = []
    for im in ims[:k]:
        x, y, w, h = boxes[im["id"]]
        out.append(Image.open(RAW / split / im["file_name"]).convert("RGB").crop((x, y, x + w, y + h)))
    return out


def add_extra_training_frames():
    rng = random.Random(SEED + 1)
    for split, pool_dir, n_neg, n_comp, n_free, crop_args in (
            ("train", "data_v2/coarse/train", N_NEG, N_COMP, N_COMP_FREE, ("train", N_COMP, None)),
            ("val", "data_v2/coarse/val", 200, 200, 100, ("val", 200, False))):
        pool = [q for c in Path(pool_dir).iterdir() if c.name != "Battery" for q in c.iterdir()]
        img_dir, lab_dir = DET / "images" / split, DET / "labels" / split
        for i, q in enumerate(rng.sample(pool, min(n_neg, len(pool)))):           # plain battery-free photos
            Image.open(q).convert("RGB").save(img_dir / f"neg_{i}.jpg"); (lab_dir / f"neg_{i}.txt").write_text("")
        bats = crops(*crop_args[:2], rng, crop_args[2])
        for i, b in enumerate(bats):                                               # cluttered frames with one battery
            img, box = make_frame(rng, pool, b, rng.choice(OCCLUSIONS))
            img.save(img_dir / f"comp_{i}.jpg")
            x0, y0, x1, y1 = box
            (lab_dir / f"comp_{i}.txt").write_text(f"0 {(x0 + x1) / 1280:.6f} {(y0 + y1) / 1280:.6f} {(x1 - x0) / 640:.6f} {(y1 - y0) / 640:.6f}")
        for i in range(n_free):                                                    # cluttered battery-free frames
            img, _ = make_frame(rng, pool); img.save(img_dir / f"compfree_{i}.jpg"); (lab_dir / f"compfree_{i}.txt").write_text("")
        print(f"{split}: +{min(n_neg, len(pool))} negatives, +{len(bats)} cluttered, +{n_free} cluttered battery-free", flush=True)


def train():
    from ultralytics import YOLO
    m = YOLO("yolov8n.pt")
    m.train(data=str((DET / "data.yaml").resolve()), epochs=30, imgsz=640, batch=16, workers=int(os.environ.get("NUM_WORKERS", 0)),
            seed=SEED, deterministic=True, project=str(Path("runs_det").resolve()), name="yolov8n_recybat", exist_ok=True, plots=False, verbose=False)


def make_frame(rng, clutter_pool, battery=None, occl=0.0):
    W = H = 640
    belt = np.clip(np.random.RandomState(rng.randint(0, 2**31 - 1)).normal(110, 12, (H, W, 3)), 0, 255).astype(np.uint8)
    img = Image.fromarray(belt)
    for _ in range(rng.randint(3, 9)):
        c = Image.open(rng.choice(clutter_pool)).convert("RGB"); s = rng.randint(110, 220)
        c = c.resize((s, s)); img.paste(c, (rng.randint(0, W - s), rng.randint(0, H - s)))
    box = None
    if battery is not None:
        b = battery.copy(); scale = rng.uniform(110, 190) / max(b.size); b = b.resize((max(8, int(b.size[0] * scale)), max(8, int(b.size[1] * scale))))
        x, y = rng.randint(0, W - b.size[0]), rng.randint(0, H - b.size[1]); img.paste(b, (x, y)); box = (x, y, x + b.size[0], y + b.size[1])
        if occl > 0:                                   # another item covers the left part of the battery
            c = Image.open(rng.choice(clutter_pool)).convert("RGB"); ow = int(b.size[0] * occl)
            c = c.resize((max(ow, 1), b.size[1] + 20)); img.paste(c, (x, max(0, y - 10)))
    return img, box


def clutter():
    import pandas as pd, torch
    from ultralytics import YOLO
    from evaluate_model import load_checkpoint, get_test_transform, BATTERY_THRESHOLD, device
    rng = random.Random(SEED)
    pool = [p for c in Path("data_v2/coarse/test").iterdir() if c.name != "Battery" for p in c.iterdir()]
    bats = crops("val", N_FRAMES, rng, test=True)
    det = YOLO("runs_det/yolov8n_recybat/weights/best.pt")
    cnn, classes, _ = load_checkpoint("EfficientNet-B0", "coarse"); bi = classes.index("Battery"); tf = get_test_transform()
    rows = []
    for occl in OCCLUSIONS + ("none",):
        for i in range(N_FRAMES):
            if occl == "none":
                img, box = make_frame(rng, pool)
            else:
                img, box = make_frame(rng, pool, bats[i], occl)
            r = det.predict(img, conf=0.25, verbose=False)[0]
            xyxy = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
            hit = False
            if box is not None:
                cx, cy = (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2
                hit = bool(np.any((cx >= box[0]) & (cx <= box[2]) & (cy >= box[1]) & (cy <= box[3])))
            with torch.no_grad():
                p = torch.softmax(cnn(tf(img).unsqueeze(0).to(device)), 1)[0, bi].item()
            rows.append(dict(occlusion=occl, frame=i, battery=box is not None, det_hit=hit, det_boxes=len(xyxy),
                             cls_P=p, cls_alarm=p >= BATTERY_THRESHOLD))
        print("done occlusion", occl, flush=True)
    df = pd.DataFrame(rows); OUT.mkdir(parents=True, exist_ok=True); df.to_csv(OUT / "detector_clutter.csv", index=False)
    summ = {}
    for occl, g in df.groupby("occlusion", sort=False):
        if occl == "none":
            summ["battery_free"] = {"det_false_alarm_frames": float((g.det_boxes > 0).mean()), "cls_false_alarm_frames": float(g.cls_alarm.mean()), "n": len(g)}
        else:
            summ[f"occlusion_{occl}"] = {"det_recall": float(g.det_hit.mean()), "cls_recall": float(g.cls_alarm.mean()), "n": len(g)}
    try:
        res = json.load(open("runs_det/yolov8n_recybat/results.json")) if Path("runs_det/yolov8n_recybat/results.json").exists() else {}
    except Exception:
        res = {}
    m = YOLO("runs_det/yolov8n_recybat/weights/best.pt").val(data=str((DET / "data.yaml").resolve()), verbose=False, plots=False)
    summ["val_clean"] = {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map), "precision": float(m.box.mp), "recall": float(m.box.mr)}
    json.dump(summ, open(OUT / "detector_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("prepare", "all"): prepare()
    if step in ("train", "all"): train()
    if step in ("clutter", "all"): clutter()
