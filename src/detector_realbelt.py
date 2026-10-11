"""
detector_realbelt.py -- the YOLOv8n battery detector on a REAL recycling conveyor.

The clutter test of detector_yolo.py pastes items on a synthetic grey belt. Here the background is
real: frames from ZeroWaste-w (Bashkirova et al., CVPR 2022, doi:10.5281/zenodo.6412647, CC BY 4.0),
filmed above the conveyor of a single-stream recycling plant in Massachusetts ("before" = unsorted
mixed waste, "after" = sorted paper). ZeroWaste has no battery class.

  A. real frames as they are -> frames with a battery box = false alarms on a real belt
     (an upper bound: a real battery in a ZeroWaste frame would also count as a false alarm)
  B. one battery crop pasted on a real frame (RecyBat24 val photos of the clutter-test half, never
     used in training or model selection), covered 0 / 25 / 50 / 75 % by a patch cut from another
     real frame -> recall on a real belt background

The first run (detector v2, trained without real frames) boxed white paper rectangles as pouch cells
in about 30 % of real frames. Fine-tuning (mode "finetune") adds real frames from HALF of the
ZeroWaste videos to training: as battery-free frames and with a pasted RecyBat24 TRAIN crop. Frames
of one video are near-identical, so the split is by video, and both detectors are evaluated only on
frames from the other videos (mode "eval").

Run: python src/detector_realbelt.py --frames C:/ewaste_ext/zerowaste finetune
     python src/detector_realbelt.py --frames C:/ewaste_ext/zerowaste eval
Outputs: runs_det/yolov8n_realbelt/, results/pub/detector_realbelt.csv, detector_realbelt_summary.json
"""
import argparse, hashlib, json, os, random, re, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from PIL import Image

from detector_yolo import crops, SEED, OCCLUSIONS, OUT, DET

V2 = "runs_det/yolov8n_recybat/weights/best.pt"
V3 = "runs_det/yolov8n_realbelt/weights/best.pt"
RB = Path("data_det_realbelt")


def video(p):
    """ZeroWaste frame name 'before__03_frame_000891.PNG' -> video 'before_03'."""
    m = re.match(r"(before|after)__(\d+)_", Path(p).name); return f"{m.group(1)}_{m.group(2)}"


def is_test(p):
    return int(hashlib.md5(video(p).encode()).hexdigest(), 16) % 2 == 0

W = H = 640


def load_frame(p):
    im = Image.open(p).convert("RGB"); s = min(im.size)        # centre square crop, as the detector sees 640x640
    l, t = (im.size[0] - s) // 2, (im.size[1] - s) // 2
    return im.crop((l, t, l + s, t + s)).resize((W, H))


def paste(rng, img, battery, occl, occluder):
    b = battery.copy(); sc = rng.uniform(110, 190) / max(b.size)
    b = b.resize((max(8, int(b.size[0] * sc)), max(8, int(b.size[1] * sc))))
    x, y = rng.randint(0, W - b.size[0]), rng.randint(0, H - b.size[1]); img.paste(b, (x, y))
    if occl > 0:
        ow = max(1, int(b.size[0] * occl)); oh = b.size[1] + 20
        ox, oy = rng.randint(0, W - ow), rng.randint(0, H - oh)
        img.paste(occluder.crop((ox, oy, ox + ow, oy + oh)), (x, max(0, y - 10)))
    return img, (x, y, x + b.size[0], y + b.size[1])


def finetune(frames):
    from ultralytics import YOLO
    rng = random.Random(SEED + 7)
    tr = [p for p in frames if not is_test(p)]
    if RB.exists(): shutil.rmtree(RB)
    (RB / "images").mkdir(parents=True); (RB / "labels").mkdir(parents=True)
    bats = crops("train", len(tr), rng)                     # RecyBat24 TRAIN split only
    for i, p in enumerate(tr):
        img = load_frame(p); img.save(RB / "images" / f"rb_neg_{i}.jpg"); (RB / "labels" / f"rb_neg_{i}.txt").write_text("")
        img, (x0, y0, x1, y1) = paste(rng, load_frame(p), bats[i % len(bats)], rng.choice(OCCLUSIONS), load_frame(rng.choice(tr)))
        img.save(RB / "images" / f"rb_comp_{i}.jpg")
        (RB / "labels" / f"rb_comp_{i}.txt").write_text(f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    yaml = RB / "data.yaml"
    yaml.write_text(f"train: [{(DET / 'images/train').resolve().as_posix()}, {(RB / 'images').resolve().as_posix()}]\n"
                    f"val: {(DET / 'images/val').resolve().as_posix()}\nnames:\n  0: battery\n")
    print(f"fine-tune: {len(tr)} real frames from {len({video(p) for p in tr})} videos (x2: empty + pasted battery)", flush=True)
    YOLO(V2).train(data=str(yaml.resolve()), epochs=10, imgsz=640, batch=16, workers=int(os.environ.get("NUM_WORKERS", 0)),
                   seed=SEED, deterministic=True, project=str(Path("runs_det").resolve()), name="yolov8n_realbelt",
                   exist_ok=True, plots=False, verbose=False)


def evaluate(frames):
    from ultralytics import YOLO
    te = [p for p in frames if is_test(p)]
    rows = []
    for tag, w in (("v2", V2), ("v3_realbelt", V3)):
        det = YOLO(w); rng = random.Random(SEED)
        bats = crops("val", len(te), rng, test=True)

        def run(img, box, cond, src):
            r = det.predict(img, conf=0.25, verbose=False)[0]
            xyxy = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
            hit = False
            if box is not None and len(xyxy):
                cx, cy = (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2
                hit = bool(np.any((cx >= box[0]) & (cx <= box[2]) & (cy >= box[1]) & (cy <= box[3])))
            rows.append(dict(detector=tag, condition=cond, frame=src.name, video=video(src), battery=box is not None,
                             det_boxes=len(xyxy), det_hit=hit, max_conf=float(r.boxes.conf.max()) if len(r.boxes) else 0.0))

        for p in te:
            run(load_frame(p), None, "real_frame", p)
        for occl in OCCLUSIONS:
            for i, p in enumerate(te):
                img, box = paste(rng, load_frame(p), bats[i % len(bats)], occl, load_frame(rng.choice(te)))
                run(img, box, f"battery_occl_{occl}", p)
        print("done", tag, flush=True)
    d = pd.DataFrame(rows); OUT.mkdir(parents=True, exist_ok=True); d.to_csv(OUT / "detector_realbelt.csv", index=False)
    s = {"n_test_frames": len(te), "test_videos": sorted({video(p) for p in te}),
         "n_train_frames": len(frames) - len(te), "train_videos": sorted({video(p) for p in frames if not is_test(p)})}
    for tag, g in d.groupby("detector"):
        real = g[g.condition == "real_frame"]
        s[tag] = {"false_alarm_frames": float((real.det_boxes > 0).mean()),
                  **{f"recall_occl_{o}": float(g[g.condition == f"battery_occl_{o}"].det_hit.mean()) for o in OCCLUSIONS}}
    try:
        from ultralytics import YOLO
        m = YOLO(V3).val(data=str((DET / "data.yaml").resolve()), verbose=False, plots=False)
        s["v3_realbelt"]["synthetic_val_mAP50"] = float(m.box.map50)
    except Exception as e:
        print("val skipped:", e)
    print(json.dumps(s, indent=1)); json.dump(s, open(OUT / "detector_realbelt_summary.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--frames", required=True); ap.add_argument("mode", choices=["finetune", "eval"])
    a = ap.parse_args()
    frames = sorted(p for p in Path(a.frames).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    finetune(frames) if a.mode == "finetune" else evaluate(frames)


if __name__ == "__main__":
    main()
