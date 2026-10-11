"""
bench_speed.py -- how fast is each stage, and can it keep up with a belt? (run on an otherwise idle machine)

Per-image latency, batch 1, 20 warm-up + 100 timed runs: coarse EfficientNet-B0 and MobileNetV2 on GPU and on CPU
with 1 and 4 threads (a rough stand-in for an edge CPU; a Raspberry Pi will be slower), YOLOv8n on a 640 px frame,
and the per-step cost of fuzzy inference + gated memory + FSM. Items per second the camera stage can classify
follow directly. Output: results/pub/speed.json
"""
import sys, time, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, torch


def t_ms(fn, n=100, warm=20, cuda=False):
    for _ in range(warm): fn()
    if cuda: torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n): fn()
    if cuda: torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


if __name__ == "__main__":
    from evaluate_model import load_checkpoint
    out = {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "torch": torch.__version__}
    x = torch.randn(1, 3, 224, 224)
    for name, path in (("EfficientNet-B0", None), ("MobileNetV2", Path("runs_cnn/MobileNetV2_s1/checkpoints/best_MobileNetV2_coarse.pth"))):
        if path is not None and not path.exists(): continue
        m, _, _ = load_checkpoint(name, "coarse", path); m.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                mg, xg = m.cuda(), x.cuda()
                out[f"{name}_gpu_ms"] = t_ms(lambda: mg(xg), cuda=True)
            mc = m.cpu()
            for th in (1, 4):
                torch.set_num_threads(th)
                out[f"{name}_cpu{th}_ms"] = t_ms(lambda: mc(x), n=50, warm=10)
    torch.set_num_threads(4)
    w = Path("runs_det/yolov8n_recybat/weights/best.pt")
    if w.exists():
        from ultralytics import YOLO
        det = YOLO(str(w)); img = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)
        if torch.cuda.is_available():
            out["YOLOv8n_gpu_ms"] = t_ms(lambda: det.predict(img, device=0, verbose=False), n=50, warm=10)
        out["YOLOv8n_cpu4_ms"] = t_ms(lambda: det.predict(img, device="cpu", verbose=False), n=30, warm=5)
    # control loop per 0.1 s step: fuzzy inference + memory + FSM
    import sensor_simulation as ss
    from fuzzy_inference import MamdaniFIS
    from adaptive_risk import AdaptiveRiskModel
    from supervisory_control import SupervisoryFSM
    df = ss.run_simulation(duration_s=60.0, battery_arrives_at=5.0, seed=1)
    fis, arm, fsm = MamdaniFIS(), AdaptiveRiskModel(), SupervisoryFSM()
    rows = list(df.itertuples()); t0 = time.perf_counter()
    for r in rows:
        s, _ = fis.infer(r.temperature_C, r.mq2_ppm, r.mq135_ppm, bool(r.metal_present))
        _, _, rf = arm.update(S_t=s, P_battery=0.5, object_present=bool(r.object_present), temperature=r.temperature_C)
        fsm.step(r.time, rf, r.temperature_C, r.mq2_ppm, r.mq135_ppm)
    out["control_step_ms"] = (time.perf_counter() - t0) / len(rows) * 1000
    out["control_budget_ms"] = 100.0
    for k in [k for k in out if k.endswith("_ms") and not k.startswith("control")]:
        out[k.replace("_ms", "_items_per_s")] = 1000.0 / out[k]
    Path("results/pub").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("results/pub/speed.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
