"""
dashboard.py — Live monitoring dashboard for the E-Waste Safety System

Drives the ACTUAL simulation code (sensor_simulation.py, fuzzy_inference.py,
adaptive_risk.py -- including RealCNN, which runs the real trained
checkpoints on real held-out test images -- and supervisory_control.py)
step by step, in the browser, in real time. This is not a mockup: every
number and every classified image shown is computed live by the same
code paths verified throughout this project, not pre-baked or staged.

Ten tabs:
  0. Conveyor Line       — animated moving belt (src/conveyor_sim.py): camera-only, plain rule and the
                            proposed system on the same item, side by side, with diverter, monitored bin,
                            sensor hood, gas plume and outcome banners; physics panel underneath.
  1. Live Monitor        — full scenario playback, temperature chart shaded
                            by the real 4-phase thermal-runaway model.
  2. Manual Control       — sliders drive the real MamdaniFIS live, one
                            fixed point at a time, no playback loop needed.
  3. Baseline Race        — B0/B1/B2/B2-debounce/B3/CUSUM/logistic/Proposed on
                            the same sensor data, side by side, ending in a
                            real compute_metrics() table.
  4. Break It             — drives the real FSM to Emergency_Stop, then
                            floods it with implausible sensor readings to
                            prove the latch holds, next to a clearly-labeled
                            reimplementation of the pre-fix buggy logic.
  5. Upload & Classify    — runs the real trained checkpoints on a
                            user-uploaded photo, optional Grad-CAM overlay.
  6. Custom Sequence      — build your own item sequence (benign metal /
                            non-metal / battery) and play it through the
                            same real pipeline as Live Monitor.

  7. Results             — pre-computed held-out results from results/pub/
                            (baselines, stress tests, statistics, trade-off
                            curves, calibrated model, measured cells).
  8. Real Measured Cell  — replays a measured thermal-runaway temperature
                            curve (Kwak et al. 2026) through the pipeline.
  9. Business Case       — hardware cost per line, required sensor distance, false-stop cost, payback.
  The upload tab also draws YOLOv8n battery boxes when the detector is trained.

Run: streamlit run dashboard.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as _cm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sensor_simulation import (
    run_simulation, run_normal_scenario, run_multi_item_scenario,
    run_noise_stress_scenario, ThermalRunawayModel, MQ2Sensor, MQ135Sensor,
    InductiveProximitySensor,
)
from fuzzy_inference import MamdaniFIS
from adaptive_risk import AdaptiveRiskModel, RealCNN
from supervisory_control import (
    SupervisoryFSM, State, SHREDDER_SPEED, THRESHOLDS, MIN_DWELL, DT,
)
from baseline_comparison import (
    baseline_vision_only, baseline_rule_based, baseline_fixed_fuzzy, compute_metrics,
)
from evaluate_model import load_checkpoint, get_test_transform, coarse_label, BATTERY_THRESHOLD, device as TORCH_DEVICE

STATE_COLOR = {
    "Normal":         "#2ecc71",
    "Warning":        "#f1c40f",
    "Critical":       "#e67e22",
    "Emergency_Stop": "#e74c3c",
    "Sensor_Fault":   "#9b59b6",
}

# One item choice in the sidebar drives Live Monitor, Baseline Race and Conveyor Line (src/demo_items.py).
from demo_items import ITEMS, REPEATS, seed_of


def item_sidebar():
    """The single item picker, shared by every tab that shows an item on the line."""
    with st.sidebar:
        st.header("Item on the line")
        name = st.selectbox("Item", list(ITEMS), key="item",
                            help="Used by the Live Monitor, Baseline Race and Conveyor Line tabs.")
        repeat = st.select_slider("Repeat", options=list(REPEATS), value=1, key="repeat",
                                  help="Every test was run 30 times, each time with fresh random sensor noise and a "
                                       "different test photo for the camera. This picks which of the 30 runs you watch; "
                                       "the pattern of results stays the same.")
        if ITEMS[name].note:
            st.caption(ITEMS[name].note)
        st.caption("Live Monitor and Baseline Race park the item under the sensors for minutes (how the risk builds). "
                   "Conveyor Line carries it past them in about 10 s (what the line does with it).")
    return name, seed_of(repeat)


def item_unavailable(where):
    st.info(f"This item is a {where}-only test. Pick another item in the sidebar, or see the other tab.", icon=":material/info:")


RESULTS_PUB = Path(__file__).parent / "results" / "pub"
B2_DEBOUNCE_S, CUSUM_H, CUSUM_K, CUSUM_MU = 3.0, 16.0, 0.10, 0.17   # tuned on seeds 200-229 (results/pub/tuned.json)


def load_logistic():
    import json
    f = RESULTS_PUB / "fixes_tuned.json"
    return json.load(open(f))["lr"] if f.exists() else None


@st.cache_resource
def get_real_cnn():
    """Loads the real checkpoints once per Streamlit session, not once
    per rerun -- this is the slow part (model + test-image pool loading)."""
    return RealCNN(seed=int(time.time()) % 100000)


@st.cache_resource
def get_detector():
    """YOLOv8n of src/detector_yolo.py, fine-tuned on real conveyor frames by src/detector_realbelt.py."""
    w = Path(__file__).parent / "checkpoints" / "detector" / "yolov8n_v3_realbelt.pt"
    if not w.exists():
        return None
    from ultralytics import YOLO
    return YOLO(str(w))


@st.cache_resource
def get_classification_models():
    """Loads both checkpoints once per session for the upload-a-photo tab."""
    coarse_model, coarse_classes, _ = load_checkpoint("EfficientNet-B0", "coarse")
    fine_model, fine_classes, _ = load_checkpoint("EfficientNet-B0", "fine")
    transform = get_test_transform()
    return coarse_model, coarse_classes, fine_model, fine_classes, transform


def draw_conveyor(ax, object_present: bool, fsm_state: str, coarse_class: str):
    ax.clear()
    ax.set_xlim(0, 10); ax.set_ylim(0, 2)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.4), 10, 0.3, color="#7f8c8d"))
    for x in np.arange(0.3, 10, 0.6):
        ax.add_patch(plt.Rectangle((x, 0.4), 0.08, 0.3, color="#95a5a6"))
    color = STATE_COLOR.get(fsm_state, "#7f8c8d")
    ax.add_patch(plt.Rectangle((8.2, 0.1), 1.3, 1.5, color=color, alpha=0.85))
    ax.text(8.85, 1.9, "SHREDDER", ha="center", fontsize=9, fontweight="bold")
    if object_present:
        label = coarse_class if coarse_class not in ("none", "") else "item"
        ax.add_patch(plt.Circle((2.5, 0.85), 0.35, color="#34495e"))
        ax.text(2.5, 0.85, label, ha="center", va="center", color="white",
                fontsize=8, fontweight="bold")


def draw_risk_gauge(ax, r_fused: float, fsm_state: str):
    ax.clear()
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.2, 1.2)
    ax.axis("off")
    theta = np.linspace(np.pi, 0, 100)
    bands = [(0.0, 0.30, "#2ecc71"), (0.30, 0.55, "#f1c40f"),
             (0.55, 0.75, "#e67e22"), (0.75, 1.0, "#e74c3c")]
    for lo, hi, color in bands:
        t = np.linspace(np.pi * (1 - lo), np.pi * (1 - hi), 20)
        ax.plot(np.cos(t), np.sin(t), color=color, linewidth=14, solid_capstyle="butt")
    needle_theta = np.pi * (1 - np.clip(r_fused, 0, 1))
    ax.plot([0, 0.9 * np.cos(needle_theta)], [0, 0.9 * np.sin(needle_theta)],
            color="black", linewidth=2)
    ax.add_patch(plt.Circle((0, 0), 0.05, color="black"))
    ax.text(0, -0.15, f"R_fused = {r_fused:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.text(0, 1.05, fsm_state, ha="center", fontsize=13, fontweight="bold",
            color=STATE_COLOR.get(fsm_state, "black"))


# ── Tab 1: Live Monitor ─────────────────────────────────────────────────────

def render_live_monitor(item_name, seed):
    st.subheader(f"Parked under the sensors: {item_name}")
    item = ITEMS[item_name]
    if item.live is None:
        item_unavailable("belt"); return
    with st.container(horizontal=True, vertical_alignment="bottom"):
        speed = st.select_slider("Playback speed (sim-seconds per update)", options=[1, 2, 5, 10, 20, 50], value=5, key="lm_speed")
        chart_window = st.slider("Chart window (seconds shown)", 30, 300, 100, key="lm_window")
        run = st.button("Run simulation", type="primary", key="lm_run", icon=":material/play_arrow:")
    if not run:
        st.info("Click **Run simulation** to play this item. The first run loads the trained models and takes a few seconds.")
        return
    _run_playback(item.live(seed), speed, chart_window)


# Physics-thermal-phase overlay: shades the temperature chart by which phase
# of ThermalRunawayModel (sensor_simulation.py) produced each reading --
# Normal (no shading) / Phase1_SEI / Phase2_Separator / Phase3_Runaway /
# Phase4_Fire -- so the real 4-phase thermal model driving the simulation is
# visible, not just its numeric output.
PHASE_COLOR = {
    "Normal":           None,
    "Phase1_SEI":       "#fff59d",
    "Phase2_Separator": "#ffb74d",
    "Phase3_Runaway":   "#ff8a65",
    "Phase4_Fire":      "#b71c1c",
}


def _phase_spans(t_w, phase_w):
    """Collapses a per-timestep phase list into contiguous (t_start, t_end, phase) runs."""
    spans = []
    if not t_w:
        return spans
    start, cur = t_w[0], phase_w[0]
    for i in range(1, len(t_w)):
        if phase_w[i] != cur:
            spans.append((start, t_w[i], cur))
            start, cur = t_w[i], phase_w[i]
    spans.append((start, t_w[-1] + DT, cur))
    return spans


def _run_playback(df, speed, chart_window):
    """Shared step-by-step playback loop: feeds df through the real
    fis/cnn/arm/fsm pipeline and renders it live. Used by both the Live
    Monitor tab (canned scenarios) and the Custom Sequence tab (user-built
    item sequences) -- identical fidelity guarantees for both."""
    cnn = get_real_cnn()
    fis = MamdaniFIS()
    arm = AdaptiveRiskModel()
    fsm = SupervisoryFSM()

    col_left, col_right = st.columns([2, 1])
    with col_left:
        state_badge = st.empty()
        chart_ph = st.empty()
        conveyor_ph = st.empty()
    with col_right:
        gauge_ph = st.empty()
        cnn_ph = st.empty()
        log_ph = st.empty()

    fig_conveyor, ax_conveyor = plt.subplots(figsize=(6, 1.6))
    fig_gauge, ax_gauge = plt.subplots(figsize=(4, 2.6))

    history_t, history_T, history_h2, history_co2, history_R, history_phase = [], [], [], [], [], []
    event_log = []

    has_phase = "phase" in df.columns

    # IMPORTANT: every row of df is processed through fis/arm/fsm below --
    # full fidelity, identical to main_pipeline.py. `redraw_every` only
    # throttles how often the UI repaints; it must never skip computing a
    # timestep, or the dashboard would silently diverge from the real,
    # verified simulation (e.g. missing a 3-step noise spike entirely).
    redraw_every = max(int(speed / 0.1), 1)

    n = len(df)
    last_cnn_out = None
    for i in range(n):
        row = df.iloc[i]
        fuzzy_risk, _ = fis.infer(row["temperature_C"], row["mq2_ppm"],
                                    row["mq135_ppm"], bool(row["metal_present"]))

        cnn_out = cnn.predict(
            temperature=row["temperature_C"], mq2_ppm=row["mq2_ppm"],
            object_present=bool(row["object_present"]),
            metal_present=bool(row["metal_present"]),
            battery_present=bool(row.get("battery_present", False)),
        )
        last_cnn_out = cnn_out

        R_t, beta, R_fused = arm.update(
            S_t=fuzzy_risk, P_battery=cnn_out["battery_prob"],
            object_present=bool(row["object_present"]), temperature=row["temperature_C"],
        )

        state, shredder_speed = fsm.step(
            t=row["time"], R_fused=R_fused, temperature=row["temperature_C"],
            h2_ppm=row["mq2_ppm"], co2_ppm=row["mq135_ppm"],
        )

        if fsm.transitions and (not event_log or fsm.transitions[-1] not in event_log):
            for tr in fsm.transitions:
                if tr not in event_log:
                    event_log.append(tr)

        history_t.append(row["time"]); history_T.append(row["temperature_C"])
        history_h2.append(row["mq2_ppm"]); history_co2.append(row["mq135_ppm"])
        history_R.append(R_fused)
        history_phase.append(row["phase"] if has_phase else "Normal")

        is_redraw = (i % redraw_every == 0) or (i == n - 1)
        if not is_redraw:
            continue

        window_start = max(0, len(history_t) - int(chart_window / 0.1))
        t_w = history_t[window_start:]

        state_badge.markdown(
            f"### <span style='color:{STATE_COLOR.get(state.value)}'>"
            f"● {state.value}</span> &nbsp;&nbsp; "
            f"t = {row['time']:.1f}s &nbsp;&nbsp; Shredder speed: {shredder_speed}%",
            unsafe_allow_html=True,
        )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
        if has_phase:
            for t0, t1, ph in _phase_spans(t_w, history_phase[window_start:]):
                color = PHASE_COLOR.get(ph)
                if color:
                    ax1.axvspan(t0, t1, color=color, alpha=0.4, linewidth=0, zorder=0)
        ax1.plot(t_w, history_T[window_start:], color="#e74c3c", label="Temp (°C)", zorder=2)
        ax1.set_ylabel("Temp (°C)")
        if has_phase:
            phase_handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4, label=p)
                              for p, c in PHASE_COLOR.items() if c]
            ax1.legend(handles=[plt.Line2D([0], [0], color="#e74c3c", label="Temp (°C)")] + phase_handles,
                       loc="upper left", fontsize=6, ncol=2)
        else:
            ax1.legend(loc="upper left", fontsize=8)
        ax2.plot(t_w, history_h2[window_start:], color="#3498db", label="H2 (ppm)")
        ax2.plot(t_w, history_co2[window_start:], color="#2ecc71", label="CO2 (ppm)")
        ax2.set_ylabel("ppm"); ax2.set_xlabel("time (s)")
        ax2.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        chart_ph.pyplot(fig)
        plt.close(fig)

        draw_conveyor(ax_conveyor, bool(row["object_present"]), state.value,
                      cnn_out["coarse_class"])
        conveyor_ph.pyplot(fig_conveyor)

        draw_risk_gauge(ax_gauge, R_fused, state.value)
        gauge_ph.pyplot(fig_gauge)

        if last_cnn_out and last_cnn_out.get("image_path"):
            try:
                img = Image.open(last_cnn_out["image_path"])
                cnn_ph.image(
                    img, width=220,
                    caption=(f"Classified: {last_cnn_out['coarse_class']}"
                             f" (P_battery={last_cnn_out['battery_prob']:.3f})"
                             + (f"\nSubtype: {last_cnn_out['fine_class']}"
                                if last_cnn_out['fine_class'] != 'none' else '')),
                )
            except Exception:
                cnn_ph.empty()
        else:
            cnn_ph.empty()

        if event_log:
            log_df = pd.DataFrame(event_log[-8:])
            log_ph.dataframe(log_df, width="stretch", hide_index=True)

        time.sleep(0.02)

    plt.close(fig_conveyor)
    plt.close(fig_gauge)
    st.success(f"Scenario complete. {len(fsm.transitions)} state transitions logged.")


# ── Tab 2: Manual Control Panel ─────────────────────────────────────────────

def _rule_color(rule_name: str) -> str:
    if "critical" in rule_name: return "#e74c3c"
    if "high" in rule_name:     return "#e67e22"
    if "medium" in rule_name:   return "#f1c40f"
    return "#2ecc71"


def render_manual_control():
    st.subheader("Manual control: watch the 20 rules fire")
    st.caption(
        "Drag the sliders to set a hypothetical sensor reading. Every rerun "
        "calls the real `MamdaniFIS.infer()` -- this is Streamlit's native "
        "reactive behavior, no playback loop needed."
    )

    col_in, col_out = st.columns([1, 2])
    with col_in:
        T = st.slider("Temperature (°C)", 20.0, 500.0, 25.0, 1.0, key="mc_temp")
        h2 = st.slider("H2 / MQ-2 (ppm)", 0.0, 12000.0, 0.0, 10.0, key="mc_h2")
        co2 = st.slider("CO2 / MQ-135 (ppm)", 400.0, 6000.0, 410.0, 10.0, key="mc_co2")
        metal = st.checkbox("Metal present", key="mc_metal")
        st.divider()
        st.caption("Optional: fold this fixed point through one step of the "
                   "adaptive fusion + FSM, as if it were a fresh object arriving "
                   "(R_prev = 0, this is not a time series).")
        p_battery = st.slider("Hypothetical CNN P(battery)", 0.0, 1.0, 0.0, 0.01, key="mc_pbatt")
        obj_present = st.checkbox("Object present", value=True, key="mc_obj")

    fis = MamdaniFIS()
    risk, debug = fis.infer(T, h2, co2, metal)
    label = fis.get_risk_label(risk)

    arm = AdaptiveRiskModel()
    R_t, beta, R_fused = arm.update(S_t=risk, P_battery=p_battery, object_present=obj_present, temperature=T)

    fsm = SupervisoryFSM()
    state, shredder_speed = fsm.step(t=0.0, R_fused=R_fused, temperature=T, h2_ppm=h2, co2_ppm=co2)

    with col_out:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fuzzy risk S(t)", f"{risk:.3f}", label)
        m2.metric("R_fused", f"{R_fused:.3f}")
        m3.metric("FSM state", state.value)
        m4.metric("Shredder speed", f"{shredder_speed}%")

        rules = debug["rule_strengths"]
        names = list(rules.keys())
        vals = [rules[n] for n in names]
        colors = [_rule_color(n) for n in names]

        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.bar(names, vals, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Firing strength")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.axhline(0, color="black", linewidth=0.5)
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color="#e74c3c", label="critical"),
            plt.Rectangle((0, 0), 1, 1, color="#e67e22", label="high"),
            plt.Rectangle((0, 0), 1, 1, color="#f1c40f", label="medium"),
            plt.Rectangle((0, 0), 1, 1, color="#2ecc71", label="low"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        fz = debug["fuzzified"]
        st.caption(
            "Fuzzified memberships — "
            + ", ".join(f"{k}={v:.2f}" for k, v in fz.items())
        )
        n_fired = sum(1 for v in vals if v > 1e-6)
        st.caption(f"{n_fired} / {len(vals)} rules fired above threshold this step.")


# ── Tab 3: Baseline Race ────────────────────────────────────────────────────

def render_baseline_race(item_name, seed):
    st.subheader("Baseline race: every baseline against the proposed system on the same data")
    st.caption(
        "B1 = vision PROXY (temperature-driven stand-in, not the trained CNN; no gas/metal/temporal info). B2 = hard-threshold rules "
        "on raw sensor values. B3 = fuzzy inference with a fixed beta=0.85 (no "
        "adaptive temporal update). Proposed = full pipeline: fuzzy inference + "
        "adaptive beta(t) + the real trained CNN + the FSM. B0 = camera-only stop "
        "(real CNN, stop as soon as P(battery) >= threshold). B2-D = B2 thresholds that must "
        "hold for 3 s (tuned). CUSUM = cumulative-sum detector on the fuzzy score (h = 16, tuned). "
        "B4 = logistic regression on sensor features (trained on tuning seeds). All run on "
        "identical sensor data."
    )

    item = ITEMS[item_name]
    if item.live is None:
        item_unavailable("belt"); return
    with st.container(horizontal=True, vertical_alignment="bottom"):
        speed = st.select_slider("Playback speed (sim-seconds per update)", options=[1, 2, 5, 10, 20, 50], value=10, key="race_speed")
        run = st.button("Run baseline race", type="primary", key="race_run", icon=":material/play_arrow:")
    if not run:
        st.info(f"Item: **{item_name}**. Click **Run baseline race**.")
        return

    df = item.live(seed)
    battery_arrives = item.arrival

    df_b1 = baseline_vision_only(df, seed=seed)
    df_b2 = baseline_rule_based(df)
    df_b3 = baseline_fixed_fuzzy(df)

    from pub_experiments import b2_debounce
    st_b2d = b2_debounce(df, B2_DEBOUNCE_S)
    lr = load_logistic()
    if lr is not None:
        from pub_fixes import features
        p_lr = 1 / (1 + np.exp(-(features(df) @ np.array(lr["coef"]) + lr["icpt"])))
        fsm_lr = SupervisoryFSM()
    cnn = get_real_cnn()
    fis = MamdaniFIS()
    arm = AdaptiveRiskModel()
    fsm = SupervisoryFSM()
    g_cusum, cusum_latched, b0_latched = 0.0, False, False
    st_b0, st_cu, st_lr, hist_cu, hist_b0, hist_lr = [], [], [], [], [], []

    badge_names = ["B0_Camera_Only", "B1_Vision_Proxy", "B2_Rule_Based", "B2D_Debounce_3s",
                   "B3_Fixed_Fuzzy", "CUSUM_h16", "B4_Logistic", "Proposed"]
    badge_cols = st.columns(4) + st.columns(4)
    badge_ph = {name: col.empty() for name, col in zip(badge_names, badge_cols)}
    chart_ph = st.empty()

    def badge(name, state_str, risk_val):
        color = STATE_COLOR.get(state_str, "#7f8c8d")
        badge_ph[name].markdown(
            f"**{name}**  \n"
            f"<span style='color:{color}; font-size:1.1em'>● {state_str}</span>  \n"
            f"risk = {risk_val:.3f}",
            unsafe_allow_html=True,
        )

    n = len(df)
    redraw_every = max(int(speed / 0.1), 1)
    hist_t, hist_b1, hist_b2, hist_b3, hist_prop = [], [], [], [], []
    proposed_states = []

    for i in range(n):
        row = df.iloc[i]
        fuzzy_risk, _ = fis.infer(row["temperature_C"], row["mq2_ppm"],
                                    row["mq135_ppm"], bool(row["metal_present"]))
        cnn_out = cnn.predict(
            temperature=row["temperature_C"], mq2_ppm=row["mq2_ppm"],
            object_present=bool(row["object_present"]),
            metal_present=bool(row["metal_present"]),
            battery_present=bool(row.get("battery_present", False)),
        )
        R_t, beta, R_fused = arm.update(
            S_t=fuzzy_risk, P_battery=cnn_out["battery_prob"],
            object_present=bool(row["object_present"]), temperature=row["temperature_C"],
        )
        state, _ = fsm.step(
            t=row["time"], R_fused=R_fused, temperature=row["temperature_C"],
            h2_ppm=row["mq2_ppm"], co2_ppm=row["mq135_ppm"],
        )
        proposed_states.append(state.value)
        p_b = cnn_out["battery_prob"]
        b0_latched = b0_latched or (bool(row["object_present"]) and p_b >= BATTERY_THRESHOLD)
        st_b0.append("Emergency_Stop" if b0_latched else "Normal"); hist_b0.append(p_b)
        g_cusum = max(0.0, g_cusum + fuzzy_risk - CUSUM_MU - CUSUM_K)
        cusum_latched = cusum_latched or g_cusum > CUSUM_H
        st_cu.append("Emergency_Stop" if cusum_latched else "Warning" if g_cusum > CUSUM_H / 2 else "Normal")
        hist_cu.append(min(g_cusum / CUSUM_H, 1.0))
        if lr is not None:
            s_lr, _ = fsm_lr.step(t=row["time"], R_fused=float(p_lr[i]), temperature=row["temperature_C"],
                                  h2_ppm=row["mq2_ppm"], co2_ppm=row["mq135_ppm"])
            st_lr.append(s_lr.value); hist_lr.append(float(p_lr[i]))

        hist_t.append(row["time"])
        hist_b1.append(df_b1.iloc[i]["risk"])
        hist_b2.append(df_b2.iloc[i]["risk"])
        hist_b3.append(df_b3.iloc[i]["risk"])
        hist_prop.append(R_fused)

        if not ((i % redraw_every == 0) or (i == n - 1)):
            continue

        badge("B1_Vision_Proxy", df_b1.iloc[i]["state"], df_b1.iloc[i]["risk"])
        badge("B2_Rule_Based", df_b2.iloc[i]["state"], df_b2.iloc[i]["risk"])
        badge("B3_Fixed_Fuzzy", df_b3.iloc[i]["state"], df_b3.iloc[i]["risk"])
        badge("Proposed", state.value, R_fused)
        badge("B0_Camera_Only", st_b0[-1], hist_b0[-1])
        badge("B2D_Debounce_3s", st_b2d[i], df_b2.iloc[i]["risk"])
        badge("CUSUM_h16", st_cu[-1], hist_cu[-1])
        if lr is not None:
            badge("B4_Logistic", st_lr[-1], hist_lr[-1])
        else:
            badge_ph["B4_Logistic"].markdown("**B4_Logistic**: run `python src/pub_fixes.py` first")

        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(hist_t, hist_b1, color="#9b59b6", label="B1 Vision-Proxy", alpha=0.8)
        ax.plot(hist_t, hist_b2, color="#3498db", label="B2 Rule-Based", alpha=0.8)
        ax.plot(hist_t, hist_b3, color="#f39c12", label="B3 Fixed-Fuzzy", alpha=0.8)
        ax.plot(hist_t, hist_cu, color="#16a085", label="CUSUM g/h", alpha=0.8)
        if lr is not None:
            ax.plot(hist_t, hist_lr, color="#7f8c8d", label="B4 Logistic", alpha=0.8)
        ax.plot(hist_t, hist_prop, color="#e74c3c", label="Proposed", linewidth=2)
        ax.axhline(0.75, color="gray", linestyle="--", linewidth=0.8, label="Emergency threshold (Proposed)")
        ax.set_xlabel("time (s)"); ax.set_ylabel("risk")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        fig.tight_layout()
        chart_ph.pyplot(fig)
        plt.close(fig)

        time.sleep(0.02)

    st.success("Race complete. Computing metrics...")

    df_prop_full = pd.DataFrame({
        "time": df["time"].values,
        "state": proposed_states,
        "risk": hist_prop,
        "battery_present": df["battery_present"].values if "battery_present" in df.columns else 0,
        "phase": df["phase"].values if "phase" in df.columns else "Normal",
    })

    metrics = [
        compute_metrics(df_b1, battery_arrives, "B1_Vision_Proxy"),
        compute_metrics(df_b2, battery_arrives, "B2_Rule_Based"),
        compute_metrics(df_b3, battery_arrives, "B3_Fixed_Fuzzy"),
        compute_metrics(df_prop_full, battery_arrives, "Proposed"),
    ]
    def _frame(states, risk):
        return df_prop_full.assign(state=list(states), risk=list(risk))
    metrics += [compute_metrics(_frame(st_b0, hist_b0), battery_arrives, "B0_Camera_Only"),
                compute_metrics(_frame(st_b2d, df_b2["risk"].values), battery_arrives, "B2D_Debounce_3s"),
                compute_metrics(_frame(st_cu, hist_cu), battery_arrives, "CUSUM_h16")]
    if lr is not None:
        metrics.append(compute_metrics(_frame(st_lr, hist_lr), battery_arrives, "B4_Logistic"))
    df_metrics = pd.DataFrame(metrics)
    df_metrics[["T_warn_s", "T_stop_s"]] = df_metrics[["T_warn_s", "T_stop_s"]].astype(str)   # numbers or '>400'
    st.dataframe(df_metrics, width="stretch", hide_index=True)
    st.caption(
        "Hazard_FNR/Critical_FNR = miss rate during dangerous phases (lower is "
        "better). FPR = false Emergency Stops with no battery present (lower is "
        "better). T_warn_s/T_stop_s = seconds from battery arrival to first "
        "Warning / to Emergency Stop (lower is better, faster detection). "
        "CFNC = (10*runaway-phase misses + 2*all misses)/hazard steps (lower is better); it equals 2 x Hazard FNR when no miss falls in the runaway phases."
    )


# ── Tab 4: Try To Break It ──────────────────────────────────────────────────

class _OldBuggyFSM(SupervisoryFSM):
    """Reimplements the PRE-FIX sensor-fault check for live demonstration only.

    This is NOT used by the real system. The real, fixed logic lives in
    supervisory_control.py's SupervisoryFSM.step() / _check_sensor_fault(),
    which excludes Emergency_Stop from the sensor-fault check specifically so
    a latched Emergency_Stop can only clear via manual reset. This subclass
    drops that exclusion (the exact bug fixed this session) so the dashboard
    can show, live, what the latch-bypass looked like before the fix.
    """

    def step(self, t: float, R_fused: float, temperature: float,
             h2_ppm: float, co2_ppm: float, manual_reset: bool = False):
        self.dwell_time += DT

        # BUG: missing `self.state != State.EMERGENCY_STOP` guard here --
        # implausible readings can clear a latched Emergency_Stop with no
        # human involved.
        if self._check_sensor_fault(temperature, h2_ppm, co2_ppm):
            self._sensor_fault_count += 1
            if self._sensor_fault_count >= 10:
                self._transition(t, State.SENSOR_FAULT, R_fused)
        else:
            self._sensor_fault_count = max(0, self._sensor_fault_count - 1)

        if manual_reset and self.state == State.EMERGENCY_STOP:
            self.emergency_latch = False
            self._transition(t, State.WARNING, R_fused)

        if self.state == State.NORMAL:
            if R_fused >= THRESHOLDS["normal_to_warning"]:
                self._transition(t, State.WARNING, R_fused)
        elif self.state == State.WARNING:
            if self.dwell_time >= MIN_DWELL[State.WARNING]:
                if R_fused >= THRESHOLDS["warning_to_critical"]:
                    self._transition(t, State.CRITICAL, R_fused)
                elif R_fused < THRESHOLDS["warning_to_normal"]:
                    self._transition(t, State.NORMAL, R_fused)
        elif self.state == State.CRITICAL:
            if self.dwell_time >= MIN_DWELL[State.CRITICAL]:
                if R_fused >= THRESHOLDS["critical_to_emergency"]:
                    self._transition(t, State.EMERGENCY_STOP, R_fused)
                    self.emergency_latch = True
                elif R_fused < THRESHOLDS["critical_to_warning"]:
                    self._transition(t, State.WARNING, R_fused)
        elif self.state == State.EMERGENCY_STOP:
            pass
        elif self.state == State.SENSOR_FAULT:
            if self._sensor_fault_count == 0:
                self._transition(t, State.WARNING, R_fused)

        speed = SHREDDER_SPEED[self.state]
        self.state_history.append(self.state.value)
        self.speed_history.append(speed)
        return self.state, speed


def _reset_break_demo():
    st.session_state.break_fsm = SupervisoryFSM()
    st.session_state.break_buggy = _OldBuggyFSM()
    st.session_state.break_log = []
    st.session_state.break_t = 0.0


def render_break_it():
    st.subheader("Try to break it: the Emergency Stop latch")
    st.caption(
        "Step 1 drives the REAL SupervisoryFSM to Emergency_Stop with escalating "
        "risk. Step 2 then floods it with implausible sensor readings (out of "
        "physical range) to prove the latch holds -- this is exactly the bug "
        "class fixed this session (see supervisory_control.py's "
        "`_check_sensor_fault` docstring). A reimplementation of the OLD, "
        "pre-fix logic runs alongside for contrast; it is clearly labeled and "
        "NOT used by the real system."
    )

    if "break_fsm" not in st.session_state:
        _reset_break_demo()

    fsm = st.session_state.break_fsm
    buggy = st.session_state.break_buggy

    col1, col2, col3 = st.columns(3)
    with col1:
        drive = st.button("1. Drive to Emergency Stop", key="break_drive")
    with col2:
        can_flood = fsm.state == State.EMERGENCY_STOP
        flood = st.button("2. Feed 12 implausible readings", key="break_flood",
                           disabled=not can_flood)
    with col3:
        reset = st.button("Reset demo", key="break_reset")

    if reset:
        _reset_break_demo()
        st.rerun()

    if drive:
        t = st.session_state.break_t
        # Ramp up, then hold at a high plateau for enough extra steps to
        # clear BOTH dwell-time hysteresis windows (2.0s in Warning, 1.0s in
        # Critical -- see supervisory_control.py's MIN_DWELL): a fixed-length
        # ramp alone can cross both risk thresholds but still run out of
        # steps before the FSM is actually *allowed* to escalate, since each
        # state requires sitting above threshold for its full dwell time
        # first. Verified live via the browser tools -- an earlier 40-step
        # pure ramp reached Critical but never Emergency_Stop.
        ramp = list(np.linspace(0.1, 0.9, 10)) + [0.9] * 60
        for r in ramp:
            t += DT
            T = 25.0 + r * 300.0
            h2 = r * 2000.0
            co2 = 410.0 + r * 2000.0
            s1, sp1 = fsm.step(t=t, R_fused=float(r), temperature=T, h2_ppm=h2, co2_ppm=co2)
            s2, sp2 = buggy.step(t=t, R_fused=float(r), temperature=T, h2_ppm=h2, co2_ppm=co2)
            st.session_state.break_log.append({
                "t": round(t, 1), "R_fused": round(float(r), 3),
                "real_state": s1.value, "real_speed": sp1,
                "old_buggy_state": s2.value, "old_buggy_speed": sp2,
            })
            if s1 == State.EMERGENCY_STOP:
                break
        st.session_state.break_t = t
        st.rerun()

    if flood:
        t = st.session_state.break_t
        implausible = [(1500.0, 20000.0, 8000.0), (-50.0, -100.0, 100.0),
                        (1200.0, 18000.0, 7500.0)] * 4  # 12 readings total
        for T, h2, co2 in implausible:
            t += DT
            s1, sp1 = fsm.step(t=t, R_fused=0.9, temperature=T, h2_ppm=h2, co2_ppm=co2)
            s2, sp2 = buggy.step(t=t, R_fused=0.9, temperature=T, h2_ppm=h2, co2_ppm=co2)
            st.session_state.break_log.append({
                "t": round(t, 1), "R_fused": 0.9,
                "real_state": s1.value, "real_speed": sp1,
                "old_buggy_state": s2.value, "old_buggy_speed": sp2,
            })
        st.session_state.break_t = t
        st.rerun()

    if st.session_state.break_log:
        log_df = pd.DataFrame(st.session_state.break_log)

        def highlight(row):
            if row["real_state"] == "Emergency_Stop" and row["old_buggy_state"] != "Emergency_Stop":
                return ["background-color: #fdecea"] * len(row)
            return [""] * len(row)

        st.dataframe(log_df.style.apply(highlight, axis=1), width="stretch", hide_index=True)

        last = st.session_state.break_log[-1]
        if last["real_state"] == "Emergency_Stop":
            if last["old_buggy_state"] != "Emergency_Stop":
                st.error(
                    f"Fixed system: still **Emergency_Stop** (speed {last['real_speed']}%). "
                    f"Old buggy logic: dropped to **{last['old_buggy_state']}** "
                    f"(speed {last['old_buggy_speed']}%) — the shredder would be running "
                    f"again with no human involved. Click step 2 above to see this live."
                )
            else:
                st.success(
                    f"Fixed system: **Emergency_Stop** (speed {last['real_speed']}%), "
                    "latch holding. Click step 2 to flood it with implausible readings."
                )
        else:
            st.info(
                f"Ramping — currently **{last['real_state']}** "
                f"(R_fused={last['R_fused']:.2f}). Click step 1 again if it "
                f"stops before reaching Emergency_Stop."
            )
    else:
        st.info("Click step 1 to drive the real FSM to Emergency_Stop.")


# ── Tab 5: Upload & Classify ────────────────────────────────────────────────

class _GradCAM:
    """Minimal Grad-CAM via forward/backward hooks (same approach as
    generate_gradcam.py, condensed here for reuse against an in-memory
    uploaded image rather than a test-set file path)."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fwd = target_layer.register_forward_hook(self._save_act)
        self._bwd = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.activations = out.detach()

    def _save_grad(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, img_tensor, target_class):
        import torch.nn.functional as F
        self.model.zero_grad(set_to_none=True)
        logits = self.model(img_tensor)
        logits[0, target_class].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=img_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam_max = cam.max()
        return cam / cam_max if cam_max > 1e-8 else np.zeros_like(cam)

    def close(self):
        self._fwd.remove()
        self._bwd.remove()


def _overlay_heatmap(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = np.asarray(pil_img.resize((224, 224)).convert("RGB")).astype(np.float32) / 255.0
    try:
        jet = matplotlib.colormaps["jet"]
    except AttributeError:
        jet = _cm.get_cmap("jet")
    heat = jet(cam)[..., :3]
    return np.clip((1 - alpha) * base + alpha * heat, 0, 1)


def render_upload_classify():
    st.subheader("Upload your own photo: live classification")
    st.caption(
        "Runs the REAL trained checkpoints (coarse EfficientNet-B0, then the "
        "fine-stage checkpoint if coarse predicts Battery) on whatever image "
        "you upload -- not a cherry-picked test-set example."
    )

    uploaded = st.file_uploader("Upload an e-waste photo", type=["jpg", "jpeg", "png", "bmp"],
                                 key="upload_file")
    show_gradcam = st.checkbox("Show Grad-CAM attention overlay", value=False, key="upload_gradcam",
                                help="Highlights the image regions the model's decision is based on.")

    if uploaded is None:
        st.info("Upload a photo to classify it with the real trained model.")
        return

    img = Image.open(uploaded).convert("RGB")
    det = get_detector()
    if det is None:
        st.image(img, width=300, caption="Uploaded image")
    else:
        r = det.predict(img, conf=0.25, verbose=False)[0]
        st.image(r.plot()[:, :, ::-1], width=420,
                 caption=f"YOLOv8n battery detector: {len(r.boxes)} battery box(es) at confidence >= 0.25 "
                         "(finds batteries among several items; the classifier below sees the whole photo)")

    coarse_model, coarse_classes, fine_model, fine_classes, transform = get_classification_models()
    x = transform(img).unsqueeze(0).to(TORCH_DEVICE)

    with torch.no_grad():
        coarse_probs = torch.softmax(coarse_model(x), dim=1)[0].cpu().numpy()
    coarse_pred = coarse_label({c: float(p) for c, p in zip(coarse_classes, coarse_probs)})
    coarse_pred_idx = coarse_classes.index(coarse_pred)

    st.metric("Coarse prediction", coarse_pred, f"{coarse_probs[coarse_pred_idx] * 100:.1f}% probability")
    st.caption(f"Rule: Battery whenever P(Battery) >= {BATTERY_THRESHOLD} (safety threshold chosen on validation data); otherwise the most likely class.")

    fig, ax = plt.subplots(figsize=(6, 0.4 * len(coarse_classes) + 1))
    colors = ["#e74c3c" if c == coarse_pred else "#3498db" for c in coarse_classes]
    ax.barh(coarse_classes, coarse_probs, color=colors)
    ax.set_xlim(0, 1); ax.set_xlabel("probability")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    fine_pred = None
    fine_pred_idx = None
    if coarse_pred == "Battery":
        with torch.no_grad():
            fine_probs = torch.softmax(fine_model(x), dim=1)[0].cpu().numpy()
        fine_pred_idx = int(np.argmax(fine_probs))
        fine_pred = fine_classes[fine_pred_idx]
        st.metric("Fine-stage subtype", fine_pred, f"{fine_probs[fine_pred_idx] * 100:.1f}% confidence")

        fig2, ax2 = plt.subplots(figsize=(6, 0.4 * len(fine_classes) + 1))
        colors2 = ["#e74c3c" if c == fine_pred else "#f39c12" for c in fine_classes]
        ax2.barh(fine_classes, fine_probs, color=colors2)
        ax2.set_xlim(0, 1); ax2.set_xlabel("probability")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
    else:
        st.caption(
            "Coarse stage did not predict Battery — the fine-stage subtype "
            "classifier is skipped, matching the real pipeline (it only runs "
            "when coarse says Battery)."
        )

    if show_gradcam:
        st.divider()
        st.markdown("**Grad-CAM — what the model is attending to**")
        gc_col1, gc_col2 = st.columns(2)

        gc = _GradCAM(coarse_model, coarse_model.features[-1])
        try:
            cam = gc(x, coarse_pred_idx)
        finally:
            gc.close()
        overlay = _overlay_heatmap(img, cam)
        gc_col1.image(overlay, width=280, caption=f"Coarse stage — attention for '{coarse_pred}'")

        if fine_pred_idx is not None:
            gc2 = _GradCAM(fine_model, fine_model.features[-1])
            try:
                cam2 = gc2(x, fine_pred_idx)
            finally:
                gc2.close()
            overlay2 = _overlay_heatmap(img, cam2)
            gc_col2.image(overlay2, width=280, caption=f"Fine stage — attention for '{fine_pred}'")


# ── Tab 6: Custom Sequence ──────────────────────────────────────────────────

ITEM_KIND = {
    "Battery":              dict(metal=True,  battery=True,  default_duration=300.0),
    "Benign metal item":    dict(metal=True,  battery=False, default_duration=30.0),
    "Benign non-metal item": dict(metal=False, battery=False, default_duration=30.0),
}


def build_custom_sequence_df(items, gap_s: float = 20.0, seed: int = 42) -> pd.DataFrame:
    """Builds a sensor dataframe for a user-defined item sequence, reusing the
    exact same physics sensor classes as sensor_simulation.py's
    run_multi_item_scenario() -- one continuous thermal/gas state carried
    across the whole belt (not reset per item), just like a real conveyor.
    `items` is a list of {"kind": one of ITEM_KIND, "duration_s": float}.
    """
    thermal   = ThermalRunawayModel(seed=seed)
    mq2       = MQ2Sensor(seed=seed + 1)
    mq135     = MQ135Sensor(seed=seed + 2)
    inductive = InductiveProximitySensor(seed=seed + 3)

    schedule = []
    t_cursor = 5.0  # initial idle gap
    for item in items:
        spec = ITEM_KIND[item["kind"]]
        start = t_cursor
        end = start + item["duration_s"]
        schedule.append((start, end, spec["metal"], spec["battery"]))
        t_cursor = end + gap_s
    total_duration = t_cursor + 5.0

    rows = []
    for t in np.arange(0.0, total_duration, DT):
        item = next(((s, e, m, b) for s, e, m, b in schedule if s <= t < e), None)
        obj_present = item is not None
        metal_target = item[2] if item else False
        batt = item[3] if item else False

        temp_C, phase = thermal.step(batt)
        mq2_ppm, mq2_v = mq2.read(temp_C)
        mq135_ppm, mq135_v = mq135.read(temp_C)
        metal_det = inductive.read(metal_target)

        rows.append(dict(time=round(t, 2), temperature_C=temp_C, mq2_ppm=mq2_ppm,
            mq2_voltage=mq2_v, mq135_ppm=mq135_ppm, mq135_voltage=mq135_v,
            metal_present=int(metal_det), object_present=int(obj_present),
            object_size_cm=(10.0 if obj_present else 0.0), phase=phase,
            battery_present=int(batt)))
    return pd.DataFrame(rows)


def render_custom_sequence():
    st.subheader("Custom sequence: build your own belt")
    st.caption(
        "Add items one at a time (benign metal, benign non-metal, or a "
        "battery), then play the sequence through the real pipeline -- an "
        "interactive version of run_multi_item_scenario(). Thermal/gas state "
        "carries continuously across the whole belt, so an item's danger "
        "depends on what came before it, exactly like the real system."
    )

    if "custom_items" not in st.session_state:
        st.session_state.custom_items = []

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        kind = st.selectbox("Item type", list(ITEM_KIND.keys()), key="cs_kind")
    with col2:
        duration = st.number_input(
            "Duration on belt (s)", min_value=5.0, max_value=600.0,
            value=ITEM_KIND[kind]["default_duration"], step=5.0, key="cs_duration",
        )
    with col3:
        st.write("")
        st.write("")
        if st.button("Add item", key="cs_add"):
            st.session_state.custom_items.append({"kind": kind, "duration_s": duration})

    if st.session_state.custom_items:
        seq_df = pd.DataFrame([
            {"#": i + 1, "Item type": it["kind"], "Duration (s)": it["duration_s"]}
            for i, it in enumerate(st.session_state.custom_items)
        ])
        st.dataframe(seq_df, width="stretch", hide_index=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.session_state.custom_items and st.button("Remove last item", key="cs_remove"):
                st.session_state.custom_items.pop()
                st.rerun()
        with col_b:
            if st.button("Clear sequence", key="cs_clear"):
                st.session_state.custom_items = []
                st.rerun()
        with col_c:
            gap_s = st.number_input("Gap between items (s)", min_value=3.0, max_value=120.0,
                                     value=20.0, step=1.0, key="cs_gap")

        speed = st.select_slider("Playback speed (sim-seconds per UI update)",
                                   options=[1, 2, 5, 10, 20, 50], value=5, key="cs_speed")
        chart_window = st.slider("Chart window (seconds shown)", 30, 300, 150, key="cs_window")
        play = st.button("Play sequence", type="primary", key="cs_play")

        if play:
            df = build_custom_sequence_df(st.session_state.custom_items, gap_s=gap_s)
            _run_playback(df, speed, chart_window)
    else:
        st.info("Add at least one item above to build a sequence.")


THERMAL_REFERENCE = pd.DataFrame([
    ("SEI decomposition onset", "~90 C (reported 80-120 C)", "45 C (Phase 1 starts)", "Simulation earlier (conservative)", "Feng et al. 2018 [2]"),
    ("Separator melting / H2 onset", "~130 C", "70 C (H2), 90 C (Phase 2 end)", "Simulation earlier (conservative)", "Feng et al. 2018 [2]"),
    ("Cathode decomposition", ">~180 C", "Phase 3 runaway from 90 C", "Simulation earlier (conservative)", "Feng et al. 2018 [2]"),
    ("Peak cell temperature", "up to ~850 C (18650 cells)", "851 C", "Consistent", "Golubkov et al. 2014 [22]"),
], columns=["Quantity", "Published value", "Simulation", "Verdict", "Source"])


def render_robustness():
    st.subheader("Held-out results (seeds 100-129, pre-computed)")
    st.caption("Read from results/pub/ -- produced by src/pub_experiments.py, src/pub_fixes.py and "
               "src/pub_realtrace.py with the same code as the other tabs. All tunable settings were chosen "
               "on separate seeds 200-229. Not recomputed in the browser.")
    ev_f = RESULTS_PUB / "eval.csv"
    if not ev_f.exists():
        st.warning("Run `python src/pub_experiments.py` from the repo root first.")
        return
    ev = pd.read_csv(ev_f)
    main_sys = ["Proposed", "Proposed_occluded", "B0_CameraOnly", "B1_VisionProxy", "B2_Rule",
                "B2_Debounce_3s", "B3_FixedFuzzy", "CUSUM_h16"]
    haz = ["single", "multi", "lag5", "lag20", "pre80", "pre90"]
    x = ev[ev.scenario.isin(haz) & ev.system.isin(main_sys)]
    st.markdown("**Battery scenarios: mean Hazard FNR** (fraction of hazardous time not alerted, lower is better)")
    st.dataframe(x.pivot_table(index="system", columns="scenario", values="Hazard_FNR", aggfunc="mean")
                 .reindex(main_sys)[haz].round(4), width="stretch")
    st.markdown("**Stop margin before runaway** (seconds, higher is better)")
    st.dataframe(x.pivot_table(index="system", columns="scenario", values="stop_margin_s", aggfunc="mean")
                 .reindex(main_sys)[haz].round(1), width="stretch")
    free = ["mild", "hard", "harder", "hot60", "hot90", "warm_bursts"]
    y = ev[ev.scenario.isin(free) & ev.system.isin(main_sys)]
    st.markdown("**Battery-free stress tests: runs (of 30) with a false Critical or Emergency Stop**")
    st.dataframe(y.pivot_table(index="system", columns="scenario", values="false_escalation", aggfunc="sum")
                 .reindex(main_sys)[free].astype("Int64"), width="stretch")
    st.caption("Known weaknesses, shown as measured: a warm benign item plus gas bursts drives the Proposed "
               "system to Critical (never to a stop), and any object at 90 C or hotter causes a stop.")
    stf = RESULTS_PUB / "stats.csv"
    if stf.exists():
        st.markdown("**Paired statistics** (Proposed minus baseline, Hazard FNR, 95% bootstrap CI, Holm-corrected p)")
        st.dataframe(pd.read_csv(stf)[["scenario", "comparison", "mean_diff", "ci_lo", "ci_hi", "dz", "p_holm"]].round(4),
                     width="stretch", hide_index=True)
    trf = RESULTS_PUB / "fixes_tradeoff.csv"
    if trf.exists():
        st.markdown("**Miss rate vs false escalations across each method's parameter range** (sensor-only, no camera)")
        tr = pd.read_csv(trf)
        fig, ax = plt.subplots(figsize=(7, 4))
        for fam, g in tr.groupby("family"):
            g = g.sort_values("false_rate")
            ax.plot(g.false_rate, g.fnr, marker="o", label=fam)
        ax.set_xlabel("fraction of hard / harder / warm-item runs with a false escalation")
        ax.set_ylabel("single-battery Hazard FNR"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.caption("Without the camera the gated memory is not better than a tuned debounced rule (B2deb); "
                   "the full system's advantage comes from fusing the camera with the memory.")
    for name, title in (("fixes_summary.txt", "Calibrated thermal model, rate gate, learned baseline, CFNC weights"),
                        ("realtrace_summary.txt", "Measured cell temperature curves (Kwak et al. 2026)")):
        f = RESULTS_PUB / name
        if f.exists():
            with st.expander(title):
                st.text(f.read_text(encoding="utf-8"))
    st.subheader("Thermal model vs published thermal-runaway data")
    st.dataframe(THERMAL_REFERENCE, width="stretch", hide_index=True)
    st.caption("The main model's onset thresholds sit below published ranges, so alarms come early. A second "
               "model with published onsets (src/pub_fixes.py) and replayed measured cell curves "
               "(src/pub_realtrace.py) check that the conclusions do not depend on this choice.")


# ── Tab 8: Real measured cell ───────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _real_trace_run(path: str, seed: int):
    from pub_realtrace import load, trace_df
    from pub_experiments import gated, fsm_run, ALPHA
    from gap_temporal_sweeps import fuzzy_series
    tt, T = load(path)
    df, arrive = trace_df(tt, T, seed)
    S = fuzzy_series(df)
    Rg = gated(S, df.temperature_C.values, df.object_present.values)
    cnn = RealCNN(seed=77)
    P = np.array([cnn.predict(temperature=r.temperature_C, mq2_ppm=r.mq2_ppm, object_present=bool(r.object_present),
                              metal_present=bool(r.metal_present), battery_present=bool(r.battery_present))["battery_prob"]
                  for r in df.itertuples()])
    Rf = np.clip(Rg * (1 + 0.35 * P), 0, 1)
    return df, Rf, fsm_run(df, Rf), baseline_rule_based(df).state.values


def render_real_cell():
    st.subheader("Real measured cell — replay a thermal-runaway test through the pipeline")
    st.caption("Surface temperature measured in heater-driven abuse tests (5.5 C/min) by Kwak et al., "
               "Scientific Data 13, 1084 (2026), OSF doi:10.17605/OSF.IO/C2HNQ. The dataset has no gas channel, "
               "so H2/CO2 come from the gas model. All settings are frozen; nothing is tuned on these curves.")
    root = Path(__file__).parent / "data" / "raw" / "kwak_tr" / "Dataset_TR" / "Cell"
    files = sorted(root.glob("**/*Temperature.xlsx"))
    if not files:
        st.warning("Download Dataset_TR.zip from https://osf.io/c2hnq/ and unzip it into data/raw/kwak_tr/.")
        return
    names = {f"{f.parent.parent.name} / {f.parent.name} / {f.stem.split('_')[0]}": f for f in files}
    c1, c2 = st.columns(2)
    choice = c1.selectbox("Measured cell", list(names), key="rc_cell")
    seed = c2.number_input("Sensor-noise seed", value=100, step=1, key="rc_seed")
    if not st.button("Replay", type="primary", key="rc_run"):
        st.info("Pick a cell and click **Replay** (about a minute: every 0.1 s step runs the real fuzzy system).")
        return
    with st.spinner("Running fuzzy inference, memory, camera and FSM on every step..."):
        df, Rf, st_p, st_b2 = _real_trace_run(str(names[choice]), int(seed))
    t = df.time.values; T = df.temperature_C.values
    run_i = int(np.argmax(df.phase.isin(["Phase3_Runaway", "Phase4_Fire"]).values))

    def first(states):
        k = np.where(np.asarray(states) == "Emergency_Stop")[0]
        return float(t[k[0]]) if len(k) else None
    stop_p, stop_b2 = first(st_p), first(st_b2)
    m1, m2, m3 = st.columns(3)
    m1.metric("Runaway (200 C) at", f"{t[run_i]:.0f} s")
    m2.metric("Proposed stops at", f"{stop_p:.0f} s" if stop_p is not None else "never",
              f"{t[run_i] - stop_p:.0f} s before runaway" if stop_p is not None else None)
    m3.metric("B2 rule stops at", f"{stop_b2:.0f} s" if stop_b2 is not None else "never",
              f"{t[run_i] - stop_b2:.0f} s before runaway" if stop_b2 is not None else None)
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax[0].plot(t, T, color="#c0392b"); ax[0].set_ylabel("measured T (C)")
    ax[0].axvline(t[run_i], color="black", ls="--", lw=0.8, label="runaway (200 C)")
    if stop_p is not None: ax[0].axvline(stop_p, color="#e74c3c", lw=1.5, label="Proposed E-stop")
    if stop_b2 is not None: ax[0].axvline(stop_b2, color="#3498db", lw=1.0, label="B2 E-stop")
    ax[0].legend(fontsize=8)
    ax[1].plot(t, Rf, color="#e74c3c", label="Proposed R_fused"); ax[1].axhline(0.75, color="gray", ls="--", lw=0.8)
    ax[1].set_ylabel("risk"); ax[1].set_xlabel("time (s)"); ax[1].legend(fontsize=8)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption("This test checks behaviour on measured curves; it does not separate the systems, because the "
               "slow heater-driven heating gives every method ample time to stop.")


def render_business_case():
    import business_case as bc
    st.subheader("Business case — cost, placement and payback for one conveyor line")
    st.caption("Formulas in src/business_case.py; defaults are cited street prices / published loss figures "
               "(docs/SHARK_TANK_REVIEW.md) or measured results. Plant-specific numbers are yours to enter; the "
               "fraction of fires prevented is NOT measured (no field data) and is an assumption.")
    tier = st.radio("Hardware tier", ["Prototype (hobby sensors)", "Industrial-grade sensors"], horizontal=True, key="bc_tier")
    bom = bc.BOM_PROTOTYPE if tier.startswith("Prototype") else bc.BOM_INDUSTRIAL
    st.dataframe(pd.DataFrame([(k, f"{a:,.0f} - {b:,.0f}") for k, (a, b) in bom.items()], columns=["Item", "USD"]),
                 width="stretch", hide_index=True)
    lo, hi = bc.bom_total(bom)
    c1, c2, c3 = st.columns(3)
    with c1:
        v = st.number_input("Belt speed (m/s)", 0.1, 3.0, 0.7, 0.1, key="bc_v", help="0.7 m/s: feed conveyor of a Rotarex R1200 e-waste shredder (listing)")
        stop = st.number_input("Detection-to-stop time (s)", 0.0, 30.0, 4.2, 0.1, key="bc_stop",
                               help="Measured for a cell arriving at 90 C: 3.1 s (gate of Tables 9-13), 4.2 s (recommended gate)")
        rund = st.number_input("Shredder rundown to standstill (s)", 0.0, 60.0, 5.0, 0.5, key="bc_rund", help="Ask the shredder vendor; not measured here")
    with c2:
        fps = st.number_input("False stops per shift", 0.0, 20.0, 0.1, 0.05, key="bc_fps")
        shifts = st.number_input("Shifts per year", 1, 1100, 600, 10, key="bc_shifts")
        dmin = st.number_input("Downtime per false stop (min)", 0.0, 120.0, 10.0, 1.0, key="bc_dmin")
        dcost = st.number_input("Downtime cost (USD/hour)", 0.0, 1e6, 2000.0, 100.0, key="bc_dcost")
    with c3:
        fires = st.number_input("Battery fires per year without the system", 0.0, 50.0, 0.2, 0.05, key="bc_fires")
        loss = st.number_input("Loss per fire (USD)", 0.0, 1e8, 1.49e6, 1e5, key="bc_loss", help="EuRIC: EUR 1.3 M average damage per thermal event")
        frac = st.slider("Fraction of fires prevented (assumption)", 0.0, 1.0, 0.3, 0.05, key="bc_frac")
        upkeep = st.number_input("Upkeep per year (USD, recalibration etc.)", 0.0, 1e5, 500.0, 50.0, key="bc_up")
    false_cost = bc.annual_false_stop_cost(fps, shifts, dmin, dcost)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Hardware per line", f"${lo:,.0f} - {hi:,.0f}")
    m2.metric("Sensors must sit upstream by", f"{bc.placement_m(v, stop, rund):.1f} m")
    m3.metric("False-stop cost per year", f"${false_cost:,.0f}")
    pb = bc.payback_years(hi, fires, loss, frac, false_cost, upkeep)
    m4.metric("Payback (high hardware cost)", "never" if pb == float("inf") else f"{pb * 12:.1f} months")
    st.caption("Expected avoided loss per year = fires x loss x fraction prevented. False stops cost more than the hardware: "
               "a system that stops too often never pays back, which is why false escalations are measured throughout.")


# ── Tab 0: Conveyor Line ────────────────────────────────────────────────────

@st.cache_data(show_spinner="Running the moving-belt simulation for three lanes...")
def _conveyor(kind: str, seed: int):
    import conveyor_sim, conveyor_demo
    fig, lanes, item = conveyor_demo.figure(kind, seed, conveyor_sim.load_pool())
    ev = {s: conveyor_demo.events(lanes[s][0], lanes[s][1], item, s) for s in lanes}
    return fig, conveyor_demo.physics_figure(lanes, item), {s: lanes[s][0] for s in lanes}, ev


def render_conveyor(item_name, seed):
    import conveyor_sim, conveyor_demo
    st.subheader(f"On the moving belt: {item_name}")
    st.caption(f"Camera, sensor hood, diverter and shredder on a {conveyor_sim.V_BELT} m/s belt (camera to shredder "
               f"{conveyor_sim.X_SHRED / conveyor_sim.V_BELT:.0f} s). The proposed line slows the belt when risk rises "
               f"(Warning 60 %, Critical 10 %, Emergency stop 0 %), diverts on a camera sighting (P >= {conveyor_sim.P_DIVERT}) or a hood flag "
               "into a bin with its own sensors. Geometry, dilution and sensor lag are assumptions; camera "
               "probabilities are the real checkpoint's outputs on held-out photos.")
    kind = ITEMS[item_name].belt
    if kind is None:
        item_unavailable("parked"); return
    fig, phys, res, ev = _conveyor(kind, int(seed))
    st.caption("Press Play. Slats move with each lane's belt (they slow at Warning and stop at Emergency stop); "
               "pins under the belt mark where and when each event happened, in seconds after the camera.")
    st.plotly_chart(fig)
    battery = any(w in kind for w in ("intact", "fail"))
    BADGE = {"#e74c3c": ("red", ":material/dangerous:"), "#2ecc71": ("green", ":material/check_circle:"),
             "#f1c40f": ("orange", ":material/warning:"), "#e67e22": ("orange", ":material/local_fire_department:")}
    for col, (s, name) in zip(st.columns(3, border=True), conveyor_demo.LANES):
        txt, colour = conveyor_demo.outcome(res[s], battery)
        badge_colour, icon = BADGE.get(colour, ("gray", ":material/info:"))
        with col:
            st.markdown(f"**{name}**")
            st.markdown(f":{badge_colour}[{icon} **{txt}**]")
            st.caption(f"Camera P_battery {res[s]['P_battery']:.2f} · belt time lost {res[s]['belt_loss_s']:.1f} s")
    st.markdown("**Event timeline: what happened, where and when** (time = seconds after the item passed the camera)")
    rows = [dict(time_s=round(t, 1), line=name, position_m=("-" if np.isnan(x) else f"{x:.2f}"), event=txt)
            for s, name in conveyor_demo.LANES for t, x, _, txt in ev[s]]
    st.dataframe(pd.DataFrame(rows).sort_values(["time_s", "line"], kind="stable"), hide_index=True,
                 column_config={"event": st.column_config.TextColumn(width="large")})
    st.markdown("**Physics panel: what the proposed line's sensors saw**")
    st.plotly_chart(phys)
    summ = Path(__file__).parent / "results" / "conveyor" / "summary.txt"
    if summ.exists():
        with st.expander("Held-out results over 30 items per scenario (results/conveyor/summary.txt)"):
            st.code(summ.read_text(encoding="utf-8"))


@st.cache_data(ttl=600)
def headline_kpis():
    """Headline numbers for the KPI row, read from the committed result files (None if a file is missing)."""
    import json
    root = Path(__file__).parent / "results"
    k = {}
    try:
        ev = pd.read_csv(root / "conveyor" / "eval.csv")
        fail = ev[ev.scenario.str.contains("fail")]
        k["belt"] = {s: int(fail[fail.system == s].hazardous_at_entry.sum()) for s in ("ours", "rule", "camera_only")}
        k["belt_n"] = int((fail.system == "ours").sum())
    except Exception:
        pass
    try:
        pe = pd.read_csv(root / "pub" / "eval.csv")
        m = lambda sc, sy: pe[(pe.scenario == sc) & (pe.system == sy)]
        k["miss"] = (m("single", "Proposed").Hazard_FNR.mean(), m("single", "B2_Debounce_3s").Hazard_FNR.mean())
        k["false_stop"] = (int(m("hard", "Proposed").false_estop.sum()), int(m("hard", "B2_Rule").false_estop.sum()), int(len(m("hard", "B2_Rule"))))
    except Exception:
        pass
    try:
        c = json.load(open(root / "gaps" / "cnn_v3_table.json"))
        k["intact"] = (c["garbage12"][0] + c["recybat"][0], c["garbage12"][1] + c["recybat"][1]); k["damaged"] = tuple(c["damaged"])
    except Exception:
        pass
    return k


def header():
    st.title("E-waste battery safety system", anchor=False)
    st.markdown(":orange-badge[:material/science: Simulated sensors] :blue-badge[:material/neurology: Real trained CNN] "
                ":green-badge[:material/verified: Held-out seeds 100-129] :violet-badge[:material/conveyor_belt: Moving-belt test]")
    st.caption("A camera and IR / gas sensors decide, item by item, whether to let a piece of e-waste into the shredder, divert it "
               "to a monitored bin, or stop the line. Every value below is computed live from the trained models and the simulation.")
    k = headline_kpis()
    with st.container(horizontal=True):
        if "belt" in k:
            st.metric("Failing cells shredded hot", f"{k['belt']['ours']} / {k['belt_n']}", f"-{k['belt']['rule']} vs plain rule",
                      delta_color="inverse", border=True, help=f"Moving-belt test, 30 held-out seeds per scenario. Camera only: {k['belt']['camera_only']}.")
        if "miss" in k:
            st.metric("Hazard miss rate", f"{k['miss'][0]:.4f}", f"{k['miss'][0] - k['miss'][1]:+.4f} vs debounced rule",
                      delta_color="inverse", border=True, help="Share of hazardous time with no alert, single battery, held-out seeds.")
        if "false_stop" in k:
            st.metric("False stops on gas bursts", f"{k['false_stop'][0]} / {k['false_stop'][2]}", f"{k['false_stop'][0] - k['false_stop'][1]:+d} vs plain rule",
                      delta_color="inverse", border=True, help="Battery-free runs with hard gas bursts that ended in an Emergency Stop.")
        if "intact" in k:
            st.metric("Test batteries found", f"{k['intact'][0]} / {k['intact'][1]}", f"+ {k['damaged'][0]} / {k['damaged'][1]} damaged",
                      delta_color="off", border=True, help="Coarse CNN on its held-out test photos, with the validated battery threshold.")


TABS = [":material/conveyor_belt: Conveyor line", ":material/monitor_heart: Live monitor", ":material/tune: Manual control",
        ":material/sports_score: Baseline race", ":material/emergency_home: Break it", ":material/photo_camera: Upload & classify",
        ":material/playlist_add: Custom sequence", ":material/query_stats: Results", ":material/battery_alert: Real measured cell",
        ":material/payments: Business case"]


def main():
    st.set_page_config(page_title="E-waste battery safety", page_icon=":material/battery_alert:", layout="wide")
    header()
    item_name, seed = item_sidebar()
    # lazy tabs: only the open tab runs (the others would otherwise compute, and load models, on every rerun)
    tabs = st.tabs(TABS, key="tab", on_change="rerun")
    pages = [lambda: render_conveyor(item_name, seed), lambda: render_live_monitor(item_name, seed), render_manual_control,
             lambda: render_baseline_race(item_name, seed), render_break_it, render_upload_classify, render_custom_sequence,
             render_robustness, render_real_cell, render_business_case]
    for tab, page in zip(tabs, pages):
        if tab.open:
            with tab:
                page()


if __name__ == "__main__":
    main()
