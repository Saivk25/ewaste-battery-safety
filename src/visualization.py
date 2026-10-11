"""
visualization.py — All publication-ready figures for the paper and slides
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import json, os
OUT = "outputs/figures"
os.makedirs(OUT, exist_ok=True)

COLORS = {
    "Normal":        "#2ecc71",
    "Warning":       "#f39c12",
    "Critical":      "#e74c3c",
    "Emergency_Stop":"#8e44ad",
    "Phase1_SEI":    "#f1c40f",
    "Phase2_Sep":    "#e67e22",
    "Phase3_Run":    "#e74c3c",
    "Phase4_Fire":   "#8e44ad",
    "proposed":      "#2980b9",
    "b1":            "#e74c3c",
    "b2":            "#27ae60",
    "b3":            "#f39c12",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── FIG 1: Fuzzy Membership Functions ────────────────────────────────────────
def plot_fuzzy_memberships():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Mamdani FIS — Membership Functions\n(Calibrated to Lithium-Ion Thermal Runaway Characteristics)",
                 fontsize=14, fontweight='bold', y=1.01)

    def trimf(x, a, b, c):
        y = np.zeros_like(x)
        m1 = (x > a) & (x <= b); y[m1] = (x[m1]-a)/(b-a+1e-9)
        m2 = (x > b) & (x < c);  y[m2] = (c-x[m2])/(c-b+1e-9)
        return y
    def trapmf(x, a, b, c, d):
        y = np.zeros_like(x)
        m1 = (x > a) & (x < b);  y[m1] = (x[m1]-a)/(b-a+1e-9)
        m2 = (x >= b) & (x <= c); y[m2] = 1.0
        m3 = (x > c) & (x < d);  y[m3] = (d-x[m3])/(d-c+1e-9)
        return y

    # Temperature
    ax = axes[0,0]; T = np.linspace(0, 500, 2000)
    ax.plot(T, trapmf(T,0,0,30,55),    color='#2ecc71', lw=2.5, label='Low (0–55°C)')
    ax.plot(T, trimf(T,40,70,100),     color='#f39c12', lw=2.5, label='Medium (40–100°C)')
    ax.plot(T, trapmf(T,80,110,500,500),color='#e74c3c', lw=2.5, label='High (>80°C)')
    ax.axvline(70, color='gray', ls='--', lw=1.2, alpha=0.6, label='H₂ onset (70°C)')
    ax.axvline(90, color='gray', ls=':',  lw=1.2, alpha=0.6, label='Runaway onset (90°C)')
    ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Membership μ")
    ax.set_title("Input: Temperature — MLX90614"); ax.legend(fontsize=9); ax.set_ylim(-0.05,1.15)

    # MQ-2 H2
    ax = axes[0,1]; H = np.linspace(0, 3000, 2000)
    ax.plot(H, trapmf(H,0,0,50,150),       color='#2ecc71', lw=2.5, label='Low (0–150 ppm)')
    ax.plot(H, trimf(H,80,400,1200),       color='#f39c12', lw=2.5, label='Medium (80–1200 ppm)')
    ax.plot(H, trapmf(H,800,1500,3000,3000),color='#e74c3c', lw=2.5, label='High (>800 ppm)')
    ax.set_xlabel("H₂ Concentration (ppm)"); ax.set_ylabel("Membership μ")
    ax.set_title("Input: H₂/CO — MQ-2 Sensor"); ax.legend(fontsize=9); ax.set_ylim(-0.05,1.15)

    # MQ-135 CO2
    ax = axes[1,0]; C = np.linspace(300, 6000, 2000)
    ax.plot(C, trapmf(C,0,0,450,650),       color='#2ecc71', lw=2.5, label='Low (ambient ~400 ppm)')
    ax.plot(C, trimf(C,500,1200,2500),      color='#f39c12', lw=2.5, label='Medium (500–2500 ppm)')
    ax.plot(C, trapmf(C,2000,3000,6000,6000),color='#e74c3c', lw=2.5, label='High (>2000 ppm)')
    ax.set_xlabel("CO₂ Concentration (ppm)"); ax.set_ylabel("Membership μ")
    ax.set_title("Input: CO₂ — MQ-135 Sensor"); ax.legend(fontsize=9); ax.set_ylim(-0.05,1.15)

    # Risk output
    ax = axes[1,1]; R = np.linspace(0, 1, 2000)
    ax.plot(R, trapmf(R,0,0,0.25,0.40),    color='#2ecc71', lw=2.5, label='Low Risk')
    ax.plot(R, trimf(R,0.30,0.50,0.70),    color='#f39c12', lw=2.5, label='Medium Risk')
    ax.plot(R, trimf(R,0.60,0.75,0.90),    color='#e74c3c', lw=2.5, label='High Risk')
    ax.plot(R, trapmf(R,0.82,0.92,1.0,1.0),color='#8e44ad', lw=2.5, label='Critical Risk')
    for thresh, lbl in [(0.30,'Warning'),(0.55,'Critical'),(0.75,'Emergency')]:
        ax.axvline(thresh, color='black', ls='--', lw=1.0, alpha=0.5)
        ax.text(thresh+0.01, 0.92, lbl, fontsize=8, color='black', alpha=0.7)
    ax.set_xlabel("Risk Level R [0,1]"); ax.set_ylabel("Membership μ")
    ax.set_title("Output: Risk — FSM Thresholds Shown"); ax.legend(fontsize=9); ax.set_ylim(-0.05,1.15)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig1_fuzzy_memberships.png", bbox_inches='tight')
    plt.close()
    print("✓ Fig 1: Fuzzy membership functions")


# ── FIG 2: Sensor Time Series (Temperature & Gas) ────────────────────────────
def plot_sensor_timeseries(df):
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(4, 1, hspace=0.45)

    phases_colors = {
        "Normal": "#ecf0f1", "Phase1_SEI": "#fef9e7",
        "Phase2_Separator": "#fef5e4", "Phase3_Runaway": "#fdedec",
        "Phase4_Fire": "#f9ebea",
    }
    t = df["time"].values

    def shade_phases(ax):
        prev_p, prev_t = df["phase"].iloc[0], t[0]
        for i, (ph, ti) in enumerate(zip(df["phase"], t)):
            if ph != prev_p or i == len(t)-1:
                col = phases_colors.get(prev_p, "#ffffff")
                ax.axvspan(prev_t, ti, color=col, alpha=0.6, zorder=0)
                prev_p, prev_t = ph, ti

    # Temperature
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, df["temperature_C"], color='#c0392b', lw=2, label='MLX90614 Temperature')
    shade_phases(ax1)
    ax1.axhline(70,  color='#f39c12', ls='--', lw=1.2, label='H₂ onset (70°C)', alpha=0.8)
    ax1.axhline(90,  color='#e74c3c', ls='--', lw=1.2, label='Runaway onset (90°C)', alpha=0.8)
    ax1.set_ylabel("Temperature (°C)"); ax1.set_title("Sensor Time Series — Thermal Runaway Scenario")
    ax1.legend(loc='upper left', fontsize=9); ax1.set_xlim(t[0], t[-1])

    # MQ-2 H2
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(t, df["mq2_ppm"], color='#8e44ad', lw=2, label='MQ-2: H₂/CO (ppm)')
    ax2.fill_between(t, df["mq2_ppm"], alpha=0.2, color='#8e44ad')
    shade_phases(ax2)
    ax2.set_ylabel("H₂/CO (ppm)"); ax2.legend(loc='upper left', fontsize=9)
    ax2.set_xlim(t[0], t[-1])

    # MQ-135 CO2
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(t, df["mq135_ppm"], color='#16a085', lw=2, label='MQ-135: CO₂ (ppm)')
    ax3.fill_between(t, df["mq135_ppm"], 400, where=df["mq135_ppm"]>400, alpha=0.2, color='#16a085')
    shade_phases(ax3)
    ax3.axhline(400, color='gray', ls=':', lw=1.0, alpha=0.5, label='Atmospheric baseline')
    ax3.set_ylabel("CO₂ (ppm)"); ax3.legend(loc='upper left', fontsize=9)

    # Metal detection
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.fill_between(t, df["metal_present"], step='post', alpha=0.7,
                      color='#2980b9', label='LJ18A3 Metal Detection')
    shade_phases(ax4)
    ax4.set_ylabel("Metal\nDetected"); ax4.set_xlabel("Time (seconds)")
    ax4.set_ylim(-0.1, 1.4); ax4.legend(loc='upper left', fontsize=9)
    ax4.set_yticks([0,1]); ax4.set_yticklabels(['No','Yes'])

    # Phase legend
    patches = [mpatches.Patch(color=v, label=k.replace("_"," "), alpha=0.7)
               for k,v in phases_colors.items()]
    fig.legend(handles=patches, loc='lower center', ncol=5, fontsize=9,
               title="Thermal Runaway Phase", bbox_to_anchor=(0.5, -0.01))

    plt.savefig(f"{OUT}/fig2_sensor_timeseries.png", bbox_inches='tight')
    plt.close()
    print("✓ Fig 2: Sensor time series")


# ── FIG 3: Risk vs Time + FSM State ──────────────────────────────────────────
def plot_risk_and_fsm(df):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                                          gridspec_kw={'hspace':0.35, 'height_ratios':[2,2,1]})
    t = df["time"].values

    # Risk curves
    ax1.plot(t, df["fuzzy_risk"],  color='#27ae60', lw=2.0, label='Fuzzy Risk S(t)', alpha=0.8)
    ax1.plot(t, df["R_temporal"],  color='#2980b9', lw=2.0, label='Temporal R(t)', alpha=0.9)
    ax1.plot(t, df["R_fused"],     color='#c0392b', lw=2.5, label='Fused R_fused(t) [CNN+Fuzzy]')
    for y, lbl, col in [(0.30,'Warning','#f39c12'),(0.55,'Critical','#e74c3c'),(0.75,'Emergency Stop','#8e44ad')]:
        ax1.axhline(y, ls='--', color=col, lw=1.3, alpha=0.9)
        ax1.text(5, y+0.015, lbl, color=col, fontsize=9, fontweight='bold')
    ax1.set_ylabel("Risk Level [0, 1]")
    ax1.set_title("Adaptive Temporal Risk: R(t) = β(t)·R(t−1) + (1−β(t))·S(t)")
    ax1.legend(loc='upper left', fontsize=9); ax1.set_ylim(-0.05, 1.10)

    # Beta
    ax2.step(t, df["beta"], where='post', color='#7f8c8d', lw=2.0, label='β(t) used at each step')
    ax2.set_ylabel("β(t)"); ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("β(t): corroboration-gated adaptive memory (fast below the gate or when temperature "
                  "corroborates; slow only for short gas-only bursts)", fontsize=10)
    ax2.legend(loc='upper right', fontsize=9)

    # FSM State
    state_map = {"Normal": 0, "Warning": 1, "Critical": 2, "Emergency_Stop": 3, "Sensor_Fault": 4}
    state_colors_list = ['#2ecc71','#f39c12','#e74c3c','#8e44ad','#95a5a6']
    state_nums = df["fsm_state"].map(state_map).fillna(0).values
    ax3.step(t, state_nums, where='post', color='#2c3e50', lw=1.5)
    ax3.fill_between(t, state_nums, step='post', alpha=0.4,
                      color=[state_colors_list[int(s)] for s in state_nums])
    ax3.set_yticks([0,1,2,3]); ax3.set_yticklabels(['Normal','Warning','Critical','Emg.Stop'])
    ax3.set_xlabel("Time (seconds)"); ax3.set_ylabel("FSM State")
    ax3.set_title("Supervisory Control State Machine")

    plt.savefig(f"{OUT}/fig3_risk_vs_time.png", bbox_inches='tight')
    plt.close()
    print("✓ Fig 3: Risk vs time + FSM state")


# ── FIG 4: Emergency Stop Trigger Detail ─────────────────────────────────────
def plot_emergency_stop_detail(df, df_trans):
    # Focus on ±60s around emergency stop
    if "Emergency_Stop" not in df["fsm_state"].values:
        print("  (No emergency stop in data — skipping Fig 4)")
        return

    t_emg = df[df["fsm_state"]=="Emergency_Stop"]["time"].iloc[0]
    mask  = (df["time"] >= t_emg - 30) & (df["time"] <= t_emg + 60)
    d     = df[mask]
    t     = d["time"].values

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True,
                              gridspec_kw={'hspace': 0.40, 'height_ratios':[2,1.5,1.5,1]})

    # Risk
    ax = axes[0]
    ax.plot(t, d["R_fused"], color='#c0392b', lw=2.5, label='R_fused(t)')
    ax.axhline(0.75, ls='--', color='#8e44ad', lw=1.5, label='Emergency Threshold (0.75)')
    ax.axvline(t_emg, color='#8e44ad', lw=2.5, ls='-', alpha=0.8, label=f'Emergency Stop t={t_emg:.1f}s')
    ax.fill_between(t, d["R_fused"], 0.75, where=d["R_fused"]>=0.75, alpha=0.25, color='#8e44ad')
    ax.set_ylabel("Fused Risk"); ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9); ax.set_title(f"Emergency Stop Trigger — Detail View (±30/60s around t={t_emg:.0f}s)")

    # Temperature
    ax = axes[1]
    ax.plot(t, d["temperature_C"], color='#c0392b', lw=2)
    ax.axvline(t_emg, color='#8e44ad', lw=2, ls='--', alpha=0.7)
    ax.axhline(90, color='#e74c3c', ls=':', lw=1.2, alpha=0.7, label='Runaway onset 90°C')
    ax.set_ylabel("Temperature (°C)"); ax.legend(fontsize=9)

    # Gas
    ax = axes[2]
    ax2b = ax.twinx()
    ax.plot(t, d["mq2_ppm"],    color='#8e44ad', lw=2, label='H₂/CO (MQ-2)')
    ax2b.plot(t, d["mq135_ppm"], color='#16a085', lw=2, ls='--', label='CO₂ (MQ-135)')
    ax.axvline(t_emg, color='#8e44ad', lw=2, ls='--', alpha=0.7)
    ax.set_ylabel("H₂/CO (ppm)"); ax2b.set_ylabel("CO₂ (ppm)")
    ax.legend(loc='upper left', fontsize=9); ax2b.legend(loc='upper right', fontsize=9)

    # Shredder speed
    ax = axes[3]
    ax.fill_between(t, d["shredder_speed"], step='post', alpha=0.7, color='#2980b9')
    ax.step(t, d["shredder_speed"], where='post', color='#1a5276', lw=2)
    ax.axvline(t_emg, color='#8e44ad', lw=2, ls='--', alpha=0.7)
    ax.set_ylabel("Shredder\nSpeed (%)"); ax.set_xlabel("Time (seconds)")
    ax.set_ylim(-5, 115)

    # Annotate transitions
    for _, tr in df_trans.iterrows():
        if mask.any() and d["time"].min() <= tr["time"] <= d["time"].max():
            for a in axes:
                a.axvline(tr["time"], color='black', lw=1.0, ls=':', alpha=0.5)

    plt.savefig(f"{OUT}/fig4_emergency_stop_detail.png", bbox_inches='tight')
    plt.close()
    print("✓ Fig 4: Emergency stop detail")


# ── FIG 5: CNN Training Curves + Backbone Comparison ─────────────────────────
def plot_cnn_results():
    with open("outputs/training_curves.json") as f:
        curves = json.load(f)
    df_comp = pd.read_csv("outputs/cnn_comparison_table.csv")

    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.35)
    epochs = list(range(1, 26))
    bb_colors = {"MobileNetV2":"#e74c3c", "EfficientNet-B0":"#2980b9", "EfficientNet-B3":"#27ae60"}

    # Coarse validation curves
    ax = fig.add_subplot(gs[0,:2])
    for bb, col in bb_colors.items():
        ax.plot(epochs, curves[bb]["val_coarse"], color=col, lw=2.5, label=f'{bb}')
        ax.plot(epochs, curves[bb]["train_coarse"], color=col, lw=1.5, ls='--', alpha=0.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("Coarse Classification — Validation Accuracy (solid) / Train (dashed)")
    ax.legend(); ax.set_ylim(0.5, 1.02); ax.axhline(0.9, color='gray', ls=':', alpha=0.5)

    # Fine validation curves
    ax = fig.add_subplot(gs[1,:2])
    for bb, col in bb_colors.items():
        ax.plot(epochs, curves[bb]["val_fine"], color=col, lw=2.5, label=f'{bb}')
        ax.plot(epochs, curves[bb]["train_fine"], color=col, lw=1.5, ls='--', alpha=0.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("Fine-Grained Battery Classification — Validation Accuracy")
    ax.legend(); ax.set_ylim(0.5, 1.02)

    # Radar: Accuracy vs Latency vs Params
    ax = fig.add_subplot(gs[0,2])
    x = df_comp["Backbone"]; idx = np.arange(len(x))
    w = 0.25
    bars_acc = ax.bar(idx - w, df_comp["Fine_Acc"], w, color=[bb_colors[b] for b in x], label='Fine Acc')
    bars_f1  = ax.bar(idx,     df_comp["Fine_F1"],  w, color=[bb_colors[b] for b in x], alpha=0.7, label='Fine F1')
    ax.set_xticks(idx); ax.set_xticklabels([b.replace("EfficientNet-","EB") for b in x], fontsize=9)
    ax.set_ylabel("Score"); ax.set_title("Accuracy & F1 — Fine Classification")
    ax.set_ylim(0.7, 1.0); ax.legend(fontsize=9)

    # Latency vs FNR
    ax = fig.add_subplot(gs[1,2])
    for i, row in df_comp.iterrows():
        bb = row["Backbone"]
        ax.scatter(row["Infer_ms"], row["Battery_FNR_Fine"],
                    s=row["Params(M)"]*30, color=bb_colors[bb],
                    zorder=5, edgecolors='black', linewidth=1.2)
        ax.annotate(bb.replace("EfficientNet-","EB"),
                    (row["Infer_ms"], row["Battery_FNR_Fine"]),
                    textcoords="offset points", xytext=(8,4), fontsize=9)
    ax.set_xlabel("Inference Latency (ms)"); ax.set_ylabel("Battery FNR (lower=safer)")
    ax.set_title("Latency vs Battery Miss Rate\n(bubble size = #params)")
    ax.annotate("← Safer", xy=(0.65,0.08), xycoords='axes fraction', fontsize=9, color='green')
    ax.annotate("Faster →", xy=(0.65,0.02), xycoords='axes fraction', fontsize=9, color='blue')

    plt.savefig(f"{OUT}/fig5_cnn_results.png", bbox_inches='tight')
    plt.close()
    print("✓ Fig 5: CNN training curves + comparison")


# ── FIG 6: Confusion Matrices ─────────────────────────────────────────────────
def plot_confusion_matrices():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    fig.suptitle("EfficientNet-B0 — Hierarchical CNN Confusion Matrices", fontsize=14, fontweight='bold')

    cmap = plt.cm.Blues
    for ax, tag, labels, title, battery_idx in [
        (axes[0], "EfficientNetB0_coarse",
         ["Battery","Cardboard_Trash","Glass","Metal","Paper","Plastic"],
         "Stage 1: Coarse Classifier (6-class)", 0),
        (axes[1], "EfficientNetB0_fine",
         ["LiIon\n18650","LiIon\nPouch","LiPo"],
         "Stage 2: Fine Battery Subtype Classifier (3-class)", None),
    ]:
        cm = np.load(f"outputs/cm_{tag}.npy")
        n_total = int(cm.sum())
        acc = float(np.trace(cm)) / n_total
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9); ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = cm_norm[i, j]
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        color='white' if val > 0.5 else 'black', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, label='Normalised')

        # Macro F1 (unweighted mean of per-class F1)
        f1s = []
        for i in range(len(labels)):
            tp = cm[i, i]; fn = cm[i, :].sum() - tp; fp = cm[:, i].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        macro_f1 = float(np.mean(f1s))

        if battery_idx is not None:
            tp = cm[battery_idx, battery_idx]
            fn = cm[battery_idx, :].sum() - tp
            fp = cm[:, battery_idx].sum() - tp
            fnr = fn / (fn + tp) if (fn + tp) else 0.0
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            stat_text = (f"Accuracy: {acc*100:.2f}%   Battery FNR: {fnr:.4f}   "
                         f"Battery Precision: {prec*100:.2f}%   Macro F1: {macro_f1*100:.2f}%   "
                         f"Test images: {n_total}")
        else:
            fnr_macro = 1.0 - np.mean([cm[i, i] / cm[i, :].sum() for i in range(len(labels))])
            stat_text = (f"Accuracy: {acc*100:.2f}%   Macro FNR: {fnr_macro:.4f}   "
                         f"Macro F1: {macro_f1*100:.2f}%   Test images: {n_total}")
        ax.text(0.5, -0.30, stat_text, transform=ax.transAxes, ha='center', va='top',
                fontsize=8.5, family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='#999999'))

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig6_confusion_matrices.png", bbox_inches='tight')
    plt.close()
    print("✓ Fig 6: Confusion matrices")


# ── FIG 7: Baseline Comparison ───────────────────────────────────────────────
def plot_baseline_comparison():
    df = pd.read_csv("outputs/baseline_comparison.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    fig.suptitle("Baseline System Comparison", fontsize=14, fontweight='bold')

    colors = ['#e74c3c','#27ae60','#f39c12','#2980b9']
    labels = [s.replace("_"," ") for s in df["System"]]
    x = np.arange(len(labels))

    # FNR
    ax = axes[0]
    bars = ax.bar(x, df["Hazard_FNR"], color=colors, edgecolor='black', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel("Hazard False Negative Rate"); ax.set_title("Miss Rate During Dangerous Phases\n(Lower = Safer ↓)")
    for bar, val in zip(bars, df["Hazard_FNR"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Time to Warning
    t_warn = pd.to_numeric(df["T_warn_s"], errors='coerce').fillna(400)
    ax = axes[1]
    bars = ax.bar(x, t_warn, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel("Time to First Warning (s from battery arrival)")
    ax.set_title("Response Speed\n(Lower = Faster ↓)")
    for bar, val in zip(bars, t_warn):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                f'{val:.0f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # CFNC
    ax = axes[2]
    bars = ax.bar(x, df["CFNC"], color=colors, edgecolor='black', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel("CFNC Score"); ax.set_title("Catastrophic False Negative Cost\n(Lower = Better ↓)")
    for bar, val in zip(bars, df["CFNC"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig7_baseline_comparison.png", bbox_inches='tight')
    plt.close()
    print("✓ Fig 7: Baseline comparison")


# ── FIG 8: Normal vs Hazardous Scenario Overlay ──────────────────────────────
def plot_scenario_overlay(df_hazard, df_normal):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False, gridspec_kw={'hspace':0.4})

    ax = axes[0]
    ax.plot(df_hazard["time"], df_hazard["temperature_C"],
            color='#e74c3c', lw=2.5, label='Hazardous (Li-ion battery)')
    ax.plot(df_normal["time"], df_normal["temperature_C"],
            color='#2ecc71', lw=2, ls='--', label='Normal (PCB/plastic e-waste)')
    ax.axhline(70, color='#f39c12', ls=':', lw=1.5, label='H₂ onset (70°C)')
    ax.axhline(90, color='#e74c3c', ls=':', lw=1.5, label='Runaway onset (90°C)')
    ax.set_ylabel("Temperature (°C)"); ax.set_xlabel("Time (s)")
    ax.set_title("Temperature Profile Comparison: Normal vs Hazardous Item")
    ax.legend(fontsize=10); ax.set_xlim(0, min(df_hazard["time"].max(), 400))

    ax = axes[1]
    ax.plot(df_hazard["time"], df_hazard["mq2_ppm"],
            color='#8e44ad', lw=2.5, label='Hazardous — H₂/CO (MQ-2)')
    ax.plot(df_normal["time"], df_normal["mq2_ppm"],
            color='#95a5a6', lw=2, ls='--', label='Normal — H₂/CO (MQ-2)')
    ax.set_ylabel("H₂/CO (ppm)"); ax.set_xlabel("Time (s)")
    ax.set_title("Gas Sensor Comparison: H₂ Off-Gassing Signature (90s Early Warning Window)")
    ax.legend(fontsize=10)

    plt.savefig(f"{OUT}/fig8_scenario_overlay.png", bbox_inches='tight')
    plt.close()
    print("✓ Fig 8: Normal vs hazardous overlay")


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nGenerating all publication figures...")
    df_full  = pd.read_csv("outputs/full_pipeline_output.csv")
    df_trans = pd.read_csv("outputs/state_transitions.csv")
    df_norm  = pd.read_csv("outputs/sensor_data_normal.csv")

    plot_fuzzy_memberships()
    plot_sensor_timeseries(df_full)
    plot_risk_and_fsm(df_full)
    plot_emergency_stop_detail(df_full, df_trans)
    plot_cnn_results()
    plot_confusion_matrices()
    plot_baseline_comparison()
    plot_scenario_overlay(df_full, df_norm)

    print(f"\n✓ All 8 figures saved to {OUT}/")
