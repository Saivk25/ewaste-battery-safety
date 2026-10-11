"""
architecture_diagram.py — Full system architecture figure (publication-ready)
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

OUT = "outputs/figures"
os.makedirs(OUT, exist_ok=True)

def draw_box(ax, x, y, w, h, label, sublabel="", color="#2980b9",
             fontsize=10, alpha=0.92, style="round,pad=0.1", text_color="white"):
    box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                          boxstyle=style, facecolor=color, edgecolor='white',
                          linewidth=1.8, alpha=alpha, zorder=3)
    ax.add_patch(box)
    ax.text(x, y + (0.012 if sublabel else 0), label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color, zorder=4)
    if sublabel:
        ax.text(x, y-0.025, sublabel, ha='center', va='center',
                fontsize=8, color=text_color, alpha=0.85, zorder=4)

def arrow(ax, x1, y1, x2, y2, color='#2c3e50', lw=2.0, label="", style="->"):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle="arc3,rad=0.0"))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.01, my+0.015, label, fontsize=8, color=color,
                ha='center', va='bottom', fontstyle='italic')

fig, ax = plt.subplots(1, 1, figsize=(18, 11))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis('off')
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('#f8f9fa')

ax.text(0.5, 0.97, "E-Waste Safety System — Complete Pipeline Architecture",
        ha='center', va='top', fontsize=16, fontweight='bold', color='#2c3e50')
ax.text(0.5, 0.935, "Hierarchical CNN + Adaptive Fuzzy Temporal Supervisory Risk Model for Pre-Shred Ignition Control",
        ha='center', va='top', fontsize=10, color='#7f8c8d', style='italic')

# ── Layer 1: Physical Input Layer ─────────────────────────────────────────────
ax.text(0.5, 0.90, "① SENSING LAYER", ha='center', fontsize=9, color='#7f8c8d',
        fontweight='bold', style='italic')
draw_box(ax, 0.10, 0.82, 0.14, 0.06, "HC-SR04", "Ultrasonic\nObject Trigger", color="#1a5276")
draw_box(ax, 0.27, 0.82, 0.14, 0.06, "Camera", "1080p / 30fps\nImage Capture", color="#1a5276")
draw_box(ax, 0.50, 0.82, 0.14, 0.06, "MLX90614", "IR Thermometer\n±0.5°C, 10Hz", color="#6c3483")
draw_box(ax, 0.67, 0.82, 0.14, 0.06, "MQ-2 + MQ-135", "H₂/CO + CO₂\nGas Sensors", color="#6c3483")
draw_box(ax, 0.84, 0.82, 0.12, 0.06, "LJ18A3", "Inductive\nMetal Detect", color="#6c3483")

# ── Layer 2: Processing ───────────────────────────────────────────────────────
ax.text(0.5, 0.74, "② PROCESSING LAYER", ha='center', fontsize=9, color='#7f8c8d',
        fontweight='bold', style='italic')

# CNN Branch
draw_box(ax, 0.185, 0.66, 0.25, 0.065, "Hierarchical CNN", "EfficientNet-B0 Backbone", color="#117a65")
draw_box(ax, 0.10, 0.55, 0.13, 0.065, "Coarse Classifier", "6-class e-waste\nAcc: 91.2%", color="#1e8449", fontsize=9)
draw_box(ax, 0.27, 0.55, 0.13, 0.065, "Fine Classifier", "Battery sub-type\nAcc: 87.1%", color="#1e8449", fontsize=9)
draw_box(ax, 0.185, 0.445, 0.20, 0.055, "P_battery", "CNN Battery Prob\n[0, 1]", color="#27ae60", fontsize=9)

# Sensor Fusion Branch
draw_box(ax, 0.68, 0.66, 0.28, 0.065, "Sensor Pre-processing", "Normalisation + Health Monitor", color="#7d6608")
draw_box(ax, 0.68, 0.555, 0.28, 0.065, "Mamdani FIS", "20 Rules | Centroid Defuzz", color="#9a7d0a", fontsize=9)
draw_box(ax, 0.68, 0.445, 0.20, 0.055, "S(t) Fuzzy Risk", "[0, 1]", color="#d4ac0d", text_color="#2c3e50", fontsize=9)

# ── Layer 3: Fusion + Temporal ────────────────────────────────────────────────
ax.text(0.5, 0.39, "③ FUSION & TEMPORAL LAYER", ha='center', fontsize=9, color='#7f8c8d',
        fontweight='bold', style='italic')
draw_box(ax, 0.50, 0.325, 0.44, 0.07, "Adaptive Temporal Risk Update",
         "R(t) = β(t)·R(t−1) + (1−β(t))·S(t)   |   β(t) = β_max·exp(−λ·|ΔS(t)|)",
         color="#c0392b", fontsize=10)
draw_box(ax, 0.50, 0.235, 0.36, 0.06, "CNN-Confidence Fusion",
         "R_fused(t) = R(t) · (1 + α · P_battery)   [clipped to 1]",
         color="#e74c3c", fontsize=9)

# ── Layer 4: Supervisory Control ──────────────────────────────────────────────
ax.text(0.5, 0.175, "④ SUPERVISORY CONTROL LAYER", ha='center', fontsize=9,
        color='#7f8c8d', fontweight='bold', style='italic')

# FSM States
state_x = [0.18, 0.36, 0.54, 0.72, 0.90]
state_c  = ["#2ecc71","#f39c12","#e74c3c","#8e44ad","#95a5a6"]
state_n  = ["NORMAL\n100% speed","WARNING\n60% speed","CRITICAL\n10% speed",
             "EMERGENCY\nSTOP  0%","SENSOR\nFAULT  50%"]
for x, col, n in zip(state_x, state_c, state_n):
    draw_box(ax, x, 0.115, 0.14, 0.065, n, color=col, fontsize=8.5)

# FSM arrows
for i in range(3):
    arrow(ax, state_x[i]+0.07, 0.115, state_x[i+1]-0.07, 0.115, color='black', lw=1.5)
for i in range(3, 0, -1):
    pass  # hysteresis arrows implied

# Latching note
ax.text(0.72, 0.065, "⊗ LATCHING\n(manual reset)", ha='center', fontsize=8,
        color='#8e44ad', fontweight='bold')

# ── Layer 5: Output ───────────────────────────────────────────────────────────
draw_box(ax, 0.50, 0.025, 0.55, 0.04, "SHREDDER CONTROL OUTPUT — Hardware Interrupt Signal (≤100ms)",
         color="#2c3e50", fontsize=9)

# ── Arrows between layers ─────────────────────────────────────────────────────
# Sensors → CNN
arrow(ax, 0.27, 0.79, 0.20, 0.695, color='#1a5276')
# Sensors → FIS
arrow(ax, 0.50, 0.79, 0.60, 0.695, color='#6c3483')
arrow(ax, 0.67, 0.79, 0.65, 0.695, color='#6c3483')
arrow(ax, 0.84, 0.79, 0.76, 0.695, color='#6c3483')
# CNN → coarse/fine
arrow(ax, 0.14, 0.628, 0.115, 0.583, color='#1e8449')
arrow(ax, 0.24, 0.628, 0.265, 0.583, color='#1e8449')
# coarse+fine → P_battery
arrow(ax, 0.115, 0.518, 0.175, 0.473, color='#27ae60')
arrow(ax, 0.265, 0.518, 0.205, 0.473, color='#27ae60')
# FIS → S(t)
arrow(ax, 0.68, 0.522, 0.68, 0.473, color='#d4ac0d', lw=2)
# P_battery + S(t) → Temporal
arrow(ax, 0.185, 0.418, 0.35, 0.362, color='#c0392b', lw=2, label='P_battery')
arrow(ax, 0.68, 0.418, 0.62, 0.362, color='#c0392b', lw=2, label='S(t)')
# Temporal → Fusion
arrow(ax, 0.50, 0.29, 0.50, 0.268, color='#e74c3c', lw=2.5, label='R(t)')
# Fusion → FSM
arrow(ax, 0.50, 0.205, 0.50, 0.150, color='#8e44ad', lw=2.5, label='R_fused')
# FSM → Output
arrow(ax, 0.50, 0.082, 0.50, 0.048, color='#2c3e50', lw=2.5)

# ── Legend boxes ──────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color='#1a5276', label='Physical Sensors'),
    mpatches.Patch(color='#117a65', label='Vision / CNN'),
    mpatches.Patch(color='#9a7d0a', label='Fuzzy Inference'),
    mpatches.Patch(color='#c0392b', label='Temporal Fusion'),
    mpatches.Patch(color='#8e44ad', label='Emergency State'),
]
ax.legend(handles=legend_items, loc='lower left', fontsize=9,
           bbox_to_anchor=(0.0, 0.0), framealpha=0.9)

plt.tight_layout()
plt.savefig(f"{OUT}/fig0_system_architecture.png", bbox_inches='tight', dpi=180)
plt.close()
print("✓ Architecture diagram saved.")

if __name__ == "__main__":
    pass
