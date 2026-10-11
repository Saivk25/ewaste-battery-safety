"""
adaptive_risk.py — Adaptive Temporal Risk Update + Confidence-Weighted Fusion

R(t) = beta(t)*R(t-1) + (1-beta(t))*S(t)

beta(t) adapts to the evidence behind a rise in the fuzzy risk S(t)
(corroboration-gated adaptive memory, the default since the Oct 2026 gap analysis):
    S(t) <= C_GATE                                   -> beta = BETA_FAST  (early warning, cheap)
    S(t) >  C_GATE and temperature >= T_CORR         -> beta = BETA_FAST  (hot cell: follow at once)
    S(t) >  C_GATE, gas-only, lasted >= M_PERSIST     -> beta = BETA_FAST  (sustained: follow)
    S(t) >  C_GATE, gas-only, shorter than M_PERSIST -> beta = BETA_SLOW  (likely a sensor burst)
Parameters were tuned on seeds 200-229 and evaluated on held-out seeds 100-129
(src/gap_corroborated_memory.py, results/gaps/corroborated_memory_summary.*).

The original rule, beta(t) = beta_max * exp(-lambda * |dS(t)|), is kept as mode="exp"
for comparison; it is no longer the default.

Conservative fusion with CNN confidence:
R_fused(t) = R(t) * (1 + alpha * P_battery)

Object-presence reset: if no object for >3s, the temporal state is reset (fresh item).
"""
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import pandas as pd

# corroboration-gated adaptive memory (default)
BETA_FAST  = 0.0    # memory when the rise is cheap to act on or corroborated
BETA_SLOW  = 0.99   # memory for an uncorroborated (gas-only) excursion into the Critical zone
C_GATE     = 0.5    # fuzzy-risk level above which corroboration or persistence is required
M_PERSIST  = 20     # steps (2 s at DT = 0.1 s) a gas-only excursion must last
T_CORR     = 40.0   # temperature (C) that corroborates a rise
# original exponential rule (mode="exp")
BETA_MAX   = 0.85   # max temporal smoothing
LAMBDA_B   = 0.15   # sensitivity of beta to sensor change rate
ALPHA_CNN  = 0.35   # CNN confidence amplification weight
RESET_TIME = 3.0    # seconds without object → reset temporal risk
DT         = 0.1


class AdaptiveRiskModel:
    def __init__(self, mode: str = "corroborated"):
        assert mode in ("corroborated", "exp")
        self.mode          = mode
        self.R_prev        = 0.0
        self.S_prev        = 0.0
        self.run           = 0      # consecutive gas-only steps above C_GATE
        self.no_object_t   = 0.0
        self.beta_history  = []
        self.risk_history  = []
        self.fused_history = []

    def gated_beta(self, S_t: float, temperature) -> float:
        """Corroboration-gated memory (default). temperature=None counts as uncorroborated."""
        if S_t <= C_GATE:
            self.run = 0
            return BETA_FAST
        self.run += 1
        corroborated = temperature is not None and temperature >= T_CORR
        return BETA_FAST if (corroborated or self.run >= M_PERSIST) else BETA_SLOW

    def compute_beta(self, S_current: float, S_previous: float) -> float:
        """
        Original rule (mode="exp"): beta(t) = beta_max * exp(-lambda * |dS/dt|)
        When sensors spike (large delta), beta drops → trust new reading more.
        When sensors stable, beta stays high → smooth historical average.
        """
        delta_S = abs(S_current - S_previous)
        return BETA_MAX * np.exp(-LAMBDA_B * delta_S)

    def update(self, S_t: float, P_battery: float,
               object_present: bool, temperature=None) -> Tuple[float, float, float]:
        """
        Parameters
        ----------
        S_t            : current fuzzy risk score [0,1]
        P_battery      : CNN battery detection probability [0,1]
        object_present : from ultrasonic/proximity sensor
        temperature    : IR temperature (C) used to corroborate a rise (corroborated mode)

        Returns
        -------
        (R_temporal, beta, R_fused)
        """
        # Object-presence reset
        if not object_present:
            self.no_object_t += DT
            if self.no_object_t >= RESET_TIME:
                self.R_prev = 0.0   # reset for next fresh item
                self.run = 0
        else:
            self.no_object_t = 0.0

        # Compute adaptive beta
        beta = self.gated_beta(S_t, temperature) if self.mode == "corroborated" else self.compute_beta(S_t, self.S_prev)

        # Temporal update
        R_t = beta * self.R_prev + (1.0 - beta) * S_t

        # CNN confidence fusion (conservative: amplifies risk when battery detected)
        R_fused = R_t * (1.0 + ALPHA_CNN * P_battery)
        R_fused = float(np.clip(R_fused, 0.0, 1.0))

        # Update state
        self.R_prev = R_t
        self.S_prev = S_t
        self.beta_history.append(beta)
        self.risk_history.append(R_t)
        self.fused_history.append(R_fused)

        return float(R_t), float(beta), R_fused


class SimulatedCNN:
    """
    DEPRECATED — kept only for reference/comparison, no longer used by
    run_adaptive_pipeline(). This fabricates battery_prob from a hand-picked
    temperature/gas threshold curve plus injected noise; it never touches
    the real trained checkpoints, so nothing about the "hierarchical CNN"
    the paper describes was actually exercised by the end-to-end simulation
    that produces R_fused / the FSM / the Emergency Stop timing. Superseded
    by RealCNN below, which runs the actual best_EfficientNetB0_{coarse,fine}
    checkpoints on real held-out test images.
    """
    def __init__(self, seed=77):
        self._rng = np.random.RandomState(seed)
        self._smooth = 0.0

    def predict(self, temperature: float, mq2_ppm: float,
                object_present: bool, metal_present: bool) -> Dict:
        """Returns coarse_class, fine_class, battery_prob, cnn_confidence."""
        if not object_present:
            self._smooth = 0.0
            return dict(coarse_class="none", fine_class="none",
                        battery_prob=0.0, cnn_confidence=0.0)

        if temperature > 150:
            base_prob = 0.92
        elif temperature > 90:
            base_prob = 0.78
        elif temperature > 45:
            base_prob = 0.55
        elif mq2_ppm > 200:
            base_prob = 0.40
        elif metal_present:
            base_prob = 0.25
        else:
            base_prob = 0.05

        noise = self._rng.normal(0, 0.06)
        prob  = float(np.clip(base_prob + noise, 0.0, 1.0))
        self._smooth = 0.7 * self._smooth + 0.3 * prob
        prob = float(np.clip(self._smooth, 0.0, 1.0))

        if prob > 0.6:
            coarse = "battery_waste"; fine = "lithium_ion_battery"
        elif metal_present:
            coarse = "metal_ewaste"; fine = "pcb_or_component"
        else:
            coarse = "non_metal_ewaste"; fine = "plastic_housing"

        conf = prob if prob > 0.5 else 1.0 - prob
        return dict(coarse_class=coarse, fine_class=fine,
                    battery_prob=prob, cnn_confidence=conf)


class RealCNN:
    """
    Runs the actual trained EfficientNet-B0 checkpoints (best_EfficientNetB0_
    coarse.pth / _fine.pth) on real held-out test images, instead of
    fabricating a probability from a hand-tuned curve. Classifies once per
    object arrival (object_present False->True edge) and holds that result
    steady while the object remains on the belt -- matching how a real
    camera would classify a frame once per item, and avoiding thousands of
    redundant forward passes over a 4000-step, 0.1s-resolution simulation.

    Which real image gets sampled is driven by the simulation's own ground-
    truth `battery_present` flag: a battery scenario draws a random real
    image from data/fine/test/{LiIon_18650,LiIon_Pouch,LiPo}, a non-battery
    scenario draws from data/coarse/test/{Electronics,Glass,Metal,Paper,
    Plastic}. Both checkpoints then run for real; battery_prob is the coarse
    model's actual softmax probability for the "Battery" class.
    """
    COARSE_NONBATTERY = ["Electronics", "Glass", "Metal", "Paper", "Plastic"]
    FINE_CLASSES      = ["LiIon_18650", "LiIon_Pouch", "LiPo"]

    def __init__(self, seed=77, data_dir="data", checkpoint_dir="checkpoints"):
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from evaluate_model import load_checkpoint, get_test_transform, coarse_label, device as _device
        self._coarse_label = coarse_label
        import torch
        from PIL import Image

        self._torch = torch
        self._Image = Image
        self._device = _device
        self._transform = get_test_transform()
        self._rng = np.random.RandomState(seed)

        self._coarse_model, self._coarse_classes, _ = load_checkpoint("EfficientNet-B0", "coarse")
        self._fine_model, self._fine_classes, _ = load_checkpoint("EfficientNet-B0", "fine")
        self._battery_idx = self._coarse_classes.index("Battery")

        data_dir = Path(data_dir)
        self._coarse_pool = {
            cls: sorted((data_dir / "coarse" / "test" / cls).glob("*"))
            for cls in self.COARSE_NONBATTERY
        }
        self._fine_pool = {
            cls: sorted((data_dir / "fine" / "test" / cls).glob("*"))
            for cls in self.FINE_CLASSES
        }

        self._was_present = False
        self._cached = dict(coarse_class="none", fine_class="none",
                             battery_prob=0.0, cnn_confidence=0.0)

    def _classify_image(self, model, img_path, class_names):
        img = self._Image.open(img_path).convert("RGB")
        x = self._transform(img).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            probs = self._torch.softmax(model(x), dim=1)[0].cpu().numpy()
        return {c: float(p) for c, p in zip(class_names, probs)}

    def predict(self, temperature: float, mq2_ppm: float,
                object_present: bool, metal_present: bool,
                battery_present: bool = False) -> Dict:
        """Returns coarse_class, fine_class, battery_prob, cnn_confidence."""
        if not object_present:
            self._was_present = False
            self._cached = dict(coarse_class="none", fine_class="none",
                                 battery_prob=0.0, cnn_confidence=0.0, image_path=None)
            return self._cached

        if not self._was_present:
            # New object arrival -- classify once for real, then hold.
            self._was_present = True
            if battery_present:
                fine_cls = self._fine_pool[self._rng.choice(self.FINE_CLASSES)]
                img_path = fine_cls[self._rng.randint(len(fine_cls))]
                coarse_probs = self._classify_image(self._coarse_model, img_path, self._coarse_classes)
                fine_probs = self._classify_image(self._fine_model, img_path, self._fine_classes)
                fine_top = max(fine_probs, key=fine_probs.get)
                # report what the coarse model actually decides for this battery photo
                coarse_top = self._coarse_label(coarse_probs)
            else:
                cls = self._rng.choice(self.COARSE_NONBATTERY)
                pool = self._coarse_pool[cls]
                img_path = pool[self._rng.randint(len(pool))]
                coarse_probs = self._classify_image(self._coarse_model, img_path, self._coarse_classes)
                coarse_top = self._coarse_label(coarse_probs)
                fine_top = "none"

            battery_prob = coarse_probs[self._coarse_classes[self._battery_idx]]
            conf = battery_prob if battery_prob > 0.5 else 1.0 - battery_prob
            self._cached = dict(coarse_class=coarse_top, fine_class=fine_top,
                                 battery_prob=battery_prob, cnn_confidence=conf,
                                 image_path=str(img_path))

        return self._cached


def run_adaptive_pipeline(df: pd.DataFrame, use_real_cnn: bool = True) -> pd.DataFrame:
    """Apply adaptive risk model to full sensor dataframe.

    use_real_cnn=True (default) runs the actual trained checkpoints on real
    test images via RealCNN. Set False only to reproduce old SimulatedCNN
    behavior for comparison -- SimulatedCNN never touches a real checkpoint.
    """
    from fuzzy_inference import MamdaniFIS
    fis   = MamdaniFIS()
    arm   = AdaptiveRiskModel()
    cnn   = RealCNN(seed=77) if use_real_cnn else SimulatedCNN(seed=77)

    results = []
    for _, row in df.iterrows():
        # Fuzzy inference
        fuzzy_risk, fz_debug = fis.infer(
            temperature   = row["temperature_C"],
            h2_ppm        = row["mq2_ppm"],
            co2_ppm       = row["mq135_ppm"],
            metal_present = bool(row["metal_present"]),
        )
        fuzzy_label = fis.get_risk_label(fuzzy_risk)

        # CNN inference (real checkpoint, or simulated if use_real_cnn=False)
        cnn_kwargs = dict(
            temperature   = row["temperature_C"],
            mq2_ppm       = row["mq2_ppm"],
            object_present= bool(row["object_present"]),
            metal_present = bool(row["metal_present"]),
        )
        if isinstance(cnn, RealCNN):
            cnn_kwargs["battery_present"] = bool(row.get("battery_present", False))
        cnn_out = cnn.predict(**cnn_kwargs)

        # Adaptive temporal update
        R_t, beta, R_fused = arm.update(
            S_t           = fuzzy_risk,
            P_battery     = cnn_out["battery_prob"],
            object_present= bool(row["object_present"]),
            temperature   = row["temperature_C"],
        )

        results.append({
            **row.to_dict(),
            "fuzzy_risk":   round(fuzzy_risk, 4),
            "fuzzy_label":  fuzzy_label,
            "cnn_battery_prob": round(cnn_out["battery_prob"], 4),
            "cnn_confidence":   round(cnn_out["cnn_confidence"], 4),
            "coarse_class":     cnn_out["coarse_class"],
            "R_temporal":   round(R_t, 4),
            "beta":         round(beta, 4),
            "R_fused":      round(R_fused, 4),
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    import sys; sys.path.insert(0, "outputs/ewaste_system")
    from sensor_simulation import run_simulation
    df_raw = run_simulation(duration_s=400.0, battery_arrives_at=20.0)
    df_out = run_adaptive_pipeline(df_raw)
    df_out.to_csv("outputs/ewaste_system/adaptive_risk_output.csv", index=False)
    print("Adaptive risk output saved.")
    cols = ["time","temperature_C","fuzzy_risk","R_temporal","beta","R_fused","phase"]
    print(df_out[cols].iloc[::200].to_string(index=False))
