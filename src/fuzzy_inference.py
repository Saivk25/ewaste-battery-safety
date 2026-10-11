"""
fuzzy_inference.py — Mamdani Fuzzy Inference System (no external library)
Inputs: temperature, mq2_ppm (H2), mq135_ppm (CO2), metal_present (binary)
Output: risk_level [0, 1]
Membership functions calibrated from lithium-ion thermal runaway literature.
"""
import numpy as np
from typing import Dict, Tuple

# ── Membership Functions ──────────────────────────────────────────────────────

def trimf(x, a, b, c):
    """Triangular membership function."""
    if x <= a or x >= c: return 0.0
    if x <= b: return (x - a) / (b - a + 1e-12)
    return (c - x) / (c - b + 1e-12)

def trapmf(x, a, b, c, d):
    """Trapezoidal membership function.

    Boundary note: several calls here use a degenerate edge (a==b, e.g.
    trapmf(v, 0, 0, 50, 150), or c==d, e.g. trapmf(v, 800, 1500, 12000, 12000))
    to model a plateau that starts/ends exactly at the domain boundary. The
    old `x <= a or x >= d: return 0.0` short-circuit fired on x==a==b (or
    x==c==d) before the plateau check ran, incorrectly returning 0.0 exactly
    at the boundary instead of 1.0 - e.g. h2_low(0) returned 0.0 for the
    single most common H2 reading (0 ppm, ~72% of simulation timesteps),
    and risk_critical(1.0) returned 0.0 at the most dangerous end of the
    output universe. Using strict '<'/'>' plus explicit degenerate-edge
    handling fixes both without changing behavior anywhere the edges aren't
    degenerate.

    A degenerate edge also means the plateau continues past the domain
    boundary (open shoulder): a reading beyond it is still fully "high" (or
    "low"). Without this, an IR reading above 500 C -- a cell already on
    fire -- had zero membership in every temperature set, and with dead or
    diluted gas sensors the fused risk fell to 0.5 and never reached
    Emergency Stop.
    """
    if x > d: return 1.0 if c == d else 0.0
    if x < a: return 1.0 if a == b else 0.0
    if b <= x <= c: return 1.0
    if x < b: return (x - a) / (b - a) if b > a else 1.0
    return (d - x) / (d - c) if d > c else 1.0

def sigmf(x, c, sigma):
    """Sigmoid membership function (for sharp transitions)."""
    return 1.0 / (1.0 + np.exp(-sigma * (x - c)))


class FuzzyMemberships:
    """
    Membership functions calibrated to real sensor ranges:
      Temperature (MLX90614): 25–400°C operating concern range
      MQ-2 H2: 0–10000 ppm
      MQ-135 CO2: 400–5000 ppm
      Metal: binary → mapped to 0 or 1 fuzzy value directly
    """

    # -- Temperature (°C) --
    @staticmethod
    def temp_low(T):    return trapmf(T, 0, 0, 30, 55)
    @staticmethod
    def temp_medium(T): return trimf(T, 40, 70, 100)
    @staticmethod
    def temp_high(T):   return trapmf(T, 80, 110, 500, 500)

    # -- H2 ppm (MQ-2) --
    @staticmethod
    def h2_low(v):    return trapmf(v, 0, 0, 50, 150)
    @staticmethod
    def h2_medium(v): return trimf(v, 80, 400, 1200)
    @staticmethod
    def h2_high(v):   return trapmf(v, 800, 1500, 12000, 12000)

    # -- CO2 ppm (MQ-135) --
    @staticmethod
    def co2_low(v):    return trapmf(v, 0, 0, 450, 650)
    @staticmethod
    def co2_medium(v): return trimf(v, 500, 1200, 2500)
    @staticmethod
    def co2_high(v):   return trapmf(v, 2000, 3000, 6000, 6000)

    # -- Risk output (0–1) --
    @staticmethod
    def risk_low(r):      return trapmf(r, 0, 0, 0.25, 0.40)
    @staticmethod
    def risk_medium(r):   return trimf(r, 0.30, 0.50, 0.70)
    @staticmethod
    def risk_high(r):     return trimf(r, 0.60, 0.75, 0.90)
    @staticmethod
    def risk_critical(r): return trapmf(r, 0.82, 0.92, 1.0, 1.0)


class MamdaniFIS:
    """
    Mamdani Fuzzy Inference System with 20 physically-motivated rules.
    Conservative fusion: sensor disagreement is treated as higher risk.
    Defuzzification: centroid method over 1000-point output universe.
    """

    def __init__(self):
        self.mf = FuzzyMemberships()
        self._output_universe = np.linspace(0.0, 1.0, 1000)

    def _fuzzify(self, T: float, h2: float, co2: float, metal: bool) -> Dict:
        m = self.mf
        return {
            "T_low":    m.temp_low(T),
            "T_med":    m.temp_medium(T),
            "T_high":   m.temp_high(T),
            "H2_low":   m.h2_low(h2),
            "H2_med":   m.h2_medium(h2),
            "H2_high":  m.h2_high(h2),
            "CO2_low":  m.co2_low(co2),
            "CO2_med":  m.co2_medium(co2),
            "CO2_high": m.co2_high(co2),
            "metal":    1.0 if metal else 0.0,
        }

    def _apply_rules(self, fz: Dict) -> Dict:
        """
        20 rules derived from lithium-ion thermal runaway behaviour.
        Conservative: any high-severity sensor alone triggers elevated risk.
        """
        rules = {}

        # ── CRITICAL rules — any single catastrophic indicator ───────────────
        # R1: High temp alone → critical (thermal runaway confirmed)
        rules["R1_critical"] = fz["T_high"]

        # R2: High H2 + metal → critical (battery venting detected)
        rules["R2_critical"] = min(fz["H2_high"], fz["metal"])

        # R3: High H2 + high CO2 → critical (multi-gas runaway signature)
        rules["R3_critical"] = min(fz["H2_high"], fz["CO2_high"])

        # R4: Med temp + high H2 + metal → critical (early runaway, conservative)
        rules["R4_critical"] = min(fz["T_med"], fz["H2_high"], fz["metal"])

        # ── HIGH rules ───────────────────────────────────────────────────────
        # R5: Med temp + med H2 + metal → high
        rules["R5_high"] = min(fz["T_med"], fz["H2_med"], fz["metal"])

        # R6: High H2 alone → high (gas sensor positive, vision may miss battery)
        rules["R6_high"] = fz["H2_high"]

        # R7: Med temp + high CO2 → high
        rules["R7_high"] = min(fz["T_med"], fz["CO2_high"])

        # R8: High temp (even without gas — sensor might lag) → high minimum
        rules["R8_high"] = min(fz["T_high"], fz["H2_low"])

        # R9: Med H2 + high CO2 + metal → high
        rules["R9_high"] = min(fz["H2_med"], fz["CO2_high"], fz["metal"])

        # R10: Med temp + med CO2 + metal → high
        rules["R10_high"] = min(fz["T_med"], fz["CO2_med"], fz["metal"])

        # ── MEDIUM rules ─────────────────────────────────────────────────────
        # R11: Metal detected + low temp + low gas → medium (unconfirmed).
        # Weighted at 0.4 certainty, not full strength: metal alone (a PCB,
        # can, or any inert metal item) is weak evidence on its own -- with
        # the trapmf boundary fix (see trapmf() docstring), this rule now
        # correctly fires the instant ANY metal object is detected, and at
        # full strength it pushed R_fused across the Warning threshold for
        # ordinary non-battery metal e-waste (~8.85% false-Warning rate
        # measured against a battery-free scenario). Dampening it lets a
        # genuine battery's temp/gas signature (which arrives and escalates
        # this via R1-R10) still drive fast detection, without every metal
        # object alone tripping Warning.
        rules["R11_medium"] = 0.4 * min(fz["metal"], fz["T_low"], fz["H2_low"])

        # R12: Med temp + low gas + no metal → medium (warming, unconfirmed)
        rules["R12_medium"] = min(fz["T_med"], fz["H2_low"], 1 - fz["metal"])

        # R13: Med H2 alone → medium
        rules["R13_medium"] = fz["H2_med"]

        # R14: Med CO2 + metal → medium
        rules["R14_medium"] = min(fz["CO2_med"], fz["metal"])

        # R15: Low temp + med H2 → medium (early H2 off-gassing detected)
        rules["R15_medium"] = min(fz["T_low"], fz["H2_med"])

        # ── LOW rules ────────────────────────────────────────────────────────
        # R16: All low, no metal → low
        rules["R16_low"] = min(fz["T_low"], fz["H2_low"], fz["CO2_low"], 1 - fz["metal"])

        # R17: Low temp + low H2 + low CO2 + metal (PCB, not battery) → low-medium
        rules["R17_low"] = min(fz["T_low"], fz["H2_low"], fz["CO2_low"], fz["metal"])

        # R18: Low temp + low gas + no metal → low
        rules["R18_low"] = min(fz["T_low"], fz["H2_low"], 1 - fz["metal"])

        # R19: Med CO2 alone (background variation) → low
        rules["R19_low"] = min(fz["CO2_med"], fz["H2_low"], fz["T_low"])

        # R20: Low everything → low
        rules["R20_low"] = min(fz["T_low"], fz["H2_low"], fz["CO2_low"])

        return rules

    def _aggregate_and_defuzzify(self, rules: Dict) -> float:
        """Centroid defuzzification over aggregated output sets."""
        u = self._output_universe
        m = self.mf

        # Map rule strengths to output MFs
        agg = np.zeros_like(u)
        for label, strength in rules.items():
            if strength < 1e-6:
                continue
            if "critical" in label:
                mf_vals = np.array([min(strength, m.risk_critical(r)) for r in u])
            elif "high" in label:
                mf_vals = np.array([min(strength, m.risk_high(r)) for r in u])
            elif "medium" in label:
                mf_vals = np.array([min(strength, m.risk_medium(r)) for r in u])
            else:  # low
                mf_vals = np.array([min(strength, m.risk_low(r)) for r in u])
            agg = np.maximum(agg, mf_vals)

        # Centroid
        denom = np.sum(agg)
        if denom < 1e-9:
            # No rule fired above threshold -- an input combination the rule
            # base doesn't cover (e.g. metal present + moderately elevated
            # temperature + gas not yet risen). Defaulting to 0.0 (Low) would
            # read an unrecognised/ambiguous state as "confirmed safe", which
            # is backwards for a fire-safety system. Default to the Medium
            # band's centre instead: elevated caution without triggering an
            # emergency escalation off pure ambiguity alone.
            return 0.5
        return float(np.sum(u * agg) / denom)

    def infer(self, temperature: float, h2_ppm: float, co2_ppm: float,
              metal_present: bool) -> Tuple[float, Dict]:
        """
        Run complete Mamdani inference.
        Returns (risk_level [0,1], debug_dict)
        """
        fz    = self._fuzzify(temperature, h2_ppm, co2_ppm, metal_present)
        rules = self._apply_rules(fz)
        risk  = self._aggregate_and_defuzzify(rules)
        debug = {"fuzzified": fz, "rule_strengths": rules, "risk": risk}
        return risk, debug

    def get_risk_label(self, risk: float) -> str:
        if risk < 0.30: return "Low"
        if risk < 0.55: return "Medium"
        if risk < 0.75: return "High"
        return "Critical"


if __name__ == "__main__":
    fis = MamdaniFIS()
    test_cases = [
        ("Normal PCB",         25.0,    0.0,  410.0, False),
        ("PCB with metal",     27.0,    0.0,  405.0, True),
        ("Early warming",      45.0,   10.0,  430.0, True),
        ("H2 off-gassing",     72.0,  120.0,  440.0, True),
        ("Phase 2 onset",      95.0,  800.0,  900.0, True),
        ("Runaway",           250.0, 5000.0, 3500.0, True),
        ("Fire",              500.0, 9000.0, 5000.0, True),
    ]
    print(f"\n{'Scenario':<22} {'Temp':>6} {'H2':>7} {'CO2':>7} {'Metal':>6} {'Risk':>7} {'Label':>10}")
    print("-" * 75)
    for name, T, h2, co2, metal in test_cases:
        risk, _ = fis.infer(T, h2, co2, metal)
        label = fis.get_risk_label(risk)
        print(f"{name:<22} {T:>6.1f} {h2:>7.1f} {co2:>7.1f} {str(metal):>6} {risk:>7.4f} {label:>10}")
