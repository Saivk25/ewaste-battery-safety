"""
Tests for fuzzy_inference.py.

Several of these are direct regression guards for a real bug found and
fixed in this project: trapmf()'s `x <= a or x >= d: return 0.0` check
fired before the plateau check for degenerate edges (a==b or c==d),
which several real membership functions rely on. Concretely, h2_low(0)
returned 0.0 instead of 1.0 for the single most common H2 reading (~72%
of simulation timesteps), and risk_critical(1.0) returned 0.0 at the
most dangerous end of the output universe. See trapmf()'s docstring.
"""
import numpy as np
import pytest

from fuzzy_inference import trimf, trapmf, FuzzyMemberships, MamdaniFIS


class TestTrapmfBoundaries:
    """trapmf() at its boundary points, including degenerate edges."""

    def test_degenerate_left_edge_at_a_equals_b(self):
        # h2_low = trapmf(v, 0, 0, 50, 150): a==b==0, so x=0 is the START
        # of the plateau and must be full membership, not zero.
        assert trapmf(0, 0, 0, 50, 150) == 1.0

    def test_degenerate_right_edge_at_c_equals_d(self):
        # risk_critical = trapmf(r, 0.82, 0.92, 1.0, 1.0): c==d==1.0, so
        # r=1.0 is still WITHIN the plateau, not past it.
        assert trapmf(1.0, 0.82, 0.92, 1.0, 1.0) == 1.0

    def test_temp_high_degenerate_edge(self):
        # temp_high = trapmf(T, 80, 110, 500, 500)
        assert trapmf(500, 80, 110, 500, 500) == 1.0

    def test_normal_left_edge_still_zero_outside_support(self):
        # A genuinely non-degenerate left edge: x strictly below a is 0.
        assert trapmf(-1, 0, 10, 50, 150) == 0.0

    def test_normal_right_edge_still_zero_outside_support(self):
        assert trapmf(151, 0, 10, 50, 150) == 0.0

    def test_plateau_interior_is_one(self):
        # Plateau is [b, c] = [10, 50]; 30 is inside it, 75 is on the
        # falling edge (between c=50 and d=150), not the plateau.
        assert trapmf(30, 0, 10, 50, 150) == 1.0

    def test_rising_edge_interpolates(self):
        # a=0, b=10: halfway should be ~0.5
        assert trapmf(5, 0, 10, 50, 150) == pytest.approx(0.5, abs=1e-6)

    def test_falling_edge_interpolates(self):
        # c=50, d=150: halfway should be ~0.5
        assert trapmf(100, 0, 10, 50, 150) == pytest.approx(0.5, abs=1e-6)


class TestTrimf:
    def test_zero_outside_support(self):
        assert trimf(0, 10, 20, 30) == 0.0
        assert trimf(31, 10, 20, 30) == 0.0

    def test_peak_is_one(self):
        # trimf's rising-edge division has a +1e-12 fudge factor to avoid
        # divide-by-zero on degenerate triangles, so the peak is 1.0 only
        # up to floating-point precision, not exactly.
        assert trimf(20, 10, 20, 30) == pytest.approx(1.0, abs=1e-9)

    def test_symmetric_slopes(self):
        assert trimf(15, 10, 20, 30) == pytest.approx(0.5, abs=1e-6)
        assert trimf(25, 10, 20, 30) == pytest.approx(0.5, abs=1e-6)


class TestFuzzyMembershipsRealValues:
    """The actual membership functions used by MamdaniFIS, at the specific
    input values that occur constantly in the real simulation."""

    def test_h2_zero_ppm_is_fully_low(self):
        # 0 ppm H2 is the resting/baseline reading before thermal runaway
        # onset -- must read as fully "low", not ambiguous or zero.
        assert FuzzyMemberships.h2_low(0) == 1.0

    def test_co2_zero_ppm_is_fully_low(self):
        assert FuzzyMemberships.co2_low(0) == 1.0

    def test_risk_critical_at_max_output(self):
        assert FuzzyMemberships.risk_critical(1.0) == 1.0

    def test_ambient_temperature_is_fully_low(self):
        assert FuzzyMemberships.temp_low(25.0) == 1.0
        assert FuzzyMemberships.temp_medium(25.0) == 0.0
        assert FuzzyMemberships.temp_high(25.0) == 0.0

    def test_extreme_fire_temperature_is_fully_high(self):
        # temp_high = trapmf(T, 80, 110, 500, 500) is an open right shoulder:
        # a fire at 850 C must stay fully "high", not fall out of every set.
        assert FuzzyMemberships.temp_high(300.0) == pytest.approx(1.0, abs=1e-6)
        assert FuzzyMemberships.temp_high(850.0) == 1.0
        assert FuzzyMemberships.temp_medium(850.0) == 0.0

    def test_fire_without_gas_reaches_critical_risk(self):
        risk, _ = MamdaniFIS().infer(temperature=850.0, h2_ppm=0.0, co2_ppm=400.0, metal_present=True)
        assert risk >= 0.75


class TestMamdaniFISEndToEnd:
    """Sanity checks on full inference, not just membership functions."""

    def test_ambient_quiet_scenario_is_low_risk(self):
        fis = MamdaniFIS()
        risk, _ = fis.infer(temperature=25.0, h2_ppm=0.0, co2_ppm=400.0, metal_present=False)
        assert fis.get_risk_label(risk) == "Low"

    def test_extreme_fire_scenario_is_critical(self):
        fis = MamdaniFIS()
        risk, _ = fis.infer(temperature=500.0, h2_ppm=9000.0, co2_ppm=5000.0, metal_present=True)
        assert fis.get_risk_label(risk) == "Critical"

    def test_metal_alone_does_not_reach_critical(self):
        # Just an object with metal detected, nothing else elevated --
        # should not be treated the same as a confirmed hazard.
        fis = MamdaniFIS()
        risk, _ = fis.infer(temperature=25.0, h2_ppm=0.0, co2_ppm=400.0, metal_present=True)
        assert risk < 0.55  # below the warning_to_critical FSM threshold

    def test_metal_alone_stays_below_warning_threshold(self):
        # Regression guard for the R11 retune: an ordinary metal object
        # (PCB, can, etc.) with no thermal/gas signature must not alone
        # push risk across the Normal->Warning FSM threshold (0.30).
        # Averaged over repeated calls since risk composition involves a
        # centroid over a rule aggregate, not literally random, but this
        # guards the specific value directly.
        fis = MamdaniFIS()
        risk, _ = fis.infer(temperature=25.0, h2_ppm=0.0, co2_ppm=400.0, metal_present=True)
        assert risk < 0.30

    def test_output_always_in_valid_range(self):
        fis = MamdaniFIS()
        rng = np.random.RandomState(0)
        for _ in range(50):
            T = rng.uniform(0, 900)
            h2 = rng.uniform(0, 12000)
            co2 = rng.uniform(0, 6000)
            metal = bool(rng.randint(0, 2))
            risk, _ = fis.infer(T, h2, co2, metal)
            assert 0.0 <= risk <= 1.0

    def test_risk_label_boundaries(self):
        fis = MamdaniFIS()
        assert fis.get_risk_label(0.0) == "Low"
        assert fis.get_risk_label(0.29) == "Low"
        assert fis.get_risk_label(0.30) == "Medium"
        assert fis.get_risk_label(0.55) == "High"
        assert fis.get_risk_label(0.75) == "Critical"
