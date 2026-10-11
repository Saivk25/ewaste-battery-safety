"""
Tests for sensor_simulation.py, centered on the RNG-independence
regression: every sensor class previously called the global
np.random.seed(seed) in __init__, so constructing multiple objects in
sequence made each constructor clobber the previous one's seed --
the per-object `seed=X` was cosmetic, and what a class's .step()/.read()
actually drew from depended on how many *other* objects had been built
and in what order. Confirmed concretely: the same object-construction
pattern run_simulation() has always used, run in isolation, produced a
DECREASING temperature trend under continuous battery_present=True
instead of the expected thermal runaway.
"""
import numpy as np
import pytest

from sensor_simulation import (
    ThermalRunawayModel, MQ2Sensor, MQ135Sensor,
    InductiveProximitySensor, UltrasonicSensor,
    run_simulation, run_normal_scenario, run_multi_item_scenario,
)


class TestRNGIndependence:
    def test_thermal_model_unaffected_by_other_object_construction(self):
        thermal_alone = ThermalRunawayModel(seed=42)
        readings_alone = [thermal_alone.step(True)[0] for _ in range(50)]

        thermal_with_siblings = ThermalRunawayModel(seed=42)
        # Constructing other sensor objects afterward must not change
        # thermal_with_siblings' own draws.
        MQ2Sensor(seed=43)
        MQ135Sensor(seed=44)
        InductiveProximitySensor(seed=45)
        UltrasonicSensor(seed=46)
        readings_with_siblings = [thermal_with_siblings.step(True)[0] for _ in range(50)]

        assert readings_alone == readings_with_siblings

    def test_thermal_model_rises_under_continuous_battery_presence(self):
        # The actual symptom of the bug: temperature must trend upward,
        # not flatten or decrease, when battery_present=True continuously.
        thermal = ThermalRunawayModel(seed=42)
        MQ2Sensor(seed=43)
        MQ135Sensor(seed=44)
        InductiveProximitySensor(seed=45)
        temps = [thermal.step(True)[0] for _ in range(3000)]  # 300s
        assert temps[-1] > temps[0] + 50  # meaningfully hotter, not flat/noise

    def test_construction_order_does_not_matter(self):
        # thermal built first vs. built last -- same seed, same result.
        t1 = ThermalRunawayModel(seed=42)
        MQ2Sensor(seed=43); MQ135Sensor(seed=44)
        r1 = [t1.step(True)[0] for _ in range(20)]

        MQ2Sensor(seed=43); MQ135Sensor(seed=44)
        t2 = ThermalRunawayModel(seed=42)
        r2 = [t2.step(True)[0] for _ in range(20)]

        assert r1 == r2

    def test_two_independently_seeded_sensors_differ(self):
        mq2_a = MQ2Sensor(seed=1)
        mq2_b = MQ2Sensor(seed=2)
        readings_a = [mq2_a.read(100.0) for _ in range(20)]
        readings_b = [mq2_b.read(100.0) for _ in range(20)]
        assert readings_a != readings_b


class TestThermalRunawayModel:
    def test_no_battery_decays_toward_ambient(self):
        thermal = ThermalRunawayModel(seed=42)
        for _ in range(1000):
            thermal.step(True)
        hot_temp = thermal.T
        assert hot_temp > 30

        for _ in range(500):
            temp, phase = thermal.step(False)
        assert phase == "Normal"
        assert temp < hot_temp

    def test_battery_departure_resets_runaway_state(self):
        thermal = ThermalRunawayModel(seed=42)
        for _ in range(1000):
            thermal.step(True)
        thermal.step(False)
        assert thermal.t_batt == 0.0
        assert thermal.past_peak is False


class TestScenarioGeneration:
    def test_run_simulation_is_reproducible(self):
        df1 = run_simulation(duration_s=20.0, battery_arrives_at=5.0, seed=42)
        df2 = run_simulation(duration_s=20.0, battery_arrives_at=5.0, seed=42)
        assert (df1["temperature_C"] == df2["temperature_C"]).all()

    def test_run_normal_scenario_has_no_battery(self):
        df = run_normal_scenario(duration_s=10.0)
        assert (df["battery_present"] == 0).all()

    def test_multi_item_scenario_has_three_distinct_windows(self):
        df = run_multi_item_scenario(item3_gap_s=10.0)
        item1 = df[(df["time"] >= 15) & (df["time"] < 45)]
        gap = df[(df["time"] >= 45) & (df["time"] < 60)]
        item2 = df[(df["time"] >= 60) & (df["time"] < 460)]

        assert (item1["object_present"] == 1).all()
        assert (item1["battery_present"] == 0).all()
        assert (gap["object_present"] == 0).all()
        assert (item2["battery_present"] == 1).all()

    def test_multi_item_scenario_item3_position_follows_gap_param(self):
        short_gap = run_multi_item_scenario(item3_gap_s=10.0)
        long_gap = run_multi_item_scenario(item3_gap_s=900.0)
        assert long_gap["time"].max() > short_gap["time"].max()
