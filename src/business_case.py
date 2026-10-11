"""
business_case.py -- per-line cost, placement and payback calculator (answers "what does it cost, how far upstream
must it sit, when does it pay back"). Every default is either a cited figure (docs/SHARK_TANK_REVIEW.md) or
a measured result of this project; plant-specific values (downtime cost, fire probability) are inputs.
"""
# Bill of materials per conveyor line, USD, (low, high) street prices found Oct 2026 (sources in SHARK_TANK_REVIEW.md)
BOM_PROTOTYPE = {"MQ-2 gas module": (1, 6), "MQ-135 gas module": (1, 6), "MLX90614 IR temperature": (1, 27),
                 "LJ18A3 inductive sensor": (2, 9), "HC-SR04 ultrasonic gate": (0.3, 8), "Camera (Pi Camera Module 3)": (26, 40),
                 "Edge computer (Jetson Orin Nano Super)": (249, 399)}
BOM_INDUSTRIAL = {"Fixed H2 detector (0-1000 ppm)": (1629, 2371), "IR pyrometer (Optris CT LT)": (295, 590),
                  "Inductive + ultrasonic (industrial)": (50, 200), "Industrial camera": (300, 800),
                  "Edge computer (Jetson Orin Nano Super)": (249, 399)}


def bom_total(bom):
    return sum(v[0] for v in bom.values()), sum(v[1] for v in bom.values())


def placement_m(belt_speed_m_s, stop_time_s, shredder_rundown_s):
    """Minimum distance from the sensing point to the shredder inlet so the shredder has stopped before the item arrives."""
    return belt_speed_m_s * (stop_time_s + shredder_rundown_s)


def annual_false_stop_cost(false_stops_per_shift, shifts_per_year, downtime_min_per_stop, downtime_cost_per_hour):
    return false_stops_per_shift * shifts_per_year * downtime_min_per_stop / 60.0 * downtime_cost_per_hour


def payback_years(system_cost, fires_per_year_without, loss_per_fire, fraction_prevented, annual_false_cost, annual_upkeep):
    """Years until avoided fire losses repay the system; inf if it never pays back."""
    net = fires_per_year_without * loss_per_fire * fraction_prevented - annual_false_cost - annual_upkeep
    return system_cost / net if net > 0 else float("inf")


if __name__ == "__main__":
    lo, hi = bom_total(BOM_PROTOTYPE); assert 280 < lo < hi < 500, (lo, hi)
    assert abs(placement_m(0.7, 3.1, 2.0) - 3.57) < 1e-9          # 0.7 m/s belt, 3.1 s stop, 2 s rundown
    assert annual_false_stop_cost(1, 600, 10, 3000) == 300000.0
    assert payback_years(5000, 0.1, 1.3e6, 0.5, 0, 0) == 5000 / 65000
    assert payback_years(5000, 0.0, 1.3e6, 0.5, 0, 100) == float("inf")
    print("prototype BOM", bom_total(BOM_PROTOTYPE), "industrial BOM", bom_total(BOM_INDUSTRIAL), "checks passed")
