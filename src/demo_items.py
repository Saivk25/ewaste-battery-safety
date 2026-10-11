"""
demo_items.py -- the items the dashboard can put on the line, shared by the Live Monitor, Baseline Race and
Conveyor Line tabs so one sidebar choice drives all three.

Each item has a parked version (live: the item sits under the sensors for minutes, used by Live Monitor and
Baseline Race) and/or a belt version (belt: a conveyor_sim scenario, the item rides past the sensors in about
10 s). A battery hidden inside a device is modelled in the parked version by telling the camera there is no
battery (it sees the device), while the sensors still see the cell.

"Repeat" r (1-30) selects held-out seed 99 + r: every test was run 30 times with fresh sensor noise and a
different test photo for the camera, and the repeat picks which of those runs is shown.
"""
from dataclasses import dataclass
from typing import Callable, Optional


def _prewarmed(seed, T0, scale, hidden=False, gas_dead=False):
    from gap_prewarmed import sim
    df = sim(seed, T0, scale)
    if hidden:
        df["battery_present"] = 0
    if gas_dead:
        df["mq2_ppm"] = 0.0; df["mq135_ppm"] = 400.0
    return df


def _sensor(name, **kw):
    import sensor_simulation as ss
    return lambda seed: getattr(ss, name)(seed=seed, **kw)


def _warm_bursts(seed):
    from pub_experiments import scen
    return scen("warm_bursts", seed)[0]


@dataclass(frozen=True)
class Item:
    live: Optional[Callable] = None     # seed -> sensor DataFrame (parked under the sensors)
    arrival: float = 0.0                # s at which the item reaches the sensors in the parked version
    belt: Optional[str] = None          # conveyor_sim scenario
    note: str = ""


ITEMS = {
    "Failing cell hidden in a device (60 C, slow)": Item(lambda s: _prewarmed(s, 60, 3, hidden=True), 20.0, "fail60_hidden",
        "The camera sees only the device; heat and gas give it away."),
    "Failing cell hidden in a device (80 C)": Item(lambda s: _prewarmed(s, 80, 6, hidden=True), 20.0, "fail80_hidden"),
    "Single battery from room temperature": Item(_sensor("run_simulation", duration_s=400.0, battery_arrives_at=20.0), 20.0, "intact",
        "A visible, healthy battery: no heat or gas at first, so only the camera can see it."),
    "Failing cell arriving at 80 C": Item(lambda s: _prewarmed(s, 80, 6), 20.0, "fail80"),
    "Gas sensors dead + hidden 80 C cell": Item(lambda s: _prewarmed(s, 80, 6, hidden=True, gas_dead=True), 20.0, "gasdead_fail80_hidden",
        "Both gas sensors read nothing; only the IR thermometer is left."),
    "Warm benign item + gas burst": Item(_warm_bursts, 0.0, "warm_burst", "No battery: a 50 C item while the gas sensors see bursts."),
    "Gas spikes, no battery": Item(_sensor("run_noise_stress_scenario", duration_s=300.0, n_spikes=8), 0.0, "benign_burst"),
    "Benign items only": Item(_sensor("run_normal_scenario", duration_s=200.0), 0.0, "benign"),
    "Hot plant (45 C) + gas burst": Item(belt="hot45_burst", note="A belt-only test: the plant itself is hot."),
    "Open belt (no hood) + hidden cell": Item(belt="open_fail80_hidden", note="A belt-only test: no hood, so the gas is 30 times more diluted."),
    "Multi-item: benign, battery, benign": Item(_sensor("run_multi_item_scenario", item3_gap_s=900.0), 60.0,
        note="A parked-only test: three items in a row under the sensors."),
}
REPEATS = range(1, 31)


def seed_of(repeat: int) -> int:
    return 99 + int(repeat)


if __name__ == "__main__":          # self-check: every parked item builds a sensor table
    for name, it in ITEMS.items():
        assert it.live or it.belt, name
        if it.live:
            df = it.live(seed_of(1))
            assert {"time", "temperature_C", "mq2_ppm", "mq135_ppm", "object_present", "metal_present"} <= set(df.columns), name
        print(f"{name:<45} parked={'yes' if it.live else 'no ':3}  belt={it.belt or '-'}")
