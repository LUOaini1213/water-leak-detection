import numpy as np
import pytest

from leakdet.network import (
    hydraulic_areas,
    link_distance,
    node_distances,
    nominal_demand,
    parse_inp,
    pattern_value,
    tank_area_m2,
)

TOY = """
[JUNCTIONS]
 S1 10 0
 A1 10 1.0 PAT
 A2 10 2.0
 A3 10 0
 C1 10 1.5
[RESERVOIRS]
 R1 50
[TANKS]
 T1 40 2 0 4 10 0
[PIPES]
 P0 R1 S1 10 200 100 0 Open
 P1 A1 A2 100 200 100 0 Open
 P2 A2 A3 200 200 100 0 Open
 P3 A1 A3 500 200 100 0 Open
 P4 T1 C1 50 200 100 0 Open
[PUMPS]
 PU1 A3 T1 HEAD 1
[VALVES]
 V1 S1 A1 200 PRV 40 0
[PATTERNS]
 PAT 1.0 2.0
 PAT 3.0
[OPTIONS]
 Units CMH
 Demand Multiplier 1.0
"""


@pytest.fixture
def toy():
    return parse_inp(TOY)


def test_parse_sections(toy):
    assert set(toy.pipes) == {"P0", "P1", "P2", "P3", "P4"}
    assert toy.pumps == {"PU1": ("A3", "T1")}
    assert toy.valves["V1"][2] == "PRV"
    assert toy.patterns["PAT"] == [1.0, 2.0, 3.0]
    assert toy.base_demand["A2"] == 2.0
    assert toy.link_length("PU1") == 0.0 and toy.link_length("P3") == 500


def test_demands_section_overrides_junction_column():
    net = parse_inp(TOY + "\n[DEMANDS]\n A1 0.5 PAT\n A1 0.25 \n")
    assert net.demands == {"A1": {"PAT": 0.5, "": 0.25}}


def test_areas(toy):
    areas = hydraulic_areas(toy)
    assert areas["C1"] == "C" and areas["T1"] == "C"
    assert areas["A1"] == areas["A2"] == areas["A3"] == "A"
    assert areas["R1"] == areas["S1"] == "source"


def test_shortest_path_ignores_pumps_and_valves(toy):
    d, idx = node_distances(toy)
    assert d[idx["A1"], idx["A3"]] == pytest.approx(300.0)  # via A2, not the 500 m pipe
    assert np.isinf(d[idx["A1"], idx["C1"]])  # only a pump connects A and C


def test_link_distance_battledim_definition(toy):
    d, idx = node_distances(toy)
    assert link_distance(toy, d, idx, "P1", "P1") == 0.0
    # P1 (A1-A2, 100 m) and P2 (A2-A3, 200 m) share A2: 0 + 50 + 100
    assert link_distance(toy, d, idx, "P1", "P2") == pytest.approx(150.0)


def test_patterns_and_nominal_demand(toy):
    steps = np.array([0, 1, 2, 3])
    assert list(pattern_value(toy, "PAT", steps)) == [1.0, 2.0, 3.0, 1.0]
    assert list(pattern_value(toy, "", steps)) == [1.0] * 4
    # A1 uses PAT (base 1), A2 has no pattern (base 2)
    assert list(nominal_demand(toy, ["A1", "A2"], steps)) == [3.0, 4.0, 5.0, 3.0]


def test_tank_area(toy):
    assert tank_area_m2(toy.tanks["T1"]) == pytest.approx(np.pi * 25)


def test_ltown_structure(lnet):
    from collections import Counter

    areas = hydraulic_areas(lnet)
    assert len(lnet.junctions) == 782 and len(lnet.pipes) == 905
    assert Counter(areas.values()) == {"A": 657, "C": 93, "B": 31, "source": 4}
    assert lnet.pumps["PUMP_1"] == ("n54", "T1")
    assert lnet.options["UNITS"] == "CMH"
    assert len(lnet.patterns["P-Residential"]) == 2016  # one week at 5 min
