import json

import numpy as np
import pandas as pd
import pytest

from conftest import FIX, RESULTS
from leakdet import battledim as bd
from leakdet.io import Leak
from leakdet.network import parse_inp

TOY = """
[JUNCTIONS]
 a 0 0
 b 0 0
 c 0 0
 d 0 0
 e 0 0
[PIPES]
 L1 a b 100 100 100 0 Open
 L2 b c 200 100 100 0 Open
 L3 c d 300 100 100 0 Open
 L4 d e 1000 100 100 0 Open
"""
T = pd.Timestamp("2019-01-01")


@pytest.fixture
def setup():
    net = parse_inp(TOY)
    leak = Leak("L2", T + pd.Timedelta(hours=1), T + pd.Timedelta(days=2), 0.01, "abrupt", T + pd.Timedelta(hours=1))
    series = {"L2": np.full(288 * 3, 12.0)}  # 12 m3/h -> 1 m3 per 5-min step
    return net, [leak], series


def test_time2steps():
    assert bd.time2steps(T) == 0
    assert bd.time2steps(T + pd.Timedelta(minutes=5)) == 1
    assert bd.time2steps(T - pd.Timedelta(hours=1)) == -12


def test_exact_hit_scores_remaining_water(setup):
    net, leaks, series = setup
    r = bd.score_detections([("L2", T + pd.Timedelta(hours=2))], leaks, series, net)
    # step 24 -> MATLAB timeSeries(24:end) -> python [23:], 864 - 23 = 841 steps of 1 m3
    assert r["detections"][0]["score"] == pytest.approx(841 * 0.8)
    assert (r["true_positives"], r["false_positives"], r["false_negatives"]) == (1, 0, 0)


def test_neighbour_pipe_pays_distance_cost(setup):
    net, leaks, series = setup
    r = bd.score_detections([("L1", T + pd.Timedelta(hours=2))], leaks, series, net)
    # L1-L2 share node b: x = 0 + 100/2 + 200/2 = 150 m -> crew cost 150/300 * 500
    assert r["detections"][0]["distance_m"] == pytest.approx(150.0)
    assert r["detections"][0]["score"] == pytest.approx(841 * 0.8 - 250.0)


def test_far_pipe_wrong_time_and_repeat(setup):
    net, leaks, series = setup
    dets = [
        ("L4", T + pd.Timedelta(hours=2)),  # x = 200/2 + 300 + 1000/2 = 900 m > 300: false alarm
        ("L2", T + pd.Timedelta(minutes=30)),  # before the leak starts: false alarm
        ("L2", T + pd.Timedelta(hours=3)),  # first hit
        ("L3", T + pd.Timedelta(hours=4)),  # x = 250 m, leak already found: repeat, 0 EUR
        ("L2", T + pd.Timedelta(days=5)),  # after repair: false alarm
        ("L2", T - pd.Timedelta(days=1)),  # 2018 detections are ignored
    ]
    r = bd.score_detections(dets, leaks, series, net)
    outcomes = [d["outcome"] for d in r["detections"]]
    assert outcomes == ["FP", "FP", "TP", "repeat", "FP"]
    assert r["false_positives"] == 3 and r["true_positives"] == 1
    assert r["total_score_eur"] == pytest.approx(round(-1500 + (864 - 35) * 0.8, 2))


def test_parse_submission():
    subs = bd.parse_submission("# header\np257, 2019-01-01 00:05\n\np523, 2019-01-15 23:00\n")
    assert subs == [("p257", pd.Timestamp("2019-01-01 00:05")), ("p523", pd.Timestamp("2019-01-15 23:00"))]


def test_active_leaks(leaks):
    act = bd.active_leaks(leaks)
    assert len(act) == 23
    assert {"p257", "p427", "p810", "p654"} <= {lk.link for lk in act}


def test_candidate_pipe():
    net = parse_inp(TOY)
    assert bd.candidate_pipe(net, "b") == "L1"  # shortest of L1 (100 m) and L2 (200 m)
    assert bd.candidate_pipe(net, "e") == "L4"


def test_port_reproduces_official_reports():
    """Committed validation: the port's totals equal the organisers' published totals."""
    ref = json.loads((FIX / "battledim_reference.json").read_text(encoding="utf-8"))
    val = json.loads((RESULTS / "scoring_port_validation.json").read_text(encoding="utf-8"))
    assert set(val) == set(ref["teams"])
    for team, spec in ref["teams"].items():
        assert val[team]["official_total_eur"] == spec["official_total_eur"]
        assert val[team]["port_total_eur"] == pytest.approx(spec["official_total_eur"], abs=0.005)
        assert val[team]["port_detection_scores"] == pytest.approx(spec["official_detection_scores"], abs=1e-3)
        if spec["official_tp_fp_fn"] is not None:
            assert val[team]["port_tp_fp_fn"] == spec["official_tp_fp_fn"]
