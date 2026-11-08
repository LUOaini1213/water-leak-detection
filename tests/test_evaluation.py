import numpy as np
import pandas as pd
import pytest

from leakdet.evaluation import (
    Alarm,
    carry_over,
    chance_level,
    events_in,
    localisation_hits,
    match_alarms,
    match_by_area,
    window_end,
)
from leakdet.io import Leak

D = pd.Timestamp("2019-01-01")
day = lambda n: D + pd.Timedelta(days=n)  # noqa: E731
START, END = D, D + pd.Timedelta(days=100)

L1 = Leak("p1", day(10), day(40), 0.01, "abrupt", day(10))
L2 = Leak("p2", day(20), day(60), 0.01, "incipient", day(30))
OLD = Leak("p0", D - pd.Timedelta(days=30), day(90), 0.01, "incipient", D - pd.Timedelta(days=5))
LEAKS = [L1, L2, OLD]


def test_windows():
    assert window_end(L1) == day(17)  # abrupt: start + grace
    assert window_end(L2) == day(37)  # incipient: peak + grace
    short = Leak("p3", day(10), day(12), 0.01, "abrupt", day(10))
    assert window_end(short) == day(12)  # cut at repair


def test_events_and_carry_over():
    assert [lk.link for lk in events_in(LEAKS, START, END)] == ["p1", "p2"]
    assert [lk.link for lk in carry_over(LEAKS, START)] == ["p0"]


def test_match_tp_duplicate_fp_and_delay():
    alarms = [day(11), day(12), day(25), day(50), day(-3)]
    r = match_alarms(alarms, LEAKS, START, END)
    assert r["n_events"] == 2
    assert r["detected"] == 2 and r["missed"] == 0
    assert r["duplicates"] == 1  # day 12 inside the window of already-detected p1
    assert r["false_positives"] == 1  # day 50 is after both windows; day -3 is outside the period
    assert r["n_alarms"] == 4
    pl = r["per_leak"].set_index("link")
    assert pl.loc["p1", "delay_h"] == pytest.approx(24.0)
    assert pl.loc["p2", "delay_h"] == pytest.approx(5 * 24.0)
    assert r["median_delay_h"] == pytest.approx(72.0)
    assert r["fp_per_30d"] == pytest.approx(1 / 100 * 30)
    assert [lab for _, lab in r["alarm_labels"]] == ["TP", "duplicate", "TP", "FP"]


def test_overlapping_windows_credit_earliest_start():
    r = match_alarms([day(21), day(22)], LEAKS, START, END, grace_days=30)
    pl = r["per_leak"].set_index("link")
    assert pl.loc["p1", "alarm_time"] == day(21)
    assert pl.loc["p2", "alarm_time"] == day(22)


def test_alarm_before_leak_start_is_false():
    r = match_alarms([day(9)], LEAKS, START, END)
    assert r["false_positives"] == 1 and r["detected"] == 0


def test_alarm_exactly_at_window_end_counts():
    r = match_alarms([day(17)], LEAKS, START, END)
    assert r["detected"] == 1


def test_area_aware_matching():
    area = {"p1": "C", "p2": "AB", "p0": "AB"}
    alarms = [Alarm(day(11), "m", "AB"), Alarm(day(12), "m", "C"), Alarm(day(25), "m", None)]
    r = match_by_area(alarms, LEAKS, START, END, area)
    pl = r["per_leak"].set_index("link")
    assert pl.loc["p1", "alarm_time"] == day(12)  # the AB alarm on day 11 cannot detect a C leak
    assert not pl.loc["p2", "detected"]
    assert r["false_positives"] == 2  # wrong-area alarm + alarm without area
    assert r["n_alarms"] == 3


def test_chance_level_is_reproducible_and_sane():
    a = chance_level(5, LEAKS, START, END, n_sim=200, seed=1)
    b = chance_level(5, LEAKS, START, END, n_sim=200, seed=1)
    assert a == b
    assert 0 < a["mean_detected"] <= 2
    assert a["mean_detected"] + a["mean_false_positives"] <= 5
    # windows cover 7 + 17 of 100 days: expected detections are well below 2
    assert a["mean_detected"] < 1.5
    z = chance_level(0, LEAKS, START, END, n_sim=10)
    assert z["mean_detected"] == 0 and z["mean_false_positives"] == 0


def test_chance_level_area_aware_is_stricter():
    area = {"p1": "C", "p2": "AB", "p0": "AB"}
    plain = chance_level(20, LEAKS, START, END, n_sim=300, seed=3)
    aware = chance_level(20, LEAKS, START, END, n_sim=300, seed=3, area_of=area)
    assert aware["mean_false_positives"] > plain["mean_false_positives"]


def test_localisation_hits():
    class Net:
        def link_nodes(self, link):
            return {"pA": ("a", "b"), "pB": ("c", "d")}[link]

    sensors = ["s1", "s2", "s3"]
    idx = {"a": 0, "b": 1, "c": 2, "d": 3}
    dist = np.array([[5, 9, 50, 60], [40, 30, 1, 70], [20, 20, 20, 20]], dtype=float)
    matched = [("pA", Alarm(D, "m", ranking=("s2", "s1", "s3"))), ("pB", Alarm(D, "m", ranking=("s2", "s3", "s1")))]
    out = localisation_hits(matched, sensors, dist, idx, Net(), ks=(1, 2))
    assert list(out["per_leak"]["nearest_sensor"]) == ["s1", "s2"]
    assert list(out["per_leak"]["rank"]) == [2, 1]
    assert out["hit_at_1"] == 1 and out["hit_at_2"] == 2
    assert out["chance_at_1"] == pytest.approx(2 / 3)
