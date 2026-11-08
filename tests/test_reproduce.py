"""Offline reproduction of committed results from committed derived inputs."""
import json

import numpy as np
import pandas as pd
import pytest

from conftest import FIX, load_json
from leakdet.evaluation import match_alarms, match_by_area
from leakdet.features import fit_residual_model, leak_free_end
from leakdet.network import hydraulic_areas
from leakdet.train import CUTOFF, run_all

EVAL_END = pd.Timestamp("2020-01-01")


@pytest.fixture(scope="module")
def alarms(daily_inputs, lnet):
    areas = hydraulic_areas(lnet)
    sensor_area = {c[4:]: areas[c[4:]] for c in daily_inputs.columns if c.startswith("res_")}
    tuned = load_json("tuning_2018.json")
    al = run_all(daily_inputs, tuned, sensor_area)
    to_ab = {"A": "AB", "B": "AB", "C": "C"}
    for lst in al.values():
        for a in lst:
            a.area = to_ab.get(a.area, a.area)
    return al


def _leak_area(lnet, leaks):
    areas = hydraulic_areas(lnet)
    return {lk.link: ("C" if areas[lnet.pipes[lk.link][0]] == "C" else "AB") for lk in leaks}


def test_detection_summary_reproduced(alarms, leaks, lnet):
    det = load_json("detection_summary.json")
    leak_area = _leak_area(lnet, leaks)
    for method, al in alarms.items():
        scope = [lk for lk in leaks if leak_area[lk.link] == "C"] if method == "balance_cusum" else leaks
        r = match_alarms([a.time for a in al], scope, CUTOFF, EVAL_END)
        ra = match_by_area(al, scope, CUTOFF, EVAL_END, leak_area)
        want, want_a = det["methods"][method]["eval_2019"], det["methods"][method]["eval_2019_area_aware"]
        for key in ("n_events", "detected", "missed", "false_positives", "duplicates", "n_alarms"):
            assert r[key] == want[key], (method, key)
            assert ra[key] == want_a[key], (method, key)
        if want["median_delay_h"] is not None:
            assert r["median_delay_h"] == pytest.approx(want["median_delay_h"])


def test_alarm_log_reproduced(alarms):
    log = pd.read_csv(__import__("conftest").RESULTS / "alarms.csv", parse_dates=["time"])
    for method, al in alarms.items():
        assert list(log.loc[log["method"] == method, "time"]) == [a.time for a in al]
        tops = log.loc[log["method"] == method, "top_sensors"].fillna("").tolist()
        assert tops == [" ".join(a.ranking[:3]) for a in al]


def test_residual_model_reproduced_from_training_window(lnet, leaks):
    hourly = pd.read_csv(FIX / "hourly_train_window.csv.gz", index_col=0, parse_dates=True)
    areas = hydraulic_areas(lnet)
    sensors = [c for c in hourly.columns if c != "T1"]
    end = leak_free_end([lk for lk in leaks if lk.start < CUTOFF])
    assert hourly.index.max() < end
    model = fit_residual_model(hourly, sensors, areas, end, CUTOFF)
    ref = load_json("residual_model.json")
    assert model.features == ref["features"]
    for s, coef in ref["coefficients"].items():
        # the fixture is rounded to 1e-6 m; coefficients agree to well below sensor resolution
        pred_ref = np.column_stack([hourly[ref["features"][s]].to_numpy(), np.ones(len(hourly))]) @ np.array(coef)
        pred = np.column_stack([hourly[model.features[s]].to_numpy(), np.ones(len(hourly))]) @ model.coefs[s]
        assert np.max(np.abs(pred - pred_ref)) < 1e-3, s
