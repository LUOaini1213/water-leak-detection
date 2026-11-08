"""Guards against using evaluation-year (2019) data or labels for fitting and tuning."""
import json

import numpy as np
import pandas as pd
import pytest

from conftest import RESULTS, load_json
from leakdet.io import Leak
from leakdet.network import hydraulic_areas
from leakdet.train import CUTOFF, training_view, tune


def _scramble(daily: pd.DataFrame, leaks: list[Leak]):
    """Destroy every 2019 value and move every 2019 label; 2018 stays untouched."""
    rng = np.random.default_rng(42)
    d = daily.copy()
    late = d.index >= CUTOFF
    d.loc[late] = rng.normal(0, 100, size=(int(late.sum()), d.shape[1]))
    shifted = []
    for lk in leaks:
        if lk.start >= CUTOFF:
            off = pd.Timedelta(days=int(rng.integers(1, 60)))
            lk = Leak(lk.link, lk.start - off, lk.end, lk.diameter_m, lk.kind, lk.peak - off)
            if lk.start < CUTOFF:  # keep it in the evaluation year
                lk = Leak(lk.link, CUTOFF + off, lk.end, lk.diameter_m, lk.kind, CUTOFF + off)
        shifted.append(lk)
    return d, shifted


def test_training_view_drops_evaluation_rows_and_labels(daily_inputs, leaks):
    d, lk = training_view(daily_inputs, leaks)
    assert d.index.max() < CUTOFF
    assert all(x.start < CUTOFF for x in lk)
    assert len(lk) == 14


def test_tune_refuses_evaluation_rows(daily_inputs, leaks, lnet):
    areas = hydraulic_areas(lnet)
    _, train_leaks = training_view(daily_inputs, leaks)
    with pytest.raises(ValueError, match="evaluation period"):
        tune(daily_inputs, train_leaks, areas)


def test_tune_refuses_evaluation_labels(daily_inputs, leaks, lnet):
    areas = hydraulic_areas(lnet)
    d, _ = training_view(daily_inputs, leaks)
    with pytest.raises(ValueError, match="leak labels"):
        tune(d, leaks, areas)


def test_tuning_is_invariant_to_evaluation_year_and_reproduces_committed(daily_inputs, leaks, lnet):
    """Scrambling all 2019 data and labels must not change anything that is tuned."""
    areas = hydraulic_areas(lnet)
    sensor_area = {c[4:]: areas[c[4:]] for c in daily_inputs.columns if c.startswith("res_")}
    scrambled, moved = _scramble(daily_inputs, leaks)
    d, lk = training_view(scrambled, moved)
    tuned = json.loads(json.dumps(tune(d, lk, sensor_area), default=list))
    committed = load_json("tuning_2018.json")
    for method in ("mnf_threshold", "pressure_cusum"):
        assert tuned[method]["params"] == committed[method]["params"]
        assert tuned[method]["top"] == committed[method]["top"]
    assert tuned["pressure_scale"] == pytest.approx(committed["pressure_scale"])
    assert tuned["balance_scale"] == pytest.approx(committed["balance_scale"])


def test_residual_model_trained_before_first_training_leak(leaks):
    m = load_json("residual_model.json")
    end = pd.Timestamp(m["train_end_exclusive"])
    first = min(lk.start for lk in leaks if lk.start < CUTOFF)
    assert end <= first and end < CUTOFF
    assert pd.Timestamp(m["train_start"]) < end
