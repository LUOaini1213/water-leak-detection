import numpy as np
import pandas as pd
import pytest

from leakdet.detectors import ResidualModel
from leakdet.features import build_daily_inputs, fit_residual_model, hourly_inputs, leak_free_end
from leakdet.io import Leak


def test_build_daily_inputs_synthetic(lnet):
    idx = pd.date_range("2018-01-01", periods=3 * 288, freq="5min")
    flows = pd.DataFrame({"p227": 60.0, "p235": 40.0, "PUMP_1": 20.0}, index=idx)
    levels = pd.DataFrame({"T1": 3.0}, index=idx)
    amr = pd.DataFrame({"n2": 10000.0, "n3": 5000.0}, index=idx)  # 15 m3/h
    hourly = hourly_inputs(pd.DataFrame({"s1": 40.0, "s2": 41.0}, index=idx), levels)
    assert list(hourly.columns) == ["s1", "s2", "T1"] and len(hourly) == 72
    model = ResidualModel(k=1)
    model.features = {"s1": ["s2"]}
    model.coefs = {"s1": np.array([1.0, -1.5])}  # s1 = s2 - 1.5 -> residual +0.5
    daily = build_daily_inputs(flows, levels, amr, hourly, model, lnet, unmetered_ratio=0.1)
    assert list(daily.columns) == ["mnf_AB", "mnf_C", "loss_C", "res_s1"]
    assert len(daily) == 3
    assert daily["mnf_AB"].to_numpy() == pytest.approx(80.0)
    assert daily["mnf_C"].to_numpy() == pytest.approx(20.0)
    assert daily["loss_C"].to_numpy() == pytest.approx(20.0 - 15.0 * 1.1)
    assert daily["res_s1"].to_numpy() == pytest.approx(0.5)


def test_leak_free_end_floors_to_hour():
    lk = [
        Leak("a", pd.Timestamp("2018-03-01 10:00"), pd.Timestamp("2018-04-01"), 0.01, "abrupt", pd.Timestamp("2018-03-01 10:00")),
        Leak("b", pd.Timestamp("2018-01-08 13:30"), pd.Timestamp("2018-04-01"), 0.01, "incipient", pd.Timestamp("2018-02-01")),
    ]
    assert leak_free_end(lk) == pd.Timestamp("2018-01-08 13:00")


def test_fit_residual_model_uses_only_window(lnet):
    from leakdet.network import hydraulic_areas

    areas = hydraulic_areas(lnet)
    idx = pd.date_range("2018-01-01", periods=300, freq="h")
    rng = np.random.default_rng(0)
    base = rng.normal(size=300)
    hourly = pd.DataFrame({"n105": base, "n114": 2 * base + 1, "n1": base, "n4": base, "T1": base}, index=idx)
    hourly.loc[idx[200]:, "n114"] += 5.0  # a change after the training window must not affect the fit
    m = fit_residual_model(hourly, ["n105", "n114", "n1", "n4"], areas, idx[200], pd.Timestamp("2019-01-01"), k=1)
    assert m.coefs["n114"][0] == pytest.approx(2.0) and m.coefs["n114"][1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        fit_residual_model(hourly, ["n105"], areas, pd.Timestamp("2019-02-01"), pd.Timestamp("2019-01-01"))
