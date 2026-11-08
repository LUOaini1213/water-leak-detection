import numpy as np
import pandas as pd
import pytest

from leakdet.network import hydraulic_areas, tank_area_m2
from leakdet.waterbalance import (
    STEP_H,
    area_inflows,
    daily_mnf,
    daily_mnf_hour,
    leakage_ndf,
    pattern_steps,
    storage_inflow,
    true_leak_volumes,
    water_balance,
)


def _idx(days=2, start="2019-01-01"):
    return pd.date_range(start, periods=days * 288, freq="5min")


def test_storage_inflow_rate():
    idx = _idx(1)
    level = pd.Series(2.0 + 0.01 * np.arange(len(idx)), index=idx)  # +1 cm per 5 min
    q = storage_inflow(level, tank_area=100.0)
    assert q.iloc[0] == 0.0
    assert q.iloc[1:].to_numpy() == pytest.approx(0.01 * 100.0 / STEP_H)  # 12 m3/h


def test_area_inflows_identities(lnet):
    idx = _idx(1)
    flows = pd.DataFrame({"p227": 60.0, "p235": 40.0, "PUMP_1": 20.0}, index=idx)
    area = tank_area_m2(lnet.tanks["T1"])
    level = 3.0 + (0.5 / area) * np.arange(len(idx))  # storage rises 0.5 m3 per step = 6 m3/h
    levels = pd.DataFrame({"T1": level}, index=idx)
    q = area_inflows(flows, levels, lnet)
    assert q["AB"].to_numpy() == pytest.approx(80.0)
    assert q["C"].iloc[1:].to_numpy() == pytest.approx(14.0)
    assert q["system"].iloc[1:].to_numpy() == pytest.approx(94.0)
    assert (q["system"] - q["AB"] - q["C"]).abs().max() < 1e-9


def test_daily_mnf_uses_night_window_and_hourly_means():
    idx = _idx(2)
    hour = idx.hour + idx.minute / 60
    q = pd.Series(100.0, index=idx)
    q[(hour >= 3) & (hour < 4)] = 40.0  # night minimum
    q[(hour >= 13) & (hour < 14)] = 10.0  # daytime dip must be ignored
    q[idx == pd.Timestamp("2019-01-01 02:00")] = 0.0  # a single low sample is averaged out
    mnf = daily_mnf(q)
    assert list(mnf.round(6)) == [40.0, 40.0]
    assert list(daily_mnf_hour(q).dt.hour) == [3, 3]


def test_ndf_constant_pressure_is_24h():
    idx = _idx(2)
    p = pd.Series(40.0, index=idx)
    assert leakage_ndf(p, night_hour=3) == pytest.approx(24.0)
    lower_day = pd.Series(np.where(idx.hour == 3, 40.0, 10.0), index=idx)
    assert leakage_ndf(lower_day, night_hour=3) == pytest.approx(1 + 23 * 0.5)  # (10/40)^0.5


def test_pattern_steps_origin():
    idx = pd.DatetimeIndex(["2018-01-01 00:00", "2018-01-01 00:10", "2019-01-01 00:00"])
    assert list(pattern_steps(idx)) == [0, 2, 365 * 288]


def test_water_balance_synthetic(lnet):
    areas = hydraulic_areas(lnet)
    idx = _idx(2)
    flows = pd.DataFrame({"p227": 60.0, "p235": 40.0, "PUMP_1": 20.0}, index=idx)
    levels = pd.DataFrame({"T1": 3.0}, index=idx)
    amr_nodes = [n for n, a in areas.items() if a == "C" and lnet.base_demand.get(n, 0) > 0][:82]
    demands = pd.DataFrame(15000.0 / len(amr_nodes), index=idx, columns=amr_nodes)  # 15 m3/h in L/h
    pressures = pd.DataFrame({"n105": 50.0, "n1": 30.0}, index=idx)
    wb = water_balance(flows, levels, demands, pressures, lnet, areas)
    c = wb["area_C"]
    hours = len(idx) * STEP_H
    assert wb["hours"] == pytest.approx(hours)
    assert c["system_input_m3"] == pytest.approx(20.0 * hours)
    assert c["billed_metered_m3"] == pytest.approx(15.0 * hours)
    assert c["water_losses_m3"] == pytest.approx(
        c["system_input_m3"] - c["billed_metered_m3"] - c["unbilled_unmetered_estimate_m3"]
    )
    assert 0 < c["unbilled_unmetered_estimate_m3"] < 0.5 * c["billed_metered_m3"]
    assert c["non_revenue_water_m3"] == pytest.approx(5.0 * hours)
    assert c["water_losses_pct_of_input"] == pytest.approx(100 * c["water_losses_m3"] / c["system_input_m3"])
    assert c["n_demand_nodes"] == 92 and c["n_metered_nodes"] == 82
    ab = wb["area_AB"]
    assert ab["system_input_m3"] == pytest.approx(80.0 * hours)
    assert ab["water_losses_m3"] == pytest.approx(ab["system_input_m3"] - ab["authorised_consumption_estimate_m3"])
    assert ab["water_losses_low_m3h"] < ab["water_losses_m3h"] < ab["water_losses_high_m3h"]
    assert ab["mnf_mean_m3h"] == pytest.approx(80.0)
    assert ab["night_day_factor_h"] == pytest.approx(24.0)
    # with constant inflow the MNF method and the top-down balance see the same night leakage
    assert ab["mnf_method_losses_m3h"] == pytest.approx(
        ab["night_leakage_mean_m3h"] * ab["night_day_factor_h"] / 24.0, rel=1e-9
    )
    assert wb["system"]["system_input_m3"] == pytest.approx(100.0 * hours)


def test_true_leak_volumes(lnet):
    areas = hydraulic_areas(lnet)
    idx = _idx(1)
    lk = pd.DataFrame({"p257": 6.0, "p427": 3.0}, index=idx)  # p257 is in area C, p427 in A
    t = true_leak_volumes(lk, lnet, areas)
    assert t["area_C"]["leak_mean_m3h"] == pytest.approx(6.0)
    assert t["area_AB"]["leak_mean_m3h"] == pytest.approx(3.0)
    assert t["system"]["leak_volume_m3"] == pytest.approx(9.0 * 24)
