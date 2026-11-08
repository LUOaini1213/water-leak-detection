import numpy as np
import pandas as pd
import pytest

from leakdet.detectors import (
    DAILY_DECISION_OFFSET,
    MNF_DECISION_OFFSET,
    ResidualModel,
    assert_before,
    balance_cusum_alarms,
    cusum_alarms,
    mnf_threshold_alarms,
    robust_daily_scale,
    sensor_groups,
)

DAYS = pd.date_range("2018-01-01", periods=60, freq="D")


def test_mnf_threshold_alarm_and_rearm():
    x = pd.Series(50.0, index=DAYS)
    x.iloc[20:] = 75.0  # +25 step
    x.iloc[40:] = 90.0  # +15 step: below delta
    al = mnf_threshold_alarms(x, delta=20.0, ref_days=7, min_ref=3, area="AB")
    assert [a.time for a in al] == [DAYS[20] + MNF_DECISION_OFFSET]
    assert al[0].area == "AB" and al[0].stat == pytest.approx(25.0)


def test_mnf_threshold_needs_min_ref_days_after_reset():
    x = pd.Series(50.0, index=DAYS)
    x.iloc[10:] = 80.0
    x.iloc[12:] = 110.0  # second step only 2 days after the first alarm
    x.iloc[30:] = 140.0
    al = mnf_threshold_alarms(x, delta=20.0, ref_days=7, min_ref=5)
    assert [a.time.normalize() for a in al] == [DAYS[10], DAYS[30]]


def test_mnf_threshold_reference_is_median_of_recent_days():
    x = pd.Series(50.0, index=DAYS)
    x.iloc[2] = 200.0  # outlier while the detector is still learning its reference
    x.iloc[8] = 72.0
    # the median of days 1..7 stays 50, so 72 exceeds it by 22 (a mean reference would be ~71)
    al = mnf_threshold_alarms(x, delta=20.0, ref_days=7, min_ref=7)
    assert [a.time.normalize() for a in al] == [DAYS[8]]


def test_mnf_threshold_ignores_decrease():
    x = pd.Series(50.0, index=DAYS)
    x.iloc[20:] = 10.0
    assert mnf_threshold_alarms(x, delta=5.0) == []


def _resid_frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(0, 0.001, (60, 3)), index=DAYS, columns=["s1", "s2", "s3"])


def test_cusum_detects_drop_on_right_sensor_and_day():
    df = _resid_frame()
    df.loc[DAYS[30]:, "s2"] -= 0.05
    scale = pd.Series(0.001, index=df.columns)
    al = cusum_alarms(df, scale, k=1.0, h=10.0, ref_days=7, area_of={"s1": "A", "s2": "C", "s3": "A"})
    assert al[0].time == DAYS[30] + DAILY_DECISION_OFFSET
    assert al[0].ranking[0] == "s2" and al[0].area == "C"
    assert set(al[0].ranking) == {"s1", "s2", "s3"}


def test_cusum_one_sided():
    df = _resid_frame()
    df.loc[DAYS[30]:, "s2"] += 0.05  # pressure rise (e.g. a repair) must not alarm
    scale = pd.Series(0.001, index=df.columns)
    assert cusum_alarms(df, scale, k=1.0, h=10.0) == []
    assert len(cusum_alarms(df, scale, k=1.0, h=10.0, sign=+1.0)) >= 1


def test_cusum_accumulates_small_shift_and_allowance_matters():
    df = pd.DataFrame(0.0, index=DAYS, columns=["s1"])
    df.loc[DAYS[20]:, "s1"] = -0.002  # 2 sigma per day
    scale = pd.Series(0.001, index=["s1"])
    al = cusum_alarms(df, scale, k=1.0, h=4.5, ref_days=30, min_ref=7)
    # S grows by 2 - 1 = 1 per day while the sliding median stays at 0: 1, 2, 3, 4, 5
    assert al[0].time == DAYS[24] + DAILY_DECISION_OFFSET
    assert cusum_alarms(df, scale, k=2.5, h=4.5, ref_days=30, min_ref=7) == []


def test_cusum_resets_after_alarm():
    df = pd.DataFrame(0.0, index=DAYS, columns=["s1"])
    df.loc[DAYS[10]:, "s1"] = -0.05
    scale = pd.Series(0.001, index=["s1"])
    al = cusum_alarms(df, scale, k=1.0, h=5.0, ref_days=7, min_ref=3)
    # after re-baselining on the new level there is nothing more to detect
    assert len(al) == 1


def test_cusum_rebaselines_from_post_alarm_data_only():
    df = pd.DataFrame(0.0, index=DAYS, columns=["s1"])
    df.loc[DAYS[10]:, "s1"] = -0.05
    df.loc[DAYS[30]:, "s1"] = -0.10
    scale = pd.Series(0.001, index=["s1"])
    al = cusum_alarms(df, scale, k=1.0, h=5.0, ref_days=7, min_ref=3)
    assert [a.time.normalize() for a in al] == [DAYS[11], DAYS[31]]


def test_cusum_skips_nan_rows():
    df = _resid_frame()
    df.iloc[5] = np.nan
    scale = pd.Series(0.001, index=df.columns)
    assert cusum_alarms(df, scale, k=3.0, h=50.0) == []


def test_balance_cusum_detects_increase():
    x = pd.Series(10.0, index=DAYS)
    x.iloc[25:] = 12.0
    al = balance_cusum_alarms(x, scale=0.2, k=0.5, h=5.0, ref_days=7)
    assert al[0].time.normalize() == DAYS[26] and al[0].method == "balance_cusum" and al[0].area == "C"
    x2 = pd.Series(10.0, index=DAYS)
    x2.iloc[25:] = 8.0
    assert balance_cusum_alarms(x2, scale=0.2, k=0.5, h=5.0) == []


def test_robust_scale():
    df = pd.DataFrame({"a": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], "b": [0.0] * 7}, index=DAYS[:7])
    sc = robust_daily_scale(df)
    # day-to-day changes alternate +1/-1; median 0, MAD 1
    assert sc["a"] == pytest.approx(1.4826 / np.sqrt(2))
    assert sc["b"] == pytest.approx(1e-4)  # floor


def test_residual_model_recovers_linear_relation():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2018-01-01", periods=400, freq="h")
    a, b, c = rng.normal(size=(3, 400))
    X = pd.DataFrame({"a": a, "b": b, "c": 0.01 * c, "s": 2 * a - 3 * b + 1.0}, index=idx)
    m = ResidualModel(k=2).fit(X.iloc[:200], {"s": ["a", "b", "c"]}, cutoff=pd.Timestamp("2019-01-01"))
    assert set(m.features["s"]) == {"a", "b"}
    coef = dict(zip(m.features["s"], m.coefs["s"][:2]))
    assert coef["a"] == pytest.approx(2.0) and coef["b"] == pytest.approx(-3.0)
    assert m.coefs["s"][2] == pytest.approx(1.0)
    assert m.residuals(X)["s"].abs().max() < 1e-9
    X2 = X.copy()
    X2.loc[idx[300]:, "s"] -= 0.5
    assert m.residuals(X2)["s"].iloc[350] == pytest.approx(-0.5)


def test_residual_model_keeps_k_most_correlated():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2018-01-01", periods=300, freq="h")
    base = rng.normal(size=300)
    X = pd.DataFrame(
        {"near": base + 0.01 * rng.normal(size=300), "far": rng.normal(size=300), "s": base}, index=idx
    )
    m = ResidualModel(k=1).fit(X, {"s": ["far", "near"]}, cutoff=pd.Timestamp("2019-01-01"))
    assert m.features["s"] == ["near"]


def test_residual_model_refuses_evaluation_data():
    idx = pd.date_range("2018-12-30", periods=100, freq="h")
    X = pd.DataFrame({"a": np.arange(100.0), "s": np.arange(100.0)}, index=idx)
    with pytest.raises(ValueError):
        ResidualModel(k=1).fit(X, {"s": ["a"]}, cutoff=pd.Timestamp("2019-01-01"))


def test_assert_before():
    assert_before(pd.DatetimeIndex(["2018-12-31 23:00"]), pd.Timestamp("2019-01-01"), "x")
    with pytest.raises(ValueError):
        assert_before(pd.DatetimeIndex(["2019-01-01 00:00"]), pd.Timestamp("2019-01-01"), "x")


def test_sensor_groups():
    areas = {"s1": "A", "s2": "A", "s3": "B", "s4": "C", "s5": "C"}
    g = sensor_groups(list(areas), areas, {"C": ["T1"]})
    assert g["s1"] == ["s2"]
    assert g["s3"] == ["s1", "s2"]  # alone in B -> uses area A sensors
    assert g["s4"] == ["s5", "T1"]
