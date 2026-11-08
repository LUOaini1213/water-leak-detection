import numpy as np
import pandas as pd
import pytest

from leakdet.metering import (
    hourly_profiles,
    identical_share,
    isolated_spikes,
    longest_run,
    night_use,
    node_category,
    quality_checks,
    runs_at_least,
)


def _readings(days=3, seed=0):
    idx = pd.date_range("2019-01-07", periods=days * 288, freq="5min")  # starts on a Monday
    rng = np.random.default_rng(seed)
    base = 100 + 50 * np.sin(2 * np.pi * idx.hour / 24)
    df = pd.DataFrame({m: base + rng.normal(0, 5, len(idx)) for m in ("m1", "m2", "m3")}, index=idx)
    return df


def test_run_helpers():
    m = np.array([1, 1, 0, 1, 1, 1, 0, 1], dtype=bool)
    assert longest_run(m) == 3
    assert runs_at_least(m, 2) == 2
    assert runs_at_least(m, 3) == 1
    assert runs_at_least(np.array([1, 1, 1], dtype=bool), 3) == 1  # run at the end counts


def test_isolated_spikes_vs_level_shift():
    v = np.r_[np.full(50, 10.0), np.full(50, 500.0)] + np.tile([0.0, 1.0], 50)
    assert isolated_spikes(v, z=20).sum() == 0  # a sustained jump is not a spike
    v[20] = 400.0
    mask = isolated_spikes(v, z=20)
    assert mask.sum() == 1 and mask[20]


def test_quality_checks_counts():
    df = _readings()
    df.loc[df.index[10], "m1"] = np.nan
    df.loc[df.index[20], "m1"] = -3.0
    df.loc[df.index[100], "m2"] = 5000.0  # spike
    df.loc[df.index[300]:df.index[300 + 84], "m3"] = 77.0  # 85 identical readings ~ 7 h
    df.loc[df.index[600]:df.index[611], "m3"] = 0.0
    df.loc[df.index[450]:df.index[549], "m2"] = 0.0  # 100 zeros (~8 h): no use, not a stuck register
    df = df.drop(df.index[[400, 401]])  # two missing timestamps
    q = quality_checks(df, flat_hours=6.0, spike_z=20.0)
    assert q.attrs["index_gaps"] == 2
    assert q.loc["m1", "missing"] == 3 and q.loc["m2", "missing"] == 2
    assert q.loc["m1", "negative"] == 1
    assert q.loc["m2", "spikes"] == 1 and q.loc["m1", "spikes"] == 0
    assert q.loc["m3", "flat_runs"] == 1 and q.loc["m1", "flat_runs"] == 0
    assert q.loc["m2", "flat_runs"] == 0 and q.loc["m2", "longest_flat_h"] < 1.0
    assert q.loc["m2", "zero_share"] == pytest.approx(100 / (3 * 288))
    assert q.loc["m3", "longest_flat_h"] == pytest.approx(85 * 5 / 60)
    assert q.loc["m3", "zero_share"] == pytest.approx(12 / (3 * 288))


def test_identical_share():
    a = _readings(seed=1)
    b = a.copy()
    b.iloc[: len(b) // 4, 0] += 1.0
    s = identical_share(a, b)
    assert s["m1"] == pytest.approx(0.75)
    assert s["m2"] == 1.0


def test_hourly_profiles_shape_and_units():
    df = _readings(days=7)
    prof = hourly_profiles(df, {"m1": "residential", "m2": "residential", "m3": "commercial"})
    assert set(prof["category"]) == {"residential", "commercial"}
    assert set(prof["day_type"]) == {"weekday", "weekend"}
    wk = prof[(prof["category"] == "residential") & (prof["day_type"] == "weekday")].set_index("hour")["m3h"]
    assert len(wk) == 24
    assert wk.loc[6] == pytest.approx(2 * 0.150, rel=0.05)  # two meters at ~150 L/h at 06:00


def test_night_use_flags_within_category():
    idx = pd.date_range("2019-01-01", periods=10 * 288, freq="5min")
    day = np.where((idx.hour >= 7) & (idx.hour < 22), 100.0, 0.0) + 1.0
    cols = {f"r{i}": day + i * 0.1 for i in range(8)}
    cols["leaky"] = day + 60.0  # constant extra flow: high night ratio
    cols["shop1"] = np.full(len(idx), 50.0)  # commercial with flat use
    cols["shop2"] = np.full(len(idx), 52.0)
    df = pd.DataFrame(cols, index=idx)
    cat = {c: "residential" for c in cols} | {"shop1": "commercial", "shop2": "commercial"}
    nu = night_use(df, cat)
    assert nu.loc["leaky", "flagged"]
    assert not nu.loc["shop1", "flagged"] and not nu.loc["r3", "flagged"]
    assert nu.loc["shop1", "night_ratio"] == pytest.approx(1.0)
    assert nu["continuous_share"].min() == 1.0


def test_node_category():
    cat = node_category({"n1": {"P-Industrial": 0.6}, "n2": {"P-Residential": 0.1, "P-Commercial": 0.3}, "n3": {}})
    assert cat == {"n1": "industrial", "n2": "commercial"}
