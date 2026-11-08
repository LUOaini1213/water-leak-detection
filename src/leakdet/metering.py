"""Customer-meter (AMR) analytics: data-quality checks, consumption profiles, night use.

Input is a DataFrame of meter readings (L/h) on a regular 5-min index, one
column per meter. No ground truth for customer-side leaks or meter faults
exists in the benchmark, so the outputs are screening lists, not accuracy claims.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STEP = pd.Timedelta(minutes=5)


def longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of True values."""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def runs_at_least(mask: np.ndarray, n: int) -> int:
    """Number of runs of True values with length >= n."""
    count = cur = 0
    for v in mask:
        if v:
            cur += 1
        else:
            if cur >= n:
                count += 1
            cur = 0
    return count + (1 if cur >= n else 0)


def isolated_spikes(v: np.ndarray, z: float) -> np.ndarray:
    """Boolean mask of single-reading spikes.

    A reading is a spike when it exceeds *both* neighbours by more than ``z``
    robust standard deviations of the meter's reading-to-reading changes.
    Sustained level changes (e.g. an industrial shift starting) are not spikes.
    """
    d = np.diff(v)
    d = d[~np.isnan(d)]
    # changes of a stationary meter series are centred on zero: scale = 1.4826 * median |change|
    sigma = 1.4826 * np.median(np.abs(d)) if len(d) else 0.0
    sigma = max(sigma, 1e-9)
    out = np.zeros(len(v), dtype=bool)
    if len(v) >= 3:
        up_prev = v[1:-1] - v[:-2]
        up_next = v[1:-1] - v[2:]
        out[1:-1] = np.minimum(up_prev, up_next) > z * sigma
    return out


def quality_checks(
    readings: pd.DataFrame,
    flat_hours: float = 6.0,
    spike_z: float = 20.0,
) -> pd.DataFrame:
    """Per-meter data-quality indicators.

    * ``missing``: expected timestamps absent from the index plus NaN readings;
    * ``negative``: readings below zero;
    * ``zero_share``: share of zero readings;
    * ``longest_flat_h`` / ``flat_runs``: repeated identical non-zero readings
      (a stuck register) - longest run in hours and number of runs >= ``flat_hours``;
    * ``spikes``: isolated single-reading spikes (see :func:`isolated_spikes`).
    """
    full = pd.date_range(readings.index.min(), readings.index.max(), freq=STEP)
    gaps = len(full) - readings.index.nunique()
    x = readings.reindex(full)
    n_flat = int(round(pd.Timedelta(hours=flat_hours) / STEP))
    rows = []
    for col in x.columns:
        v = x[col].to_numpy(dtype=float)
        same = np.zeros(len(v), dtype=bool)
        same[1:] = (v[1:] == v[:-1]) & (v[1:] != 0) & ~np.isnan(v[1:])
        # a run of k equal successive differences means k+1 identical readings
        rows.append(
            {
                "meter": col,
                "missing": int(np.isnan(v).sum()),
                "negative": int((v < 0).sum()),
                "zero_share": float(np.mean(v == 0)),
                "longest_flat_h": (longest_run(same) + 1) * STEP / pd.Timedelta(hours=1) if same.any() else 0.0,
                "flat_runs": runs_at_least(same, n_flat - 1),
                "spikes": int(isolated_spikes(v, spike_z).sum()),
                "mean_lph": float(np.nanmean(v)),
                "max_lph": float(np.nanmax(v)),
            }
        )
    out = pd.DataFrame(rows).set_index("meter")
    out.attrs["index_gaps"] = int(gaps)
    return out


def identical_share(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """Per-meter share of readings identical between two equally long periods.

    A high share between two different years indicates replayed or duplicated data.
    """
    n = min(len(a), len(b))
    cols = [c for c in a.columns if c in b.columns]
    eq = a[cols].to_numpy()[:n] == b[cols].to_numpy()[:n]
    return pd.Series(eq.mean(axis=0), index=cols)


def hourly_profiles(readings: pd.DataFrame, category: dict[str, str]) -> pd.DataFrame:
    """Mean consumption (m3/h) by hour of day, for weekdays and weekends, per category."""
    hourly = readings.resample("h").mean() / 1000.0
    groups: dict[str, list[str]] = {}
    for m in hourly.columns:
        groups.setdefault(category.get(m, "unknown"), []).append(m)
    out = []
    for cat, members in groups.items():
        s = hourly[members].sum(axis=1)
        weekend = s.index.day_of_week >= 5
        for label, mask in (("weekday", ~weekend), ("weekend", weekend)):
            prof = s[mask].groupby(s[mask].index.hour).mean()
            out.append(pd.DataFrame({"category": cat, "day_type": label, "hour": prof.index, "m3h": prof.values}))
    return pd.concat(out, ignore_index=True)


def night_use(
    readings: pd.DataFrame,
    category: dict[str, str] | None = None,
    night: tuple[int, int] = (2, 4),
    robust_z: float = 3.5,
) -> pd.DataFrame:
    """Night-use screening per meter.

    ``night_ratio`` = median over days of (mean night flow / mean daily flow).
    ``continuous_share`` = share of days on which every hourly mean was > 0
    (no zero-consumption hour: the classic 'continuous flow' rule).
    The robust z-score of the night ratio is computed within each customer
    category (commercial users legitimately use more water at night than
    households); meters with z > ``robust_z`` are flagged for review.
    """
    hourly = readings.resample("h").mean()
    day = hourly.index.normalize()
    is_night = (hourly.index.hour >= night[0]) & (hourly.index.hour < night[1])
    night_mean = hourly[is_night].groupby(day[is_night]).mean()
    daily_mean = hourly.groupby(day).mean()
    ratio = (night_mean / daily_mean.replace(0, np.nan)).median()
    continuous = (hourly.groupby(day).min() > 0).mean()
    cats = pd.Series({m: (category or {}).get(m, "all") for m in ratio.index})
    z = pd.Series(0.0, index=ratio.index)
    for _, members in cats.groupby(cats):
        r = ratio[members.index]
        med = r.median()
        mad = (r - med).abs().median() * 1.4826
        z[members.index] = (r - med) / (mad if mad > 0 else 1.0)
    return pd.DataFrame(
        {
            "category": cats,
            "night_ratio": ratio,
            "continuous_share": continuous,
            "robust_z": z,
            "flagged": z > robust_z,
        }
    )


def node_category(demands: dict[str, dict[str, float]]) -> dict[str, str]:
    """Dominant demand pattern per node, e.g. 'P-Residential' -> 'residential'."""
    out = {}
    for node, d in demands.items():
        if not d or max(d.values()) <= 0:
            continue
        pat = max(d, key=d.get)
        out[node] = pat.replace("P-", "").lower() or "unknown"
    return out
