"""Training-period tuning and detector runs on daily inputs.

The *daily inputs* table has one row per day and these columns:
``mnf_AB``, ``mnf_C`` (minimum night flow, m3/h), ``loss_C`` (area-C daily
metered balance, m3/h) and ``res_<sensor>`` (daily-mean pressure residual, m).

Everything that is fitted or tuned goes through :func:`training_view`, which
keeps only rows and leak labels strictly before the evaluation cutoff.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .detectors import balance_cusum_alarms, cusum_alarms, mnf_threshold_alarms, robust_daily_scale
from .evaluation import Alarm, match_alarms
from .io import Leak

CUTOFF = pd.Timestamp("2019-01-01")  # first instant of the evaluation year
TRAIN_START = pd.Timestamp("2018-01-01")

MNF_GRID = {
    "delta_AB": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
    "delta_C": [2.0, 4.0, 6.0, 8.0, None],  # None = area C not monitored
    "ref": [(7, 7), (14, 14), (7, 3), (14, 3), (3, 1)],  # (ref_days, min_ref)
}
CUSUM_GRID = {
    "k": [0.5, 1.0, 2.0, 3.0, 5.0],
    "h": [5.0, 10.0, 20.0, 40.0, 80.0],
    "ref": [(7, 7), (14, 14), (7, 3), (14, 3), (3, 1)],
}
# Area-C balance CUSUM: textbook defaults, not tuned (only two area-C leaks per year)
BALANCE_PARAMS = {"k": 0.5, "h": 5.0, "ref_days": 7, "min_ref": 7}


def training_view(daily: pd.DataFrame, leaks: list[Leak], cutoff: pd.Timestamp = CUTOFF):
    """Rows and leak labels that are allowed for fitting and tuning."""
    return daily.loc[daily.index < cutoff], [lk for lk in leaks if lk.start < cutoff]


def _check_training(daily: pd.DataFrame, leaks: list[Leak], cutoff: pd.Timestamp) -> None:
    if len(daily) and daily.index.max() >= cutoff:
        raise ValueError("tuning data reaches into the evaluation period")
    late = [lk.link for lk in leaks if lk.start >= cutoff]
    if late:
        raise ValueError(f"evaluation-period leak labels passed to tuning: {late}")


def f1(r: dict) -> float:
    denom = 2 * r["detected"] + r["false_positives"] + r["missed"]
    return 2 * r["detected"] / denom if denom else 0.0


def _rank_key(row: dict):
    delay = row["median_delay_h"] if not np.isnan(row["median_delay_h"]) else np.inf
    return (-row["f1"], row["false_positives"], delay)


def residual_columns(daily: pd.DataFrame) -> list[str]:
    return [c for c in daily.columns if c.startswith("res_")]


def run_mnf(daily: pd.DataFrame, p: dict) -> list[Alarm]:
    alarms = []
    rd, mr = p["ref"]
    for area, key in (("AB", "delta_AB"), ("C", "delta_C")):
        if p[key] is None:
            continue
        alarms += mnf_threshold_alarms(daily[f"mnf_{area}"], p[key], rd, mr, area=area)
    return sorted(alarms, key=lambda a: a.time)


def run_cusum(daily: pd.DataFrame, p: dict, scale: pd.Series, area_of: dict[str, str]) -> list[Alarm]:
    cols = residual_columns(daily)
    frame = daily[cols].rename(columns=lambda c: c[4:])
    rd, mr = p["ref"]
    return cusum_alarms(frame, scale, p["k"], p["h"], rd, mr, area_of=area_of)


def run_balance(daily: pd.DataFrame, scale: float, p: dict = BALANCE_PARAMS) -> list[Alarm]:
    return balance_cusum_alarms(daily["loss_C"], scale, p["k"], p["h"], p["ref_days"], p["min_ref"])


def pressure_scale(daily_train: pd.DataFrame) -> pd.Series:
    frame = daily_train[residual_columns(daily_train)].rename(columns=lambda c: c[4:])
    return robust_daily_scale(frame)


def balance_scale(daily_train: pd.DataFrame) -> float:
    return float(robust_daily_scale(daily_train[["loss_C"]]).iloc[0])


def tune(
    daily_train: pd.DataFrame,
    train_leaks: list[Leak],
    area_of: dict[str, str],
    cutoff: pd.Timestamp = CUTOFF,
    start: pd.Timestamp = TRAIN_START,
) -> dict:
    """Grid-search both detectors on the training year; maximise event F1.

    Ties are broken by fewer false positives, then shorter median delay.
    """
    _check_training(daily_train, train_leaks, cutoff)
    out = {}
    rows = []
    for d_ab, d_c, ref in itertools.product(MNF_GRID["delta_AB"], MNF_GRID["delta_C"], MNF_GRID["ref"]):
        p = {"delta_AB": d_ab, "delta_C": d_c, "ref": ref}
        r = match_alarms([a.time for a in run_mnf(daily_train, p)], train_leaks, start, cutoff)
        rows.append({**p, **_summary(r)})
    rows.sort(key=_rank_key)
    out["mnf_threshold"] = {"params": {k: rows[0][k] for k in ("delta_AB", "delta_C", "ref")}, "top": rows[:5], "n_grid": len(rows)}

    scale = pressure_scale(daily_train)
    rows = []
    for k, h, ref in itertools.product(CUSUM_GRID["k"], CUSUM_GRID["h"], CUSUM_GRID["ref"]):
        p = {"k": k, "h": h, "ref": ref}
        r = match_alarms([a.time for a in run_cusum(daily_train, p, scale, area_of)], train_leaks, start, cutoff)
        rows.append({**p, **_summary(r)})
    rows.sort(key=_rank_key)
    out["pressure_cusum"] = {"params": {k: rows[0][k] for k in ("k", "h", "ref")}, "top": rows[:5], "n_grid": len(rows)}
    out["pressure_scale"] = {s: float(v) for s, v in scale.items()}
    out["balance_scale"] = balance_scale(daily_train)
    out["balance_params"] = dict(BALANCE_PARAMS)
    return out


def _summary(r: dict) -> dict:
    return {
        "f1": round(f1(r), 6),
        "detected": r["detected"],
        "missed": r["missed"],
        "false_positives": r["false_positives"],
        "duplicates": r["duplicates"],
        "n_alarms": r["n_alarms"],
        "median_delay_h": r["median_delay_h"],
    }


def run_all(daily: pd.DataFrame, tuned: dict, area_of: dict[str, str]) -> dict[str, list[Alarm]]:
    """Run the three detectors over the whole daily table with tuned parameters."""
    scale = pd.Series(tuned["pressure_scale"])
    return {
        "mnf_threshold": run_mnf(daily, tuned["mnf_threshold"]["params"]),
        "pressure_cusum": run_cusum(daily, tuned["pressure_cusum"]["params"], scale, area_of),
        "balance_cusum": run_balance(daily, tuned["balance_scale"], tuned["balance_params"]),
    }
