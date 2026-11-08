"""Leak detectors.

* ``mnf_threshold_alarms`` - baseline: daily minimum night flow exceeds its
  recent reference level by more than ``delta`` m3/h.
* ``ResidualModel`` + ``cusum_alarms`` - each pressure sensor is predicted from
  correlated sensors in the same supply area by least squares fitted on a
  training window; daily-mean residuals are monitored with a one-sided CUSUM.
* ``balance_cusum_alarms`` - area C only: CUSUM on the daily metered water
  balance (inflow minus AMR consumption).

All detectors are causal: a decision for day d uses data up to the end of the
data window that defines it. The reference level is a sliding median of the
last ``ref_days`` days observed since the last alarm; after an alarm at least
``min_ref`` new (unlabelled) days are needed before the next decision,
mimicking an operator who acknowledges an alarm and re-baselines.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .evaluation import Alarm

MNF_DECISION_OFFSET = pd.Timedelta(hours=6)  # night window ends at 06:00
DAILY_DECISION_OFFSET = pd.Timedelta(hours=24)  # daily mean known at midnight


def assert_before(index: pd.DatetimeIndex, cutoff: pd.Timestamp, what: str) -> None:
    """Guard against fitting on data from the evaluation period."""
    if len(index) and index.max() >= cutoff:
        raise ValueError(f"{what}: training data reaches {index.max()}, beyond cutoff {cutoff}")


# --------------------------------------------------------------------------- MNF
def mnf_threshold_alarms(
    mnf: pd.Series,
    delta: float,
    ref_days: int = 7,
    min_ref: int | None = None,
    area: str | None = None,
    method: str = "mnf_threshold",
) -> list[Alarm]:
    """Alarm when today's MNF exceeds the median of recent MNFs by more than ``delta``.

    The reference is the median of the last ``ref_days`` daily MNFs recorded
    since the last alarm; at least ``min_ref`` (default ``ref_days``) such days
    are needed before the detector can alarm again.
    """
    min_ref = ref_days if min_ref is None else min_ref
    alarms: list[Alarm] = []
    ref: list[float] = []
    for day, value in mnf.items():
        if np.isnan(value):
            continue
        if len(ref) >= min_ref:
            base = float(np.median(ref[-ref_days:]))
            if value - base > delta:
                alarms.append(Alarm(day + MNF_DECISION_OFFSET, method, area, stat=value - base))
                ref = []
                continue
        ref.append(float(value))
    return alarms


# ---------------------------------------------------------------- residual model
@dataclass
class ResidualModel:
    """Per-sensor linear model predicting each pressure from k correlated inputs."""

    k: int = 6
    features: dict[str, list[str]] = field(default_factory=dict)
    coefs: dict[str, np.ndarray] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, groups: dict[str, list[str]], cutoff: pd.Timestamp) -> "ResidualModel":
        """Fit on hourly data ``X`` (sensors + auxiliary inputs such as tank level).

        ``groups[s]`` lists the candidate inputs for sensor ``s``; the ``k`` with the
        largest absolute correlation on the training data are kept.
        """
        assert_before(X.index, cutoff, "ResidualModel.fit")
        corr = X.corr()
        for s, cands in groups.items():
            ranked = corr.loc[cands, s].abs().sort_values(ascending=False)
            feats = list(ranked.index[: self.k])
            A = np.column_stack([X[feats].to_numpy(), np.ones(len(X))])
            coef, *_ = np.linalg.lstsq(A, X[s].to_numpy(), rcond=None)
            self.features[s] = feats
            self.coefs[s] = coef
        return self

    def residuals(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for s, feats in self.features.items():
            A = np.column_stack([X[feats].to_numpy(), np.ones(len(X))])
            out[s] = X[s].to_numpy() - A @ self.coefs[s]
        return pd.DataFrame(out, index=X.index)


def sensor_groups(sensors: list[str], areas: dict[str, str], aux: dict[str, list[str]]) -> dict[str, list[str]]:
    """Candidate inputs per sensor: other sensors in the same area plus area inputs.

    A sensor alone in its area (area B in L-Town) uses the sensors of area A.
    """
    groups = {}
    for s in sensors:
        a = areas[s]
        same = [o for o in sensors if o != s and areas[o] == a]
        if not same:
            same = [o for o in sensors if areas[o] == "A"]
        groups[s] = same + aux.get(a, [])
    return groups


def robust_daily_scale(daily_resid: pd.DataFrame) -> pd.Series:
    """Per-sensor noise scale of daily residuals: MAD of day-to-day changes / sqrt(2)."""
    d = daily_resid.diff().dropna()
    mad = (d - d.median()).abs().median() * 1.4826 / np.sqrt(2)
    return mad.clip(lower=1e-4)


def cusum_alarms(
    daily: pd.DataFrame,
    scale: pd.Series,
    k: float,
    h: float,
    ref_days: int = 7,
    min_ref: int | None = None,
    sign: float = -1.0,
    method: str = "pressure_cusum",
    area_of: dict[str, str] | None = None,
    offset: pd.Timedelta = DAILY_DECISION_OFFSET,
) -> list[Alarm]:
    """One-sided multivariate CUSUM against a sliding, re-baselined reference.

    For every column: S = max(0, S + sign*(x - mu)/scale - k), where mu is the
    median of the last ``ref_days`` rows observed since the last alarm (at least
    ``min_ref`` rows, default ``ref_days``, are required before testing). With
    ``sign = -1`` a drop below the reference accumulates (pressure loss). An
    alarm is raised when any column exceeds ``h``; its ranking lists the columns
    by CUSUM value. Statistics and reference are reset after an alarm.
    """
    min_ref = ref_days if min_ref is None else min_ref
    cols = list(daily.columns)
    sc = scale.reindex(cols).to_numpy()
    values = daily.to_numpy()
    alarms: list[Alarm] = []
    buf: list[np.ndarray] = []
    S = np.zeros(len(cols))
    for i, day in enumerate(daily.index):
        row = values[i]
        if np.isnan(row).any():
            continue
        if len(buf) < min_ref:
            buf.append(row)
            continue
        mu = np.median(np.vstack(buf[-ref_days:]), axis=0)
        z = sign * (row - mu) / sc
        S = np.maximum(0.0, S + z - k)
        if S.max() > h:
            order = np.argsort(-S)
            ranking = tuple(cols[j] for j in order)
            area = area_of.get(ranking[0]) if area_of else None
            alarms.append(Alarm(day + offset, method, area, ranking, float(S.max())))
            S = np.zeros(len(cols))
            buf = []
            continue
        buf.append(row)
    return alarms


# ----------------------------------------------------------- metered balance (C)
def balance_cusum_alarms(
    daily_loss: pd.Series, scale: float, k: float, h: float, ref_days: int = 7, min_ref: int | None = None
) -> list[Alarm]:
    """CUSUM on the daily area-C balance (inflow - AMR); an increase means new losses."""
    df = daily_loss.to_frame("C")
    return cusum_alarms(
        df, pd.Series({"C": scale}), k, h, ref_days, min_ref, sign=+1.0, method="balance_cusum", area_of={"C": "C"}
    )
