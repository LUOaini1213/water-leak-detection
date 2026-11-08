"""Build the daily input table used by all detectors from 5-min SCADA/AMR data."""
from __future__ import annotations

import pandas as pd

from .detectors import ResidualModel, sensor_groups
from .io import Leak
from .waterbalance import area_inflows, daily_mnf


def hourly_inputs(pressures: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
    """Hourly means of all pressure sensors plus the tank level (model inputs)."""
    h = pressures.resample("h").mean()
    for col in levels.columns:
        h[col] = levels[col].resample("h").mean()
    return h


def leak_free_end(train_leaks: list[Leak]) -> pd.Timestamp:
    """End (exclusive, hour-aligned) of the leak-free window at the start of the training year."""
    return min(lk.start for lk in train_leaks).floor("h")


def fit_residual_model(
    hourly: pd.DataFrame,
    sensors: list[str],
    areas: dict[str, str],
    train_end: pd.Timestamp,
    cutoff: pd.Timestamp,
    k: int = 6,
) -> ResidualModel:
    """Fit the per-sensor regression on hourly rows strictly before ``train_end``."""
    if train_end > cutoff:
        raise ValueError("residual-model training window extends past the cutoff")
    groups = sensor_groups(sensors, areas, {"C": ["T1"]})
    return ResidualModel(k=k).fit(hourly.loc[hourly.index < train_end], groups, cutoff)


def build_daily_inputs(
    flows: pd.DataFrame,
    levels: pd.DataFrame,
    amr_lph: pd.DataFrame,
    hourly: pd.DataFrame,
    model: ResidualModel,
    net,
    unmetered_ratio: float,
) -> pd.DataFrame:
    """One row per day: MNF per area, area-C balance and daily-mean pressure residuals."""
    inflow = area_inflows(flows, levels, net)
    amr = amr_lph.sum(axis=1) / 1000.0
    loss_c = inflow["C"] - amr * (1.0 + unmetered_ratio)
    daily = pd.DataFrame(
        {
            "mnf_AB": daily_mnf(inflow["AB"]),
            "mnf_C": daily_mnf(inflow["C"]),
            "loss_C": loss_c.resample("D").mean(),
        }
    )
    res = model.residuals(hourly).resample("D").mean()
    res.columns = [f"res_{c}" for c in res.columns]
    out = daily.join(res, how="inner")
    out.index.name = "day"
    return out
