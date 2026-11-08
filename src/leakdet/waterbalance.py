"""Area inflows, minimum night flow (MNF) and an IWA-style water balance.

L-Town supply structure (derived from the .inp topology, see network.hydraulic_areas):
* reservoirs R1/R2 feed the network through pipes p227 and p235 (flow sensors);
* PUMP_1 (flow sensor) lifts water from area A into tank T1 (level sensor);
* tank T1 alone supplies area C, where 82 of the 92 demand nodes carry AMR meters;
* area B is fed from area A through a PRV and has no inflow meter of its own.
Hence: inflow(A+B) = p227 + p235 - PUMP_1 and inflow(C) = PUMP_1 - dV(T1)/dt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .network import Network, nominal_demand, tank_area_m2

STEP_H = 5.0 / 60.0  # sampling interval in hours
PATTERN_ORIGIN = pd.Timestamp("2018-01-01 00:00")  # EPANET simulation start of the dataset
NIGHT_HOURS = (0, 6)  # MNF search window [start, end) in local hours
NDF_EXPONENT = 0.5  # leakage-pressure exponent N1 used for the night-day factor
NOMINAL_BAND = 0.10  # dataset README: nominal base demands within 10% of actual


def storage_inflow(levels: pd.Series, tank_area: float) -> pd.Series:
    """Rate of water going into tank storage (m3/h) from successive level readings."""
    return (levels.diff().fillna(0.0) * tank_area / STEP_H).rename("storage")


def area_inflows(flows: pd.DataFrame, levels: pd.DataFrame, net: Network, tank: str = "T1") -> pd.DataFrame:
    """Net inflow (consumption + losses) per supply area, m3/h at the native 5-min step."""
    dv = storage_inflow(levels[tank], tank_area_m2(net.tanks[tank]))
    source = flows["p227"] + flows["p235"]
    return pd.DataFrame(
        {
            "system": source - dv,
            "AB": source - flows["PUMP_1"],
            "C": flows["PUMP_1"] - dv,
        }
    )


def daily_mnf(q: pd.Series, night: tuple[int, int] = NIGHT_HOURS) -> pd.Series:
    """Minimum night flow: lowest hourly-mean flow within the night window, per day."""
    hourly = q.resample("h").mean()
    sel = hourly[(hourly.index.hour >= night[0]) & (hourly.index.hour < night[1])]
    return sel.groupby(sel.index.normalize()).min()


def daily_mnf_hour(q: pd.Series, night: tuple[int, int] = NIGHT_HOURS) -> pd.Series:
    """Timestamp (hour start) at which the daily MNF occurs."""
    hourly = q.resample("h").mean()
    sel = hourly[(hourly.index.hour >= night[0]) & (hourly.index.hour < night[1])]
    return sel.groupby(sel.index.normalize()).idxmin()


def pattern_steps(index: pd.DatetimeIndex) -> np.ndarray:
    """Pattern step (5-min) of each timestamp since the dataset's simulation start."""
    return np.asarray((index - PATTERN_ORIGIN) // pd.Timedelta(minutes=5), dtype=np.int64)


def leakage_ndf(pressure: pd.Series, night_hour: int, exponent: float = NDF_EXPONENT) -> float:
    """Night-day factor (hours/day): sum over 24 h of (P(h)/P(night))^N1.

    ``pressure`` is the average zone pressure (m). Uses the mean diurnal profile.
    N1 = 0.5 corresponds to fixed-area orifice leaks (FAVAD lower bound).
    """
    prof = pressure.groupby(pressure.index.hour).mean()
    return float(((prof / prof.loc[night_hour]) ** exponent).sum())


def water_balance(
    flows: pd.DataFrame,
    levels: pd.DataFrame,
    demands: pd.DataFrame,
    pressures: pd.DataFrame,
    net: Network,
    areas: dict[str, str],
) -> dict:
    """IWA-style annual balance for one year of SCADA + AMR data (no leak labels used).

    Returns volumes in m3 and mean rates in m3/h. Components that the data
    cannot support are reported as None with a reason.
    """
    hours = len(flows) * STEP_H
    inflow = area_inflows(flows, levels, net)
    area = tank_area_m2(net.tanks["T1"])
    d_storage = float((levels["T1"].iloc[-1] - levels["T1"].iloc[0]) * area)
    vol = lambda s: float(s.sum() * STEP_H)  # noqa: E731  (m3 from m3/h samples)

    steps = pattern_steps(flows.index)
    metered_nodes = list(demands.columns)
    c_nodes = [n for n, a in areas.items() if a == "C" and net.base_demand.get(n, 0) > 0]
    unmetered_c = [n for n in c_nodes if n not in metered_nodes]
    ab_nodes = [n for n, a in areas.items() if a in ("A", "B", "source") and net.base_demand.get(n, 0) > 0]

    amr = demands.sum(axis=1) / 1000.0  # L/h -> m3/h
    nominal_metered = nominal_demand(net, metered_nodes, steps)
    calib = float(amr.sum() / nominal_metered.sum())  # measured / nominal on metered nodes

    # ---- area C: closed balance (inflow metered, 82 of 92 consumers metered)
    siv_c = vol(flows["PUMP_1"]) - d_storage
    billed_c = vol(amr)
    nominal_unmetered_c = nominal_demand(net, unmetered_c, steps)
    unmetered_c_est = float(nominal_unmetered_c.sum() * STEP_H * calib)
    losses_c = siv_c - billed_c - unmetered_c_est

    # ---- area A+B: inflow metered, consumption only from the nominal model
    siv_ab = vol(inflow["AB"])
    nominal_ab = float(nominal_demand(net, ab_nodes, steps).sum() * STEP_H)
    cons_ab = nominal_ab * calib
    losses_ab = siv_ab - cons_ab
    band = NOMINAL_BAND * cons_ab

    # ---- MNF method for A+B: night leakage x night-day factor
    mnf_ab = daily_mnf(inflow["AB"])
    mnf_hour = daily_mnf_hour(inflow["AB"])
    hourly_nominal = pd.Series(nominal_demand(net, ab_nodes, steps), index=flows.index).resample("h").mean() * calib
    legit_night = hourly_nominal.reindex(mnf_hour.values).to_numpy()
    night_leak = mnf_ab.to_numpy() - legit_night
    a_sensors = [s for s in pressures.columns if areas.get(s) == "A"]
    zone_p = pressures[a_sensors].mean(axis=1)
    typical_hour = int(pd.Series(mnf_hour.dt.hour).mode().iloc[0])
    ndf = leakage_ndf(zone_p, typical_hour)
    mnf_losses_ab = float(np.nansum(night_leak) * ndf)  # m3 over the year

    system_siv = vol(flows["p227"] + flows["p235"]) - d_storage
    return {
        "hours": hours,
        "system": {
            "system_input_m3": system_siv,
            "system_input_m3h": system_siv / hours,
            "authorised_consumption_m3": None,
            "note": "areas A and B have no consumer meters in the dataset, so the network-wide "
            "balance relies on the nominal-demand estimate below",
        },
        "area_C": {
            "system_input_m3": siv_c,
            "billed_metered_m3": billed_c,
            "unbilled_unmetered_estimate_m3": unmetered_c_est,
            "n_demand_nodes": len(c_nodes),
            "n_metered_nodes": len([n for n in c_nodes if n in metered_nodes]),
            "water_losses_m3": losses_c,
            "water_losses_m3h": losses_c / hours,
            "water_losses_pct_of_input": 100 * losses_c / siv_c,
            "non_revenue_water_m3": siv_c - billed_c,
            "non_revenue_water_pct": 100 * (siv_c - billed_c) / siv_c,
            "apparent_losses_m3": None,
            "apparent_losses_note": "no meter-error or theft information in the dataset; "
            "all water losses are treated as real losses",
        },
        "area_AB": {
            "system_input_m3": siv_ab,
            "authorised_consumption_estimate_m3": cons_ab,
            "nominal_to_measured_factor": calib,
            "water_losses_m3": losses_ab,
            "water_losses_m3h": losses_ab / hours,
            "water_losses_low_m3h": (losses_ab - band) / hours,
            "water_losses_high_m3h": (losses_ab + band) / hours,
            "water_losses_pct_of_input": 100 * losses_ab / siv_ab,
            "mnf_mean_m3h": float(mnf_ab.mean()),
            "mnf_typical_hour": typical_hour,
            "night_leakage_mean_m3h": float(np.nanmean(night_leak)),
            "night_day_factor_h": ndf,
            "mnf_method_losses_m3h": mnf_losses_ab / hours,
        },
    }


def true_leak_volumes(leak_flows: pd.DataFrame, net: Network, areas: dict[str, str]) -> dict:
    """Ground-truth leak volume per area (m3 and mean m3/h) — for validation only."""
    hours = len(leak_flows) * STEP_H
    out = {}
    for name, members in (("area_C", ("C",)), ("area_AB", ("A", "B")), ("system", ("A", "B", "C"))):
        cols = [p for p in leak_flows.columns if areas[net.pipes[p][0]] in members]
        v = float(leak_flows[cols].sum(axis=1).sum() * STEP_H)
        out[name] = {"leak_volume_m3": v, "leak_mean_m3h": v / hours, "n_leaking_pipes": len(cols)}
    return out
