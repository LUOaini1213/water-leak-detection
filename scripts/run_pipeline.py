"""End-to-end pipeline: raw BattLeDIM data -> committed results/ files.

Run ``python scripts/fetch_data.py`` first. Every number quoted in the README
is read from a file this script writes.

Stages
  1. dataset summary and network areas
  2. IWA-style water balance and MNF (2018 and 2019), validated against labels
  3. residual model fit on the leak-free start of 2018; daily input table
  4. detector tuning on 2018 only; one evaluation run on 2019
  5. sensor-level localisation and the official BattLeDIM score (Python port)
  6. AMR metering analytics
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from leakdet import battledim as bd  # noqa: E402
from leakdet.evaluation import (  # noqa: E402
    carry_over,
    chance_level,
    events_in,
    localisation_hits,
    match_alarms,
    match_by_area,
)
from leakdet.features import build_daily_inputs, fit_residual_model, hourly_inputs, leak_free_end  # noqa: E402
from leakdet.io import RAW, load_leak_config, load_leak_flows, load_year, parse_sensor_lists  # noqa: E402
from leakdet.metering import hourly_profiles, identical_share, night_use, node_category, quality_checks  # noqa: E402
from leakdet.network import hydraulic_areas, load_network, node_distances, nominal_demand  # noqa: E402
from leakdet.train import CUTOFF, TRAIN_START, run_all, training_view, tune  # noqa: E402
from leakdet.waterbalance import pattern_steps, true_leak_volumes, water_balance  # noqa: E402

RESULTS = ROOT / "results"
FIXTURES = ROOT / "tests" / "fixtures"
EVAL_END = pd.Timestamp("2020-01-01")
GZIP = {"method": "gzip", "mtime": 0}  # byte-stable compressed outputs
METERING = {"flat_hours": 6.0, "spike_z": 20.0, "night_window_h": (2, 4), "night_robust_z": 3.5}


def dump(name: str, obj) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(type(o))

    def clean(o):
        if isinstance(o, float) and np.isnan(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o

    (RESULTS / name).write_text(json.dumps(clean(obj), indent=2, default=default) + "\n", encoding="utf-8")


def area_of_link(net, areas, link):
    a = areas[net.pipes[link][0]]
    return "AB" if a in ("A", "B") else a


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "figures").mkdir(exist_ok=True)
    net = load_network(RAW / "L-TOWN.inp")
    areas = hydraulic_areas(net)
    cfg_text = (RAW / "dataset_configuration.yaml").read_text(encoding="utf-8")
    sensors_cfg = parse_sensor_lists(cfg_text)
    leaks = load_leak_config()

    data = {y: load_year(y) for y in (2018, 2019)}
    cat = lambda k: pd.concat([data[2018][k], data[2019][k]])  # noqa: E731
    pressures, flows, levels, amr = cat("pressures"), cat("flows"), cat("levels"), cat("demands")
    sensors = list(pressures.columns)
    leak_flows = {y: load_leak_flows(y) for y in (2018, 2019)}

    # ------------------------------------------------------------ 1. summary
    events19 = events_in(leaks, CUTOFF, EVAL_END)
    carry19 = carry_over(leaks, CUTOFF)
    demand_nodes = [n for n, d in net.base_demand.items() if d > 0]
    summary = {
        "n_junctions": len(net.junctions),
        "n_pipes": len(net.pipes),
        "n_pressure_sensors": len(sensors_cfg["pressure_sensors"]),
        "n_flow_sensors": len(sensors_cfg["flow_sensors"]),
        "n_amr": len(sensors_cfg["amrs"]),
        "n_demand_nodes": len(demand_nodes),
        "pressure_sensors_per_area": {a: sum(areas[s] == a for s in sensors) for a in ("A", "B", "C")},
        "demand_nodes_per_area": {a: sum(areas[n] == a for n in demand_nodes) for a in ("A", "B", "C")},
        "amr_in_area_C": sum(areas[n] == "C" for n in sensors_cfg["amrs"]),
        "leaks_total": len(leaks),
        "leaks_starting_2018": len(events_in(leaks, TRAIN_START, CUTOFF)),
        "leaks_starting_2019": len(events19),
        "carry_over_into_2019": len(carry19),
        "carry_over_links": [lk.link for lk in carry19],
        "battledim_scored_leaks": len(bd.active_leaks(leaks)),
        "events_2019_by_kind": {k: sum(lk.kind == k for lk in events19) for k in ("abrupt", "incipient")},
        "events_2019_by_area": {a: sum(area_of_link(net, areas, lk.link) == a for lk in events19) for a in ("AB", "C")},
        "samples_per_year": len(data[2019]["pressures"]),
        "sampling_minutes": int(pd.Series(data[2019]["pressures"].index).diff().median() / pd.Timedelta(minutes=1)),
        "missing_values": int(sum(int(df.isna().sum().sum()) for d in data.values() for df in d.values())),
    }
    dump("dataset_summary.json", summary)

    # -------------------------------------------------- 2. water balance + MNF
    wb = {}
    for y in (2018, 2019):
        d = data[y]
        est = water_balance(d["flows"], d["levels"], d["demands"], d["pressures"], net, areas)
        truth = true_leak_volumes(leak_flows[y], net, areas)
        est["truth"] = truth
        est["error_vs_truth_m3h"] = {
            "area_C": est["area_C"]["water_losses_m3h"] - truth["area_C"]["leak_mean_m3h"],
            "area_AB_balance": est["area_AB"]["water_losses_m3h"] - truth["area_AB"]["leak_mean_m3h"],
            "area_AB_mnf": est["area_AB"]["mnf_method_losses_m3h"] - truth["area_AB"]["leak_mean_m3h"],
        }
        total_losses = est["area_C"]["water_losses_m3"] + est["area_AB"]["water_losses_m3"]
        est["system"]["water_losses_estimate_m3h"] = total_losses / est["hours"]
        est["system"]["water_losses_estimate_pct"] = 100 * total_losses / est["system"]["system_input_m3"]
        est["system"]["true_leak_pct"] = 100 * truth["system"]["leak_volume_m3"] / est["system"]["system_input_m3"]
        wb[str(y)] = est
    dump("water_balance.json", wb)

    # ------------------------------------- 3. residual model and daily inputs
    hourly = hourly_inputs(pressures, levels)
    _, train_leaks = training_view(pd.DataFrame(index=pd.DatetimeIndex([])), leaks)
    lf_end = leak_free_end(train_leaks)
    model = fit_residual_model(hourly, sensors, areas, lf_end, CUTOFF)
    dump(
        "residual_model.json",
        {
            "train_start": hourly.index.min(),
            "train_end_exclusive": lf_end,
            "n_train_hours": int((hourly.index < lf_end).sum()),
            "k_features": model.k,
            "features": model.features,
            "coefficients": {s: [float(c) for c in v] for s, v in model.coefs.items()},
        },
    )
    FIXTURES.mkdir(parents=True, exist_ok=True)
    hourly.loc[hourly.index < lf_end].round(6).to_csv(
        FIXTURES / "hourly_train_window.csv.gz", float_format="%.6f", compression=GZIP
    )

    c_nodes = [n for n in demand_nodes if areas[n] == "C"]
    unmet = [n for n in c_nodes if n not in amr.columns]
    steps18 = pattern_steps(data[2018]["flows"].index)
    unmetered_ratio = float(
        nominal_demand(net, unmet, steps18).sum() / nominal_demand(net, list(amr.columns), steps18).sum()
    )
    daily = build_daily_inputs(flows, levels, amr, hourly, model, net, unmetered_ratio)
    daily.to_csv(RESULTS / "daily_inputs.csv.gz", float_format="%.6f", compression=GZIP)
    daily = pd.read_csv(RESULTS / "daily_inputs.csv.gz", index_col=0, parse_dates=True)  # identical to CI input

    # ------------------------------------------------- 4. tuning + evaluation
    sensor_area = {s: areas[s] for s in sensors}
    daily_train, train_leaks = training_view(daily, leaks)
    tuned = tune(daily_train, train_leaks, sensor_area)
    tuned["unmetered_ratio_C"] = unmetered_ratio
    dump("tuning_2018.json", tuned)

    alarms = run_all(daily, tuned, sensor_area)
    to_ab = {"A": "AB", "B": "AB", "AB": "AB", "C": "C"}
    for al in alarms.values():
        for a in al:
            a.area = to_ab.get(a.area, a.area)
    leak_area = {lk.link: area_of_link(net, areas, lk.link) for lk in leaks}
    det = {"grace_days": 7, "methods": {}}
    per_leak_rows = []
    alarm_rows = []
    c_events19 = [lk for lk in events19 if area_of_link(net, areas, lk.link) == "C"]
    c_events18 = [lk for lk in events_in(leaks, TRAIN_START, CUTOFF) if area_of_link(net, areas, lk.link) == "C"]
    strip = lambda r: {k: v for k, v in r.items() if k not in ("per_leak", "alarm_labels")}  # noqa: E731
    for method, al in alarms.items():
        times = [a.time for a in al]
        # the area-C balance only watches area C, so it is scored against area-C leaks only
        scope = [lk for lk in leaks if leak_area[lk.link] == "C"] if method == "balance_cusum" else leaks
        scope_train = [lk for lk in scope if lk.start < CUTOFF]
        r19 = match_alarms(times, scope, CUTOFF, EVAL_END)
        r18 = match_alarms(times, scope_train, TRAIN_START, CUTOFF)
        c_times = [a.time for a in al if a.area == "C"]
        rc19 = match_alarms(c_times, c_events19, CUTOFF, EVAL_END)
        rc18 = match_alarms(c_times, c_events18, TRAIN_START, CUTOFF)
        ra19 = match_by_area(al, scope, CUTOFF, EVAL_END, leak_area)
        ch = chance_level(r19["n_alarms"], scope, CUTOFF, EVAL_END, n_sim=2000, seed=0)
        cha = chance_level(r19["n_alarms"], scope, CUTOFF, EVAL_END, n_sim=2000, seed=0, area_of=leak_area)
        abrupt = r19["per_leak"][r19["per_leak"]["kind"] == "abrupt"]
        incip = r19["per_leak"][r19["per_leak"]["kind"] == "incipient"]
        det["methods"][method] = {
            "scope": "area C leaks only" if method == "balance_cusum" else "all leaks",
            "eval_2019": strip(r19),
            "eval_2019_area_aware": strip(ra19),
            "train_2018": strip(r18),
            "chance_2019": ch,
            "chance_2019_area_aware": cha,
            "area_C_2019": strip(rc19),
            "area_C_2018": strip(rc18),
            "abrupt_events_2019": int(len(abrupt)),
            "abrupt_detected_2019": int(abrupt["detected"].sum()),
            "abrupt_median_delay_h_2019": float(abrupt["delay_h"].median()) if abrupt["detected"].any() else None,
            "incipient_events_2019": int(len(incip)),
            "incipient_detected_2019": int(incip["detected"].sum()),
            "incipient_median_delay_h_2019": float(incip["delay_h"].median()) if incip["detected"].any() else None,
        }
        area_detected = dict(zip(ra19["per_leak"]["link"], ra19["per_leak"]["detected"]))
        for _, row in r19["per_leak"].iterrows():
            per_leak_rows.append({"method": method, **row.to_dict(), "detected_area_aware": area_detected[row["link"]]})
        labels = dict(r19["alarm_labels"] + r18["alarm_labels"])
        for a in al:
            alarm_rows.append(
                {
                    "method": method,
                    "time": a.time,
                    "area": a.area,
                    "top_sensors": " ".join(a.ranking[:3]),
                    "stat": a.stat,
                    "label": labels.get(a.time, ""),
                }
            )
    det["n_events_2019"] = len(events19)
    det["n_events_area_C_2019"] = len(c_events19)
    dump("detection_summary.json", det)
    pl = pd.DataFrame(per_leak_rows)
    pl["area"] = [area_of_link(net, areas, lk) for lk in pl["link"]]
    pl.to_csv(RESULTS / "detection_2019_per_leak.csv", index=False, float_format="%.2f")
    pd.DataFrame(alarm_rows).to_csv(RESULTS / "alarms.csv", index=False, float_format="%.4f")

    # ------------------------------------- 5. localisation + BattLeDIM score
    dist_s, idx = node_distances(net, sensors)
    r19 = match_alarms([a.time for a in alarms["pressure_cusum"]], leaks, CUTOFF, EVAL_END)
    by_time = {a.time: a for a in alarms["pressure_cusum"]}
    matched = [(row.link, by_time[row.alarm_time]) for row in r19["per_leak"].itertuples() if row.detected]
    loc = localisation_hits(matched, sensors, dist_s, idx, net)
    loc["per_leak"].to_csv(RESULTS / "localisation_2019.csv", index=False)
    loc_out = {k: v for k, v in loc.items() if k != "per_leak"}
    kinds = {lk.link: lk.kind for lk in leaks}
    lp = loc["per_leak"]
    for kind in ("abrupt", "incipient"):
        sub = lp[[kinds[x] == kind for x in lp["link"]]]
        loc_out[f"n_{kind}"] = int(len(sub))
        loc_out[f"hit_at_1_{kind}"] = int((sub["rank"] == 1).sum())
    rm = match_alarms([a.time for a in alarms["mnf_threshold"]], leaks, CUTOFF, EVAL_END)
    mnf_by_time = {a.time: a for a in alarms["mnf_threshold"]}
    area_hits = [
        mnf_by_time[row.alarm_time].area == area_of_link(net, areas, row.link)
        for row in rm["per_leak"].itertuples()
        if row.detected
    ]
    loc_out["mnf_area_correct"] = int(sum(area_hits))
    loc_out["mnf_area_n"] = len(area_hits)
    dump("localisation.json", loc_out)

    series19 = {c: leak_flows[2019][c].to_numpy() for c in leak_flows[2019].columns}
    dist_all, idx_all = node_distances(net)
    scored = bd.active_leaks(leaks)
    subs = [(bd.candidate_pipe(net, a.ranking[0]), a.time) for a in alarms["pressure_cusum"] if a.time >= CUTOFF]
    sc = bd.score_detections(subs, scored, series19, net, dist_all, idx_all)
    perfect = [(lk.link, max(lk.start, CUTOFF + pd.Timedelta(minutes=5))) for lk in scored]
    sc_perfect = bd.score_detections(perfect, scored, series19, net, dist_all, idx_all)
    dump(
        "battledim_score.json",
        {
            "method": "pressure_cusum + nearest-pipe-to-top-sensor",
            "n_submissions": len(subs),
            "total_score_eur": sc["total_score_eur"],
            "true_positives": sc["true_positives"],
            "false_positives": sc["false_positives"],
            "false_negatives": sc["false_negatives"],
            "repeats": sum(r["outcome"] == "repeat" for r in sc["detections"]),
            "perfect_score_eur": sc_perfect["total_score_eur"],
            "share_of_perfect_pct": 100 * sc["total_score_eur"] / sc_perfect["total_score_eur"],
            "detections": sc["detections"],
        },
    )
    ref = json.loads((FIXTURES / "battledim_reference.json").read_text(encoding="utf-8"))
    port = {}
    for team, spec in ref["teams"].items():
        res = bd.score_detections(bd.parse_submission(spec["submission"]), scored, series19, net, dist_all, idx_all)
        port[team] = {
            "official_total_eur": spec["official_total_eur"],
            "port_total_eur": res["total_score_eur"],
            "official_tp_fp_fn": spec["official_tp_fp_fn"],
            "port_tp_fp_fn": [res["true_positives"], res["false_positives"], res["false_negatives"]],
            "official_detection_scores": spec["official_detection_scores"],
            "port_detection_scores": [round(r["score"], 4) for r in res["detections"]],
        }
    dump("scoring_port_validation.json", port)

    # ------------------------------------------------------------- 6. AMR
    amr19, amr18 = data[2019]["demands"], data[2018]["demands"]
    catmap = node_category(net.demands)
    q = quality_checks(amr19, flat_hours=METERING["flat_hours"], spike_z=METERING["spike_z"])
    q.insert(0, "category", [catmap.get(m, "unknown") for m in q.index])
    q.to_csv(RESULTS / "metering_quality_2019.csv", float_format="%.4f")
    nu = night_use(amr19, catmap, night=METERING["night_window_h"], robust_z=METERING["night_robust_z"])
    nu.to_csv(RESULTS / "metering_night_use_2019.csv", float_format="%.4f")
    prof = hourly_profiles(amr19, catmap)
    prof.to_csv(RESULTS / "metering_profiles_2019.csv", index=False, float_format="%.4f")
    same = identical_share(amr18, amr19)
    weekday = prof[prof["day_type"] == "weekday"]
    peak = weekday.loc[weekday.groupby("category")["m3h"].idxmax()].set_index("category")["hour"]
    dump(
        "metering_summary.json",
        {
            "n_meters": int(amr19.shape[1]),
            "meters_per_category": {k: int(v) for k, v in q["category"].value_counts().items()},
            "index_gaps": int(q.attrs["index_gaps"]),
            "missing_readings": int(q["missing"].sum()),
            "negative_readings": int(q["negative"].sum()),
            "median_zero_share_pct": float(100 * q["zero_share"].median()),
            "max_zero_share_pct": float(100 * q["zero_share"].max()),
            "max_zero_share_meter": str(q["zero_share"].idxmax()),
            "meters_with_flat_runs_6h": int((q["flat_runs"] > 0).sum()),
            "longest_flat_h": float(q["longest_flat_h"].max()),
            "meters_with_spikes": int((q["spikes"] > 0).sum()),
            "spike_meters": sorted(q.index[q["spikes"] > 0].tolist()),
            "spike_meter_categories": sorted(set(q.loc[q["spikes"] > 0, "category"])),
            "total_spikes": int(q["spikes"].sum()),
            "identical_2018_2019_pct_median": float(100 * same.median()),
            "identical_2018_2019_pct_mean": float(100 * same.mean()),
            "meters_fully_identical_2018_2019": int((same == 1.0).sum()),
            "night_ratio_median_by_category": {k: float(v) for k, v in nu.groupby("category")["night_ratio"].median().items()},
            "night_use_flagged": sorted(nu.index[nu["flagged"]].tolist()),
            "continuous_flow_meters": int((nu["continuous_share"] >= 0.99).sum()),
            "weekday_peak_hour_by_category": {k: int(v) for k, v in peak.items()},
            "amr_share_of_area_C_inflow_pct_2019": float(
                100 * wb["2019"]["area_C"]["billed_metered_m3"] / wb["2019"]["area_C"]["system_input_m3"]
            ),
        },
    )
    # ------------------------------------------------- method constants used
    from leakdet import detectors as dmod
    from leakdet import evaluation as emod
    from leakdet import train as tmod
    from leakdet import waterbalance as wmod

    dump(
        "config_used.json",
        {
            "mnf_night_window_h": list(wmod.NIGHT_HOURS),
            "mnf_decision_hour": int(dmod.MNF_DECISION_OFFSET / pd.Timedelta(hours=1)),
            "daily_decision_hour": int(dmod.DAILY_DECISION_OFFSET / pd.Timedelta(hours=1)),
            "ndf_exponent": wmod.NDF_EXPONENT,
            "nominal_demand_band_pct": 100 * wmod.NOMINAL_BAND,
            "grace_days": emod.GRACE_DAYS,
            "chance_simulations": 2000,
            "residual_k_features": model.k,
            "residual_train_hours": int((hourly.index < lf_end).sum()),
            "residual_train_days": round(int((hourly.index < lf_end).sum()) / 24, 1),
            "mnf_grid_size": tuned["mnf_threshold"]["n_grid"],
            "cusum_grid_size": tuned["pressure_cusum"]["n_grid"],
            "balance_params": tmod.BALANCE_PARAMS,
            "battledim": {"xmax_m": bd.ScoreParams().xmax, "cost_water_eur_m3": bd.ScoreParams().cost_water,
                          "cost_crew_eur": bd.ScoreParams().cost_crew},
            "metering": METERING,
            "unmetered_ratio_C_pct": 100 * unmetered_ratio,
            "cutoff": str(CUTOFF.date()),
            "pressure_scale_median_mm": 1000 * float(np.median(list(tuned["pressure_scale"].values()))),
        },
    )
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    dump(
        "data_size.json",
        {
            "fetched_files": len(manifest["files"]),
            "fetched_mb": sum(f["size"] for f in manifest["files"].values()) / 1e6,
            "not_fetched": manifest["not_fetched"],
        },
    )

    # ---------------------------------------------------------- 7. figures
    from figures import alarm_timeline, area_c_balance

    ev = pd.DataFrame({"start": [lk.start for lk in events19]})
    al_df = pd.read_csv(RESULTS / "alarms.csv", parse_dates=["time"])
    alarm_timeline(al_df, ev, RESULTS / "figures" / "alarms_2019.png")
    c_pipes = [p for p in leak_area if leak_area[p] == "C"]
    true_c = pd.concat([leak_flows[y][[p for p in c_pipes if p in leak_flows[y].columns]].sum(axis=1) for y in (2018, 2019)])
    est_c = daily["loss_C"]
    wbc = wb["2019"]["area_C"]
    title = f"Area C (tank-fed, {wbc['n_metered_nodes']} of {wbc['n_demand_nodes']} consumers on AMR): daily water losses"
    area_c_balance(est_c, true_c.resample("D").mean(), RESULTS / "figures" / "area_c_balance.png", title)
    print("pipeline finished; results written to results/")


if __name__ == "__main__":
    main()
