"""Checks that need the full BattLeDIM download (skipped in CI)."""
import json

import pandas as pd
import pytest

from conftest import FIX, RAW, RESULTS, load_json, rawdata


@rawdata
def test_manifest_hashes_match_download():
    import subprocess
    import sys

    from conftest import ROOT

    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_data.py"), "--verify"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@rawdata
def test_water_balance_recomputed(lnet):
    from leakdet.io import load_leak_flows, load_year
    from leakdet.network import hydraulic_areas
    from leakdet.waterbalance import true_leak_volumes, water_balance

    areas = hydraulic_areas(lnet)
    d = load_year(2019, RAW)
    wb = water_balance(d["flows"], d["levels"], d["demands"], d["pressures"], lnet, areas)
    ref = load_json("water_balance.json")["2019"]
    assert wb["area_C"]["water_losses_m3"] == pytest.approx(ref["area_C"]["water_losses_m3"])
    assert wb["area_AB"]["mnf_method_losses_m3h"] == pytest.approx(ref["area_AB"]["mnf_method_losses_m3h"])
    truth = true_leak_volumes(load_leak_flows(2019, RAW), lnet, areas)
    assert truth["system"]["leak_volume_m3"] == pytest.approx(ref["truth"]["system"]["leak_volume_m3"])


@rawdata
def test_scoring_port_on_raw_leak_series(lnet, leaks):
    from leakdet import battledim as bd
    from leakdet.io import load_leak_flows

    lf = load_leak_flows(2019, RAW)
    series = {c: lf[c].to_numpy() for c in lf.columns}
    ref = json.loads((FIX / "battledim_reference.json").read_text(encoding="utf-8"))
    for team, spec in ref["teams"].items():
        r = bd.score_detections(bd.parse_submission(spec["submission"]), bd.active_leaks(leaks), series, lnet)
        assert r["total_score_eur"] == pytest.approx(spec["official_total_eur"], abs=0.005), team


@rawdata
def test_daily_inputs_match_raw_mnf():
    from leakdet.io import load_year
    from leakdet.network import load_network
    from leakdet.waterbalance import area_inflows, daily_mnf

    net = load_network(RAW / "L-TOWN.inp")
    d = load_year(2018, RAW)
    mnf = daily_mnf(area_inflows(d["flows"], d["levels"], net)["AB"])
    daily = pd.read_csv(RESULTS / "daily_inputs.csv.gz", index_col=0, parse_dates=True)
    assert daily.loc[mnf.index, "mnf_AB"].to_numpy() == pytest.approx(mnf.to_numpy(), abs=1e-5)
