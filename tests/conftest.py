from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
RESULTS = ROOT / "results"
RAW = ROOT / "data" / "raw"


def load_json(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def lnet():
    from leakdet.network import load_network

    return load_network(FIX / "L-TOWN.inp.gz")


@pytest.fixture(scope="session")
def leaks():
    from leakdet.io import parse_leak_config

    return parse_leak_config((FIX / "dataset_configuration.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def daily_inputs():
    return pd.read_csv(RESULTS / "daily_inputs.csv.gz", index_col=0, parse_dates=True)


def raw_available() -> bool:
    return (RAW / "2019_SCADA_Pressures.csv").exists() and (RAW / "2019_Leakages.csv").exists()


rawdata = pytest.mark.skipif(not raw_available(), reason="raw BattLeDIM files not downloaded")
