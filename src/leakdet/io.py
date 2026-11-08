"""Readers for the BattLeDIM 2020 files (semicolon separated, decimal comma)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
STEP_MIN = 5  # sampling interval of all BattLeDIM series (minutes)


def read_scada_csv(path: str | Path) -> pd.DataFrame:
    """Read one BattLeDIM CSV into a float DataFrame indexed by timestamp."""
    df = pd.read_csv(path, sep=";", decimal=",", index_col=0, parse_dates=[0])
    df.index.name = "timestamp"
    return df.astype(float)


def load_year(year: int, raw: Path = RAW) -> dict[str, pd.DataFrame]:
    """Load SCADA pressures, flows, tank levels and AMR demands for one year."""
    out = {}
    for kind in ("Pressures", "Flows", "Levels", "Demands"):
        out[kind.lower()] = read_scada_csv(raw / f"{year}_SCADA_{kind}.csv")
    return out


def load_leak_flows(year: int, raw: Path = RAW) -> pd.DataFrame:
    """Ground-truth leak outflow per pipe (m3/h). Evaluation use only for 2019."""
    return read_scada_csv(raw / f"{year}_Leakages.csv")


@dataclass(frozen=True)
class Leak:
    link: str
    start: pd.Timestamp
    end: pd.Timestamp
    diameter_m: float
    kind: str  # "abrupt" or "incipient"
    peak: pd.Timestamp


_LEAK_LINE = re.compile(
    r"^-\s*(p\d+),\s*([\d\- :]+),\s*([\d\- :]+),\s*([\d.eE\-]+),\s*(\w+),\s*([\d\- :]+)\s*$"
)


def parse_leak_config(text: str) -> list[Leak]:
    """Parse the ``leakages:`` block of dataset_configuration.yaml.

    The file is not strictly valid YAML (comma-separated rows), so it is parsed
    line by line.
    """
    leaks = []
    for line in text.splitlines():
        m = _LEAK_LINE.match(line.strip())
        if m:
            link, s, e, d, kind, p = m.groups()
            leaks.append(
                Leak(link, pd.Timestamp(s.strip()), pd.Timestamp(e.strip()), float(d), kind, pd.Timestamp(p.strip()))
            )
    return leaks


def load_leak_config(raw: Path = RAW) -> list[Leak]:
    return parse_leak_config((raw / "dataset_configuration.yaml").read_text(encoding="utf-8"))


def parse_sensor_lists(text: str) -> dict[str, list[str]]:
    """Return pressure/flow/level/amr sensor ID lists from the configuration file."""
    lists: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        head = re.match(r"^(pressure_sensors|flow_sensors|level_sensors|amrs):\s*$", line)
        if head:
            current = head.group(1)
            lists[current] = []
            continue
        item = re.match(r"^-\s*(\S+)\s*$", line)
        if current and item:
            lists[current].append(item.group(1))
        elif line.strip() and not line.startswith("-"):
            current = None
    return lists
