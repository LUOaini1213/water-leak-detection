"""Python port of the official BattLeDIM 2020 scoring algorithm.

Ported from KIOS-Research/BattLeDIM "Scoring Algorithm" (MATLAB, EUPL-1.1/1.2):
Scoring_Algorithm.m, functions/scoring_function.m, functions/time2steps.m and
functions/nodeTopologicalDistance.m. Logic reproduced here:

* times are 5-minute steps since 2019-01-01 00:00; detections before 2019 are ignored;
* a detection (pipe, time) succeeds for leak j if t_start <= t <= t_end and the
  pipe distance x <= 300 m (closest end-node pair + half of both pipe lengths,
  shortest paths over pipes only);
* a successful first detection of a leak scores (remaining leak volume from the
  detection step) * 0.8 EUR/m3 - (x / 300) * 500 EUR; if several undetected leaks
  match, the closest one is credited;
* a detection that only matches already-detected leaks scores 0 (repeat);
* any other detection is a false positive and scores -500 EUR.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io import Leak
from .network import Network, link_distance, node_distances

T0 = pd.Timestamp("2019-01-01 00:00")


@dataclass(frozen=True)
class ScoreParams:
    xmax: float = 300.0  # metres
    cost_water: float = 0.80  # EUR per m3
    cost_crew: float = 500.0  # EUR per assignment


def time2steps(t: pd.Timestamp) -> int:
    return int(round((pd.Timestamp(t) - T0) / pd.Timedelta(minutes=5)))


def parse_submission(text: str) -> list[tuple[str, pd.Timestamp]]:
    """Read a BattLeDIM results file: '<linkID>, <yyyy-mm-dd HH:MM>' per line."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        link, when = line.split(",", 1)
        out.append((link.strip(), pd.Timestamp(when.strip())))
    return out


def active_leaks(leaks: list[Leak]) -> list[Leak]:
    """Leaks scored by BattLeDIM: those still running on or after 2019-01-01."""
    return [lk for lk in leaks if lk.end >= T0]


def score_detections(
    detections: list[tuple[str, pd.Timestamp]],
    leaks: list[Leak],
    leak_series: dict[str, np.ndarray],
    net: Network,
    dist: np.ndarray | None = None,
    idx: dict[str, int] | None = None,
    params: ScoreParams = ScoreParams(),
) -> dict:
    """Score detections exactly as the official MATLAB code does.

    ``leak_series[link]`` holds the 2019 leak outflow (m3/h) at 5-min steps,
    element 0 being 2019-01-01 00:00.
    """
    if dist is None or idx is None:
        dist, idx = node_distances(net)
    dets = [(link, time2steps(t)) for link, t in detections]
    dets = [d for d in dets if d[1] >= 0]
    dets.sort(key=lambda d: d[1])  # stable, like MATLAB sort
    t_start = [time2steps(lk.start) for lk in leaks]
    t_end = [time2steps(lk.end) for lk in leaks]

    detected: list[int] = []
    rows = []
    fp = 0
    for link, td in dets:
        x = np.array([link_distance(net, dist, idx, lk.link, link) for lk in leaks])
        ok = np.array([(td >= t_start[j]) and (td <= t_end[j]) and (abs(x[j]) <= params.xmax) for j in range(len(leaks))])
        if ok.any():
            new = [j for j in np.flatnonzero(ok) if j not in detected]
            if not new:
                rows.append({"link": link, "step": td, "leak": None, "score": 0.0, "outcome": "repeat"})
                continue
            j = min(new, key=lambda jj: x[jj])
            tdd = max(td, 1)
            water = float(np.sum(leak_series[leaks[j].link][tdd - 1 :]) / 12.0)
            s = water * params.cost_water - (x[j] / params.xmax) * params.cost_crew
            detected.append(j)
            rows.append(
                {
                    "link": link,
                    "step": td,
                    "leak": leaks[j].link,
                    "distance_m": float(x[j]),
                    "delay_steps": td - t_start[j],
                    "score": s,
                    "outcome": "TP",
                }
            )
        else:
            fp += 1
            rows.append({"link": link, "step": td, "leak": None, "score": -params.cost_crew, "outcome": "FP"})
    total = round(float(sum(r["score"] for r in rows)), 2)
    return {
        "total_score_eur": total,
        "true_positives": len(detected),
        "false_positives": fp,
        "false_negatives": len(leaks) - len(detected),
        "detections": rows,
    }


def candidate_pipe(net: Network, node: str) -> str:
    """Pipe used to report a sensor-level location: the shortest pipe attached to ``node``."""
    attached = [(length, pid) for pid, (a, b, length) in net.pipes.items() if node in (a, b)]
    return min(attached)[1]
