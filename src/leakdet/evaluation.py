"""Detection-only event matching, chance level and localisation hit rates.

Event-matching rule (documented in the README):
* events are the leaks whose start time lies inside the evaluated period;
  leaks already running at the period start (carry-over leaks) have no onset to
  detect and are listed separately;
* each event has a matching window [start, min(end, max(start, peak) + grace)].
  For abrupt leaks peak == start, so the window is the first ``grace`` days; for
  incipient leaks it runs until ``grace`` days after the leak reached full size;
* alarms are processed in time order. An alarm inside the window of at least
  one undetected event is a true positive for the earliest-starting such event;
  an alarm inside windows of already-detected events only is a duplicate
  (neither TP nor FP, as in the BattLeDIM scoring); any other alarm is a false
  positive;
* detection delay = alarm time - leak start time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .io import Leak

GRACE_DAYS = 7


@dataclass
class Alarm:
    time: pd.Timestamp
    method: str
    area: str | None = None
    ranking: tuple[str, ...] = field(default_factory=tuple)  # sensors, most suspicious first
    stat: float = float("nan")


def events_in(leaks: list[Leak], start: pd.Timestamp, end: pd.Timestamp) -> list[Leak]:
    return [lk for lk in leaks if start <= lk.start < end]


def carry_over(leaks: list[Leak], start: pd.Timestamp) -> list[Leak]:
    return [lk for lk in leaks if lk.start < start <= lk.end]


def window_end(lk: Leak, grace_days: float = GRACE_DAYS) -> pd.Timestamp:
    return min(lk.end, max(lk.start, lk.peak) + pd.Timedelta(days=grace_days))


def match_alarms(
    alarm_times: list[pd.Timestamp],
    leaks: list[Leak],
    start: pd.Timestamp,
    end: pd.Timestamp,
    grace_days: float = GRACE_DAYS,
) -> dict:
    """Match alarms to leak onsets in [start, end). See module docstring."""
    events = sorted(events_in(leaks, start, end), key=lambda lk: lk.start)
    times = sorted(t for t in alarm_times if start <= t < end)
    detected: dict[str, pd.Timestamp] = {}
    tp = fp = dup = 0
    labels = []
    for t in times:
        cands = [lk for lk in events if lk.start <= t <= window_end(lk, grace_days)]
        new = [lk for lk in cands if lk.link not in detected]
        if new:
            detected[new[0].link] = t
            tp += 1
            labels.append("TP")
        elif cands:
            dup += 1
            labels.append("duplicate")
        else:
            fp += 1
            labels.append("FP")
    rows = []
    for lk in events:
        t = detected.get(lk.link)
        rows.append(
            {
                "link": lk.link,
                "kind": lk.kind,
                "start": lk.start,
                "window_end": window_end(lk, grace_days),
                "detected": t is not None,
                "alarm_time": t,
                "delay_h": (t - lk.start) / pd.Timedelta(hours=1) if t is not None else np.nan,
            }
        )
    per_leak = pd.DataFrame(rows)
    days = (end - start) / pd.Timedelta(days=1)
    delays = per_leak["delay_h"].dropna()
    return {
        "n_events": len(events),
        "detected": tp,
        "missed": len(events) - tp,
        "false_positives": fp,
        "duplicates": dup,
        "n_alarms": len(times),
        "fp_per_30d": fp / days * 30 if days > 0 else np.nan,
        "median_delay_h": float(delays.median()) if len(delays) else np.nan,
        "per_leak": per_leak,
        "alarm_labels": list(zip(times, labels)),
    }


def chance_level(
    n_alarms: int,
    leaks: list[Leak],
    start: pd.Timestamp,
    end: pd.Timestamp,
    n_sim: int = 2000,
    seed: int = 0,
    grace_days: float = GRACE_DAYS,
    area_of: dict[str, str] | None = None,
) -> dict:
    """TP/FP expected from ``n_alarms`` alarms placed uniformly at random in the period.

    With ``area_of`` (leak pipe -> area) each random alarm also names a random
    area, drawn in proportion to the number of leak onsets per area, and is
    scored with :func:`match_by_area`.
    """
    rng = np.random.default_rng(seed)
    span = (end - start).total_seconds()
    labels, weights = None, None
    if area_of is not None:
        ev = events_in(leaks, start, end)
        labels = sorted({area_of[lk.link] for lk in ev})
        weights = np.array([sum(area_of[lk.link] == lab for lk in ev) for lab in labels], dtype=float)
        weights /= weights.sum()
    tps, fps = [], []
    for _ in range(n_sim):
        offs = np.sort(rng.uniform(0, span, n_alarms))
        ts = [start + pd.Timedelta(seconds=float(o)) for o in offs]
        if labels is None:
            r = match_alarms(ts, leaks, start, end, grace_days)
        else:
            picks = rng.choice(len(labels), size=n_alarms, p=weights)
            al = [Alarm(t, "random", labels[i]) for t, i in zip(ts, picks)]
            r = match_by_area(al, leaks, start, end, area_of, grace_days)
        tps.append(r["detected"])
        fps.append(r["false_positives"])
    return {"n_alarms": n_alarms, "mean_detected": float(np.mean(tps)), "mean_false_positives": float(np.mean(fps))}


def nearest_sensor(link: str, sensors: list[str], dist_from_sensors: np.ndarray, idx: dict[str, int], net) -> str:
    """Pressure sensor with the shortest pipe distance to either end of ``link``."""
    a, b = net.link_nodes(link)
    d = np.minimum(dist_from_sensors[:, idx[a]], dist_from_sensors[:, idx[b]])
    return sensors[int(np.argmin(d))]


def localisation_hits(
    matched: list[tuple[str, Alarm]],
    sensors: list[str],
    dist_from_sensors: np.ndarray,
    idx: dict[str, int],
    net,
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict:
    """How often the true leak's nearest sensor is in the alarm's top-k ranking."""
    rows = []
    for link, alarm in matched:
        ns = nearest_sensor(link, sensors, dist_from_sensors, idx, net)
        rank = alarm.ranking.index(ns) + 1 if ns in alarm.ranking else None
        rows.append({"link": link, "nearest_sensor": ns, "rank": rank, "top1": alarm.ranking[0] if alarm.ranking else None})
    df = pd.DataFrame(rows)
    out = {"n": len(df), "n_sensors": len(sensors)}
    for k in ks:
        out[f"hit_at_{k}"] = int(sum(r is not None and r <= k for r in df["rank"])) if len(df) else 0
        out[f"chance_at_{k}"] = len(df) * k / len(sensors)
    out["per_leak"] = df
    return out


def match_by_area(
    alarms: list[Alarm],
    leaks: list[Leak],
    start: pd.Timestamp,
    end: pd.Timestamp,
    leak_area: dict[str, str],
    grace_days: float = GRACE_DAYS,
) -> dict:
    """Area-aware variant: an alarm can only detect leaks in the supply area it names.

    ``leak_area`` maps leak pipe -> area label; ``Alarm.area`` must use the same
    labels. Alarms without an area count as false positives.
    """
    labels = sorted({leak_area[lk.link] for lk in leaks} | {a.area for a in alarms if a.area})
    parts = []
    for lab in labels:
        sub_leaks = [lk for lk in leaks if leak_area[lk.link] == lab]
        sub_times = [a.time for a in alarms if a.area == lab]
        parts.append(match_alarms(sub_times, sub_leaks, start, end, grace_days))
    orphan = sum(1 for a in alarms if not a.area and start <= a.time < end)
    keys = ("n_events", "detected", "missed", "false_positives", "duplicates", "n_alarms")
    out = {k: sum(p[k] for p in parts) for k in keys}
    out["false_positives"] += orphan
    out["n_alarms"] += orphan
    per_leak = pd.concat([p["per_leak"] for p in parts if len(p["per_leak"])], ignore_index=True)
    per_leak = per_leak.sort_values("start").reset_index(drop=True)
    delays = per_leak["delay_h"].dropna()
    days = (end - start) / pd.Timedelta(days=1)
    out["fp_per_30d"] = out["false_positives"] / days * 30
    out["median_delay_h"] = float(delays.median()) if len(delays) else np.nan
    out["per_leak"] = per_leak
    return out
