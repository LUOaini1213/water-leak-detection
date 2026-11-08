"""Render README.md from docs/README.template.md and the files in results/.

Placeholders:
  {{file.json:dotted.path|fmt}}  value from a results JSON file, formatted with
                                 Python's format spec ``fmt`` (optional)
  {{list:file.json:dotted.path}} a JSON list joined with ", "
  {{table:name}}                 a markdown table built by one of TABLES below

``python scripts/render_readme.py --check`` exits non-zero if README.md is stale.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs" / "README.template.md"
README = ROOT / "README.md"
PLACEHOLDER = re.compile(r"\{\{(.+?)\}\}")


def _load(results: Path, name: str):
    return json.loads((results / name).read_text(encoding="utf-8"))


def _get(obj, path: str):
    for key in path.split("."):
        obj = obj[int(key)] if isinstance(obj, list) else obj[key]
    return obj


def _fmt(value, spec: str) -> str:
    if value is None:
        return "n/a"
    return format(value, spec) if spec else str(value)


def _md(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def _h(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.1f}"


METHOD_NAMES = {
    "mnf_threshold": "MNF threshold (baseline)",
    "pressure_cusum": "Pressure-residual CUSUM",
    "balance_cusum": "Area-C balance CUSUM",
}


def table_detection(results: Path) -> str:
    d = _load(results, "detection_summary.json")
    rows = []
    for m, v in d["methods"].items():
        e, a, c = v["eval_2019"], v["eval_2019_area_aware"], v["chance_2019"]
        rows.append(
            {
                "Method": METHOD_NAMES[m],
                "Scope": v["scope"],
                "Leak onsets": e["n_events"],
                "Detected": e["detected"],
                "Missed": e["missed"],
                "False alarms": e["false_positives"],
                "Repeat alarms": e["duplicates"],
                "All alarms": e["n_alarms"],
                "Median delay (h)": _h(e["median_delay_h"]),
                "Detected, area-aware": a["detected"],
                "False alarms, area-aware": a["false_positives"],
                "Random alarms: detected / false": f"{c['mean_detected']:.1f} / {c['mean_false_positives']:.1f}",
            }
        )
    return _md(pd.DataFrame(rows))


def table_by_kind(results: Path) -> str:
    d = _load(results, "detection_summary.json")
    rows = []
    for m in ("mnf_threshold", "pressure_cusum"):
        v = d["methods"][m]
        rows.append(
            {
                "Method": METHOD_NAMES[m],
                "Abrupt detected": f"{v['abrupt_detected_2019']} of {v['abrupt_events_2019']}",
                "Abrupt median delay (h)": _h(v["abrupt_median_delay_h_2019"]),
                "Incipient detected": f"{v['incipient_detected_2019']} of {v['incipient_events_2019']}",
                "Incipient median delay (h)": _h(v["incipient_median_delay_h_2019"]),
            }
        )
    return _md(pd.DataFrame(rows))


def table_training(results: Path) -> str:
    d = _load(results, "detection_summary.json")
    rows = []
    for m, v in d["methods"].items():
        e = v["train_2018"]
        rows.append(
            {
                "Method": METHOD_NAMES[m],
                "Leak onsets": e["n_events"],
                "Detected": e["detected"],
                "False alarms": e["false_positives"],
                "All alarms": e["n_alarms"],
                "Median delay (h)": _h(e["median_delay_h"]),
            }
        )
    return _md(pd.DataFrame(rows))


def table_per_leak(results: Path) -> str:
    pl = pd.read_csv(results / "detection_2019_per_leak.csv")
    out = []
    for link, g in pl.groupby("link", sort=False):
        g = g.set_index("method")
        first = g.iloc[0]

        def cell(m):
            if m not in g.index:
                return "-"
            r = g.loc[m]
            if not r["detected"]:
                return "missed"
            mark = "" if r["detected_area_aware"] else " (wrong area)"
            return f"{r['delay_h']:.1f}{mark}"

        out.append(
            {
                "Leak pipe": link,
                "Type": first["kind"],
                "Area": first["area"],
                "Start": str(first["start"])[:16],
                "MNF delay (h)": cell("mnf_threshold"),
                "Pressure delay (h)": cell("pressure_cusum"),
                "Balance delay (h)": cell("balance_cusum"),
            }
        )
    df = pd.DataFrame(out).sort_values("Start")
    return _md(df)


def table_water_balance(results: Path) -> str:
    wb = _load(results, "water_balance.json")
    rows = []
    spec = [
        ("System input volume, whole network (m3/h)", "system.system_input_m3h", None),
        ("Area C: system input (m3/h)", "area_C.system_input_m3", "h"),
        ("Area C: billed metered consumption, AMR (m3/h)", "area_C.billed_metered_m3", "h"),
        ("Area C: unmetered consumption estimate (m3/h)", "area_C.unbilled_unmetered_estimate_m3", "h"),
        ("Area C: water losses, estimated (m3/h)", "area_C.water_losses_m3h", None),
        ("Area C: true leak outflow from labels (m3/h)", "truth.area_C.leak_mean_m3h", None),
        ("Area C: water losses as % of input", "area_C.water_losses_pct_of_input", None),
        ("Area C: non-revenue water as % of input", "area_C.non_revenue_water_pct", None),
        ("Areas A+B: system input (m3/h)", "area_AB.system_input_m3", "h"),
        ("Areas A+B: losses by top-down balance (m3/h)", "area_AB.water_losses_m3h", None),
        ("Areas A+B: band for +/- nominal-demand error (m3/h)", None, "band"),
        ("Areas A+B: losses by MNF x night-day factor (m3/h)", "area_AB.mnf_method_losses_m3h", None),
        ("Areas A+B: true leak outflow from labels (m3/h)", "truth.area_AB.leak_mean_m3h", None),
        ("Whole network: estimated losses as % of input", "system.water_losses_estimate_pct", None),
        ("Whole network: true leak outflow as % of input", "system.true_leak_pct", None),
    ]
    for label, path, kind in spec:
        row = {"Quantity": label}
        for y in ("2018", "2019"):
            w = wb[y]
            if kind == "band":
                row[y] = f"{w['area_AB']['water_losses_low_m3h']:.1f} to {w['area_AB']['water_losses_high_m3h']:.1f}"
                continue
            v = _get(w, path)
            if kind == "h":
                v = v / w["hours"]
            row[y] = f"{v:.2f}"
        rows.append(row)
    return _md(pd.DataFrame(rows))


def table_localisation(results: Path) -> str:
    loc = _load(results, "localisation.json")
    rows = []
    ks = sorted(int(key[7:]) for key in loc if re.fullmatch(r"hit_at_\d+", key))
    for k in ks:
        rows.append(
            {
                "k": k,
                "Nearest sensor in top-k": f"{loc[f'hit_at_{k}']} of {loc['n']}",
                "Expected by chance": f"{loc[f'chance_at_{k}']:.1f}",
            }
        )
    return _md(pd.DataFrame(rows))


def table_tuning(results: Path) -> str:
    t = _load(results, "tuning_2018.json")
    m, p = t["mnf_threshold"]["params"], t["pressure_cusum"]["params"]
    rows = [
        {
            "Detector": METHOD_NAMES["mnf_threshold"],
            "Chosen on 2018": f"delta A+B = {m['delta_AB']:g} m3/h, delta C = {_fmt(m['delta_C'], 'g')} m3/h, "
            f"reference = median of the last {m['ref'][0]} daily MNFs, at least {m['ref'][1]} needed after an alarm",
        },
        {
            "Detector": METHOD_NAMES["pressure_cusum"],
            "Chosen on 2018": f"k = {p['k']:g}, h = {p['h']:g}, reference = median of the last {p['ref'][0]} days, "
            f"at least {p['ref'][1]} needed after an alarm",
        },
        {
            "Detector": METHOD_NAMES["balance_cusum"],
            "Chosen on 2018": "not tuned: k = {k:g}, h = {h:g}, reference = median of the last {ref_days} days, "
            "at least {min_ref} needed after an alarm".format(**t["balance_params"]),
        },
    ]
    return _md(pd.DataFrame(rows))


TABLES = {
    "detection": table_detection,
    "by_kind": table_by_kind,
    "training": table_training,
    "per_leak": table_per_leak,
    "water_balance": table_water_balance,
    "localisation": table_localisation,
    "tuning": table_tuning,
}


def resolve(expr: str, results: Path) -> str:
    expr = expr.strip()
    if expr.startswith("table:"):
        return TABLES[expr[6:]](results)
    if expr.startswith("list:"):
        file, _, path = expr[5:].partition(":")
        return ", ".join(str(v) for v in _get(_load(results, file), path))
    src, _, spec = expr.partition("|")
    file, _, path = src.partition(":")
    return _fmt(_get(_load(results, file.strip()), path.strip()), spec.strip())


def render(template: str, results: Path = ROOT / "results") -> str:
    return PLACEHOLDER.sub(lambda m: resolve(m.group(1), results), template)


def main() -> int:
    text = render(TEMPLATE.read_text(encoding="utf-8"))
    if "--check" in sys.argv:
        if README.read_text(encoding="utf-8") != text:
            print("README.md is stale: run python scripts/render_readme.py")
            return 1
        print("README.md is up to date")
        return 0
    README.write_text(text, encoding="utf-8")
    print("README.md rendered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
