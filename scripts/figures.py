"""Static figures for the README (matplotlib, written to results/figures/)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# reference categorical slots 1-3 (validated all-pairs) and neutral inks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)


def alarm_timeline(alarms: pd.DataFrame, events: pd.DataFrame, out: Path) -> None:
    """2019 leak onsets (grey lines) and alarms per method (marker = outcome)."""
    methods = [("mnf_threshold", "MNF threshold (baseline)", BLUE),
               ("pressure_cusum", "Pressure residual CUSUM", ORANGE),
               ("balance_cusum", "Area-C balance CUSUM", AQUA)]
    fig, ax = plt.subplots(figsize=(9, 2.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for t in events["start"]:
        ax.axvline(pd.Timestamp(t), color=GRID, linewidth=1.2, zorder=0)
    shapes = {"TP": "o", "duplicate": "s", "FP": "x"}
    for y, (m, label, color) in enumerate(methods):
        sub = alarms[(alarms["method"] == m) & (alarms["time"] >= "2019-01-01")]
        for lab, marker in shapes.items():
            s = sub[sub["label"] == lab]
            if marker == "x":
                ax.scatter(pd.to_datetime(s["time"]), [y] * len(s), marker=marker, s=28, color=color, linewidths=1.2)
            else:
                ax.scatter(pd.to_datetime(s["time"]), [y] * len(s), marker=marker, s=28,
                           facecolors="none", edgecolors=color, linewidths=1.2)
    ax.set_yticks(range(len(methods)), [m[1] for m in methods], fontsize=8, color=INK)
    ax.set_ylim(-0.6, len(methods) - 0.4)
    handles = [plt.Line2D([], [], marker=mk, linestyle="", color=INK2, markerfacecolor="none" if mk != "x" else INK2,
                          label={"TP": "first alarm of a leak", "duplicate": "repeat alarm", "FP": "false alarm"}[k])
               for k, mk in shapes.items()]
    handles.append(plt.Line2D([], [], color=GRID, linewidth=1.2, label="leak onset"))
    ax.legend(handles=handles, fontsize=7, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              labelcolor=INK2)
    ax.set_title("Evaluation year 2019: alarms by detector", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def area_c_balance(daily_est: pd.Series, daily_true: pd.Series, out: Path, title: str) -> None:
    """Area-C water losses from the metered balance vs. the benchmark's true leak outflow."""
    fig, ax = plt.subplots(figsize=(9, 2.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.plot(daily_true.index, daily_true.values, color=BLUE, linewidth=2, label="true leak outflow (labels)")
    ax.plot(daily_est.index, daily_est.values, color=ORANGE, linewidth=1, label="estimated losses (inflow - metered - unmetered est.)")
    ax.set_ylabel("m3/h (daily mean)", fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False, loc="upper left", labelcolor=INK2)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
