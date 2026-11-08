"""Mutation-test the pytest suite: inject deliberate bugs one at a time.

For each mutant the repository (without data/raw and .venv) is copied to a
temporary directory, one exact source string is replaced, the mutated file is
byte-compiled, and the offline test suite is run. A mutant is *killed* when
the suite fails. The unmodified copy must pass first.

Writes results/mutation_report.json.
"""
from __future__ import annotations

import json
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COPY = ["src", "tests", "scripts", "results", "docs", ".github", "README.md", "LICENSE", "requirements.txt", "pyproject.toml", "data/manifest.json"]

# (id, file, original text, replacement, what the bug means)
MUTANTS = [
    ("M01", "src/leakdet/evaluation.py", "max(lk.start, lk.peak) + pd.Timedelta", "lk.start + pd.Timedelta",
     "incipient leak windows end a grace period after onset instead of after the peak"),
    ("M02", "src/leakdet/evaluation.py", "detected[new[0].link] = t", "detected[new[-1].link] = t",
     "credit the latest-starting leak instead of the earliest"),
    ("M03", "src/leakdet/evaluation.py", "dup += 1\n            labels.append(\"duplicate\")",
     "fp += 1\n            labels.append(\"FP\")", "repeat alarms counted as false alarms"),
    ("M04", "src/leakdet/evaluation.py", "return [lk for lk in leaks if start <= lk.start < end]",
     "return [lk for lk in leaks if lk.start < end]", "carry-over leaks counted as events"),
    ("M05", "src/leakdet/detectors.py", "z = sign * (row - mu) / sc", "z = (row - mu) / sc",
     "CUSUM ignores the direction of change"),
    ("M06", "src/leakdet/detectors.py", "S = np.maximum(0.0, S + z - k)", "S = np.maximum(0.0, S + z + k)",
     "CUSUM allowance added instead of subtracted"),
    ("M07", "src/leakdet/detectors.py",
     "            S = np.zeros(len(cols))\n            buf = []\n            continue",
     "            buf = []\n            continue", "CUSUM statistic not reset after an alarm"),
    ("M08", "src/leakdet/detectors.py", "if value - base > delta:", "if base - value > delta:",
     "MNF rule alarms on decreases"),
    ("M09", "src/leakdet/detectors.py", "base = float(np.median(ref[-ref_days:]))", "base = float(np.mean(ref[-ref_days:]))",
     "MNF reference uses the mean (not robust to outlier nights)"),
    ("M10", "src/leakdet/detectors.py", "* 1.4826 / np.sqrt(2)", "* 1.4826 * np.sqrt(2)",
     "wrong noise scale for daily residuals"),
    ("M11", "src/leakdet/detectors.py", "feats = list(ranked.index[: self.k])", "feats = list(ranked.index[-self.k :])",
     "regression uses the least correlated sensors"),
    ("M12", "src/leakdet/waterbalance.py", '"C": flows["PUMP_1"] - dv', '"C": flows["PUMP_1"] + dv',
     "tank storage change added to area-C inflow with the wrong sign"),
    ("M13", "src/leakdet/waterbalance.py", "return sel.groupby(sel.index.normalize()).min()",
     "return sel.groupby(sel.index.normalize()).mean()", "MNF uses the night mean instead of the minimum"),
    ("M14", "src/leakdet/waterbalance.py", "losses_c = siv_c - billed_c - unmetered_c_est", "losses_c = siv_c - billed_c",
     "unmetered consumption counted as losses"),
    ("M15", "src/leakdet/battledim.py", "leak_series[leaks[j].link][tdd - 1 :]", "leak_series[leaks[j].link][tdd:]",
     "off-by-one against MATLAB 1-based indexing"),
    ("M16", "src/leakdet/battledim.py", "- (x[j] / params.xmax) * params.cost_crew", "+ (x[j] / params.xmax) * params.cost_crew",
     "crew cost rewarded instead of charged"),
    ("M17", "src/leakdet/network.py", "return float(best + net.link_length(l1) / 2 + net.link_length(l2) / 2)",
     "return float(best)", "pipe half-lengths left out of the link distance"),
    ("M18", "src/leakdet/train.py", "return daily.loc[daily.index < cutoff], [lk for lk in leaks if lk.start < cutoff]",
     "return daily.loc[daily.index < cutoff + pd.Timedelta(days=365)], [lk for lk in leaks if lk.start < cutoff]",
     "LEAKAGE: training view includes evaluation-year rows"),
    ("M19", "src/leakdet/train.py", "return daily.loc[daily.index < cutoff], [lk for lk in leaks if lk.start < cutoff]",
     "return daily.loc[daily.index < cutoff], list(leaks)", "LEAKAGE: evaluation-year labels reach tuning"),
    ("M20", "src/leakdet/train.py", 'return (-row["f1"], row["false_positives"], delay)',
     'return (row["f1"], row["false_positives"], delay)', "tuning picks the worst F1"),
    ("M21", "src/leakdet/metering.py", "out[1:-1] = np.minimum(up_prev, up_next) > z * sigma",
     "out[1:-1] = np.maximum(up_prev, up_next) > z * sigma", "level shifts reported as spikes"),
    ("M22", "src/leakdet/metering.py", "same[1:] = (v[1:] == v[:-1]) & (v[1:] != 0) & ~np.isnan(v[1:])",
     "same[1:] = (v[1:] == v[:-1]) & ~np.isnan(v[1:])", "runs of zeros reported as stuck registers"),
    ("M23", "src/leakdet/features.py", "loss_c = inflow[\"C\"] - amr * (1.0 + unmetered_ratio)",
     "loss_c = inflow[\"C\"] - amr", "area-C balance ignores unmetered consumption"),
    ("M24", "src/leakdet/network.py", 'names[c] = "C"', 'names[c] = "A"', "tank zone not labelled as area C"),
]


def copy_repo(dest: Path) -> None:
    for item in COPY:
        src = ROOT / item
        dst = dest / item
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        else:
            shutil.copy2(src, dst)


def run_tests(where: Path, timeout: int = 900) -> tuple[int, str, float]:
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"],
        cwd=where,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return r.returncode, r.stdout[-2000:], time.time() - t0


def summary_counts(out: str) -> dict:
    counts = {}
    for n, what in re.findall(r"(\d+) (passed|failed|skipped|error)", out):
        counts[what] = int(n)
    return counts


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="leakdet_mut_"))
    try:
        base = tmp / "baseline"
        copy_repo(base)
        code, out, secs = run_tests(base)
        baseline = {"returncode": code, **summary_counts(out), "seconds": round(secs, 1)}
        print("baseline:", baseline, flush=True)
        if code != 0:
            print(out)
            print("unmodified suite fails; aborting")
            return 1
        rows = []
        for mid, rel, old, new, what in MUTANTS:
            text = (ROOT / rel).read_text(encoding="utf-8")
            occurrences = text.count(old)
            work = tmp / mid
            copy_repo(work)
            row = {"id": mid, "file": rel, "description": what, "occurrences": occurrences}
            if occurrences != 1:
                row.update(compiled=False, killed=None, note="pattern not found exactly once")
                rows.append(row)
                print(mid, "SKIPPED (pattern count", occurrences, ")", flush=True)
                continue
            (work / rel).write_text(text.replace(old, new), encoding="utf-8")
            try:
                py_compile.compile(str(work / rel), doraise=True)
                row["compiled"] = True
            except py_compile.PyCompileError as exc:
                row.update(compiled=False, killed=None, note=str(exc))
                rows.append(row)
                continue
            code, out, secs = run_tests(work)
            row["killed"] = code != 0
            row["seconds"] = round(secs, 1)
            failed = re.findall(r"^FAILED (\S+)", out, flags=re.M)
            row["first_failing_test"] = failed[0] if failed else None
            rows.append(row)
            print(mid, "killed" if row["killed"] else "SURVIVED", "-", what, flush=True)
            shutil.rmtree(work, ignore_errors=True)
        valid = [r for r in rows if r.get("compiled")]
        report = {
            "baseline": baseline,
            "n_mutants": len(valid),
            "killed": sum(bool(r["killed"]) for r in valid),
            "survived": [r["id"] for r in valid if not r["killed"]],
            "mutants": rows,
        }
        (ROOT / "results" / "mutation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"killed {report['killed']} of {report['n_mutants']}; survived: {report['survived']}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
