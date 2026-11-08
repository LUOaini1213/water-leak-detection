"""Internal consistency of the committed results files."""
import pandas as pd
import pytest

from conftest import RESULTS, ROOT, load_json


def test_detection_counts_add_up():
    det = load_json("detection_summary.json")
    pl = pd.read_csv(RESULTS / "detection_2019_per_leak.csv")
    for method, v in det["methods"].items():
        e = v["eval_2019"]
        assert e["detected"] + e["missed"] == e["n_events"]
        assert e["detected"] + e["false_positives"] + e["duplicates"] == e["n_alarms"]
        rows = pl[pl["method"] == method]
        assert len(rows) == e["n_events"]
        assert int(rows["detected"].sum()) == e["detected"]
        assert int(rows["detected_area_aware"].sum()) == v["eval_2019_area_aware"]["detected"]
        assert v["abrupt_detected_2019"] + v["incipient_detected_2019"] == e["detected"]
        if e["detected"]:
            assert rows["delay_h"].median() == pytest.approx(e["median_delay_h"], abs=0.01)
            assert (rows["delay_h"].dropna() >= 0).all()
    assert det["n_events_2019"] == load_json("dataset_summary.json")["leaks_starting_2019"]


def test_alarm_log_matches_summary():
    det = load_json("detection_summary.json")
    log = pd.read_csv(RESULTS / "alarms.csv", parse_dates=["time"])
    for method, v in det["methods"].items():
        a19 = log[(log["method"] == method) & (log["time"] >= "2019-01-01")]
        assert len(a19) == v["eval_2019"]["n_alarms"]
        assert (a19["label"] == "TP").sum() == v["eval_2019"]["detected"]
        assert (a19["label"] == "FP").sum() == v["eval_2019"]["false_positives"]


def test_water_balance_identities():
    wb = load_json("water_balance.json")
    for y, w in wb.items():
        c, ab = w["area_C"], w["area_AB"]
        assert c["water_losses_m3"] == pytest.approx(
            c["system_input_m3"] - c["billed_metered_m3"] - c["unbilled_unmetered_estimate_m3"]
        )
        assert c["water_losses_m3h"] == pytest.approx(c["water_losses_m3"] / w["hours"])
        assert ab["water_losses_m3"] == pytest.approx(ab["system_input_m3"] - ab["authorised_consumption_estimate_m3"])
        assert w["system"]["system_input_m3"] == pytest.approx(c["system_input_m3"] + ab["system_input_m3"], rel=1e-6)
        t = w["truth"]
        assert t["system"]["leak_volume_m3"] == pytest.approx(t["area_C"]["leak_volume_m3"] + t["area_AB"]["leak_volume_m3"])
        assert w["error_vs_truth_m3h"]["area_C"] == pytest.approx(c["water_losses_m3h"] - t["area_C"]["leak_mean_m3h"])
        assert w["hours"] == 8760.0


def test_localisation_and_battledim():
    loc = load_json("localisation.json")
    assert loc["hit_at_1"] <= loc["hit_at_3"] <= loc["hit_at_5"] <= loc["n"]
    per = pd.read_csv(RESULTS / "localisation_2019.csv")
    assert (per["rank"] <= 1).sum() == loc["hit_at_1"]
    bd = load_json("battledim_score.json")
    assert bd["true_positives"] + bd["false_positives"] + bd["repeats"] == bd["n_submissions"]
    assert bd["true_positives"] + bd["false_negatives"] == load_json("dataset_summary.json")["battledim_scored_leaks"]
    assert bd["total_score_eur"] == pytest.approx(sum(d["score"] for d in bd["detections"]), abs=0.01)
    assert bd["perfect_score_eur"] == pytest.approx(load_json("scoring_port_validation.json")["PERFECT_TEAM"]["official_total_eur"])


def test_metering_summary_matches_tables():
    m = load_json("metering_summary.json")
    q = pd.read_csv(RESULTS / "metering_quality_2019.csv", index_col=0)
    nu = pd.read_csv(RESULTS / "metering_night_use_2019.csv", index_col=0)
    assert len(q) == m["n_meters"] == len(nu)
    assert int(q["spikes"].sum()) == m["total_spikes"]
    assert sorted(nu.index[nu["flagged"]].tolist()) == m["night_use_flagged"]
    assert int((q["flat_runs"] > 0).sum()) == m["meters_with_flat_runs_6h"]


def test_figures_exist():
    for name in ("alarms_2019.png", "area_c_balance.png"):
        assert (RESULTS / "figures" / name).stat().st_size > 10_000


def test_no_absolute_paths_or_personal_strings_in_repo():
    bad = []
    for path in list((ROOT / "results").glob("*.json")) + list((ROOT / "results").glob("*.csv")) + [ROOT / "README.md"]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "C:\\" in text or "C:/" in text or "/Users/" in text:
            bad.append(path.name)
    assert bad == []
