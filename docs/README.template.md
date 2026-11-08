# water-leak-detection

Reproducible leak detection, water-balance and metering analytics on the public, labelled **BattLeDIM 2020 (L-Town)** benchmark.

> **Personal project on public data.** This is a self-directed study project by LUOaini1213 (MSc Civil Engineering, Transport, NUS). It uses a simulated benchmark network published for research. It is not affiliated with, endorsed by or based on the methods or data of PUB or any other water utility, and the author has not worked for a water utility.

## Why this project

Water utilities track losses with a water balance, watch minimum night flow (MNF), and increasingly use smart meters. In Singapore, PUB's Smart Water Meter Programme moves from manual meter reads to readings transmitted automatically every day, with a customer portal that alerts users to suspected leaks ([PUB press release](https://www.pub.gov.sg/Resources/News-Room/PressReleases/2021/04/PUB-appoints-SP-Services-to-install-Singapore)). This repository practises the analyses that kind of data supports, on a benchmark where the true leaks are known so every claim can be checked:

- an IWA-style **water balance** and **MNF** analysis per supply area, compared with the labelled leak outflow;
- **leak detectors** compared against a simple MNF baseline, tuned on 2018 only and evaluated on the held-out year 2019;
- rough **sensor-level localisation** and the **official BattLeDIM score** (independent Python re-implementation, checked against the organisers' published reports);
- **AMR metering analytics**: data-quality checks, consumption profiles and night-use screening.

Every number in this README is rendered from a file in [`results/`](results/) written by [`scripts/run_pipeline.py`](scripts/run_pipeline.py); a test re-renders the README and fails if any number differs.

## The benchmark in one paragraph

L-Town has {{dataset_summary.json:n_junctions}} junctions and {{dataset_summary.json:n_pipes}} pipes, {{dataset_summary.json:n_pressure_sensors}} pressure sensors, {{dataset_summary.json:n_flow_sensors}} flow sensors and {{dataset_summary.json:n_amr}} AMR customer meters, all sampled every {{dataset_summary.json:sampling_minutes}} minutes ({{dataset_summary.json:samples_per_year}} samples per year, {{dataset_summary.json:missing_values}} missing values). Reading the network topology gives three supply areas: area A (fed from two reservoirs through flow-metered pipes p227 and p235; {{dataset_summary.json:pressure_sensors_per_area.A}} pressure sensors), area B (fed from A through a pressure-reducing valve; {{dataset_summary.json:pressure_sensors_per_area.B}} sensor) and area C (a hilly zone supplied only by tank T1, which PUMP_1 fills from area A; {{dataset_summary.json:pressure_sensors_per_area.C}} pressure sensors and AMR meters on {{water_balance.json:2019.area_C.n_metered_nodes}} of its {{water_balance.json:2019.area_C.n_demand_nodes}} consumer nodes). {{dataset_summary.json:leaks_starting_2018}} leaks start in the training year 2018 and {{dataset_summary.json:leaks_starting_2019}} new leaks start in the evaluation year 2019 ({{dataset_summary.json:events_2019_by_kind.abrupt}} abrupt bursts, {{dataset_summary.json:events_2019_by_kind.incipient}} slowly growing "incipient" leaks; {{dataset_summary.json:events_2019_by_area.AB}} in areas A/B, {{dataset_summary.json:events_2019_by_area.C}} in area C). Another {{dataset_summary.json:carry_over_into_2019}} leaks start in 2018 and are never repaired, so 2019 always has several leaks running.

## Results

### Water balance and minimum night flow

Area C is a closed balance: inflow = pump flow minus the change in tank storage, and most consumers are metered. The unmetered consumption of the remaining nodes is estimated from their share of nominal demand in the network model ({{config_used.json:unmetered_ratio_C_pct|.1f}}% of metered consumption). Areas A and B have metered inflow but no customer meters in the dataset, so their consumption comes from the model's nominal demands (calibrated by the metered/nominal ratio measured in area C, {{water_balance.json:2019.area_AB.nominal_to_measured_factor|.3f}}); the dataset documentation states those nominal demands are within {{config_used.json:nominal_demand_band_pct|.0f}}% of actual, which gives the band shown. Rates are annual means. The "true" rows use the leak labels and are shown only for validation.

{{table:water_balance}}

- The area-C metered balance recovers the true leak outflow to within {{water_balance.json:2019.error_vs_truth_m3h.area_C|.2f}} m³/h in 2019 ({{water_balance.json:2018.error_vs_truth_m3h.area_C|.2f}} m³/h in 2018); the daily series tracks the labels closely (figure below). This is what full AMR coverage of a district buys.
- For areas A and B the MNF method (night leakage = MNF minus estimated legitimate night use, scaled to a day with a night-day factor of {{water_balance.json:2019.area_AB.night_day_factor_h|.1f}} h from the average zone pressure, leakage exponent N1 = {{config_used.json:ndf_exponent}}) is off by {{water_balance.json:2019.error_vs_truth_m3h.area_AB_mnf|.2f}} m³/h in 2019 and the top-down balance by {{water_balance.json:2019.error_vs_truth_m3h.area_AB_balance|.2f}} m³/h. Both are far closer than the stated uncertainty of the nominal demands, so on real data with less accurate consumption estimates the band, not the point estimate, is the honest answer.
- Apparent losses (meter under-registration, unauthorised use) cannot be separated: the dataset contains no information on them, so all water losses are treated as real losses.

![Area C daily losses](results/figures/area_c_balance.png)

### Leak detection, evaluation year 2019

Three detectors, all run causally on daily statistics; parameters chosen on 2018 only (grid of {{config_used.json:mnf_grid_size}} settings for the MNF baseline and {{config_used.json:cusum_grid_size}} for the pressure CUSUM, criterion: F1 on the 2018 leak onsets):

{{table:tuning}}

{{table:detection}}

How to read this table:

- **Detected / Missed / False alarms** use the event-matching rule in [Method](#method): an alarm counts for a leak if it falls between the leak's start and {{config_used.json:grace_days}} days after the leak reached full size. **Repeat alarms** fall inside the window of a leak that was already detected; like the official BattLeDIM scoring, they are neither hits nor false alarms, but an operator would still have to handle them.
- Because several leaks overlap for most of 2019, a time-only rule is lenient. The last column shows what the same number of alarms placed **at random** would score (mean of {{config_used.json:chance_simulations}} simulations). The **area-aware** columns additionally require the alarm to name the right supply area (A/B or C).
- The area-C balance CUSUM watches area C only and is scored against the {{detection_summary.json:n_events_area_C_2019}} area-C leaks; its parameters were not tuned (textbook k and h) because there are only {{detection_summary.json:methods.balance_cusum.train_2018.n_events}} area-C leak onsets in 2018 to tune on.

What the numbers say:

- The pressure-residual CUSUM detects {{detection_summary.json:methods.pressure_cusum.eval_2019.detected}} of {{detection_summary.json:methods.pressure_cusum.eval_2019.n_events}} leak onsets with a median delay of {{detection_summary.json:methods.pressure_cusum.eval_2019.median_delay_h|.1f}} h, against {{detection_summary.json:methods.mnf_threshold.eval_2019.detected}} of {{detection_summary.json:methods.mnf_threshold.eval_2019.n_events}} and {{detection_summary.json:methods.mnf_threshold.eval_2019.median_delay_h|.1f}} h for the MNF baseline. It is much faster on abrupt bursts (table below).
- It is **not** better on false alarms: it raises {{detection_summary.json:methods.pressure_cusum.eval_2019.false_positives}} false alarms and {{detection_summary.json:methods.pressure_cusum.eval_2019.duplicates}} repeat alarms ({{detection_summary.json:methods.pressure_cusum.eval_2019.n_alarms}} alarms in total), while the baseline raises {{detection_summary.json:methods.mnf_threshold.eval_2019.false_positives}} false alarms in {{detection_summary.json:methods.mnf_threshold.eval_2019.n_alarms}} alarms. Most repeat alarms come from incipient leaks that keep growing after the detector re-baselines.
- Compared with random alarms, the pressure CUSUM's advantage is in false alarms ({{detection_summary.json:methods.pressure_cusum.eval_2019.false_positives}} vs {{detection_summary.json:methods.pressure_cusum.chance_2019.mean_false_positives|.1f}} expected) and delay; its detection count ({{detection_summary.json:methods.pressure_cusum.eval_2019.detected}}) is only moderately above the {{detection_summary.json:methods.pressure_cusum.chance_2019.mean_detected|.1f}} that the same number of random alarms would score. The baseline's {{detection_summary.json:methods.mnf_threshold.eval_2019.detected}} detections compare with {{detection_summary.json:methods.mnf_threshold.chance_2019.mean_detected|.1f}} for random alarms.
- The area-C balance CUSUM catches {{detection_summary.json:methods.balance_cusum.eval_2019.detected}} of {{detection_summary.json:methods.balance_cusum.eval_2019.n_events}} area-C leaks (the abrupt one after {{detection_summary.json:methods.balance_cusum.abrupt_median_delay_h_2019|.1f}} h) but raises {{detection_summary.json:methods.balance_cusum.eval_2019.false_positives}} false alarms in 2019 against {{detection_summary.json:methods.balance_cusum.train_2018.false_positives}} in 2018: leak outflow depends on pressure, so the day-to-day noise of the balance grows with the amount of leakage, and a noise scale estimated in 2018 is too small for 2019.

{{table:by_kind}}

Per leak (delay in hours from leak start to the first matched alarm; "wrong area" marks hits that the area-aware rule rejects):

{{table:per_leak}}

![Alarms in 2019](results/figures/alarms_2019.png)

Training-year performance of the chosen settings (in-sample, for reference):

{{table:training}}

### Localisation (rough)

At each pressure-CUSUM alarm the sensors are ranked by their CUSUM statistic. For the {{localisation.json:n}} leaks the detector matched in 2019, how often is the sensor nearest to the true leak pipe (shortest path along pipes) among the top k of {{localisation.json:n_sensors}} sensors?

{{table:localisation}}

It works for abrupt bursts (nearest sensor ranked first for {{localisation.json:hit_at_1_abrupt}} of {{localisation.json:n_abrupt}}) and poorly for slowly growing leaks ({{localisation.json:hit_at_1_incipient}} of {{localisation.json:n_incipient}}), whose alarms are usually triggered by a different, more sensitive sensor. For the MNF baseline the only location is the supply area, which was right for {{localisation.json:mnf_area_correct}} of {{localisation.json:mnf_area_n}} detected leaks.

### Official BattLeDIM score

`src/leakdet/battledim.py` re-implements the competition's scoring rules (a hit needs the right time window **and** a reported pipe within {{config_used.json:battledim.xmax_m|.0f}} m; hits earn the value of the water saved at {{config_used.json:battledim.cost_water_eur_m3}} EUR/m³ minus a distance-weighted crew cost; false alarms cost {{config_used.json:battledim.cost_crew_eur|.0f}} EUR). Checked against the organisers' published reports, it reproduces their example totals exactly: {{scoring_port_validation.json:PERFECT_TEAM.port_total_eur|,.2f}} EUR for the perfect submission (official {{scoring_port_validation.json:PERFECT_TEAM.official_total_eur|,.2f}}) and {{scoring_port_validation.json:test_team_multiple.port_total_eur|,.2f}} EUR for their multi-detection example (official {{scoring_port_validation.json:test_team_multiple.official_total_eur|,.2f}}).

Submitting every 2019 pressure-CUSUM alarm with the shortest pipe attached to its top-ranked sensor scores **{{battledim_score.json:total_score_eur|,.0f}} EUR**, {{battledim_score.json:share_of_perfect_pct|.1f}}% of the perfect score: {{battledim_score.json:true_positives}} hits, {{battledim_score.json:false_positives}} false alarms and {{battledim_score.json:repeats}} repeats from {{battledim_score.json:n_submissions}} submissions, with {{battledim_score.json:false_negatives}} of {{dataset_summary.json:battledim_scored_leaks}} scored leaks never located. The leaderboard results are not comparable: competitors did not have the 2018 leak labels used here for tuning.

### Metering analytics (AMR, 2019)

What a metering team would check every day, on the {{metering_summary.json:n_meters}} AMR meters ({{metering_summary.json:meters_per_category.residential}} residential, {{metering_summary.json:meters_per_category.commercial}} commercial, {{metering_summary.json:meters_per_category.industrial}} industrial by dominant demand pattern in the network model). Full tables: [`results/metering_quality_2019.csv`](results/metering_quality_2019.csv), [`results/metering_night_use_2019.csv`](results/metering_night_use_2019.csv), [`results/metering_profiles_2019.csv`](results/metering_profiles_2019.csv).

- **Completeness:** {{metering_summary.json:index_gaps}} missing timestamps, {{metering_summary.json:missing_readings}} missing and {{metering_summary.json:negative_readings}} negative readings.
- **Zero readings:** median share {{metering_summary.json:median_zero_share_pct|.1f}}% per meter; highest {{metering_summary.json:max_zero_share_pct|.1f}}% on {{metering_summary.json:max_zero_share_meter}}, an industrial user with night-time shutdowns.
- **Stuck registers:** no meter repeats the same non-zero reading for {{config_used.json:metering.flat_hours|.0f}} h or more (longest run {{metering_summary.json:longest_flat_h|.2f}} h).
- **Spikes** (a reading above both neighbours by more than {{config_used.json:metering.spike_z|.0f}} robust standard deviations of the meter's reading-to-reading change): {{metering_summary.json:total_spikes}} on {{metering_summary.json:meters_with_spikes}} meters ({{list:metering_summary.json:spike_meters}}), all industrial. Their demand jumps between operating levels, so these are review items, not proven faults.
- **Replayed data:** {{metering_summary.json:meters_fully_identical_2018_2019}} of {{metering_summary.json:n_meters}} meters report exactly the same values in 2019 as in 2018 (mean {{metering_summary.json:identical_2018_2019_pct_mean|.1f}}% identical readings). In real operations that would be flagged as a data-pipeline fault; here it is how the benchmark was built, and it means consumption trends between the two years cannot be studied.
- **Night use:** median night-to-day flow ratio is {{metering_summary.json:night_ratio_median_by_category.residential|.2f}} for residential, {{metering_summary.json:night_ratio_median_by_category.commercial|.2f}} for commercial and {{metering_summary.json:night_ratio_median_by_category.industrial|.2f}} for industrial meters, so screening is done within each category. The classic "continuous flow" rule (no zero-flow hour in a day) fires for {{metering_summary.json:continuous_flow_meters}} of {{metering_summary.json:n_meters}} meters, which makes it useless on this synthetic demand. A within-category robust outlier test flags {{list:metering_summary.json:night_use_flagged}} for review; with no ground truth for customer-side leaks this is a screening list only.
- AMR consumption is {{metering_summary.json:amr_share_of_area_C_inflow_pct_2019|.1f}}% of the water entering area C in 2019; the rest is leakage and unmetered use.

## Method

**Areas and flows.** The `.inp` file is parsed directly (no hydraulic simulation). Removing pumps and valves splits the pipe graph into the supply areas. Inflow(A+B) = p227 + p235 - PUMP_1; inflow(C) = PUMP_1 - dV/dt, with V from the T1 level and the tank diameter.

**MNF.** Daily minimum of the hourly-mean inflow over the night window (from hour {{config_used.json:mnf_night_window_h.0}} to hour {{config_used.json:mnf_night_window_h.1}}); the decision for a day is available at hour {{config_used.json:mnf_decision_hour}}, while the pressure and balance detectors decide at the end of the day.

**Detectors.** All re-baseline after an alarm, using only unlabelled data that arrives after it.

- *MNF threshold (baseline):* alarm when today's MNF exceeds the median of the recent MNFs since the last alarm by more than delta (separately for A+B and C).
- *Pressure-residual CUSUM:* each pressure sensor is predicted by least squares from the {{config_used.json:residual_k_features}} most correlated sensors of its area (plus the T1 level in area C), fitted on the {{config_used.json:residual_train_hours}} hours ({{config_used.json:residual_train_days}} days) at the start of 2018 before the first training leak. Daily-mean residuals are standardised by a robust day-to-day noise scale estimated on 2018 and fed to a one-sided multivariate CUSUM (pressure drops only) against a sliding reference; an alarm fires when any sensor's statistic exceeds h.
- *Area-C balance CUSUM:* the same CUSUM on daily inflow minus metered and estimated unmetered consumption.

**Evaluation rule.** Events are leaks that start inside the evaluated year. Each has a window from its start to {{config_used.json:grace_days}} days after its peak time (abrupt leaks peak at once, incipient leaks grow until the peak time given in the dataset), cut at the repair time. Alarms are processed in time order; an alarm inside the window of an undetected event is a hit for the earliest such event, inside windows of detected events only it is a repeat, otherwise it is a false alarm. Delay = alarm time - leak start. The {{dataset_summary.json:carry_over_into_2019}} leaks that started in 2018 have no onset in 2019 and are excluded from this rule (the BattLeDIM score includes them).

**No test-set leakage.** Fitting and tuning go through `training_view`, which keeps only rows and leak labels before {{config_used.json:cutoff}}; `tune()` and the residual model refuse later data. `tests/test_leakage.py` checks that scrambling all 2019 data and labels leaves the tuned parameters unchanged, and that the guards raise.

**What could not be computed.** Authorised consumption in areas A and B (no customer meters), apparent losses (no meter-error data), infrastructure leakage index and economic level of leakage (no pipe ages, service connection counts or cost data beyond the competition's), and hydraulic-model-based localisation (EPANET is not run).

## Limitations

- **Simulated network.** Demands, leaks and sensors are synthetic and noise-light; the median day-to-day noise of the daily pressure residuals is {{config_used.json:pressure_scale_median_mm|.2f}} mm, which real sensors and models would not achieve. Results here are likely optimistic compared with what the same methods would achieve on a real network.
- **2019 was looked at during development.** The evaluation script was run on 2019 several times while the pipeline was being debugged. The detector design was then fixed, and every parameter is chosen by the 2018 criterion stored in [`results/tuning_2018.json`](results/tuning_2018.json), but this is not a strictly single-shot test.
- **Training labels.** Tuning uses the 2018 leak labels released after the competition; competitors only had a repair report. Scores are therefore not comparable with the BattLeDIM leaderboard.
- **Lenient time-only matching.** With many overlapping leaks, random alarms already score many hits; read the detection counts next to the random-alarm and area-aware columns.
- **Alarm volume.** The pressure detector raises {{detection_summary.json:methods.pressure_cusum.eval_2019.n_alarms}} alarms in 2019 for {{detection_summary.json:methods.pressure_cusum.eval_2019.n_events}} leak onsets. Grouping repeat alarms by sensor would reduce this but was not designed or tuned, so it is not claimed.
- **Daily granularity.** Decisions are made once a day (MNF in the morning, pressure and balance at midnight), so the delay depends on when in the day a leak starts; a leak starting just after a decision waits up to a day.
- **Localisation** is sensor-level only and weak for incipient leaks.

## How to run

Python 3.11. Commands from the repository root:

```bash
python -m venv .venv            # or: uv venv --python 3.11 .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python
python scripts/fetch_data.py    # downloads the CSV files from Zenodo, checks MD5 and SHA-256
python scripts/run_pipeline.py  # writes results/ (a few minutes)
python scripts/render_readme.py # README.md from docs/README.template.md + results/
python -m pytest                # offline tests on committed fixtures and results
python scripts/mutation_test.py # mutation testing of the test suite
```

## Repository layout

```
src/leakdet/      io, network (.inp parser, areas, distances), waterbalance, features,
                  detectors, train (tuning with training-only guard), evaluation,
                  battledim (official scoring rules), metering
scripts/          fetch_data, run_pipeline, figures, render_readme, mutation_test
results/          every number in this README (JSON/CSV) and the figures
tests/            pytest suite; fixtures/ holds small derived data for offline CI
data/manifest.json  dataset files, sizes, MD5 and SHA-256 (raw files are not committed)
docs/             README template and source list
```

## Quality checks

{{mutation_report.json:baseline.passed}} tests pass offline in CI. The test suite was itself mutation-tested: {{mutation_report.json:n_mutants}} deliberate bugs were injected one at a time into the source, and the suite caught {{mutation_report.json:killed}} of them ([`results/mutation_report.json`](results/mutation_report.json) lists each mutant and whether it survived).

## Data, licences and sources

- **Data:** *Dataset of BattLeDIM: Battle of the Leakage Detection and Isolation Methods*, Vrachimis et al., Zenodo, [DOI 10.5281/zenodo.4017659](https://doi.org/10.5281/zenodo.4017659), licence [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The raw files (about {{data_size.json:fetched_mb|.0f}} MB) are not committed; `scripts/fetch_data.py` downloads them and checks them against [`data/manifest.json`](data/manifest.json). The committed derived files in `results/` and `tests/fixtures/` (including a compressed copy of the network file `L-TOWN.inp`) are adaptations of that dataset under the same licence.
- **Scoring reference:** the official BattLeDIM scoring code and example reports, [KIOS-Research/BattLeDIM](https://github.com/KIOS-Research/BattLeDIM) (EUPL-1.2). [`src/leakdet/battledim.py`](src/leakdet/battledim.py) is a Python port of that MATLAB code, so it is a derivative work and is distributed under EUPL-1.2 (full text in [`LICENSES/EUPL-1.2.md`](LICENSES/EUPL-1.2.md)), as is the fixture of its example submissions and published totals, `tests/fixtures/battledim_reference.json`.
- **Code:** MIT licence, see [LICENSE](LICENSE), except `src/leakdet/battledim.py` and `tests/fixtures/battledim_reference.json`, which are EUPL-1.2 (see above).
- Full source list with retrieval dates: [`docs/sources.md`](docs/sources.md).
