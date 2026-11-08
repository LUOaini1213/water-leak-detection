# Sources

All retrieved on 2026-11-06.

## Data

| What | Where | Licence | Notes |
|---|---|---|---|
| Dataset of BattLeDIM: Battle of the Leakage Detection and Isolation Methods (Vrachimis, Eliades, Taormina, Ostfeld, Kapelan, Liu, Kyriakou, Pavlou, Qiu, Polycarpou), Zenodo record 4017659, version v1, published 2020-09-07 | https://doi.org/10.5281/zenodo.4017659 (concept DOI 10.5281/zenodo.4017658), files via https://zenodo.org/records/4017659 | CC BY 4.0 | 14 of the 17 files are used; the two `.xlsx` copies and `L-TOWN_Real.inp` (the generating model, 177 MB) are not downloaded. File sizes, Zenodo MD5 and our SHA-256 are in `data/manifest.json`. |
| BattLeDIM problem description | https://doi.org/10.5281/zenodo.3902046 | as published | Documents the competition (linked from the dataset record as "isDocumentedBy"). |
| Official scoring algorithm, example submissions and reports (`Scoring Algorithm/`) | https://github.com/KIOS-Research/BattLeDIM | EUPL-1.2 | Ported to Python in `src/leakdet/battledim.py`, which is therefore distributed under EUPL-1.2 (`LICENSES/EUPL-1.2.md`). The example submissions `PERFECT_TEAM.txt`, `test_team.txt`, `test_team_multiple.txt` and the totals and per-detection scores printed in the matching `results_reports/*_report.txt` are stored in `tests/fixtures/battledim_reference.json` to validate the re-implementation. |

## Derived files committed in this repository

* `tests/fixtures/L-TOWN.inp.gz` - gzip copy of `L-TOWN.inp` from the dataset (CC BY 4.0, unchanged content).
* `tests/fixtures/dataset_configuration.yaml` - copy of the dataset's configuration file (CC BY 4.0).
* `tests/fixtures/hourly_train_window.csv.gz` - hourly means of the 33 pressure sensors and the T1 level for the leak-free window at the start of 2018 (derived from the dataset, CC BY 4.0).
* `results/daily_inputs.csv.gz` - daily MNF, area-C balance and pressure residuals for 2018-2019 (derived, CC BY 4.0).

## Singapore context (README "Why this project")

* PUB, "PUB appoints SP Services to install Singapore’s first 300,000 Smart Water Meters", press release, 15 April 2021: https://www.pub.gov.sg/Resources/News-Room/PressReleases/2021/04/PUB-appoints-SP-Services-to-install-Singapore . Used only for the statements that meter readings are to be transmitted automatically every day (replacing manual reads once every two months) and that customers receive alerts on suspected leaks. Nothing in this repository uses PUB data or methods.

## Methods referred to

* IWA standard water balance and the minimum-night-flow / night-day-factor approach to leakage estimation are general water-loss practice; the specific formulas used here are written out in the README "Method" section and in `src/leakdet/waterbalance.py`.
* CUSUM: Page, E. S. (1954), "Continuous inspection schemes", Biometrika 41(1/2), 100-115. The untuned area-C settings (k = 0.5, h = 5 in units of the noise scale) are the common textbook choice.
