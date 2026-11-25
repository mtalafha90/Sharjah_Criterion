# Pipeline

These scripts build the data set and the model. Run them in order from the repository root; each one finds its input and output files from its own location.

1. **Scrape** (`01_scrape_iac_crescent_reports.py`): collect the reports from the ICOP pages at [astronomycenter.net](https://astronomycenter.net).
2. **Clean** (`02_clean_crescent_observations.py`): standardise the text and dates, assign the observed class and reject reports without a usable outcome.
3. **Build features** (`03_build_crescent_features.py`): geocode the sites and compute the geometry at local sunset; `03b_correct_moon_lag.py` then recomputes the lag time where the Moon had already set at sunset or no moonset was found.
4. **Train** (`04_train_two_stage_crescent_criterion.py`): fit the two logistic stages. Performance is estimated by five-fold cross-validation grouped by lunation (one ICOP monthly page), so reports from the same new moon are never in both the training and the test folds. The final models are fitted on all the modelling records: Stage 1 on the 4,470 records and Stage 2 on the 2,151 sightings among them.
5. **Thresholds** (`05_choose_thresholds_and_plot_roc.py`): ROC, threshold and calibration figures (Figs. 3-5 of the paper) and the threshold candidates.

External check: `compare_twostep_with_alrefay_rowwise.py` applies the saved models to the Alrefay tables.

```bash
python3 -m pip install -r requirements.txt
python3 pipeline/01_scrape_iac_crescent_reports.py      # needs internet access
python3 pipeline/02_clean_crescent_observations.py
python3 pipeline/03_build_crescent_features.py          # slow; see Known issues
python3 pipeline/03b_correct_moon_lag.py
python3 pipeline/04_train_two_stage_crescent_criterion.py
python3 pipeline/05_choose_thresholds_and_plot_roc.py
python3 pipeline/compare_twostep_with_alrefay_rowwise.py
```

Steps 02, 03b, 04, 05 and the Alrefay comparison reproduce the archived data, models and reports exactly. To reproduce the paper, skip steps 01 and 03. Step 01 needs `requests` and `beautifulsoup4`, which `requirements.txt` does not install, and it overwrites `data/icop/crescent_observations_raw.csv` with the reports now on the ICOP pages, so the record counts no longer match the paper. Step 03 cannot be run from this repository (see Known issues).

The outputs are described in [`data/`](../data/README.md) and [`models/`](../models/README.md); the figures go to `figures/`.

## Known issues

- **Monday reports are missing from the feature table.** `03_build_crescent_features.py` removes the weekday from the Arabic date before parsing it, but its list of weekday names has only one spelling of Monday (`الاثنين`), while the ICOP pages write `الإثنين`. The date of every Monday report therefore fails to parse, and these reports are skipped. This removed 981 of the 7,124 cleaned reports (734 of them first-crescent reports made after sunset). The lost reports have the same class shares as those kept and do not differ by year, region or latitude, so the loss is not related to visibility. The archived `crescent_features.csv` and every result derived from it do not include these reports.
- **Step 3 needs a module that is not in this repository.** `03_build_crescent_features.py` imports `moon_phases_core`, which is not included here, so step 3 cannot be rerun from this repository alone. Its output, `data/icop/crescent_features.csv`, is archived, and steps 3b to 5 start from it.
