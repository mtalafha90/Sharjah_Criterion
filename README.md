# Sharjah Criterion

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20755359.svg)](https://doi.org/10.5281/zenodo.20755359)

A two-stage empirical criterion for lunar crescent visibility, built from the reports of the International Crescent Observation Project (ICOP). Stage 1 gives the probability that the crescent is seen at all (P<sub>det</sub>); Stage 2 gives, for a sighting, the probability that it was seen with the naked eye (P<sub>eye</sub>). The criterion is checked against the published observations of Alrefay et al. (2018) and the HilalPy sighting database, and compared with classical visibility criteria.

This repository holds the data, the pipeline that builds the model, the final models, and the analysis scripts that reproduce every table and figure of the paper *Sharjah criterion for lunar crescent visibility* (submitted to *Astronomy and Computing*).

## Repository structure

```
Sharjah_Criterion/
├── pipeline/          build the data set and the model (steps 01-05), and compare it with the Alrefay tables
├── analysis/          reproduce the tables and figures of the paper
├── data/
│   ├── icop/          ICOP reports, their geometry at sunset, and the out-of-fold predictions
│   └── alrefay/       Alrefay et al. (2018) Tables I-VI and the row-by-row comparison
├── models/            final Stage 1 and Stage 2 models, their metadata, the training report and the thresholds
├── figures/           the figures of the paper
├── docs/              crescent visibility report for 17 May 2026 (Sharjah)
├── legacy/            outputs of the first version, not used by the current scripts
├── requirements.txt   package versions used
├── CITATION.cff       citation metadata
└── LICENSE
```

Every script finds its files from its own location, so it can be run from any folder.

## Data

`data/icop/`

| File | Written by | Records | Content |
|---|---|---|---|
| `crescent_observations_raw.csv` | `pipeline/01` | 7,154 | reports scraped from the ICOP pages |
| `crescent_observations_clean.csv` | `pipeline/02` | 7,124 | reports with a usable outcome and an observed class |
| `crescent_observations_rejected.csv` | `pipeline/02` | 30 | reports without a usable outcome |
| `crescent_features.csv` | `pipeline/03`, `03b` | 6,092 | geocoded reports with the geometry at local sunset |
| `two_stage_oof_predictions.csv` | `pipeline/04` | 4,470 | out-of-fold P<sub>det</sub> and P<sub>eye</sub> of the modelling records |
| `two_stage_stage1_predictions.csv` | `pipeline/04` | 4,470 | out-of-fold Stage 1 predictions |
| `two_stage_stage2_predictions.csv` | `pipeline/04` | 2,151 | out-of-fold Stage 2 predictions (sightings only) |

Observed classes: 0 = not seen, 1 = seen with optical aid only, 2 = seen with the naked eye.

`data/alrefay/`: Tables I-VI of Alrefay et al. (2018), transcribed from the scanned paper (see `data/alrefay/README.txt`, which also lists the values corrected in November 2026), and the row-by-row comparison written by `pipeline/compare_twostep_with_alrefay_rowwise.py` (425 usable records, 265 of them sightings).

## Models

`models/` holds the outputs of `pipeline/04` and `pipeline/05`:

- `stage1_detectability_model.joblib` and `stage2_naked_eye_model.joblib`: the final scikit-learn pipelines (median imputation, standardisation, logistic regression), fitted on all 4,470 modelling records with scikit-learn 1.9.1;
- `two_stage_model_meta.json`: predictors and thresholds of the two stages;
- `two_stage_crescent_criterion_report.json`: cross-validated scores and coefficients;
- `chosen_thresholds.json`: the threshold candidates of `pipeline/05`.

The predictors are evaluated at local sunset:

| Predictor | Description |
|---|---|
| `relative_altitude_deg` | ARCV: altitude of the Moon minus altitude of the Sun |
| `elongation_deg` | ARCL: angular separation of the Moon and the Sun |
| `moon_lag_minutes` | lag time: moonset minus sunset (negative if the Moon had already set) |

## Pipeline

1. **Scrape** (`01_scrape_iac_crescent_reports.py`): collect the reports from the ICOP pages at [astronomycenter.net](https://astronomycenter.net).
2. **Clean** (`02_clean_crescent_observations.py`): standardise the text and dates, assign the observed class and reject reports without a usable outcome.
3. **Build features** (`03_build_crescent_features.py`): geocode the sites and compute the geometry at local sunset; `03b_correct_moon_lag.py` then recomputes the lag time where the Moon had already set at sunset or no moonset was found.
4. **Train** (`04_train_two_stage_crescent_criterion.py`): fit the two logistic stages. Performance is estimated by five-fold cross-validation grouped by lunation (one ICOP monthly page), so reports from the same new moon are never in both the training and the test folds. The final models are fitted on all 4,470 modelling records.
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

Steps 02, 03b, 04, 05 and the Alrefay comparison reproduce the archived data, models and reports exactly.

## Analysis scripts

The scripts in `analysis/` start from the archived files in `data/` and `models/` and print the numbers given in the paper; the figure scripts write to `figures/`. Run them from the repository root, for example `python3 analysis/statistical_checks.py`. Four scripts also need the HilalPy sighting database: the file `Final 30.6.2023.csv` of the Mendeley Data deposit [10.17632/rs994bxgc4.1](https://doi.org/10.17632/rs994bxgc4.1) (CC BY 4.0), passed with `--hilalpy` (or `--input` for `hilalpy_independent_check.py`).

| Script | Reproduces |
|---|---|
| `data_selection_flowchart.py` | Fig. 1 (data selection) |
| `data_attrition_and_negatives.py` | Sections 2.1-2.2: record losses and the meaning of non-sightings |
| `sky_geometry_schematic.py` | Fig. 2 (sky geometry at sunset) |
| `threshold_and_calibration_checks.py` | Fig. 6 (calibration by band); threshold choice and error rates |
| `ordinal_model_comparison.py` | Table 2 (three-class models) |
| `class_weight_cascade_check.py` | the full cascade and class weights (Section 3.4) |
| `confusion_matrices.py` | Fig. 7 (confusion matrices) |
| `probability_contours.py` | Fig. 8 (probability contours) |
| `statistical_checks.py` | Table 3 (coefficients and intervals), Fig. 9 (precision-recall curves), the Alrefay check (Section 3.8) |
| `predictor_ablation.py` | Table 4 (predictor ablation) |
| `regional_check.py` | Fig. 10 (station map) and the leave-one-region-out test |
| `observation_time_sensitivity.py`, `subgroup_performance.py` | sensitivity to observation time, region and subgroup (Section 3.6) |
| `visibility_maps.py` | Figs. 11, 13 and 15 (visibility maps) |
| `minimum_visibility_check.py` | Fig. 14 (minimum-visibility records); needs `--hilalpy` |
| `hilalpy_independent_check.py` | Table 5 (independent check on HilalPy); needs `--input` |
| `table1_criteria_comparison.py` | Table 6 and Fig. 16 (comparison with classical criteria); run with `--sample model --include-alternatives --figure figures/criteria_comparison.png` |
| `calendar_criteria_benchmark.py` | Table 7 (calendar rules); run with `data/icop/crescent_features.csv` as argument |
| `graphical_abstract.py` | graphical abstract; `--hilalpy` fills in the HilalPy values |
| `minor_statistical_checks.py` | the checks of the third review round; needs `--hilalpy` |
| `worked_example.py` | record counts, final model parameters, a worked example and the package versions |

Fig. 12 is a photograph. The notes at the top of each script name the reviewer comment it answers; some still give the table and figure numbers of earlier versions of the paper, so use the numbers above.

## Results

Cross-validation grouped by lunation (out-of-fold, 4,470 records; accuracy at a cut-off of 0.50):

| Stage | Task | Accuracy | ROC AUC | Majority-class accuracy |
|---|---|---|---|---|
| Stage 1 | detectability (seen by any means or not) | 0.794 | 0.877 | 0.519 |
| Stage 2 | naked eye or optical aid only (sightings) | 0.896 | 0.927 | 0.765 |

External checks with the final models:

- Alrefay et al. (2018) tables, at the operational thresholds (P<sub>det</sub> ≥ 0.50 for Stage 1 and P<sub>eye</sub> ≥ 0.60 for Stage 2): Stage 1 agrees with 86.6 % of the 425 usable records, and Stage 2 with 90.6 % of the 265 sightings.
- HilalPy database (1,365 independent evening records): ROC AUC 0.883 for Stage 1 and 0.886 for Stage 2.

## Known issues

- **Monday reports are missing from the feature table.** `pipeline/03_build_crescent_features.py` removes the weekday from the Arabic date before parsing it, but its list of weekday names has only one spelling of Monday (`الاثنين`), while the ICOP pages write `الإثنين`. The date of every Monday report therefore fails to parse, and these reports are skipped. This removed 981 of the 7,124 cleaned reports (734 of them first-crescent reports made after sunset). The lost reports have the same class shares as those kept and do not differ by year, region or latitude, so the loss is not related to visibility. The archived `crescent_features.csv` and every result derived from it do not include these reports.
- **Step 3 needs a module that is not in this repository.** `pipeline/03_build_crescent_features.py` imports `moon_phases_core`, which is not included here, so step 3 cannot be rerun from this repository alone. Its output, `data/icop/crescent_features.csv`, is archived, and steps 3b to 5 start from it.

## Data sources

- **ICOP archive**: International Crescent Observation Project, [astronomycenter.net](https://astronomycenter.net). Reports from October 2009 to April 2026.
- **Alrefay et al. (2018)**: Tables I-VI of the paper (Tables I-III by text recognition of 600-dpi page images, Tables IV-VI typed by hand). See `data/alrefay/README.txt`.
- **HilalPy**: Faid, M. S., Mohd Nawawi, M. S. A. and Mohd Saadon, M. H. (2023). *Hilalpy: Database of Lunar Crescent Sighting*. Mendeley Data, version 1. https://doi.org/10.17632/rs994bxgc4.1 (not included here).

## References

- Alrefay, T. Y., Alsaab, S. A., Alreshidi, A., Alotaibi, H., Alrebdi, H. I., & Alhusari, J. (2018). Analysis of observations of earliest visibility of the lunar crescent. *The Observatory*, **138**, 267-288. [ADS: 2018Obs...138..267A](https://ui.adsabs.harvard.edu/abs/2018Obs...138..267A)
- Yallop, B. D. (1997). *A method for predicting the first sighting of the new crescent moon*. NAO Technical Note No. 69. HM Nautical Almanac Office, Royal Greenwich Observatory.
- Fotheringham, J. K. (1910). On the smallest visible phase of the Moon. *Monthly Notices of the Royal Astronomical Society*, **70**(7), 527-531.
- Maunder, E. W. (1911). On the smallest visible phase of the Moon. *Journal of the British Astronomical Association*, **21**, 355-362.

## How to cite

If you use this code or data, please cite:

> Talafha, M. (2026). *Sharjah Criterion: A Two-Stage Empirical Lunar Crescent Visibility Model*. Zenodo. https://doi.org/10.5281/zenodo.20755359

This DOI always points to the latest archived version. To cite the exact version you used, take its own DOI from the Zenodo page.

BibTeX:

```bibtex
@software{talafha_2026_sharjah_criterion,
  author    = {Talafha, Mohammed},
  title     = {{Sharjah Criterion: A Two-Stage Empirical Lunar Crescent Visibility Model}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20755359},
  url       = {https://doi.org/10.5281/zenodo.20755359}
}
```

## License

MIT — see [LICENSE](LICENSE).
