# Sharjah Criterion

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20755360.svg)](https://doi.org/10.5281/zenodo.20755360)

A data-driven two-stage empirical criterion for lunar crescent visibility, derived from ICOP crescent sighting reports. The model is validated externally against the Alrefay et al. (2018) published observation tables and benchmarked against classical visibility criteria (Maunder, Indian, Fotheringham, Yallop).

## Repository structure

```
Sharjah_Criterion/
├── Sharjah_Criterion/               # Pipeline scripts
│   ├── 01_scrape_iac_crescent_reports.py
│   ├── 02_clean_crescent_observations.py
│   ├── 03_build_crescent_features.py
│   ├── 03b_correct_moon_lag.py
│   ├── 04_train_two_stage_crescent_criterion.py
│   └── 05_choose_thresholds_and_plot_roc.py
├── ICOP_Data/                       # Raw and processed observation data
│   ├── crescent_observations_raw.csv
│   ├── crescent_observations_clean.csv
│   └── crescent_features.csv
├── Alrefay_tables_Comparison/       # External validation data and outputs
│   ├── Table_I_positive_observations.csv
│   ├── Table_II_negative_observations.csv
│   ├── Table_III_examples_observations_parameters.csv
│   ├── Table_IV_ARCV_W_equation_5.csv
│   ├── Table_V_ARCV_W_equation_6.csv
│   ├── Table_VI_not_in_agreement_with_Yallop.csv
│   ├── stage1_detectability_model.joblib
│   ├── stage2_naked_eye_model.joblib
│   ├── two_stage_crescent_criterion_report.json
│   └── compare_twostep_with_alrefay_rowwise.py
├── CITATION.cff
├── LICENSE
└── README.md
```

## Overview

The workflow follows five sequential steps:

1. **Scrape** — collect crescent sighting reports from the ICOP archive at [astronomycenter.net](https://astronomycenter.net)
2. **Clean** — normalise text, booleans, and dates; assign empirical visibility class (0 = not seen, 1 = aided, 2 = naked eye); reject invalid rows
3. **Build features** — geocode observer locations, compute sunset/moonset times, and derive astronomical parameters (ARCV, ARCL, moon lag, elongation, illumination, Yallop *q*, etc.) using the `ephem` / `skyfield` stack; then `03b_correct_moon_lag.py` recomputes the moon lag where the Moon had already set at sunset (negative lag) or no moonset was found
4. **Train** — fit two binary logistic-regression pipelines on 4 470 first-crescent observations:
   - **Stage 1** — detectability (seen by any means vs. not seen)
   - **Stage 2** — naked-eye vs. optical aid (applied only to observations in stage 1 positive class)

   Performance is estimated by five-fold cross-validation grouped by lunation (one ICOP monthly report page), so reports from the same new moon never sit in both the training and the test folds. The out-of-fold probabilities are saved to `ICOP_Data/two_stage_oof_predictions.csv`, and the final models are fitted on all 4 470 observations.
5. **Threshold selection** — sweep the out-of-fold probabilities, plot ROC, F1 and calibration curves, and export `chosen_thresholds.json`

External validation runs the saved models against the Alrefay et al. (2018) tables (277 positive, 189 negative observations) via `compare_twostep_with_alrefay_rowwise.py`.

## Model performance

### Cross-validation grouped by lunation (out-of-fold, 4 470 observations)

| Stage | Task | Accuracy | ROC AUC | Majority-class accuracy |
|---|---|---|---|---|
| Stage 1 | Detectability (seen vs. not seen) | 79.4 % | 0.877 | 51.9 % |
| Stage 2 | Naked-eye vs. aided (seen observations) | 89.6 % | 0.927 | 76.5 % |

Accuracy uses a probability cut-off of 0.50.

### External validation — Alrefay et al. (2018) tables

| Stage | Rows | Accuracy |
|---|---|---|
| Stage 1 — detectability | 425 | 86.8 % |
| Stage 2 — naked-eye vs. aided | 265 | 89.8 % |

## Known issues

- **Monday reports are missing from the feature table.** `03_build_crescent_features.py` removes the weekday from the Arabic date before parsing it, but its list of weekday names has only one spelling of Monday (`الاثنين`), while the ICOP pages write `الإثنين`. The date of every Monday report therefore fails to parse, and these reports are skipped. This removed 981 of the 7 124 cleaned reports (734 of them first-crescent reports made after sunset). The lost reports have the same class shares as those kept and do not differ by year, region or latitude, so the loss is not related to visibility. The archived `crescent_features.csv` and every result derived from it do not include these reports.
- **Step 3 needs a module that is not in this repository.** `03_build_crescent_features.py` imports `moon_phases_core`, which is not included here, so step 3 cannot be rerun from this repository alone. Its output, `ICOP_Data/crescent_features.csv`, is archived, and steps 3b to 5 start from it.

## Features used

Three astronomical parameters, all evaluated at local sunset:

| Feature | Description |
|---|---|
| `relative_altitude_deg` | Moon altitude above the horizon relative to the Sun (ARCV) |
| `elongation_deg` | Moon–Sun angular separation (ARCL) |
| `moon_lag_minutes` | Time between sunset and moonset (negative if the Moon had already set) |

## Usage

Run the scripts in order from the repository root. Each script resolves its own paths automatically.

```bash
# 1. Scrape raw reports (requires internet access)
python Sharjah_Criterion/01_scrape_iac_crescent_reports.py

# 2. Clean observations
python Sharjah_Criterion/02_clean_crescent_observations.py

# 3. Build astronomical features (slow — geocoding + ephemeris)
python Sharjah_Criterion/03_build_crescent_features.py

# 3b. Correct the moon lag where the Moon had set before sunset or no moonset was found (requires ephem)
python Sharjah_Criterion/03b_correct_moon_lag.py

# 4. Train two-stage model
python Sharjah_Criterion/04_train_two_stage_crescent_criterion.py

# 5. Select thresholds and plot ROC curves
python Sharjah_Criterion/05_choose_thresholds_and_plot_roc.py

# External validation against Alrefay tables
python Alrefay_tables_Comparison/compare_twostep_with_alrefay_rowwise.py
```

## Data sources

- **ICOP archive** — International Crescent Observation Project, hosted at [astronomycenter.net](https://astronomycenter.net). Scraped observation reports cover multiple Hijri months from participating observers worldwide (~7 100 raw rows, ~6 100 after cleaning and feature computation).
- **Alrefay et al. (2018)** — Tables I–VI extracted from the paper (Tables I–III via OCR at 600 dpi; Tables IV–VI manually transcribed). Numeric values should be verified against the source PDF before publication-grade use.

## References

- Alrefay, T. Y., Alsaab, S. A., Alreshidi, A., Alotaibi, H., Alrebdi, H. I., & Alhusari, J. (2018). Analysis of observations of earliest visibility of the lunar crescent. *The Observatory*, **138**, 267–288. [ADS: 2018Obs...138..267A](https://ui.adsabs.harvard.edu/abs/2018Obs...138..267A)
- Yallop, B. D. (1997). *A method for predicting the first sighting of the new crescent moon*. NAO Technical Note No. 69. HM Nautical Almanac Office, Royal Greenwich Observatory.
- Fotheringham, J. K. (1910). On the smallest visible phase of the Moon. *Monthly Notices of the Royal Astronomical Society*, **70**(7), 527–531.
- Maunder, E. W. (1911). On the smallest visible phase of the Moon. *Journal of the British Astronomical Association*, **21**, 355–362.

## How to cite

If you use this code or data, please cite:

> Talafha, M. (2026). *Sharjah Criterion: A Two-Stage Empirical Lunar Crescent Visibility Model* (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.20755360

BibTeX:

```bibtex
@software{talafha_2026_sharjah_criterion,
  author    = {Talafha, Mohammed},
  title     = {{Sharjah Criterion: A Two-Stage Empirical Lunar Crescent Visibility Model}},
  version   = {1.0.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20755360},
  url       = {https://doi.org/10.5281/zenodo.20755360}
}
```

## License

MIT — see [LICENSE](LICENSE).
