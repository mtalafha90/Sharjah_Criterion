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
2. **Clean** — normalize text, booleans, and dates; assign empirical visibility class (0 = not seen, 1 = aided, 2 = naked eye); reject invalid rows
3. **Build features** — geocode observer locations, compute sunset/moonset times, and derive astronomical parameters (ARCV, ARCL, moon lag, elongation, illumination, Yallop *q*, etc.) using the `ephem` / `skyfield` stack
4. **Train** — fit two binary logistic-regression pipelines on 4 470 first-crescent observations:
   - **Stage 1** — detectability (seen by any means vs. not seen)
   - **Stage 2** — naked-eye vs. optical aid (applied only to observations in stage 1 positive class)
5. **Threshold selection** — sweep probability thresholds, plot ROC and F1 curves, and export `chosen_thresholds.json`

External validation runs the saved models against the Alrefay et al. (2018) tables (277 positive, 189 negative observations) via `compare_twostep_with_alrefay_rowwise.py`.

## Model performance

### Internal test set (25 % hold-out)

| Stage | Task | Accuracy | ROC AUC |
|---|---|---|---|
| Stage 1 | Detectability (seen vs. not seen) | 78.2 % | 0.864 |
| Stage 2 | Naked-eye vs. aided | 88.1 % | 0.929 |

### External validation — Alrefay et al. (2018) tables

| Stage | Rows | Accuracy |
|---|---|---|
| Stage 1 — detectability | 425 | 89.4 % |
| Stage 2 — naked-eye vs. aided | 265 | 87.9 % |

### Comparison with classical criteria (Alrefay tables, detectability task, *n* = 302)

| Criterion | Accuracy |
|---|---|
| Alrefay Two-Boundary | 98.7 % |
| Alrefay Unaided | 97.7 % |
| **This work — Stage 1** | **93.0 %** |
| Indian | 92.1 % |
| Maunder | 90.4 % |
| Fotheringham | 85.8 % |

## Features used

Three astronomical parameters derived at the time of best observation opportunity after sunset:

| Feature | Description |
|---|---|
| `relative_altitude_deg` | Moon altitude above the horizon relative to the Sun (ARCV) |
| `elongation_deg` | Moon–Sun angular separation (ARCL) |
| `moon_lag_minutes` | Time between sunset and moonset |

## Usage

Run the scripts in order from the repository root. Each script resolves its own paths automatically.

```bash
# 1. Scrape raw reports (requires internet access)
python Sharjah_Criterion/01_scrape_iac_crescent_reports.py

# 2. Clean observations
python Sharjah_Criterion/02_clean_crescent_observations.py

# 3. Build astronomical features (slow — geocoding + ephemeris)
python Sharjah_Criterion/03_build_crescent_features.py

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
