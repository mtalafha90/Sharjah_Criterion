# Sharjah Criterion

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20755360.svg)](https://doi.org/10.5281/zenodo.20755360)

A data-driven two-stage empirical criterion for lunar crescent visibility, derived from ICOP crescent sighting reports. The model is validated externally against the Alrefay et al. (2018) published observation tables and benchmarked against classical visibility criteria (Maunder, Indian, Fotheringham, Yallop).

## Repository structure

```
Sharjah_Criterion/
├── Sharjah_Criterion/               # Pipeline scripts
│   ├── 01_scrape_iac_crescent_reports.py
│   ├── 02_clean_crescent_observations.py
│   ├── 03_build_crescent_features.py      # first feature step, kept for reference (see Corrections)
│   ├── 03b_correct_moon_lag.py            # lag correction of that table, kept for reference
│   ├── 03c_geocode_sites.py               # checks and completes the site coordinates
│   ├── 03d_rebuild_features.py            # builds the feature table and applies the validity rules
│   ├── crescent_geometry.py               # sunset geometry with PyEphem, used by 03d
│   ├── 04_train_two_stage_crescent_criterion.py
│   └── 05_choose_thresholds_and_plot_roc.py
├── ICOP_Data/                       # Raw and processed observation data
│   ├── crescent_observations_raw.csv
│   ├── crescent_observations_clean.csv
│   ├── site_geocodes.csv                  # one row per place: coordinates and how they were found
│   ├── geocode_restricted_cache.json      # the Nominatim answers used by 03c
│   ├── observation_audit.csv              # reports excluded or recoded after reading their comments
│   ├── crescent_features.csv              # the feature table used by steps 4 and 5
│   └── crescent_features_script03.csv     # the earlier table of scripts 03 and 03b
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
3. **Build features**
   - `03c_geocode_sites.py` checks the coordinates of every place (city, province, country) against the Natural Earth 1:50m boundary of its reported country. Places outside that boundary, and places without coordinates, are geocoded again with OpenStreetMap Nominatim, with the search restricted to the reported country; only a settlement, an administrative area, an island, a peak or an observatory is accepted as a match. Places that cannot be found within their reported country, including reports that name only the country, stay unresolved. Output: `ICOP_Data/site_geocodes.csv`.
   - `03d_rebuild_features.py` computes the sunset geometry of every cleaned report with `crescent_geometry.py` (ARCV, ARCL, DAZ, crescent width, Moon lag, Moon age, illumination and the other columns) and applies the validity rules below. Every cleaned report keeps its row: a report that is not used has an empty `vis_class_empirical` and the reason in `exclusion_reason`, and `vis_class_reported` keeps the class coded from the report.
4. **Train** — fit two binary logistic-regression pipelines on 4 807 first-crescent observations:
   - **Stage 1** — detectability (seen by any means vs. not seen)
   - **Stage 2** — naked-eye vs. optical aid (applied only to observations in stage 1 positive class)

   Performance is estimated by five-fold cross-validation grouped by lunation (one ICOP monthly report page), so reports from the same new moon never sit in both the training and the test folds. The out-of-fold probabilities are saved to `ICOP_Data/two_stage_oof_predictions.csv`, and the final models are fitted on all 4 807 observations.
5. **Threshold selection** — sweep the out-of-fold probabilities, plot ROC, F1 and calibration curves, and export `chosen_thresholds.json`

External validation runs the saved models against the Alrefay et al. (2018) tables (277 positive, 189 negative observations) via `compare_twostep_with_alrefay_rowwise.py`.

## Model performance

### Cross-validation grouped by lunation (out-of-fold, 4 807 observations)

| Stage | Task | Accuracy | ROC AUC | Majority-class accuracy |
|---|---|---|---|---|
| Stage 1 | Detectability (seen vs. not seen) | 81.3 % | 0.900 | 53.2 % |
| Stage 2 | Naked-eye vs. aided (seen observations) | 89.7 % | 0.929 | 76.7 % |

Accuracy uses a probability cut-off of 0.50.

### External validation — Alrefay et al. (2018) tables

| Stage | Rows | Accuracy |
|---|---|---|
| Stage 1 — detectability | 425 | 87.3 % |
| Stage 2 — naked-eye vs. aided | 265 | 90.6 % |

### Validity rules (03d)

The rules apply to the first-crescent reports made after sunset, in this order; a report is counted under the first rule that applies.

| Rule | Reports | Why |
|---|---|---|
| Site | 59 removed | the place could not be found within the reported country |
| R1 | 8 removed | ICOP's heading says the crescent was not seen, but the report records a sighting |
| R2 | 288 removed | ICOP's heading says the crescent was seen by others: a second-hand report, often from another place |
| R3 | 8 removed | a sighting on an evening when the Moon had set before the Sun |
| R4 | 81 removed, 6 recoded | the observer's comment shows a sighting made only in daytime or before sunset, a sighting at another place or by other people, or a report that is not of the evening first crescent (for example, a partial solar eclipse); or it states a different outcome from the coded one |

For R4, the comment of every first-hand sighting that has one (1 368 reports) was read against these rules, and a second reader checked every flag; `ICOP_Data/observation_audit.csv` lists the 102 confirmed flags, with the words of the comment that show the problem. Of the 5 251 first-crescent reports made after sunset, 4 807 remain: 2 555 not seen, 525 seen only with an optical aid and 1 727 seen with the naked eye.

## Corrections in this version

- **Monday reports.** `03_build_crescent_features.py` spelled Monday `الاثنين`, while the ICOP pages write `الإثنين`, so the dates of all 981 Monday reports failed to parse and these reports were dropped. The weekday list is corrected, and `03d` reads every date; 677 Monday reports are in the modelling subset.
- **Site coordinates.** Script 03 geocoded each place without restricting the search to the reported country, and 20 places (51 modelling reports) lay outside their reported country (for example, Lawrence, Massachusetts, was placed in Canada). `03c` checks every place and geocodes again the doubtful and missing ones, as described above; 45 places (71 reports) could not be found in their reported country. The Natural Earth 1:50m map has no polygon for Melilla, so its three correctly placed sites fall in Morocco's polygon; they are kept.
- **Geometry.** Script 03 imports `moon_phases_core`, which is not in this repository. `crescent_geometry.py` computes the same quantities with PyEphem. Rebuilt from the same dates and sites, all 6 092 rows of the earlier table agree to within 0.0008° in ARCV, ARCL, DAZ and Moon altitude, 0.0003′ in crescent width and 0.012 min in lag, except that 423 moonsets differ by one minute (rounding) and 15 Moon ages differ by exactly 1 h: the earlier table subtracted local clock times across a daylight-saving change, while the ages are now true elapsed hours.
- **Invalid reports.** The validity rules above are new.

The earlier feature table is kept as `ICOP_Data/crescent_features_script03.csv`.

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

# 3. Check the sites (Nominatim answers are cached in ICOP_Data/geocode_restricted_cache.json),
#    then build the feature table and apply the validity rules (requires ephem and timezonefinder)
python Sharjah_Criterion/03c_geocode_sites.py
python Sharjah_Criterion/03d_rebuild_features.py

# 4. Train two-stage model
python Sharjah_Criterion/04_train_two_stage_crescent_criterion.py

# 5. Select thresholds and plot ROC curves
python Sharjah_Criterion/05_choose_thresholds_and_plot_roc.py

# External validation against Alrefay tables
python Alrefay_tables_Comparison/compare_twostep_with_alrefay_rowwise.py
```

Scripts 03 and 03b produced the earlier table `crescent_features_script03.csv`; script 03 needs the module `moon_phases_core`, which is not included, and 03c reads the coordinates that script 03 found from that table.

## Data sources

- **ICOP archive** — International Crescent Observation Project, hosted at [astronomycenter.net](https://astronomycenter.net). Scraped observation reports cover multiple Hijri months from participating observers worldwide (7 154 scraped reports, 7 124 after cleaning).
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
