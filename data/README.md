# Data

## `icop/`: ICOP reports

| File | Written by | Records | Content |
|---|---|---|---|
| `crescent_observations_raw.csv` | `pipeline/01` | 7,154 | reports scraped from the ICOP pages |
| `crescent_observations_clean.csv` | `pipeline/02` | 7,124 | reports with a usable outcome and an observed class |
| `crescent_observations_rejected.csv` | `pipeline/02` | 30 | reports without a usable outcome |
| `crescent_features.csv` | `pipeline/03`, `03b` | 6,092 | geocoded reports with the geometry at local sunset |
| `two_stage_oof_predictions.csv` | `pipeline/04` | 4,470 | out-of-fold P<sub>det</sub> and P<sub>eye</sub> of the modelling records |
| `two_stage_stage1_predictions.csv` | `pipeline/04` | 4,470 | out-of-fold Stage 1 predictions |
| `two_stage_stage2_predictions.csv` | `pipeline/04` | 2,151 | out-of-fold Stage 2 predictions (sightings only) |

Observed classes: 0 = not seen, 1 = seen with optical aid only, 2 = seen with the naked eye. Personal phone numbers that observers wrote in their reports are replaced by `[number removed]`; observers' names are kept as published on the ICOP pages.

Reports dated on a Monday are missing from `crescent_features.csv` and from every result derived from it; see [Known issues](../pipeline/README.md#known-issues).

## `alrefay/`: Alrefay et al. (2018)

Tables I-VI of Alrefay et al. (2018), transcribed from the scanned paper (see [`alrefay/README.txt`](alrefay/README.txt), which also lists the values corrected in November 2026), and the row-by-row comparison written by `pipeline/compare_twostep_with_alrefay_rowwise.py` (425 usable records, 265 of them sightings).

## Sources

- **ICOP archive**: Islamic Crescents' Observation Project, [astronomycenter.net](https://astronomycenter.net). Reports from October 2009 to April 2026.
- **Alrefay et al. (2018)**: Tables I-VI of the paper (Tables I-III by text recognition of 600-dpi page images, Tables IV-VI typed by hand). Alrefay, T., Alsaab, S., Alshehri, F., Alghamdi, A., Hadadi, A., Alotaibi, M., Almutari, K., & Mubarki, Y. (2018). Analysis of observations of earliest visibility of the lunar crescent. *The Observatory*, **138**, 267-291. [ADS: 2018Obs...138..267A](https://ui.adsabs.harvard.edu/abs/2018Obs...138..267A)
- **HilalPy**: Faid, M. S., Mohd Nawawi, M. S. A. and Mohd Saadon, M. H. (2023). *Hilalpy: Database of Lunar Crescent Sighting*. Mendeley Data, version 1. https://doi.org/10.17632/rs994bxgc4.1 (not included here; needed by four analysis scripts).

The ICOP report texts and the Alrefay et al. (2018) tables come from these sources, and their terms apply.
