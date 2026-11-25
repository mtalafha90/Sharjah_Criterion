# Models

This folder holds the outputs of `pipeline/04` and `pipeline/05`:

- `stage1_detectability_model.joblib` and `stage2_naked_eye_model.joblib`: the final scikit-learn pipelines (median imputation, standardisation, logistic regression), fitted with scikit-learn 1.9.1: Stage 1 on all 4,470 modelling records and Stage 2 on the 2,151 sightings among them;
- `two_stage_model_meta.json`: predictors and thresholds of the two stages;
- `two_stage_crescent_criterion_report.json`: cross-validated scores and coefficients;
- `chosen_thresholds.json`: the threshold candidates of `pipeline/05`.

The predictors are evaluated at local sunset:

| Predictor | Description |
|---|---|
| `relative_altitude_deg` | ARCV: altitude of the Moon minus altitude of the Sun |
| `elongation_deg` | ARCL: angular separation of the Moon and the Sun |
| `moon_lag_minutes` | lag time: moonset minus sunset (negative if the Moon had already set) |

## Performance

Cross-validation grouped by lunation (out-of-fold, 4,470 records; accuracy at a cut-off of 0.50):

| Stage | Task | Accuracy | ROC AUC | Majority-class accuracy |
|---|---|---|---|---|
| Stage 1 | detectability (seen by any means or not) | 0.794 | 0.877 | 0.519 |
| Stage 2 | naked eye or optical aid only (sightings) | 0.896 | 0.927 | 0.765 |

External checks with the final models:

- Alrefay et al. (2018) tables, at the operational thresholds (P<sub>det</sub> ≥ 0.50 for Stage 1 and P<sub>eye</sub> ≥ 0.60 for Stage 2): Stage 1 agrees with 86.6 % of the 425 usable records, and Stage 2 with 90.6 % of the 265 sightings.
- HilalPy database (1,365 independent evening records): ROC AUC 0.883 for Stage 1 and 0.886 for Stage 2.
