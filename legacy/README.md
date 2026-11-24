# Legacy outputs

These files are outputs of the first version of the criterion, uploaded on 12 May 2026. No script in this repository writes or reads them, and the paper does not use them. They are kept only as a record.

| File | Content |
|---|---|
| `crescent_criterion_predictions.csv` | three-class predictions (class probabilities and predicted class) for a test split of 1,118 records |
| `alrefay_external_validation_predictions.csv` | the first comparison with the Alrefay et al. (2018) tables (302 records) |
| `alrefay_external_validation_report.json` | the summary of that comparison |

The current comparison with the Alrefay tables is in `data/alrefay/`, written by `pipeline/compare_twostep_with_alrefay_rowwise.py`.
