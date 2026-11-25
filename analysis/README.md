# Analysis scripts

These scripts start from the archived files in `data/` and `models/` and print the numbers given in the paper; the figure scripts write to `figures/`. Run them from the repository root, for example `python3 analysis/statistical_checks.py`.

- The first run of `regional_check.py`, `data_selection_flowchart.py`, `visibility_maps.py` or `graphical_abstract.py` needs internet access, because cartopy then downloads the Natural Earth country borders and coastlines.
- Figures redrawn on another system can differ slightly in how text and lines are rendered.
- Four scripts also need the HilalPy sighting database: the file `Final 30.6.2023.csv` of the Mendeley Data deposit [10.17632/rs994bxgc4.1](https://doi.org/10.17632/rs994bxgc4.1) (CC BY 4.0), passed with `--hilalpy` (or `--input` for `hilalpy_independent_check.py`).

| Script | Reproduces |
|---|---|
| `data_selection_flowchart.py` | Fig. 1 (data selection) |
| `data_attrition_and_negatives.py` | Sections 2.1-2.2: record losses and the meaning of non-sightings |
| `sky_geometry_schematic.py` | Fig. 2 (sky geometry at sunset) |
| `threshold_and_calibration_checks.py` | Fig. 6 (calibration by band); threshold choice and error rates |
| `ordinal_model_comparison.py` | Table 2 (three-class models), except the HilalPy columns, which `hilalpy_independent_check.py` prints in its section [6] |
| `class_weight_cascade_check.py` | the full cascade and class weights (Section 3.4) |
| `confusion_matrices.py` | Fig. 7 (confusion matrices) |
| `probability_contours.py` | Fig. 8 (probability contours) |
| `statistical_checks.py` | Table 3 (coefficients and intervals), Fig. 9 (precision-recall curves), the Alrefay check (Section 3.8) |
| `predictor_ablation.py` | Table 4 (predictor ablation); run with `data/icop/crescent_features.csv` as argument |
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

Figs. 3-5 come from `pipeline/05_choose_thresholds_and_plot_roc.py`, Fig. 12 is a photograph, and Table 1 defines the observed classes and the model categories. The notes at the top of most scripts name the review comment they answer (second review round; third round for `minor_statistical_checks.py`). Some notes, help texts and printed lines still use the table and figure numbers of earlier versions of the paper, and so does the name of `table1_criteria_comparison.py` (Table 1 of the first version is now Table 6), so use the numbers above.

## References

Classical criteria compared in `table1_criteria_comparison.py` include (the paper lists all criteria and their sources):

- Yallop, B. D. (1997). *A method for predicting the first sighting of the new crescent moon*. NAO Technical Note No. 69. HM Nautical Almanac Office, Royal Greenwich Observatory.
- Fotheringham, J. K. (1910). On the smallest visible phase of the Moon. *Monthly Notices of the Royal Astronomical Society*, **70**(7), 527-531.
- Maunder, E. W. (1911). On the smallest visible phase of the Moon. *Journal of the British Astronomical Association*, **21**, 355-362.
