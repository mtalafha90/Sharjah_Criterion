# Sharjah Criterion

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20755359.svg)](https://doi.org/10.5281/zenodo.20755359)

A two-stage empirical criterion for lunar crescent visibility, built from the reports of the Islamic Crescents' Observation Project (ICOP). Stage 1 gives the probability that the crescent is seen at all (P<sub>det</sub>); Stage 2 gives, for a sighting, the probability that it was seen with the naked eye (P<sub>eye</sub>). The criterion is checked against the published observations of Alrefay et al. (2018) and the HilalPy sighting database, and compared with classical visibility criteria.

This repository holds the data, the pipeline that builds the model, the final models, and the analysis scripts that reproduce every table and figure of the paper *Sharjah criterion for lunar crescent visibility* (submitted to *Astronomy and Computing*).

## Contents

| Folder | Content |
|---|---|
| [`pipeline/`](pipeline/README.md) | build the data set and the model (steps 01-05), and compare it with the Alrefay tables |
| [`analysis/`](analysis/README.md) | reproduce the tables and figures of the paper |
| [`data/`](data/README.md) | ICOP reports and Alrefay et al. (2018) tables, with the predictions and comparisons |
| [`models/`](models/README.md) | final Stage 1 and Stage 2 models, their metadata and their performance |
| `figures/` | the figures of the paper |
| `docs/` | crescent visibility report for 17 May 2026 (Sharjah) |

`requirements.txt` lists the package versions used.

## Quick start

```bash
python3 -m pip install -r requirements.txt
python3 analysis/worked_example.py    # record counts, final model parameters and a worked example
```

## How to cite

If you use this code or data, please cite:

> Talafha, M. (2026). *Sharjah Criterion: A Two-Stage Empirical Lunar Crescent Visibility Model*. Zenodo. https://doi.org/10.5281/zenodo.20755359

This DOI always points to the latest version. To cite the exact version you used, give its own DOI:

| Version | Date | DOI |
|---|---|---|
| 1.1.0 | 25 November 2026 | [10.5281/zenodo.24529092](https://doi.org/10.5281/zenodo.24529092) |
| 1.0.0 | 19 June 2026 | [10.5281/zenodo.20755360](https://doi.org/10.5281/zenodo.20755360) |

The [release notes](https://github.com/mtalafha90/Sharjah_Criterion/releases) list the changes in each version; version 1.1.0 moved the files into the folders above.

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

## Licence

MIT — see [LICENSE](LICENSE). It covers the code and the files it produces. The report in `docs/` is © 2026 Mohammed H. Talafha, all rights reserved. The ICOP report texts and the Alrefay et al. (2018) tables keep the terms of their sources (see [`data/`](data/README.md)).
