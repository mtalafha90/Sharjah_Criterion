# Sharjah Criterion

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20755360.svg)](https://doi.org/10.5281/zenodo.20755360)

A data-driven lunar crescent visibility project built around a **two-step empirical criterion** derived from crescent sighting reports, with supporting comparisons against classical visibility criteria and external validation using the Alrefay observation tables. The repository currently contains three top-level folders: `Sharjah_Criterion`, `ICOP_Data`, and `Alrefay_tables_Comparison`. :contentReference[oaicite:0]{index=0}

## Overview

This repository collects the code, processed data products, model files, and validation outputs used to build and test an empirical crescent-visibility criterion. The workflow is organized around:

- preparing and cleaning crescent observation data,
- computing astronomical/geometric features,
- training a **two-stage logistic-regression model**,
- selecting operational thresholds,
- comparing against classical criteria,
- and validating against external published tables. :contentReference[oaicite:1]{index=1}

The external validation material in `Alrefay_tables_Comparison` was extracted from the Alrefay et al. (2018) paper tables, including OCR-derived CSVs and row-by-row validation outputs. The folder README notes that Tables I–III were OCR-extracted from scanned pages, while Tables IV–VI were manually checked/transcribed, and it advises verifying critical values against the source before publication-grade use. :contentReference[oaicite:2]{index=2}

## Repository structure

```text
Sharjah_Criterion/
├── Sharjah_Criterion/
├── ICOP_Data/
├── Alrefay_tables_Comparison/
└── README.md
