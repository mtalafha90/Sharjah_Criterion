#!/usr/bin/env python3
"""Predictor ablation (second review round, comment 4), evaluated by lunation
(second review round, comment 6).

The script compares three predictor sets in each stage of the two-stage model.
The cross-validation keeps each lunation whole: all its reports are in the
same fold.

Input: data/icop/crescent_features.csv,
filtered as in 04_train_two_stage_crescent_criterion.py. The two-stage model
is not replaced: each predictor set is fitted with its logistic pipeline
(median imputation, standardisation, logistic regression), refitted on the
training rows of every fold. The single random 75/25 split is replaced by
repeated lunation-grouped 5-fold CV (StratifiedGroupKFold, shuffle=True,
seeds 0-9, groups = page_url, i.e. one ICOP monthly report per lunation):
  Stage 1  all rows, stratified on seen (aided or naked eye)
  Stage 2  seen rows only, stratified on naked eye, with its own grouped folds
Within a stage every predictor set sees identical folds, so the fold
differences are paired. Accuracy uses a probability cutoff of 0.50.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ARCV = "relative_altitude_deg"
ARCL = "elongation_deg"
LAG = "moon_lag_minutes"
WIDTH = "crescent_width_arcmin"
DAZ = "relative_azimuth_deg"
GROUP = "page_url"
CONFIGS = [
    ("ARCV + ARCL", [ARCV, ARCL]),
    ("ARCV + ARCL + lag", [ARCV, ARCL, LAG]),
    ("ARCV + ARCL + lag + W + DAZ", [ARCV, ARCL, LAG, WIDTH, DAZ]),
]
# Paired comparisons: (label, larger set, smaller set).
COMPARISONS = [
    ("adding lag", "ARCV + ARCL + lag", "ARCV + ARCL"),
    ("adding W and DAZ", "ARCV + ARCL + lag + W + DAZ", "ARCV + ARCL + lag"),
]
METRICS = [("acc", "Accuracy"), ("auc", "ROC AUC")]
N_FOLDS = 5
SEEDS = range(10)


def grouped_folds(df, y):
    """Repeated 5-fold splits that keep each lunation whole."""
    groups = df[GROUP].to_numpy()
    folds = []
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for train_idx, test_idx in cv.split(df, y, groups=groups):
            if set(groups[train_idx]) & set(groups[test_idx]):
                raise RuntimeError("A lunation is in both training and test rows")
            folds.append((train_idx, test_idx))
    return folds


def run_stage(df, stage_name, y):
    # Construct the folds only once: every configuration sees identical rows.
    folds = grouped_folds(df, y)
    sizes = [len(test_idx) for _, test_idx in folds]
    print(f"\n{stage_name}: N={len(df)}, lunations={df[GROUP].nunique()}, "
          f"positives={int(y.sum())}; {len(folds)} folds "
          f"({N_FOLDS} folds x {len(SEEDS)} seeds), test fold sizes "
          f"{min(sizes)}-{max(sizes)}")

    res = {}
    for name, features in CONFIGS:
        res[name] = {m: [] for m, _ in METRICS}
        for train_idx, test_idx in folds:
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=4000, solver="lbfgs")),
            ])
            model.fit(df.iloc[train_idx][features], y[train_idx])
            prob = model.predict_proba(df.iloc[test_idx][features])[:, 1]
            res[name]["acc"].append(accuracy_score(y[test_idx], prob >= 0.50))
            res[name]["auc"].append(roc_auc_score(y[test_idx], prob))
        res[name] = {m: np.array(v) for m, v in res[name].items()}

    print("Mean (SD) over folds")
    print(f"{'Predictors':30s} " + " ".join(f"{label:>17s}" for _, label in METRICS))
    for name, _ in CONFIGS:
        print(f"{name:30s} " + " ".join(
            f"{res[name][m].mean():8.4f} ({res[name][m].std(ddof=1):6.4f})"
            for m, _ in METRICS))

    print("Paired fold differences (larger set minus smaller set): mean, SD, "
          "folds improved / tied / worse")
    print(f"{'Comparison':18s} {'Metric':9s} {'Mean':>8s} {'SD':>7s} "
          f"{'Improved/Tie/Worse':>19s}")
    for label, larger, smaller in COMPARISONS:
        for m, metric in METRICS:
            d = res[larger][m] - res[smaller][m]
            tie = np.isclose(d, 0, atol=1e-12)
            better = (d > 0) & ~tie
            worse = (d < 0) & ~tie
            print(f"{label:18s} {metric:9s} {d.mean():8.4f} {d.std(ddof=1):7.4f} "
                  f"{f'{better.sum()}/{tie.sum()}/{worse.sum()}':>19s}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv", help="Path to data/icop/crescent_features.csv"
    )
    args = parser.parse_args()
    df = pd.read_csv(args.csv, low_memory=False)
    df = df[
        (df["section_type"] == "first_crescent")
        & (df["obs_relative_to_sunset"] != "before_sunset")
        & df["vis_class_empirical"].notna()
    ].copy()
    df["vis_class_empirical"] = df["vis_class_empirical"].astype(int)
    required = set(feature for _, cols in CONFIGS for feature in cols) | {GROUP}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(df) != 4470 or tuple(df["vis_class_empirical"].value_counts().sort_index()) != (2319, 505, 1646):
        raise ValueError("Source data have changed; verify cohort and splits before comparison")
    # Unique index preserves the source order, as in the published training script.
    df = df.reset_index(drop=True)
    y1 = (df["vis_class_empirical"] > 0).astype(int).to_numpy()
    run_stage(df, "Stage 1: detectability", y1)
    df2 = df[df["vis_class_empirical"] > 0].reset_index(drop=True)
    y2 = (df2["vis_class_empirical"] == 2).astype(int).to_numpy()
    run_stage(df2, "Stage 2: naked-eye among detections", y2)


if __name__ == "__main__":
    main()
