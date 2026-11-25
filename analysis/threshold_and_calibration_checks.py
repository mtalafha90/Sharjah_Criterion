#!/usr/bin/env python3
"""Threshold choice, error rates and calibration by band (reviewer comment 7).

Comment 7 (statistical review) asks:
  "Please pick the thresholds using training data or cross-validation,
  ideally nested, and then evaluate once on untouched data. The Youden-J and
  maximum-F1 thresholds that were found should be reported. The final values
  (0.30, 0.50 and 0.60) are described as 'practical', so please explain what
  error rates or costs they represent and how uncertain they are, for
  instance with a bootstrap."
  "... the observed fraction of detected or naked-eye cases in each band of
  Eq. 16 [Eq. 11 of the revised manuscript], with the number of records in each band ... Because the training
  data are not a random sample of real '29th evening' situations, a value
  such as 0.60 may not mean a 60% chance in practice."

Inputs (default paths in this repository):
  data/icop/crescent_features.csv          the modelling subset (4,470
      first-crescent reports: 2,319 not seen, 505 aided, 1,646 naked eye;
      202 lunations, one ICOP monthly page each) is filtered as in
      04_train_two_stage_crescent_criterion.py
  data/icop/two_stage_oof_predictions.csv  out-of-fold P_det and P_eye of the
      main evaluation (same rows, same order): StratifiedGroupKFold(5,
      shuffle=True, random_state=42), stratified on the three-class label,
      groups = page_url

Both stages use the pipeline of 04: median imputation, standardisation and
logistic regression (lbfgs, max_iter=4000, default L2 penalty with C=1) on
ARCV, ARCL and lag at local sunset. Stage 1 separates seen by any means from
not seen (all reports); Stage 2 separates naked eye from aided only and is
fitted on the sightings. A record is called positive when its probability is
at or above the threshold.

Threshold rules, as in 05_choose_thresholds_and_plot_roc.py:
  Youden J  the ROC threshold (an observed probability) with the largest
            J = TPR - FPR
  max F1    the value on the grid 0.050, 0.055, ..., 0.950 with the largest
            F1 of the positive class (the first one if tied)

Cluster bootstrap (sections 2 to 4): the 202 lunations are resampled with
replacement (2,000 resamples, numpy default_rng(42)), with the out-of-fold
probabilities held fixed; intervals are the 2.5th and 97.5th percentiles.

Sections
  1  Nested CV. Outer loop: StratifiedGroupKFold(5, shuffle=True,
     random_state=42), stratified on the three-class label, groups = page_url
     (the folds of the main evaluation). Inside each outer training set an
     inner StratifiedGroupKFold(5, same seed) gives out-of-fold P_det (all
     rows) and P_eye (Stage 2 fitted on the seen inner-training rows), as in
     04; the Youden-J and max-F1 thresholds of each stage are chosen from
     them (Stage 2 on the seen rows only). Both stages are then refitted on
     the whole outer training set and the chosen thresholds are applied once
     to the outer test fold. The fixed thresholds 0.50 (Stage 1) and 0.60
     (Stage 2) are scored on the same outer folds. The outer CV is repeated
     with seeds 0-4 (outer and inner seed alike).
  2  Youden-J and max-F1 thresholds of the out-of-fold probabilities, with
     bootstrap intervals; J and F1 at the adopted thresholds and their
     shortfall from the maximum.
  3  Error rates at the adopted thresholds (out-of-fold probabilities):
     Stage 1 at 0.30 and 0.50 on all reports, Stage 2 at 0.30 and 0.60 on
     the sightings, and the naked-eye rule (P_det >= 0.50 and P_eye >= 0.60)
     on all reports, with naked eye as the positive class.
  4  Calibration by band of Eq. 11 (E: P_det < 0.30; D: 0.30-0.50; for
     P_det >= 0.50, C: P_eye < 0.30; B: 0.30-0.60; A: P_eye >= 0.60) and by
     deciles, with Wilson 95% intervals (records treated as independent) and
     bootstrap intervals; calibration intercept and slope; how a probability
     would change if the share of sightings differed from the ICOP share.
     Figure calibration_by_band.png (300 dpi, no title).

Run: python3 -W ignore analysis/threshold_and_calibration_checks.py   (about one minute)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportion_confint

FEATURES = ["relative_altitude_deg", "elongation_deg", "moon_lag_minutes"]
TARGET = "vis_class_empirical"
LUNATION = "page_url"
N_FOLDS = 5
MAIN_SEED = 42
REPEAT_SEEDS = range(5)
BOOT_SEED = 42
F1_GRID = np.linspace(0.05, 0.95, 181)
# Thresholds of Eq. 11 and the cascade.
DET_LOW, DET_HIGH = 0.30, 0.50
EYE_LOW, EYE_HIGH = 0.30, 0.60
BANDS = [
    ("E", "not detectable", "P_det < 0.30"),
    ("D", "marginal detectability", "0.30 <= P_det < 0.50"),
    ("C", "aid-assisted likely", "P_det >= 0.50, P_eye < 0.30"),
    ("B", "borderline naked-eye", "P_det >= 0.50, 0.30 <= P_eye < 0.60"),
    ("A", "naked-eye likely", "P_det >= 0.50, P_eye >= 0.60"),
]

HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # the repository root
DEFAULT_INPUT = REPO / "data" / "icop" / "crescent_features.csv"
DEFAULT_OOF = REPO / "data" / "icop" / "two_stage_oof_predictions.csv"
DEFAULT_FIGURE = REPO / "figures" / "calibration_by_band.png"

# Reference palette (blue, orange) with dark ink and a recessive grid.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ----------------------------------------------------------------------------
# Data and helpers
# ----------------------------------------------------------------------------
def load_subset(path):
    """The 4,470-row modelling subset, in source order."""
    df = pd.read_csv(path, low_memory=False)
    df = df[
        (df["section_type"] == "first_crescent")
        & (df["obs_relative_to_sunset"] != "before_sunset")
        & df[TARGET].notna()
    ].copy()
    df[TARGET] = df[TARGET].astype(int)
    if len(df) != 4470 or tuple(df[TARGET].value_counts().sort_index()) != (2319, 505, 1646):
        raise ValueError("Source data have changed; verify the cohort before comparison")
    return df.reset_index(drop=True)


def load_oof(path, df):
    oof = pd.read_csv(path)
    if (len(oof) != len(df) or not (oof[LUNATION].to_numpy() == df[LUNATION].to_numpy()).all()
            or not (oof[TARGET].to_numpy() == df[TARGET].to_numpy()).all()
            or not np.allclose(oof[FEATURES].to_numpy(), df[FEATURES].to_numpy(), equal_nan=True)):
        raise ValueError("Out-of-fold file does not match the modelling subset row by row")
    return oof


def rows_by_lunation(groups):
    """Row indices of each lunation, for the cluster bootstrap."""
    codes, _ = pd.factorize(groups)
    order = np.argsort(codes, kind="stable")
    return np.split(order, np.cumsum(np.bincount(codes))[:-1])


def lunation_resamples(rows, n_boot, seed):
    """Rows of n_boot resamples of whole lunations, drawn with replacement."""
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.integers(0, len(rows), len(rows))
        yield np.concatenate([rows[i] for i in pick])


def ci(values):
    values = np.asarray(values, dtype=float)
    return np.nanpercentile(values, 2.5), np.nanpercentile(values, 97.5)


def fmt_ci(value, values):
    lo, hi = ci(values)
    return f"{value:.4f} [{lo:.4f}, {hi:.4f}]"


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan
    return proportion_confint(k, n, alpha=0.05, method="wilson")


def youden(y, p):
    """Threshold and value of the largest Youden J on the ROC curve."""
    fpr, tpr, thr = roc_curve(y, p)
    i = int(np.argmax(tpr - fpr))
    return thr[i], tpr[i] - fpr[i]


def f1_curve(y, p):
    """F1 of the positive class at every grid threshold."""
    y = np.asarray(y, dtype=bool)
    pos = np.sort(p[y])
    every = np.sort(p)
    tp = len(pos) - np.searchsorted(pos, F1_GRID, side="left")
    called = len(every) - np.searchsorted(every, F1_GRID, side="left")
    return 2 * tp / (called + len(pos))


def max_f1(y, p):
    f1 = f1_curve(y, p)
    i = int(np.argmax(f1))
    return F1_GRID[i], f1[i]


def j_at(y, p, cut):
    y = np.asarray(y, dtype=bool)
    pred = p >= cut
    return pred[y].mean() - pred[~y].mean()


def f1_at(y, p, cut):
    y = np.asarray(y, dtype=bool)
    pred = p >= cut
    return 2 * np.sum(pred & y) / (pred.sum() + y.sum())


def rates(y, pred):
    """Confusion counts and error rates of a yes/no rule."""
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tp = np.sum(pred & y)
    fp = np.sum(pred & ~y)
    fn = np.sum(~pred & y)
    tn = np.sum(~pred & ~y)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "acc": (tp + tn) / len(y),
        "sens": tp / (tp + fn),
        "spec": tn / (tn + fp),
        "prec": tp / (tp + fp) if tp + fp else np.nan,
        "f1": 2 * tp / (2 * tp + fp + fn),
        "fpr": fp / (fp + tn),
        "fnr": fn / (fn + tp),
        "share": pred.mean(),
    }


# ----------------------------------------------------------------------------
# Section 1: nested cross-validation
# ----------------------------------------------------------------------------
def fit_stage(X, y):
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, solver="lbfgs")),
    ])
    return model.fit(X, y)


def fit_predict(X_train, label_train, X_test):
    """P_det and P_eye for the test rows from both stages fitted on the training rows."""
    seen = label_train > 0
    stage1 = fit_stage(X_train, seen.astype(int))
    stage2 = fit_stage(X_train[seen], (label_train[seen] == 2).astype(int))
    return stage1.predict_proba(X_test)[:, 1], stage2.predict_proba(X_test)[:, 1]


def grouped_folds(label, groups, seed):
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    folds = list(cv.split(np.zeros(len(label)), label, groups=groups))
    for train_idx, test_idx in folds:
        if set(groups[train_idx]) & set(groups[test_idx]):
            raise RuntimeError("A lunation is in both training and test rows")
    return folds


def choose_thresholds(label, p_det, p_eye):
    """Youden-J and max-F1 thresholds of both stages from out-of-fold probabilities."""
    seen = label > 0
    eye = label[seen] == 2
    return {
        "s1_youden": youden(seen, p_det)[0],
        "s1_f1": max_f1(seen, p_det)[0],
        "s2_youden": youden(eye, p_eye[seen])[0],
        "s2_f1": max_f1(eye, p_eye[seen])[0],
    }


def nested_cv(X, label, groups, seed):
    """Outer test probabilities and, per outer fold, the thresholds chosen by the inner CV."""
    p_det = np.full(len(label), np.nan)
    p_eye = np.full(len(label), np.nan)
    folds = []
    for train_idx, test_idx in grouped_folds(label, groups, seed):
        Xtr, ltr, gtr = X[train_idx], label[train_idx], groups[train_idx]
        inner_det = np.full(len(ltr), np.nan)
        inner_eye = np.full(len(ltr), np.nan)
        for itr, ite in grouped_folds(ltr, gtr, seed):
            inner_det[ite], inner_eye[ite] = fit_predict(Xtr[itr], ltr[itr], Xtr[ite])
        chosen = choose_thresholds(ltr, inner_det, inner_eye)
        p_det[test_idx], p_eye[test_idx] = fit_predict(Xtr, ltr, X[test_idx])
        folds.append((test_idx, chosen))
    return p_det, p_eye, folds


RULES = [
    ("Nested Youden J", "s1_youden", "s2_youden"),
    ("Nested max F1", "s1_f1", "s2_f1"),
    ("Fixed 0.50 / 0.60", None, None),
]
OUTER_METRICS = [("acc", "Accuracy"), ("sens", "Sensitivity"), ("spec", "Specificity"),
                 ("prec", "Precision"), ("f1", "F1")]


def outer_scores(label, p_det, p_eye, folds):
    """Per-fold and pooled outer-test metrics of every rule and stage."""
    out = {}
    for rule, key1, key2 in RULES:
        for stage, key, fixed in [("Stage 1", key1, DET_HIGH), ("Stage 2", key2, EYE_HIGH)]:
            y_all, pred_all, per_fold = [], [], []
            for test_idx, chosen in folds:
                cut = fixed if key is None else chosen[key]
                lab = label[test_idx]
                if stage == "Stage 1":
                    y, p = lab > 0, p_det[test_idx]
                else:
                    seen = lab > 0
                    y, p = lab[seen] == 2, p_eye[test_idx][seen]
                per_fold.append(rates(y, p >= cut))
                y_all.append(y)
                pred_all.append(p >= cut)
            out[(rule, stage)] = {
                "pooled": rates(np.concatenate(y_all), np.concatenate(pred_all)),
                "folds": pd.DataFrame(per_fold),
            }
    return out


def section_nested(X, label, groups, oof):
    print("=" * 78)
    print("1. Nested cross-validation of the threshold choice")
    print("=" * 78)
    p_det, p_eye, folds = nested_cv(X, label, groups, MAIN_SEED)
    diff = max(np.abs(p_det - oof["p_det"].to_numpy()).max(),
               np.abs(p_eye - oof["p_eye"].to_numpy()).max())
    print(f"Outer seed {MAIN_SEED} (the folds of the main evaluation). Check: the outer "
          f"test-fold probabilities\nmatch two_stage_oof_predictions.csv "
          f"(largest absolute difference {diff:.1e}).")
    if diff > 1e-8:
        raise RuntimeError("Outer folds do not reproduce the main out-of-fold probabilities")

    print("\nThresholds chosen in each outer fold (inner grouped 5-fold CV on the outer "
          "training rows)")
    print(f"{'Fold':>4s} {'Test rows':>9s} {'Lunations':>9s} {'Sightings':>9s} | "
          f"{'S1 Youden':>9s} {'S1 max F1':>9s} | {'S2 Youden':>9s} {'S2 max F1':>9s}")
    for k, (test_idx, chosen) in enumerate(folds, start=1):
        print(f"{k:4d} {len(test_idx):9d} {len(set(groups[test_idx])):9d} "
              f"{int(np.sum(label[test_idx] > 0)):9d} | {chosen['s1_youden']:9.4f} "
              f"{chosen['s1_f1']:9.4f} | {chosen['s2_youden']:9.4f} {chosen['s2_f1']:9.4f}")

    scores = outer_scores(label, p_det, p_eye, folds)
    for stage, what in [("Stage 1", "all reports; positive = seen by any means"),
                        ("Stage 2", "sightings; positive = naked eye")]:
        print(f"\n{stage} on the outer test folds ({what})")
        print(f"{'Rule':20s} {'':6s} " + " ".join(f"{name:>15s}" for _, name in OUTER_METRICS))
        for rule, _, _ in RULES:
            res = scores[(rule, stage)]
            print(f"{rule:20s} {'pooled':6s} " + " ".join(
                f"{res['pooled'][m]:15.4f}" for m, _ in OUTER_METRICS))
            print(f"{'':20s} {'folds':6s} " + " ".join(
                f"{res['folds'][m].mean():7.4f} ({res['folds'][m].std(ddof=1):.4f})"
                for m, _ in OUTER_METRICS))
    print("'pooled': the five outer test folds together, each scored with its own "
          "thresholds;\n'folds': mean (SD) over the five outer folds.")

    # Repeat the whole nested CV with other seeds.
    chosen_all, pooled = [], []
    for seed in REPEAT_SEEDS:
        p_det_s, p_eye_s, folds_s = nested_cv(X, label, groups, seed)
        chosen_all += [chosen for _, chosen in folds_s]
        sc = outer_scores(label, p_det_s, p_eye_s, folds_s)
        for (rule, stage), res in sc.items():
            pooled.append({"seed": seed, "rule": rule, "stage": stage, **res["pooled"]})
    chosen_all = pd.DataFrame(chosen_all)
    pooled = pd.DataFrame(pooled)
    seeds = f"{min(REPEAT_SEEDS)}-{max(REPEAT_SEEDS)}"
    print(f"\nSeeds {seeds}: thresholds chosen in the {len(chosen_all)} outer folds")
    print(f"{'Threshold':12s} {'Mean':>7s} {'SD':>7s} {'Min':>7s} {'Max':>7s}")
    for key, name in [("s1_youden", "S1 Youden"), ("s1_f1", "S1 max F1"),
                      ("s2_youden", "S2 Youden"), ("s2_f1", "S2 max F1")]:
        v = chosen_all[key]
        print(f"{name:12s} {v.mean():7.4f} {v.std(ddof=1):7.4f} {v.min():7.4f} {v.max():7.4f}")
    print(f"\nSeeds {seeds}: pooled outer-test metrics, mean over seeds [min, max]")
    for stage in ["Stage 1", "Stage 2"]:
        print(f"{stage}")
        print(f"{'Rule':20s} " + " ".join(f"{name:>24s}" for _, name in OUTER_METRICS))
        for rule, _, _ in RULES:
            sub = pooled[(pooled["rule"] == rule) & (pooled["stage"] == stage)]
            print(f"{rule:20s} " + " ".join(
                f"{sub[m].mean():.4f} [{sub[m].min():.4f}, {sub[m].max():.4f}]".rjust(24)
                for m, _ in OUTER_METRICS))


# ----------------------------------------------------------------------------
# Section 2: the Youden-J and max-F1 thresholds and their uncertainty
# ----------------------------------------------------------------------------
def stage_arrays(label, p_det, p_eye, idx=None):
    """(Stage 1 truth, P_det) on all rows and (Stage 2 truth, P_eye) on the sightings."""
    if idx is None:
        idx = np.arange(len(label))
    lab = label[idx]
    seen = lab > 0
    return {"Stage 1": (seen, p_det[idx]), "Stage 2": (lab[seen] == 2, p_eye[idx][seen])}


ADOPTED = {"Stage 1": [DET_LOW, DET_HIGH], "Stage 2": [EYE_LOW, EYE_HIGH]}


def threshold_summary(arrays):
    out = {}
    for stage, (y, p) in arrays.items():
        out[(stage, "youden_thr")], out[(stage, "j_max")] = youden(y, p)
        out[(stage, "f1_thr")], out[(stage, "f1_max")] = max_f1(y, p)
        for cut in ADOPTED[stage]:
            out[(stage, "j", cut)] = j_at(y, p, cut)
            out[(stage, "f1", cut)] = f1_at(y, p, cut)
            out[(stage, "j_short", cut)] = out[(stage, "j_max")] - out[(stage, "j", cut)]
            out[(stage, "f1_short", cut)] = out[(stage, "f1_max")] - out[(stage, "f1", cut)]
    return out


def section_thresholds(label, p_det, p_eye, rows, n_boot):
    print("\n" + "=" * 78)
    print("2. Youden-J and max-F1 thresholds of the out-of-fold probabilities")
    print("=" * 78)
    point = threshold_summary(stage_arrays(label, p_det, p_eye))
    boot = [threshold_summary(stage_arrays(label, p_det, p_eye, idx))
            for idx in lunation_resamples(rows, n_boot, BOOT_SEED)]
    boot = {k: np.array([b[k] for b in boot]) for k in point}
    print(f"Value [95% lunation-bootstrap interval], {n_boot} resamples")
    print(f"{'Stage':8s} {'Rule':9s} {'Threshold':>26s} {'J or F1 at the optimum':>26s}")
    for stage in ["Stage 1", "Stage 2"]:
        print(f"{stage:8s} {'Youden J':9s} "
              f"{fmt_ci(point[(stage, 'youden_thr')], boot[(stage, 'youden_thr')]):>26s} "
              f"{fmt_ci(point[(stage, 'j_max')], boot[(stage, 'j_max')]):>26s}")
        print(f"{stage:8s} {'max F1':9s} "
              f"{fmt_ci(point[(stage, 'f1_thr')], boot[(stage, 'f1_thr')]):>26s} "
              f"{fmt_ci(point[(stage, 'f1_max')], boot[(stage, 'f1_max')]):>26s}")

    print("\nJ and F1 at the adopted thresholds, and the shortfall from the maximum "
          "(maximum minus value;\nthe maximum is recomputed in every resample)")
    print(f"{'Stage':8s} {'Cut':>5s} {'J':>26s} {'J shortfall':>26s} {'F1':>26s} "
          f"{'F1 shortfall':>26s}")
    for stage in ["Stage 1", "Stage 2"]:
        for cut in ADOPTED[stage]:
            print(f"{stage:8s} {cut:5.2f} " + " ".join(
                f"{fmt_ci(point[(stage, m, cut)], boot[(stage, m, cut)]):>26s}"
                for m in ["j", "j_short", "f1", "f1_short"]))

    print("\nWidth of the optimum on the out-of-fold probabilities: grid thresholds "
          "(0.050-0.950, step 0.005)\nwhose J or F1 is within 0.01 and 0.02 of the maximum")
    for stage, (y, p) in stage_arrays(label, p_det, p_eye).items():
        f1 = f1_curve(y, p)
        jj = np.array([j_at(y, p, cut) for cut in F1_GRID])
        for name, curve in [("J", jj), ("F1", f1)]:
            spans = []
            for tol in (0.01, 0.02):
                near = F1_GRID[curve >= curve.max() - tol]
                spans.append(f"within {tol:.2f}: {near.min():.4f}-{near.max():.4f}")
            print(f"  {stage} {name:2s} (grid maximum {curve.max():.4f}): " + "; ".join(spans))


# ----------------------------------------------------------------------------
# Section 3: error rates at the adopted thresholds
# ----------------------------------------------------------------------------
def adopted_rules(label, p_det, p_eye, idx=None):
    if idx is None:
        idx = np.arange(len(label))
    lab, det, eye = label[idx], p_det[idx], p_eye[idx]
    seen = lab > 0
    return {
        "S1 >= 0.30": rates(seen, det >= DET_LOW),
        "S1 >= 0.50": rates(seen, det >= DET_HIGH),
        "S2 >= 0.30": rates(lab[seen] == 2, eye[seen] >= EYE_LOW),
        "S2 >= 0.60": rates(lab[seen] == 2, eye[seen] >= EYE_HIGH),
        "Naked-eye rule": rates(lab == 2, (det >= DET_HIGH) & (eye >= EYE_HIGH)),
    }


def section_error_rates(label, p_det, p_eye, rows, n_boot):
    print("\n" + "=" * 78)
    print("3. Error rates at the adopted thresholds (out-of-fold probabilities)")
    print("=" * 78)
    point = adopted_rules(label, p_det, p_eye)
    boot = [adopted_rules(label, p_det, p_eye, idx)
            for idx in lunation_resamples(rows, n_boot, BOOT_SEED)]
    metrics = [("sens", "Sensitivity (TPR)"), ("spec", "Specificity (TNR)"),
               ("prec", "Precision (PPV)"), ("fpr", "False-positive rate"),
               ("fnr", "False-negative rate"), ("share", "Share at or above cut"),
               ("acc", "Accuracy"), ("f1", "F1")]
    blocks = [
        ("Stage 1, all 4,470 reports, positive = seen by any means", ["S1 >= 0.30", "S1 >= 0.50"]),
        (f"Stage 2, the {int(np.sum(label > 0)):,} sightings, positive = naked eye",
         ["S2 >= 0.30", "S2 >= 0.60"]),
        ("Naked-eye rule P_det >= 0.50 and P_eye >= 0.60, all 4,470 reports, "
         "positive = naked eye", ["Naked-eye rule"]),
    ]
    print(f"Value [95% lunation-bootstrap interval], {n_boot} resamples")
    for title, names in blocks:
        print(f"\n{title}")
        print(f"{'':22s} " + " ".join(f"{name:>26s}" for name in names))
        for m, mname in metrics:
            print(f"{mname:22s} " + " ".join(
                f"{fmt_ci(point[name][m], [b[name][m] for b in boot]):>26s}" for name in names))
        print(f"{'TP / FP / FN / TN':22s} " + " ".join(
            f"{int(point[n]['tp'])} / {int(point[n]['fp'])} / {int(point[n]['fn'])} / "
            f"{int(point[n]['tn'])}".rjust(26) for n in names))

    naked = (p_det >= DET_HIGH) & (p_eye >= EYE_HIGH)
    false_pos = label[naked & (label != 2)]
    print(f"\nFalse positives of the naked-eye rule: {int(np.sum(false_pos == 1))} aided "
          f"sightings and {int(np.sum(false_pos == 0))} reports with no sighting.")
    print("\nImplied costs: with calibrated probabilities, calling a record positive when "
          "p >= t minimises the\nexpected cost when a false positive costs t / (1 - t) "
          "times a false negative:")
    for t in sorted({DET_LOW, DET_HIGH, EYE_HIGH}):
        print(f"  t = {t:.2f}: false positive / false negative cost = {t / (1 - t):.4f}")


# ----------------------------------------------------------------------------
# Section 4: calibration by band and by decile
# ----------------------------------------------------------------------------
def band_of(p_det, p_eye):
    return np.select(
        [p_det < DET_LOW, p_det < DET_HIGH, p_eye < EYE_LOW, p_eye < EYE_HIGH],
        ["E", "D", "C", "B"], default="A")


def band_fractions(label, p_det, p_eye, band, idx=None):
    """Observed and mean predicted fractions per band."""
    if idx is None:
        idx = np.arange(len(label))
    lab, det, eye, bnd = label[idx], p_det[idx], p_eye[idx], band[idx]
    out = {}
    for b, _, _ in BANDS:
        m = bnd == b
        seen = m & (lab > 0)
        out[(b, "seen")] = np.mean(lab[m] > 0) if m.any() else np.nan
        out[(b, "eye_det")] = np.mean(lab[seen] == 2) if seen.any() else np.nan
        out[(b, "eye_all")] = np.mean(lab[m] == 2) if m.any() else np.nan
    return out


def recalibration(y, p):
    """Calibration intercept (logit p as offset) and slope (logistic regression on logit p)."""
    y = np.asarray(y, dtype=float)
    z = np.log(p / (1 - p))
    X = np.column_stack([np.ones_like(z), z])
    slope = sm.GLM(y, X, family=sm.families.Binomial()).fit().params[1]
    intercept = sm.GLM(y, X[:, :1], family=sm.families.Binomial(), offset=z).fit().params[0]
    return intercept, slope


def decile_table(y, p, name, prob):
    y = np.asarray(y, dtype=bool)
    dec = pd.qcut(p, 10, labels=False)
    print(f"{'Decile':>6s} {'Range of ' + prob:>17s} {'n':>5s} {'Mean ' + prob:>10s} "
          f"{'Observed ' + name:>22s} {'Wilson 95% CI':>16s}")
    for d in range(10):
        m = dec == d
        k, n = int(y[m].sum()), int(m.sum())
        lo, hi = wilson(k, n)
        print(f"{d + 1:6d} {p[m].min():8.4f}-{p[m].max():.4f} {n:5d} {p[m].mean():10.4f} "
              f"{k:>13d} {k / n:8.4f} [{lo:.4f}, {hi:.4f}]")


def section_calibration(label, p_det, p_eye, groups, rows, n_boot, figure):
    print("\n" + "=" * 78)
    print("4. Calibration by band of Eq. 11 and by decile (out-of-fold probabilities)")
    print("=" * 78)
    band = band_of(p_det, p_eye)
    seen = label > 0
    boot = [band_fractions(label, p_det, p_eye, band, idx)
            for idx in lunation_resamples(rows, n_boot, BOOT_SEED)]

    print("Wilson intervals treat records as independent; the bootstrap intervals "
          f"resample lunations ({n_boot} resamples).")
    print("\nSeen by any means")
    print(f"{'Band':4s} {'Rule':36s} {'n':>5s} {'Lun.':>5s} {'Mean P_det':>10s} {'Seen':>5s} "
          f"{'Observed':>8s} {'Wilson 95% CI':>16s} {'Bootstrap 95% CI':>16s}")
    table = {}
    for b, _, rule in BANDS:
        m = band == b
        n, k = int(m.sum()), int(np.sum(seen[m]))
        lo, hi = wilson(k, n)
        blo, bhi = ci([x[(b, "seen")] for x in boot])
        table[b] = {"n": n, "seen": k, "frac_seen": k / n, "ci_seen": (lo, hi),
                    "mean_det": p_det[m].mean()}
        print(f"{b:4s} {rule:36s} {n:5d} {len(set(groups[m])):5d} {p_det[m].mean():10.4f} "
              f"{k:5d} {k / n:8.4f} [{lo:.4f}, {hi:.4f}] [{blo:.4f}, {bhi:.4f}]")
    print(f"{'All':4s} {'':36s} {len(label):5d} {len(set(groups)):5d} {p_det.mean():10.4f} "
          f"{int(seen.sum()):5d} {seen.mean():8.4f}")

    print("\nSeen by naked eye: among the sightings in the band (compare with mean P_eye "
          "of those sightings)\nand among all reports in the band (compare with mean "
          "P_det x P_eye of all reports)")
    print(f"{'Band':4s} {'Sightings':>9s} {'Mean P_eye':>10s} {'Naked':>5s} {'Observed':>8s} "
          f"{'Wilson 95% CI':>16s} {'Bootstrap 95% CI':>16s} | {'Mean P_eye':>10s} "
          f"{'Mean PdxPe':>10s} {'Observed':>8s} {'Wilson 95% CI':>16s} "
          f"{'Bootstrap 95% CI':>16s}")
    for b, _, _ in BANDS:
        m = band == b
        ms = m & seen
        n_s, k = int(ms.sum()), int(np.sum(label[ms] == 2))
        lo, hi = wilson(k, n_s)
        blo, bhi = ci([x[(b, "eye_det")] for x in boot])
        n = int(m.sum())
        alo, ahi = wilson(k, n)
        clo, chi = ci([x[(b, "eye_all")] for x in boot])
        table[b].update({"n_seen": n_s, "eye": k, "frac_eye_det": k / n_s if n_s else np.nan,
                         "ci_eye_det": (lo, hi), "mean_eye_det": p_eye[ms].mean(),
                         "frac_eye_all": k / n, "ci_eye_all": (alo, ahi),
                         "mean_joint": np.mean(p_det[m] * p_eye[m])})
        print(f"{b:4s} {n_s:9d} {p_eye[ms].mean():10.4f} {k:5d} {table[b]['frac_eye_det']:8.4f} "
              f"[{lo:.4f}, {hi:.4f}] [{blo:.4f}, {bhi:.4f}] | {p_eye[m].mean():10.4f} "
              f"{np.mean(p_det[m] * p_eye[m]):10.4f} {k / n:8.4f} [{alo:.4f}, {ahi:.4f}] "
              f"[{clo:.4f}, {chi:.4f}]")
    print("Left of '|': among the sightings in the band. Right: among all reports in the "
          "band;\nthe first 'Mean P_eye' column there is over all reports in the band.")
    print("Bands: " + "; ".join(f"{b} {name}" for b, name, _ in BANDS) + ".")

    print("\nStage 1 by decile of P_det (all reports)")
    decile_table(seen, p_det, "seen", "P_det")
    print("\nStage 2 by decile of P_eye (sightings only)")
    decile_table(label[seen] == 2, p_eye[seen], "naked eye", "P_eye")

    print("\nCalibration intercept (logit of the probability as an offset; 0 = no overall "
          "bias) and slope\n(logistic regression of the outcome on the logit; 1 = ideal), "
          "with 95% lunation-bootstrap intervals")
    arrays = stage_arrays(label, p_det, p_eye)
    cal_boot = {stage: [] for stage in arrays}
    for idx in lunation_resamples(rows, n_boot, BOOT_SEED):
        for stage, (y, p) in stage_arrays(label, p_det, p_eye, idx).items():
            cal_boot[stage].append(recalibration(y, p))
    print(f"{'Stage':8s} {'Mean predicted':>14s} {'Observed':>8s} {'Intercept':>26s} "
          f"{'Slope':>26s}")
    for stage, (y, p) in arrays.items():
        a, s = recalibration(y, p)
        cb = np.array(cal_boot[stage])
        print(f"{stage:8s} {p.mean():14.4f} {np.mean(y):8.4f} {fmt_ci(a, cb[:, 0]):>26s} "
              f"{fmt_ci(s, cb[:, 1]):>26s}")

    print("\nIf the share of sightings among real evenings differed from the ICOP share "
          "and the reports were\nselected on the outcome (label shift), a calibrated "
          "probability p would become\np' = p r / (p r + 1 - p), with r the ratio of the "
          "new to the ICOP odds of the outcome.\nIf instead the reports were selected "
          "only on the sky geometry, P(outcome | geometry) is unchanged.")
    shares = [0.10, 0.25, 0.50, 0.75, 0.90]
    print(f"{'Probability':22s} {'ICOP share':>10s} " + " ".join(
        f"{f'share {s:.2f}':>11s}" for s in shares))
    for stage, cuts in ADOPTED.items():
        y = arrays[stage][0]
        base = np.mean(y)
        name = "P_det" if stage == "Stage 1" else "P_eye"
        for cut in cuts:
            row = []
            for s in shares:
                r = (s / (1 - s)) / (base / (1 - base))
                row.append(cut * r / (cut * r + 1 - cut))
            print(f"{f'{name} = {cut:.2f}':22s} {base:10.4f} " + " ".join(
                f"{v:11.4f}" for v in row))

    plot_bands(table, figure)
    print(f"\nSaved {figure}")


def plot_bands(table, figure):
    """Observed fractions per band (95% Wilson intervals) against the mean predicted probability."""
    letters = [b for b, _, _ in BANDS]
    names = ["not\ndetectable", "marginal", "aid-assisted\nlikely", "borderline\nnaked eye",
             "naked eye\nlikely"]
    x = np.arange(len(letters))
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.2), sharey=True)

    def draw(ax, xs, frac, lo, hi, pred, ns, colour, marker, n_above):
        """Observed fractions with intervals, mean predictions as dashes, n beside each."""
        frac, lo, hi = np.array(frac), np.array(lo), np.array(hi)
        ax.plot(xs, pred, linestyle="none", marker="_", markersize=16, markeredgewidth=2,
                color=INK, zorder=2)
        ax.errorbar(xs, frac, yerr=[frac - lo, hi - frac], fmt=marker, color=colour,
                    markersize=7, markeredgecolor="white", markeredgewidth=1.2,
                    elinewidth=1.6, capsize=3, zorder=3)
        for xi, n, l, h in zip(xs, ns, lo, hi):
            y_text, va = (h + 0.025, "bottom") if n_above else (l - 0.025, "top")
            ax.text(xi, y_text, f"n={n:,}", ha="center", va=va, fontsize=7.5, color=MUTED)

    def handle(marker, colour, label):
        if marker == "_":
            return Line2D([], [], linestyle="none", marker="_", markersize=16,
                          markeredgewidth=2, color=INK, label=label)
        return Line2D([], [], linestyle="none", marker=marker, markersize=7, color=colour,
                      markeredgecolor="white", markeredgewidth=1.2, label=label)

    # (a) seen by any means
    ax = axes[0]
    draw(ax, x, [table[b]["frac_seen"] for b in letters],
         [table[b]["ci_seen"][0] for b in letters], [table[b]["ci_seen"][1] for b in letters],
         [table[b]["mean_det"] for b in letters], [table[b]["n"] for b in letters],
         BLUE, "o", True)
    ax.set_ylabel("Fraction of reports")
    ax.set_title("(a) Seen by any means", loc="left", fontsize=9.5, fontweight="bold")
    ax.legend(handles=[handle("o", BLUE, "Observed fraction seen, 95% Wilson interval"),
                       handle("_", INK, "Mean predicted $P_\\mathrm{det}$")],
              loc="upper left", bbox_to_anchor=(0.0, -0.27), frameon=False, fontsize=8)

    # (b) seen by naked eye
    ax = axes[1]
    off = 0.17
    draw(ax, x - off, [table[b]["frac_eye_det"] for b in letters],
         [table[b]["ci_eye_det"][0] for b in letters],
         [table[b]["ci_eye_det"][1] for b in letters],
         [table[b]["mean_eye_det"] for b in letters], [table[b]["n_seen"] for b in letters],
         BLUE, "o", True)
    draw(ax, x + off, [table[b]["frac_eye_all"] for b in letters],
         [table[b]["ci_eye_all"][0] for b in letters],
         [table[b]["ci_eye_all"][1] for b in letters],
         [table[b]["mean_joint"] for b in letters], [table[b]["n"] for b in letters],
         ORANGE, "s", False)
    ax.set_title("(b) Seen by naked eye", loc="left", fontsize=9.5, fontweight="bold")
    ax.legend(handles=[
        handle("o", BLUE, "Observed naked-eye fraction among sightings, 95% Wilson interval"),
        handle("s", ORANGE, "Observed naked-eye fraction among all reports, 95% Wilson interval"),
        handle("_", INK, "Mean predicted $P_\\mathrm{eye}$ (circles) or "
                         "$P_\\mathrm{det}\\,P_\\mathrm{eye}$ (squares)")],
        loc="upper left", bbox_to_anchor=(0.0, -0.27), frameon=False, fontsize=8)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b}\n{n}" for b, n in zip(letters, names)], fontsize=8)
        ax.set_xlim(-0.6, len(letters) - 0.4)
        ax.set_ylim(-0.08, 1.08)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("Operational band")
    fig.tight_layout()
    fig.savefig(figure, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="crescent_features.csv")
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF,
                        help="two_stage_oof_predictions.csv")
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE,
                        help="output figure (PNG)")
    parser.add_argument("--n-boot", type=int, default=2000,
                        help="lunation bootstrap resamples")
    args = parser.parse_args()

    df = load_subset(args.input)
    oof = load_oof(args.oof, df)
    label = df[TARGET].to_numpy()
    groups = df[LUNATION].to_numpy()
    X = df[FEATURES].to_numpy(dtype=float)
    p_det, p_eye = oof["p_det"].to_numpy(), oof["p_eye"].to_numpy()
    rows = rows_by_lunation(groups)
    print(f"Modelling subset: {len(df):,} reports, {len(rows)} lunations, "
          f"{int(np.sum(label > 0)):,} sightings ({int(np.sum(label == 2)):,} naked eye)")

    section_nested(X, label, groups, oof)
    section_thresholds(label, p_det, p_eye, rows, args.n_boot)
    section_error_rates(label, p_det, p_eye, rows, args.n_boot)
    section_calibration(label, p_det, p_eye, groups, rows, args.n_boot, args.figure)


if __name__ == "__main__":
    main()
