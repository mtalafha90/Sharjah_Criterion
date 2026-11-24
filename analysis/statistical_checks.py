#!/usr/bin/env python3
"""Intervals, collinearity, non-linear terms and the Alrefay check (reviewer comment 7).

Comment 7 (statistical review) asks for:
  "Uncertainty is also missing from every headline number... I would ask for
  intervals, for example by bootstrap, on accuracy, AUC, precision and recall,
  and for the coefficients too."
  "Balanced accuracy, F1, Matthews correlation or kappa, and precision-recall
  curves for Stage 2, would give a fuller picture."
  "ARCV, ARCL and t_lag are strongly related, so the coefficients may be
  unstable. Please report the coefficients with confidence intervals, check
  collinearity (for example with variance inflation factors), and say whether
  any regularisation was used... interactions, squared terms or splines are
  worth testing. The authors should also say how many values were filled by
  median imputation and for which variables."
  "With 160 negative cases and no false positives, the 95% upper limit on the
  false-positive rate is still about 2% ... aided recall ... interval ...
  Please report intervals, state which thresholds were used, and check whether
  the Alrefay records overlap with ICOP."

Inputs (default paths in this repository):
  data/icop/crescent_features.csv          6,092 reports; the modelling subset
      (4,470 first-crescent reports: 2,319 not seen, 505 aided, 1,646 naked
      eye; 202 lunations, one ICOP monthly page each) is filtered as in
      04_train_two_stage_crescent_criterion.py
  data/icop/two_stage_oof_predictions.csv  out-of-fold P_det and P_eye (same
      rows, same order): StratifiedGroupKFold(5, shuffle=True, random_state=42),
      stratified on the three-class label, groups = page_url
  models/stage1_detectability_model.joblib and
      stage2_naked_eye_model.joblib (final models, fitted on all 4,470 rows),
      Table_I_positive_observations.csv and Table_II_negative_observations.csv

Predictors: ARCV = relative_altitude_deg, ARCL = elongation_deg and
lag = moon_lag_minutes, all at local sunset. Pipeline of both stages: median
imputation, standardisation, logistic regression (lbfgs, max_iter=4000, the
scikit-learn default L2 penalty with C=1).

Sections
  1  Out-of-fold metrics with 95% intervals. Cluster bootstrap over lunations:
     the 202 lunations are resampled with replacement (2,000 resamples, numpy
     default_rng(42)), with the out-of-fold probabilities held fixed, and the
     2.5th and 97.5th percentiles are reported. Stage 1 uses all reports and
     P_det >= 0.50; Stage 2 uses the sightings of the resampled lunations and
     P_eye >= 0.50 and >= 0.60. The three-class rule is the operational one
     (P_det < 0.50 -> not seen; P_det >= 0.50 and P_eye >= 0.60 -> naked eye;
     otherwise aided). The majority-class baseline is the share of the more
     common class in each resample.
  2  Coefficients of the final models: standardised and per original unit,
     with the scaler means and standard deviations, and 95% intervals from
     refitting both stages on 1,000 lunation resamples (default_rng(42));
     the resampled slopes are put on the final model's scale (per SD of the
     4,470 or 2,151 rows) before taking percentiles. Variance inflation
     factors and Pearson correlations of the predictors. The L2 (C=1) fit is
     compared with an unpenalised fit (C=inf, solver tolerance 1e-10): the
     coefficients, lunation-clustered (sandwich) Wald intervals for the
     unpenalised fit (statsmodels Logit, checked against scikit-learn), and
     the grouped-CV log-loss, AUC and Brier score on the folds of section 4.
  3  Missing predictor values filled by median imputation.
  4  Non-linear terms, by repeated grouped CV (StratifiedGroupKFold, 5 folds,
     shuffle=True, seeds 0-9, groups = page_url; Stage 1 on all reports,
     stratified on seen; Stage 2 on the sightings, stratified on naked eye,
     with its own folds, as in predictor_ablation.py). Every model sees the
     same folds, so differences against the linear model are paired. Squares
     and products are formed from the standardised predictors and then
     standardised again, so the same L2 penalty (C=1) treats all terms alike.
     Two spline bases, additive in each predictor: cubic B-splines
     (SplineTransformer, 4 knots at the 0, 1/3, 2/3 and 1 quantiles, linear
     beyond the outer knots), and natural (restricted) cubic splines with 4
     knots at the 5th, 35th, 65th and 95th percentiles (cubic between the
     knots, linear beyond the outer ones; Harrell's basis). SplineTransformer
     has no natural-spline option, hence both.
  5  Out-of-fold precision-recall curves (figure, 300 dpi, no title):
     (a) Stage 2 (sightings only), with naked eye and, scored by 1 - P_eye,
     aided as the positive class; (b) Stage 1, with seen and not seen as the
     positive class. Average precision and the no-skill line (the share of the
     positive class) are shown; markers give the operating points of the
     cutoffs (circle 0.50, square 0.60).
  6  Alrefay external check with the final saved models (Stage 1 at
     P_det >= 0.50, Stage 2 at P_eye >= 0.60, as in
     compare_twostep_with_alrefay_rowwise.py; Stage 2 also at 0.50), with
     Clopper-Pearson 95% intervals. Overlap with ICOP: each Alrefay record
     (both tables) is matched to the 6,092 ICOP reports by date (Alrefay
     Date_UT against the local date of sunset, +/-1 day) and location
     (|dLat| <= 0.5 deg and |dLong| <= 0.5 deg). For same-day matches the two
     outcomes are compared. The check is repeated without the Alrefay records
     that match any ICOP report.

Run: python3 -W ignore statistical_checks.py   (takes a few minutes)
"""
import argparse
import re
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import binomtest, rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             cohen_kappa_score, f1_score, log_loss,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, PolynomialFeatures,
                                   SplineTransformer, StandardScaler)
from statsmodels.stats.outliers_influence import variance_inflation_factor

ARCV = "relative_altitude_deg"
ARCL = "elongation_deg"
LAG = "moon_lag_minutes"
FEATURES = [ARCV, ARCL, LAG]
SHORT = {ARCV: "ARCV", ARCL: "ARCL", LAG: "lag"}
UNITS = {ARCV: "per deg", ARCL: "per deg", LAG: "per min"}
TARGET = "vis_class_empirical"
LUNATION = "page_url"
PDET, PEYE = 0.50, 0.60
N_FOLDS, SEEDS = 5, range(10)
BOOT_SEED = 42
CLASS_NAMES = ["not seen", "aided", "naked eye"]

HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # the repository root
DEFAULT_INPUT = REPO / "data" / "icop" / "crescent_features.csv"
DEFAULT_OOF = REPO / "data" / "icop" / "two_stage_oof_predictions.csv"
DEFAULT_MODELS = REPO / "models"
DEFAULT_ALREFAY = REPO / "data" / "alrefay"
DEFAULT_FIGURE = REPO / "figures" / "precision_recall_curves.png"

# Reference palette (blue, orange) with dark ink and a recessive grid.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_subset(path):
    """All feature rows and the 4,470-row modelling subset (source order kept)."""
    full = pd.read_csv(path, low_memory=False)
    df = full[
        (full["section_type"] == "first_crescent")
        & (full["obs_relative_to_sunset"] != "before_sunset")
        & full[TARGET].notna()
    ].copy()
    df[TARGET] = df[TARGET].astype(int)
    if len(df) != 4470 or tuple(df[TARGET].value_counts().sort_index()) != (2319, 505, 1646):
        raise ValueError("Source data have changed; verify the cohort before comparison")
    df["source_row"] = df.index
    return full, df.reset_index(drop=True)


def load_oof(path, df):
    oof = pd.read_csv(path)
    if (len(oof) != len(df) or not (oof[LUNATION].to_numpy() == df[LUNATION].to_numpy()).all()
            or not (oof[TARGET].to_numpy() == df[TARGET].to_numpy()).all()
            or not np.allclose(oof[FEATURES].to_numpy(), df[FEATURES].to_numpy())):
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


# ----------------------------------------------------------------------------
# Section 1: metrics
# ----------------------------------------------------------------------------
def rank_auc(y, p):
    """ROC AUC as the Mann-Whitney statistic (ties count one half)."""
    n1 = y.sum()
    n0 = len(y) - n1
    return (rankdata(p)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def binary_metrics(y, p, cut):
    y = np.asarray(y, dtype=bool)
    pred = p >= cut
    tp = float(np.sum(pred & y))
    fp = float(np.sum(pred & ~y))
    fn = float(np.sum(~pred & y))
    tn = float(np.sum(~pred & ~y))
    n = tp + fp + fn + tn
    acc = (tp + tn) / n
    rec_pos, rec_neg = tp / (tp + fn), tn / (tn + fp)
    prec_pos = tp / (tp + fp) if tp + fp else np.nan
    prec_neg = tn / (tn + fn) if tn + fn else np.nan
    chance = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    share = y.mean()
    return {
        "acc": acc,
        "majority": max(share, 1 - share),
        "gain": acc - max(share, 1 - share),
        "auc": rank_auc(y, p),
        "ap_pos": average_precision_score(y, p),
        "ap_neg": average_precision_score(~y, 1 - p),
        "prec_pos": prec_pos,
        "rec_pos": rec_pos,
        "f1_pos": 2 * tp / (2 * tp + fp + fn),
        "prec_neg": prec_neg,
        "rec_neg": rec_neg,
        "f1_neg": 2 * tn / (2 * tn + fn + fp),
        "bal_acc": (rec_pos + rec_neg) / 2,
        "mcc": (tp * tn - fp * fn) / denom if denom else np.nan,
        "kappa": (acc - chance) / (1 - chance),
        "brier": np.mean((p - y) ** 2),
        "cm": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def binary_labels(pos, neg):
    return [
        ("acc", "Accuracy"),
        ("majority", "Majority-class accuracy"),
        ("gain", "Accuracy minus majority"),
        ("auc", "ROC AUC"),
        ("ap_pos", f"Average precision ({pos})"),
        ("ap_neg", f"Average precision ({neg})"),
        ("prec_pos", f"Precision ({pos})"),
        ("rec_pos", f"Recall ({pos})"),
        ("f1_pos", f"F1 ({pos})"),
        ("prec_neg", f"Precision ({neg})"),
        ("rec_neg", f"Recall ({neg})"),
        ("f1_neg", f"F1 ({neg})"),
        ("bal_acc", "Balanced accuracy"),
        ("mcc", "Matthews correlation"),
        ("kappa", "Cohen's kappa"),
        ("brier", "Brier score"),
    ]


def three_class_metrics(y, pdet, peye):
    pred = np.where(pdet < PDET, 0, np.where(peye >= PEYE, 2, 1))
    cm = np.bincount(3 * y + pred, minlength=9).reshape(3, 3).astype(float)
    n = cm.sum()
    recall = np.diag(cm) / cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.diag(cm) / cm.sum(axis=0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    acc = np.trace(cm) / n
    chance = (cm.sum(axis=1) * cm.sum(axis=0)).sum() / n ** 2
    out = {
        "acc": acc,
        "majority": cm.sum(axis=1).max() / n,
        "gain": acc - cm.sum(axis=1).max() / n,
        "bal_acc": recall.mean(),
        "macro_f1": f1.mean(),
        "kappa": (acc - chance) / (1 - chance),
        "cm": cm.astype(int).tolist(),
    }
    for k in range(3):
        out[f"rec_{k}"] = recall[k]
        out[f"prec_{k}"] = precision[k]
    return out, pred


THREE_CLASS_LABELS = (
    [("acc", "Accuracy"), ("majority", "Majority-class accuracy"),
     ("gain", "Accuracy minus majority"), ("bal_acc", "Balanced accuracy"),
     ("macro_f1", "Macro-F1"), ("kappa", "Cohen's kappa")]
    + [(f"rec_{k}", f"Recall ({CLASS_NAMES[k]})") for k in range(3)]
    + [(f"prec_{k}", f"Precision ({CLASS_NAMES[k]})") for k in range(3)]
)


def oof_metrics(y3, pdet, peye):
    seen = y3 > 0
    return {
        "stage1": binary_metrics(seen, pdet, PDET),
        "stage2_050": binary_metrics(y3[seen] == 2, peye[seen], 0.50),
        "stage2_060": binary_metrics(y3[seen] == 2, peye[seen], PEYE),
        "three": three_class_metrics(y3, pdet, peye)[0],
    }


def check_against_sklearn(y3, pdet, peye, point):
    """The hand-written metrics must equal scikit-learn's on the full data."""
    seen = y3 > 0
    for key, y, p, cut in [("stage1", seen, pdet, PDET),
                           ("stage2_050", y3[seen] == 2, peye[seen], 0.50),
                           ("stage2_060", y3[seen] == 2, peye[seen], PEYE)]:
        m, pred = point[key], p >= cut
        ref = {
            "acc": accuracy_score(y, pred), "auc": roc_auc_score(y, p),
            "prec_pos": precision_score(y, pred), "rec_pos": recall_score(y, pred),
            "f1_pos": f1_score(y, pred), "bal_acc": balanced_accuracy_score(y, pred),
            "mcc": matthews_corrcoef(y, pred), "kappa": cohen_kappa_score(y, pred),
            "brier": brier_score_loss(y, p),
        }
        for name, value in ref.items():
            if not np.isclose(m[name], value):
                raise RuntimeError(f"{key} {name}: {m[name]} != scikit-learn {value}")
    m, pred = three_class_metrics(y3, pdet, peye)
    ref = {"acc": accuracy_score(y3, pred), "bal_acc": balanced_accuracy_score(y3, pred),
           "macro_f1": f1_score(y3, pred, average="macro"),
           "kappa": cohen_kappa_score(y3, pred)}
    ref.update({f"rec_{k}": v for k, v in enumerate(recall_score(y3, pred, average=None))})
    ref.update({f"prec_{k}": v for k, v in enumerate(precision_score(y3, pred, average=None))})
    for name, value in ref.items():
        if not np.isclose(m[name], value):
            raise RuntimeError(f"three-class {name}: {m[name]} != scikit-learn {value}")


def print_block(title, labels, point, boot):
    print(f"\n{title}")
    print(f"{'Metric':34s} {'Estimate':>8s}  95% interval")
    for key, label in labels:
        lo, hi = ci([b[key] for b in boot])
        print(f"{label:34s} {point[key]:8.4f}  [{lo:.4f}, {hi:.4f}]")


def section_metrics(df, oof, n_boot):
    y3 = df[TARGET].to_numpy()
    pdet, peye = oof["p_det"].to_numpy(), oof["p_eye"].to_numpy()
    point = oof_metrics(y3, pdet, peye)
    check_against_sklearn(y3, pdet, peye, point)
    rows = rows_by_lunation(df[LUNATION])
    boot, sizes = [], []
    for idx in lunation_resamples(rows, n_boot, BOOT_SEED):
        boot.append(oof_metrics(y3[idx], pdet[idx], peye[idx]))
        sizes.append(len(idx))

    print("=" * 78)
    print("1. Out-of-fold metrics with 95% cluster-bootstrap intervals")
    print("=" * 78)
    print(f"Reports {len(df)}, lunations {len(rows)}, sightings {int((y3 > 0).sum())}; "
          f"{n_boot} lunation resamples (default_rng({BOOT_SEED})), percentile intervals; "
          "out-of-fold probabilities held fixed. Hand-written metrics checked against "
          "scikit-learn on the full data.")
    print(f"Reports per resample: median {int(np.median(sizes))}, range {min(sizes)}-{max(sizes)}")

    blocks = [
        ("stage1", "Stage 1, seen vs not seen, all reports, cutoff P_det >= 0.50",
         binary_labels("seen", "not seen")),
        ("stage2_050", "Stage 2, naked eye vs aided, sightings only, cutoff P_eye >= 0.50",
         binary_labels("naked eye", "aided")),
        ("stage2_060", "Stage 2, naked eye vs aided, sightings only, cutoff P_eye >= 0.60 "
         "(operational)", binary_labels("naked eye", "aided")),
    ]
    for key, title, labels in blocks:
        print_block(title, labels, point[key], [b[key] for b in boot])
        cm = point[key]["cm"]
        print(f"Confusion matrix [[TN, FP], [FN, TP]] = {cm}")

    print_block("Three-class operational rule (P_det < 0.50 -> not seen; P_det >= 0.50 and "
                "P_eye >= 0.60 -> naked eye; otherwise aided)",
                THREE_CLASS_LABELS, point["three"], [b["three"] for b in boot])
    cm = np.array(point["three"]["cm"])
    print("Confusion matrix (rows = reported, columns = predicted), counts and row %")
    print(f"{'':12s}" + "".join(f"{name:>20s}" for name in CLASS_NAMES) + f"{'Total':>8s}")
    for k in range(3):
        cells = "".join(f"{cm[k, j]:>11d} ({100 * cm[k, j] / cm[k].sum():5.1f}%)" for j in range(3))
        print(f"{CLASS_NAMES[k]:12s}{cells}{cm[k].sum():>8d}")
    print(f"{'Total':12s}" + "".join(f"{cm[:, j].sum():>20d}" for j in range(3)) + f"{cm.sum():>8d}")
    return point


# ----------------------------------------------------------------------------
# Section 2: coefficients, collinearity and regularisation
# ----------------------------------------------------------------------------
def linear_model(C=1.0, tol=1e-4):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=4000, solver="lbfgs", tol=tol)),
    ])


def unpenalised_model():
    """C = inf; a tight tolerance, as the default (1e-4) stops ~0.002 short of the optimum."""
    return linear_model(C=np.inf, tol=1e-10)


def coefficients(model):
    """Intercept and slopes, standardised and per original unit."""
    clf, scaler = model.named_steps["clf"], model.named_steps["scaler"]
    beta, b0 = clf.coef_.ravel(), clf.intercept_[0]
    per_unit = beta / scaler.scale_
    return {"std": np.r_[b0, beta],
            "unit": np.r_[b0 - np.sum(per_unit * scaler.mean_), per_unit]}


def on_final_scale(unit, scaler):
    """Per-unit coefficients expressed per SD of the final model's scaling."""
    return np.r_[unit[0] + np.sum(unit[1:] * scaler.mean_), unit[1:] * scaler.scale_]


def section_coefficients(df, models, n_boot, cv_results):
    print("\n" + "=" * 78)
    print("2. Final-model coefficients, collinearity and regularisation")
    print("=" * 78)
    y3 = df[TARGET].to_numpy()
    seen = y3 > 0
    X = df[FEATURES]
    stages = [("Stage 1 (seen vs not seen; all 4,470 reports)", "stage1", np.ones(len(df), bool),
               (y3 > 0).astype(int)),
              ("Stage 2 (naked eye vs aided; 2,151 sightings)", "stage2", seen,
               (y3 == 2).astype(int))]

    # Refit on lunation resamples; Stage 2 on the sightings of each resample.
    rows = rows_by_lunation(df[LUNATION])
    boot = {"stage1": [], "stage2": []}
    for idx in lunation_resamples(rows, n_boot, BOOT_SEED):
        for _, key, mask, y in stages:
            sub = idx[mask[idx]]
            boot[key].append(coefficients(linear_model().fit(X.iloc[sub], y[sub])))

    terms = ["Intercept"] + [SHORT[f] for f in FEATURES]
    print(f"Intervals: {n_boot} lunation resamples (default_rng({BOOT_SEED})), both stages refitted "
          "on each; percentile 95% intervals. Standardised = log-odds per one SD of the "
          "predictor, using the final model's means and SDs also for the resampled fits; "
          "per unit = log-odds per degree or minute. "
          "'Positive' = share of resamples with a positive slope.")
    for title, key, mask, y in stages:
        model = models[key]
        saved = coefficients(model)
        refit = coefficients(linear_model().fit(X[mask], y[mask]))
        same = all(np.allclose(saved[s], refit[s], atol=1e-6) for s in saved)
        scaler = model.named_steps["scaler"]
        print(f"\n{title}; refit on all rows reproduces the saved model: {'yes' if same else 'NO'}")
        print(f"{'Predictor':10s} {'Mean':>9s} {'SD':>9s}   (StandardScaler, population SD)")
        for j, f in enumerate(FEATURES):
            print(f"{SHORT[f]:10s} {scaler.mean_[j]:9.4f} {scaler.scale_[j]:9.4f}")
        unit = np.array([b["unit"] for b in boot[key]])
        std = np.array([on_final_scale(u, scaler) for u in unit])
        print(f"{'Term':10s} {'Std coef':>9s} {'95% interval':>19s} {'OR per SD':>10s} "
              f"{'Per unit':>9s} {'95% interval':>19s} {'Unit':>8s} {'Positive':>9s}")
        for j, term in enumerate(terms):
            slo, shi = ci(std[:, j])
            ulo, uhi = ci(unit[:, j])
            odds = f"{np.exp(saved['std'][j]):10.4f}" if j else f"{'':10s}"
            unit_name = UNITS[FEATURES[j - 1]] if j else ""
            positive = f"{100 * np.mean(std[:, j] > 0):8.1f}%" if j else ""
            print(f"{term:10s} {saved['std'][j]:9.4f} [{slo:7.4f}, {shi:7.4f}] {odds} "
                  f"{saved['unit'][j]:9.4f} [{ulo:7.4f}, {uhi:7.4f}] {unit_name:>8s} {positive:>9s}")
        r = np.corrcoef(std[:, 1:].T)
        print("Correlation of the bootstrap slopes: "
              f"ARCV-ARCL {r[0, 1]:.4f}, ARCV-lag {r[0, 2]:.4f}, ARCL-lag {r[1, 2]:.4f}")

    print("\nCollinearity of the predictors (VIF from least squares with an intercept)")
    print(f"{'Predictor':10s} {'VIF all 4,470':>14s} {'VIF sightings 2,151':>20s}")
    vifs = {}
    for name, mask in [("all", np.ones(len(df), bool)), ("seen", seen)]:
        design = sm.add_constant(X[mask].to_numpy())
        vifs[name] = [variance_inflation_factor(design, j + 1) for j in range(len(FEATURES))]
    for j, f in enumerate(FEATURES):
        print(f"{SHORT[f]:10s} {vifs['all'][j]:14.4f} {vifs['seen'][j]:20.4f}")
    print(f"{'Pearson r':10s} {'all 4,470':>14s} {'sightings 2,151':>20s}")
    corr_all, corr_seen = np.corrcoef(X.to_numpy().T), np.corrcoef(X[seen].to_numpy().T)
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        pair = f"{SHORT[FEATURES[a]]}-{SHORT[FEATURES[b]]}"
        print(f"{pair:10s} {corr_all[a, b]:14.4f} {corr_seen[a, b]:20.4f}")

    print("\nRegularisation: the final models use scikit-learn's default L2 penalty, C = 1, "
          "on standardised predictors. Unpenalised fit (C = inf) for comparison; "
          "lunation-clustered (sandwich) Wald 95% intervals from statsmodels Logit on the "
          "same standardised predictors. SE ratio = clustered / model-based standard error.")
    for title, key, mask, y in stages:
        scaler = models[key].named_steps["scaler"]
        l2 = coefficients(models[key])["std"]
        none = coefficients(unpenalised_model().fit(X[mask], y[mask]))["std"]
        z = sm.add_constant((X[mask].to_numpy() - scaler.mean_) / scaler.scale_)
        groups = pd.factorize(df.loc[mask, LUNATION])[0]
        clustered = sm.Logit(y[mask], z).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
        naive = sm.Logit(y[mask], z).fit(disp=0)
        if not np.allclose(clustered.params, none, atol=1e-5):
            raise RuntimeError("statsmodels and scikit-learn unpenalised fits differ")
        print(f"\n{title}")
        print(f"{'Term':10s} {'L2, C=1':>9s} {'C=inf':>9s} {'Diff':>8s} "
              f"{'Clustered 95% interval':>23s} {'SE ratio':>9s}")
        bounds = clustered.conf_int()
        for j, term in enumerate(terms):
            print(f"{term:10s} {l2[j]:9.4f} {none[j]:9.4f} {l2[j] - none[j]:8.4f} "
                  f"[{bounds[j, 0]:9.4f}, {bounds[j, 1]:9.4f}] "
                  f"{clustered.bse[j] / naive.bse[j]:9.4f}")

    print("\nGrouped-CV log-loss, AUC and Brier of the two penalties (same folds as section 4)")
    for title, folds, res in cv_results:
        print_cv(title, folds, res, [MODELS[0][0], MODELS[-1][0]])


# ----------------------------------------------------------------------------
# Section 4: non-linear terms (and the L2 versus unpenalised log-loss)
# ----------------------------------------------------------------------------
def add_squares(X):
    return np.hstack([X, X ** 2])


class NaturalCubicSpline(TransformerMixin, BaseEstimator):
    """Restricted (natural) cubic spline basis for each column (Harrell's form).

    For knots t_1 < ... < t_k the basis is x and, for i = 1 .. k-2,
    [(x-t_i)+^3 - (x-t_{k-1})+^3 (t_k-t_i)/(t_k-t_{k-1})
     + (x-t_k)+^3 (t_{k-1}-t_i)/(t_k-t_{k-1})] / (t_k-t_1)^2,
    which is cubic between the knots and linear beyond the outer ones.
    """

    def __init__(self, quantiles=(0.05, 0.35, 0.65, 0.95)):
        self.quantiles = quantiles

    def fit(self, X, y=None):
        self.knots_ = np.quantile(np.asarray(X, dtype=float), self.quantiles, axis=0).T
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        cols = []
        for j, t in enumerate(self.knots_):
            x, k = X[:, j], len(t)
            cols.append(x)
            for i in range(k - 2):
                cols.append((np.clip(x - t[i], 0, None) ** 3
                             - np.clip(x - t[k - 2], 0, None) ** 3 * (t[k - 1] - t[i]) / (t[k - 1] - t[k - 2])
                             + np.clip(x - t[k - 1], 0, None) ** 3 * (t[k - 2] - t[i]) / (t[k - 1] - t[k - 2]))
                            / (t[k - 1] - t[0]) ** 2)
        return np.column_stack(cols)


def expanded_model(expansion, C=1.0):
    """Impute and standardise, expand, standardise again, then fit."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("expand", expansion),
        ("rescale", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=4000, solver="lbfgs")),
    ])


def spline_model(basis):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("spline", basis),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, solver="lbfgs")),
    ])


# (label, number of terms, model factory); the first is the reference.
MODELS = [
    ("Linear (final model)", 3, lambda: linear_model()),
    ("+ squares", 6, lambda: expanded_model(FunctionTransformer(add_squares))),
    ("+ interactions", 6, lambda: expanded_model(
        PolynomialFeatures(2, interaction_only=True, include_bias=False))),
    ("+ squares and interactions", 9, lambda: expanded_model(
        PolynomialFeatures(2, include_bias=False))),
    ("Cubic B-spline, 4 knots", 15, lambda: spline_model(SplineTransformer(
        n_knots=4, degree=3, knots="quantile", extrapolation="linear", include_bias=False))),
    ("Natural cubic spline, 4 knots", 9, lambda: spline_model(NaturalCubicSpline())),
    ("Linear, unpenalised (C=inf)", 3, unpenalised_model),
]
CV_METRICS = [("logloss", "Log-loss", -1), ("auc", "ROC AUC", 1), ("brier", "Brier", -1)]


def grouped_folds(df, y):
    """Repeated 5-fold splits that keep each lunation whole."""
    groups = df[LUNATION].to_numpy()
    folds = []
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for train_idx, test_idx in cv.split(df, y, groups=groups):
            if set(groups[train_idx]) & set(groups[test_idx]):
                raise RuntimeError("A lunation is in both training and test rows")
            folds.append((train_idx, test_idx))
    return folds


def cv_stage(df, y):
    folds = grouped_folds(df, y)
    res = {}
    for label, _, make in MODELS:
        scores = {m: [] for m, _, _ in CV_METRICS}
        for train_idx, test_idx in folds:
            model = make().fit(df.iloc[train_idx][FEATURES], y[train_idx])
            prob = model.predict_proba(df.iloc[test_idx][FEATURES])[:, 1]
            scores["logloss"].append(log_loss(y[test_idx], prob, labels=[0, 1]))
            scores["auc"].append(roc_auc_score(y[test_idx], prob))
            scores["brier"].append(brier_score_loss(y[test_idx], prob))
        res[label] = {m: np.array(v) for m, v in scores.items()}
    return folds, res


def print_cv(title, folds, res, labels):
    reference = MODELS[0][0]
    print(f"\n{title}: {len(folds)} folds ({N_FOLDS} folds x {len(SEEDS)} seeds)")
    print("Mean (SD) over folds")
    print(f"{'Model':31s} {'Terms':>5s} " + " ".join(f"{name:>17s}" for _, name, _ in CV_METRICS))
    for label, terms, _ in MODELS:
        if label not in labels:
            continue
        print(f"{label:31s} {terms:5d} " + " ".join(
            f"{res[label][m].mean():8.4f} ({res[label][m].std(ddof=1):6.4f})"
            for m, _, _ in CV_METRICS))
    print(f"Paired fold differences against '{reference}' (model minus linear): mean (SD), "
          "folds better / tied / worse")
    print(f"{'Model':31s} " + " ".join(f"{name:>27s}" for _, name, _ in CV_METRICS))
    for label in labels:
        if label == reference:
            continue
        cells = []
        for m, _, direction in CV_METRICS:
            d = res[label][m] - res[reference][m]
            tie = np.isclose(d, 0, atol=1e-12)
            better = (direction * d > 0) & ~tie
            worse = (direction * d < 0) & ~tie
            cells.append(f"{d.mean():8.4f} ({d.std(ddof=1):6.4f}) "
                         f"{f'{better.sum()}/{tie.sum()}/{worse.sum()}':>9s}")
        print(f"{label:31s} " + " ".join(f"{c:>27s}" for c in cells))


def run_cv(df):
    """Repeated grouped CV of every model in MODELS, per stage."""
    y3 = df[TARGET].to_numpy()
    seen_df = df[y3 > 0].reset_index(drop=True)
    stages = [("Stage 1: seen vs not seen, all reports", df, (y3 > 0).astype(int)),
              ("Stage 2: naked eye vs aided, sightings only", seen_df,
               (seen_df[TARGET].to_numpy() == 2).astype(int))]
    return [(title, *cv_stage(data, y)) for title, data, y in stages]


def section_cv(cv_results):
    print("\n" + "=" * 78)
    print("4. Non-linear terms: repeated lunation-grouped CV")
    print("=" * 78)
    print("Squares and products of standardised predictors; splines additive in each "
          "predictor; all models L2, C = 1, fitted on the training folds only. "
          "Better = lower log-loss or Brier, higher AUC.")
    nonlinear = [label for label, _, _ in MODELS[:-1]]
    for title, folds, res in cv_results:
        print_cv(title, folds, res, nonlinear)


# ----------------------------------------------------------------------------
# Section 3: imputation
# ----------------------------------------------------------------------------
def section_imputation(full, df, models):
    print("\n" + "=" * 78)
    print("3. Values filled by median imputation")
    print("=" * 78)
    print(f"{'Predictor':10s} {'Missing, 4,470 subset':>22s} {'Missing, 6,092 table':>21s} "
          f"{'Stage 1 median':>15s} {'Stage 2 median':>15s}")
    for j, f in enumerate(FEATURES):
        print(f"{SHORT[f]:10s} {int(df[f].isna().sum()):22d} {int(full[f].isna().sum()):21d} "
              f"{models['stage1'].named_steps['imputer'].statistics_[j]:15.4f} "
              f"{models['stage2'].named_steps['imputer'].statistics_[j]:15.4f}")
    print(f"Rows of the 6,092 table with any predictor missing: "
          f"{int(full[FEATURES].isna().any(axis=1).sum())}. The imputer is kept in the "
          "pipeline but filled no values in the final models.")


# ----------------------------------------------------------------------------
# Section 5: precision-recall curves
# ----------------------------------------------------------------------------
def pr_panel(ax, y, p, cuts, symbol, pos, neg, letter, title, legend_at):
    """Precision-recall curves with each class in turn as the positive class."""
    y = np.asarray(y, dtype=bool)
    markers = ["o", "s"]
    shares = [y.mean(), 1 - y.mean()]
    for k, (positive_is_y, name, colour, style) in enumerate(
            [(True, pos, BLUE, "-"), (False, neg, ORANGE, "--")]):
        target = y if positive_is_y else ~y
        score = p if positive_is_y else 1 - p
        precision, recall, _ = precision_recall_curve(target, score)
        ap = average_precision_score(target, score)
        ax.plot(recall, precision, drawstyle="steps-post", color=colour, lw=1.6, ls=style,
                label=f"{name} (AP = {ap:.3f})")
        # No-skill line; when the two lines are close, the lower label goes below its line.
        share = shares[k]
        below = abs(shares[0] - shares[1]) < 0.1 and share < shares[1 - k]
        ax.axhline(share, color=colour, lw=1.0, ls=":")
        ax.text(0.03, share + (-0.012 if below else 0.012), f"no skill, {name} ({share:.3f})",
                color=MUTED, fontsize=7.5, va="top" if below else "bottom")
        for cut, marker in zip(cuts, markers):
            # The positive class is called when p >= cut, the other one otherwise.
            pred = (p >= cut) if positive_is_y else (p < cut)
            tp = np.sum(pred & target)
            ax.plot(tp / target.sum(), tp / pred.sum(), marker=marker, ms=7, mfc=colour,
                    mec="white", mew=1.2, ls="none", zorder=5)
    handles, labels = ax.get_legend_handles_labels()
    for cut, marker in zip(cuts, markers):
        handles.append(plt.Line2D([], [], marker=marker, ms=7, mfc=MUTED, mec="white",
                                  mew=1.2, ls="none"))
        labels.append(f"cutoff {symbol} = {cut:.2f}")
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=legend_at, fontsize=8,
              frameon=False)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall", color=INK)
    ax.set_ylabel("Precision", color=INK)
    ax.set_title(f"({letter}) {title}", loc="left", fontsize=10, color=INK)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8.5)


def section_figure(df, oof, path):
    y3 = df[TARGET].to_numpy()
    pdet, peye = oof["p_det"].to_numpy(), oof["p_eye"].to_numpy()
    seen = y3 > 0
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    pr_panel(axes[0], y3[seen] == 2, peye[seen], [0.50, PEYE], r"$P_\mathrm{eye}$", "naked eye", "aided",
             "a", "Stage 2: naked eye vs aided (2,151 sightings)", (0.0, 0.30))
    pr_panel(axes[1], seen, pdet, [PDET], r"$P_\mathrm{det}$", "seen", "not seen",
             "b", "Stage 1: seen vs not seen (4,470 reports)", (0.0, 0.02))
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print("\n" + "=" * 78)
    print("5. Precision-recall curves")
    print("=" * 78)
    print(f"Saved {path} (300 dpi). Average precision with intervals: section 1.")


# ----------------------------------------------------------------------------
# Section 6: Alrefay external check
# ----------------------------------------------------------------------------
def read_alrefay(path, positive):
    tab = pd.read_csv(path)
    tab.columns = [re.sub(r"\s+", "_", str(c).strip()) for c in tab.columns]
    for col in ["Lat_deg", "Long_deg", "Lag_min", "ARCV_deg", "ARCL_deg"]:
        tab[col] = pd.to_numeric(tab[col], errors="coerce")
    tab["date"] = pd.to_datetime(tab["Date_UT"], format="%Y/%m/%d", errors="coerce")
    tab["seen"] = int(positive)
    if positive:
        # As in compare_twostep_with_alrefay_rowwise.py: Means ends in N or T.
        means = tab["Means"].astype(str).str.strip().str.upper()
        tab["naked_eye"] = np.where(means.str.contains(r"N\s*$"), 1.0,
                                    np.where(means.str.contains(r"T\s*$"), 0.0, np.nan))
    else:
        tab["naked_eye"] = np.nan
    tab["table"] = "I (seen)" if positive else "II (not seen)"
    return tab


def cp(k, n):
    lo, hi = binomtest(int(k), int(n)).proportion_ci(method="exact")
    return f"{k:4d}/{n:<4d} {k / n:7.4f}  [{lo:.4f}, {hi:.4f}]"


def alrefay_results(tab, stage1, stage2, label):
    """Stage 1 and Stage 2 counts with Clopper-Pearson intervals."""
    use = tab.dropna(subset=["Lag_min", "ARCV_deg", "ARCL_deg"])
    X = pd.DataFrame({ARCV: use["ARCV_deg"], ARCL: use["ARCL_deg"], LAG: use["Lag_min"]})
    pdet = stage1.predict_proba(X)[:, 1]
    peye = stage2.predict_proba(X)[:, 1]
    seen = use["seen"].to_numpy() == 1
    called = pdet >= PDET
    tn, fp = int(np.sum(~seen & ~called)), int(np.sum(~seen & called))
    fn, tp = int(np.sum(seen & ~called)), int(np.sum(seen & called))
    print(f"\n{label}")
    print(f"Stage 1, P_det >= {PDET:.2f}: {len(use)} records ({int(seen.sum())} seen, "
          f"{int((~seen).sum())} not seen); [[TN, FP], [FN, TP]] = [[{tn}, {fp}], [{fn}, {tp}]]")
    print(f"{'':32s} {'k/n':>9s} {'Rate':>7s}  Clopper-Pearson 95%")
    print(f"{'False-positive rate':32s} {cp(fp, tn + fp)}")
    if fp == 0:
        n0 = tn + fp
        print(f"{'':32s} one-sided 95% upper limit {1 - 0.05 ** (1 / n0):.4f}; "
              f"rule of three 3/{n0} = {3 / n0:.4f}")
    print(f"{'Specificity (not seen recall)':32s} {cp(tn, tn + fp)}")
    print(f"{'Sensitivity (seen recall)':32s} {cp(tp, tp + fn)}")
    print(f"{'Accuracy':32s} {cp(tn + tp, len(use))}")

    eye = use["naked_eye"].to_numpy()
    lab = ~np.isnan(eye)
    for cut in [PEYE, 0.50]:
        e, called_eye = eye[lab] == 1, peye[lab] >= cut
        tn2, fp2 = int(np.sum(~e & ~called_eye)), int(np.sum(~e & called_eye))
        fn2, tp2 = int(np.sum(e & ~called_eye)), int(np.sum(e & called_eye))
        note = " (operational)" if cut == PEYE else ""
        print(f"Stage 2, P_eye >= {cut:.2f}{note}: {int(lab.sum())} sightings with N or T "
              f"({int(e.sum())} naked eye, {int((~e).sum())} aided); "
              f"[[TN, FP], [FN, TP]] = [[{tn2}, {fp2}], [{fn2}, {tp2}]]")
        print(f"{'Accuracy':32s} {cp(tn2 + tp2, int(lab.sum()))}")
        print(f"{'Recall (aided)':32s} {cp(tn2, tn2 + fp2)}")
        print(f"{'Recall (naked eye)':32s} {cp(tp2, tp2 + fn2)}")
        print(f"{'Precision (aided)':32s} {cp(tn2, tn2 + fn2)}")
        print(f"{'Precision (naked eye)':32s} {cp(tp2, tp2 + fp2)}")
    print(f"Distinct dates: Stage 1 {use['date'].nunique()}, Stage 2 {use.loc[lab, 'date'].nunique()} "
          "(records from the same evening are not independent; the intervals assume they are)")


def section_alrefay(full, df, models, alrefay_dir):
    print("\n" + "=" * 78)
    print("6. Alrefay external check (final saved models)")
    print("=" * 78)
    tab = pd.concat([read_alrefay(alrefay_dir / "Table_I_positive_observations.csv", True),
                     read_alrefay(alrefay_dir / "Table_II_negative_observations.csv", False)],
                    ignore_index=True)
    stage1, stage2 = models["stage1"], models["stage2"]
    alrefay_results(tab, stage1, stage2, "All usable Alrefay records")

    # Match every Alrefay record against all 6,092 ICOP reports.
    icop_date = pd.to_datetime(full["sunset_local"].str[:10]).to_numpy().astype("datetime64[D]")
    offset = (icop_date[None, :] - tab["date"].to_numpy().astype("datetime64[D]")[:, None]).astype("int64")
    offset[tab["date"].isna().to_numpy()] = 99999  # an unreadable date matches nothing
    near = ((np.abs(full["latitude"].to_numpy()[None, :] - tab["Lat_deg"].to_numpy()[:, None]) <= 0.5)
            & (np.abs(full["longitude"].to_numpy()[None, :] - tab["Long_deg"].to_numpy()[:, None]) <= 0.5))
    match = near & (np.abs(offset) <= 1)
    exact = near & (offset == 0)
    tab["overlap"] = match.any(axis=1)
    in_subset = np.zeros(len(full), bool)
    in_subset[df["source_row"].to_numpy()] = True
    usable = tab[["Lag_min", "ARCV_deg", "ARCL_deg"]].notna().all(axis=1)
    no_coords = tab[["Lat_deg", "Long_deg"]].isna().any(axis=1) | tab["date"].isna()
    # ICOP class only for reports in the modelling subset (NaN, shown as '-', otherwise).
    icop_class = np.where(in_subset, full[TARGET].to_numpy(dtype=float), np.nan)

    print("\nOverlap with ICOP: Alrefay Date_UT vs local date of sunset (+/-1 day), "
          "|dLat| <= 0.5 deg and |dLong| <= 0.5 deg, against all 6,092 ICOP reports")
    print(f"Alrefay date range {tab['date'].min():%Y-%m-%d} to {tab['date'].max():%Y-%m-%d}; "
          f"ICOP sunset dates {icop_date.min()} to {icop_date.max()}")
    print(f"Records without coordinates or date (cannot be matched): {int(no_coords.sum())}")
    print(f"{'Table':14s} {'Records':>8s} {'Usable':>7s} {'Match +/-1 d':>13s} "
          f"{'Match same day':>15s} {'Usable matched':>15s}")
    for name, part in tab.groupby("table", sort=True):
        rows = part.index.to_numpy()
        print(f"{name:14s} {len(part):8d} {int(usable[rows].sum()):7d} "
              f"{int(match[rows].any(axis=1).sum()):13d} {int(exact[rows].any(axis=1).sum()):15d} "
              f"{int((usable[rows] & tab.loc[rows, 'overlap']).sum()):15d}")
    hit = match.any(axis=0)
    print(f"Matched records in total: {int(tab['overlap'].sum())} (same day: "
          f"{int(exact.any(axis=1).sum())}); ICOP reports matched: {int(hit.sum())}, of which in "
          f"the 4,470-report modelling subset: {int((hit & in_subset).sum())}")

    # Same-evening pairs: does the ICOP report give the same outcome as Alrefay?
    pairs = [(i, j) for i, j in zip(*np.nonzero(exact)) if not np.isnan(icop_class[j])]
    agree_seen = sum((tab.at[i, "seen"] == 1) == (icop_class[j] > 0) for i, j in pairs)
    both_seen = [(i, j) for i, j in pairs if tab.at[i, "seen"] == 1 and icop_class[j] > 0
                 and not np.isnan(tab.at[i, "naked_eye"])]
    agree_eye = sum((tab.at[i, "naked_eye"] == 1) == (icop_class[j] == 2) for i, j in both_seen)
    print(f"Same-day pairs with a modelling-subset ICOP report: {len(pairs)}; same seen/not seen outcome: "
          f"{agree_seen}; both seen: {len(both_seen)}, same naked eye/aided outcome: {agree_eye}")

    if tab["overlap"].any():
        print("\nMatched records. ICOP matches as class@offset: class 0 not seen, 1 aided, "
              "2 naked eye ('-' outside the modelling subset); offset = ICOP date minus Date_UT, "
              "days. ICOP lat and long of the closest-in-date match.")
        print(f"{'Table':14s} {'Date_UT':10s} {'Lat':>6s} {'Long':>6s} {'Means':>6s} "
              f"{'ICOP lat':>8s} {'ICOP long':>9s}  ICOP matches")
        for i in np.flatnonzero(tab["overlap"].to_numpy()):
            js = np.flatnonzero(match[i])
            js = js[np.argsort(np.abs(offset[i, js]), kind="stable")]
            found = " ".join(("-" if np.isnan(icop_class[j]) else str(int(icop_class[j])))
                             + f"@{offset[i, j]:+d}" for j in js)
            means = str(tab.at[i, "Means"]).strip() if tab.at[i, "seen"] else "-"
            print(f"{tab.at[i, 'table']:14s} {tab.at[i, 'Date_UT']:10s} {tab.at[i, 'Lat_deg']:6.2f} "
                  f"{tab.at[i, 'Long_deg']:6.2f} {means[-6:]:>6s} {full.at[js[0], 'latitude']:8.2f} "
                  f"{full.at[js[0], 'longitude']:9.2f}  {found}")
    alrefay_results(tab[~tab["overlap"]], stage1, stage2,
                    "Alrefay records with no ICOP match (+/-1 day, 0.5 deg)")


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="data/icop/crescent_features.csv")
    parser.add_argument("--oof", default=str(DEFAULT_OOF),
                        help="data/icop/two_stage_oof_predictions.csv")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS),
                        help="Folder with the final models (models/)")
    parser.add_argument("--alrefay-dir", default=str(DEFAULT_ALREFAY),
                        help="Folder with the Alrefay tables (data/alrefay)")
    parser.add_argument("--figure", default=str(DEFAULT_FIGURE), help="Figure path (PNG, 300 dpi)")
    parser.add_argument("--n-boot", type=int, default=2000, help="Resamples for the metrics")
    parser.add_argument("--n-boot-coef", type=int, default=1000,
                        help="Resamples for the coefficients (each refits both stages)")
    args = parser.parse_args()

    full, df = load_subset(args.input)
    oof = load_oof(args.oof, df)
    models_dir = Path(args.models_dir)
    models = {"stage1": joblib.load(models_dir / "stage1_detectability_model.joblib"),
              "stage2": joblib.load(models_dir / "stage2_naked_eye_model.joblib")}

    section_metrics(df, oof, args.n_boot)
    cv_results = run_cv(df)
    section_coefficients(df, models, args.n_boot_coef, cv_results)
    section_imputation(full, df, models)
    section_cv(cv_results)
    section_figure(df, oof, args.figure)
    section_alrefay(full, df, models, Path(args.alrefay_dir))


if __name__ == "__main__":
    main()
