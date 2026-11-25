#!/usr/bin/env python3
"""Sensitivity to the assumed observation time (second review round, comment 7).

The script recomputes ARCV and ARCL with the observation time set 15 and 30
minutes before and after local sunset, and compares the scores, coefficients
and predictions of both stages with those at sunset.

Every record of the modelling subset takes its geometry at local sunset
(obs_time_source is 'sunset_fallback' for all 4,470 rows), so there is no
subset with reported times to keep when the fallback records are dropped.
The time-shift test is therefore the one carried out here.

Inputs (default paths in this repository):
  data/icop/crescent_features.csv  the modelling subset (4,470 first-crescent
      reports: 2,319 not seen, 505 aided, 1,646 naked eye; 202 lunations, one
      ICOP monthly page each), filtered as in
      04_train_two_stage_crescent_criterion.py
  data/icop/two_stage_oof_predictions.csv  out-of-fold P_det and P_eye of the
      main evaluation, used only to check that the folds are reproduced
  models/stage1_detectability_model.joblib and
      stage2_naked_eye_model.joblib  the final models (fitted on all rows)

Both stages use the pipeline of 04: median imputation, standardisation and
logistic regression (lbfgs, max_iter=4000, default L2 penalty with C=1) on
ARCV, ARCL and lag. Stage 1 separates seen by any means from not seen (all
reports); Stage 2 separates naked eye from aided only (the sightings).

Sections
  1  Geometry. ARCV and ARCL are recomputed with PyEphem at sunset_local +
     delta, delta in {-30, -15, 0, +15, +30} min, with the archive's
     convention: ephem.Observer at the record's latitude and longitude,
     elevation 0, pressure 0 (airless, so geometric topocentric positions of
     the Moon and the Sun), ARCV = Moon altitude minus Sun altitude,
     ARCL = ephem.separation(Moon, Sun). sunset_local in the table is the
     sunset instant. Delta = 0 is first checked against relative_altitude_deg
     and elongation_deg. The lag (moonset minus sunset) does not depend on
     the observation time and is kept. Negative deltas mean observing before
     sunset, which is not realistic for evening first-crescent reports; they
     are shown only as a symmetric check.
  2  Scores. For each delta, out-of-fold Stage 1 and Stage 2 accuracy (cutoff
     0.50) and ROC AUC on the folds of the main evaluation:
     StratifiedGroupKFold(5, shuffle=True, random_state=42), stratified on the
     three-class label, groups = page_url, Stage 1 fitted on all training
     rows and Stage 2 on the seen training rows. The folds depend only on the
     labels and the lunations, so every delta sees identical folds. The
     change from delta = 0 (PyEphem) has a 95% cluster-bootstrap interval:
     the 202 lunations are resampled with replacement (2,000 resamples, numpy
     default_rng(42)), with the out-of-fold probabilities held fixed.
  3  Coefficients of both stages fitted on all records at each delta:
     standardised (as in the paper) and per degree / per minute (standardised
     coefficient divided by the feature's SD, intercept on the raw scale).
  4  Fixed model. The final model (delta = 0) applied to the shifted geometry
     of the same records: change in P_det and P_eye, and the share of records
     whose band of Eq. 11 or three-class prediction changes.
  5  Summary table.

The column '0 arch.' uses the archive values (relative_altitude_deg,
elongation_deg) instead of the PyEphem recomputation; it reproduces the
published scores and coefficients.

Run: python3 -W ignore analysis/observation_time_sensitivity.py   (well under a minute)
"""
import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import ephem
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ARCV = "relative_altitude_deg"
ARCL = "elongation_deg"
LAG = "moon_lag_minutes"
FEATURES = [ARCV, ARCL, LAG]
TARGET = "vis_class_empirical"
LUNATION = "page_url"
N_FOLDS = 5
CV_SEED = 42
BOOT_SEED = 42
DELTAS = [-30, -15, 0, 15, 30]  # minutes after the local sunset
REFERENCE = "0"
ARCHIVE = "0 arch."
# Thresholds of Eq. 11 and the cascade.
DET_LOW, DET_HIGH = 0.30, 0.50
EYE_LOW, EYE_HIGH = 0.30, 0.60
SCORES = [("acc1", "Stage 1 accuracy"), ("auc1", "Stage 1 ROC AUC"),
          ("acc2", "Stage 2 accuracy"), ("auc2", "Stage 2 ROC AUC")]

HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # the repository root
DEFAULT_INPUT = REPO / "data" / "icop" / "crescent_features.csv"
DEFAULT_OOF = REPO / "data" / "icop" / "two_stage_oof_predictions.csv"
DEFAULT_MODELS = REPO / "models"


def label(delta):
    return "0" if delta == 0 else f"{delta:+d}"


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
    if df[FEATURES].isna().any().any():
        raise ValueError("Missing ARCV, ARCL or lag in the modelling subset")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Section 1: geometry at sunset + delta
# ----------------------------------------------------------------------------
def geometry(df, deltas):
    """ARCV, ARCL and the Sun's altitude (deg) for every record and delta."""
    out = {d: np.empty((len(df), 3)) for d in deltas}
    for i, row in enumerate(df.itertuples(index=False)):
        sunset = datetime.fromisoformat(row.sunset_local)
        sunset = ephem.Date(sunset.astimezone(timezone.utc).replace(tzinfo=None))
        obs = ephem.Observer()
        obs.lat, obs.lon = str(row.latitude), str(row.longitude)  # strings are degrees
        obs.elevation = 0
        obs.pressure = 0  # airless: geometric topocentric positions
        for d in deltas:
            obs.date = ephem.Date(sunset + d * ephem.minute)
            sun, moon = ephem.Sun(obs), ephem.Moon(obs)
            out[d][i] = (math.degrees(moon.alt - sun.alt),
                         math.degrees(ephem.separation(moon, sun)),
                         math.degrees(sun.alt))
    return out


# ----------------------------------------------------------------------------
# Models and scores
# ----------------------------------------------------------------------------
def fit_logistic(X, y):
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, solver="lbfgs")),
    ])
    return model.fit(X, y)


def out_of_fold(X, y, folds):
    """P_det and P_eye for every row from models not fitted on its lunation (as in 04)."""
    p_det = np.full(len(X), np.nan)
    p_eye = np.full(len(X), np.nan)
    for train_idx, test_idx in folds:
        train, y_train = X.iloc[train_idx], y[train_idx]
        seen = y_train > 0
        stage1 = fit_logistic(train, seen.astype(int))
        stage2 = fit_logistic(train[seen], (y_train[seen] == 2).astype(int))
        p_det[test_idx] = stage1.predict_proba(X.iloc[test_idx])[:, 1]
        p_eye[test_idx] = stage2.predict_proba(X.iloc[test_idx])[:, 1]
    return p_det, p_eye


def rank_auc(y, p):
    """ROC AUC as the Mann-Whitney statistic (ties count one half)."""
    n1 = y.sum()
    n0 = len(y) - n1
    return (rankdata(p)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def scores(y, p_det, p_eye):
    seen = y > 0
    eye = y[seen] == 2
    return {
        "acc1": np.mean(seen == (p_det >= 0.5)),
        "auc1": rank_auc(seen, p_det),
        "acc2": np.mean(eye == (p_eye[seen] >= 0.5)),
        "auc2": rank_auc(eye, p_eye[seen]),
    }


def coefficients(model):
    """Standardised coefficients, and per degree / per minute with the raw-scale intercept."""
    clf, scaler = model.named_steps["clf"], model.named_steps["scaler"]
    std = np.r_[clf.intercept_[0], clf.coef_.ravel()]
    per_unit = clf.coef_.ravel() / scaler.scale_
    raw = np.r_[clf.intercept_[0] - np.sum(per_unit * scaler.mean_), per_unit]
    return std, raw


def bands(p_det, p_eye):
    """Bands of Eq. 11 (E, D, C, B, A) and the three-class cascade."""
    band = np.where(p_det < DET_LOW, "E", np.where(p_det < DET_HIGH, "D",
                    np.where(p_eye < EYE_LOW, "C", np.where(p_eye < EYE_HIGH, "B", "A"))))
    cls = np.where(p_det < DET_HIGH, 0, np.where(p_eye >= EYE_HIGH, 2, 1))
    return band, cls


def rows_by_lunation(groups):
    """Row indices of each lunation, for the cluster bootstrap."""
    codes, _ = pd.factorize(groups)
    order = np.argsort(codes, kind="stable")
    return np.split(order, np.cumsum(np.bincount(codes))[:-1])


def quartiles(x):
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return f"{q2:8.4f} [{q1:8.4f}, {q3:8.4f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="Path to data/icop/crescent_features.csv")
    parser.add_argument("--oof", default=str(DEFAULT_OOF),
                        help="Path to data/icop/two_stage_oof_predictions.csv")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS),
                        help="Folder with the two final-model .joblib files")
    parser.add_argument("--n-boot", type=int, default=2000, help="Cluster-bootstrap resamples")
    args = parser.parse_args()

    df = load_subset(args.input)
    y = df[TARGET].to_numpy()
    seen = y > 0
    print(f"Modelling subset: {len(df):,} records, {df[LUNATION].nunique()} lunations; "
          f"obs_time_source: " + ", ".join(f"{k} {v:,}" for k, v in
                                           df["obs_time_source"].value_counts().items()))

    # ---- Section 1 ----------------------------------------------------------
    geo = geometry(df, DELTAS)
    d_arcv = geo[0][:, 0] - df[ARCV].to_numpy()
    d_arcl = geo[0][:, 1] - df[ARCL].to_numpy()
    print("\n1. Geometry recomputed with PyEphem (airless, elevation 0, topocentric)")
    print("Check at delta = 0, PyEphem minus archive (absolute differences):")
    print(f"{'':6s} {'max (deg)':>11s} {'median (deg)':>13s} {'max (arcsec)':>13s}")
    for name, d in [("ARCV", d_arcv), ("ARCL", d_arcl)]:
        a = np.abs(d)
        print(f"{name:6s} {a.max():11.6f} {np.median(a):13.6f} {a.max() * 3600:13.2f}")
    print(f"Sun altitude at the stored sunset instant (PyEphem, airless): median "
          f"{np.median(geo[0][:, 2]):.4f} deg, range {geo[0][:, 2].min():.4f} to "
          f"{geo[0][:, 2].max():.4f}")

    print("\nChange from sunset (delta = 0, PyEphem): median [quartiles]")
    print(f"{'Delta (min)':>11s} {'ARCV change (deg)':>30s} {'ARCL change (deg)':>30s} "
          f"{'Sun altitude (deg)':>30s}")
    for d in DELTAS:
        print(f"{label(d):>11s} {quartiles(geo[d][:, 0] - geo[0][:, 0]):>30s} "
              f"{quartiles(geo[d][:, 1] - geo[0][:, 1]):>30s} {quartiles(geo[d][:, 2]):>30s}")
    # For scale: the spread of the sunset values across the archive.
    print("\nSpread of the archive's sunset values: SD; 5th, 50th and 95th percentiles; range")
    for name, col in [("ARCV (deg)", ARCV), ("ARCL (deg)", ARCL), ("lag (min)", LAG)]:
        v = df[col].to_numpy(float)
        p5, p50, p95 = np.percentile(v, [5, 50, 95])
        print(f"{name:11s} SD {v.std(ddof=1):6.2f}; {p5:7.2f} {p50:7.2f} {p95:7.2f}; "
              f"{v.min():7.2f} to {v.max():7.2f}")

    # Feature tables: PyEphem at each delta, and the archive values; lag kept.
    variants = {}
    for d in DELTAS:
        X = df[FEATURES].copy()
        X[ARCV], X[ARCL] = geo[d][:, 0], geo[d][:, 1]
        variants[label(d)] = X
    variants[ARCHIVE] = df[FEATURES].copy()
    names = list(variants)

    # ---- Section 2 ----------------------------------------------------------
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    folds = list(cv.split(df, y, groups=df[LUNATION]))
    groups = df[LUNATION].to_numpy()
    for train_idx, test_idx in folds:
        if set(groups[train_idx]) & set(groups[test_idx]):
            raise RuntimeError("A lunation is in both training and test rows")
    oof = {name: out_of_fold(X, y, folds) for name, X in variants.items()}

    ref_file = pd.read_csv(args.oof)
    if len(ref_file) != len(df) or not (ref_file[LUNATION].to_numpy() == groups).all():
        raise ValueError("Out-of-fold file does not match the modelling subset row by row")
    diff_det = np.abs(oof[ARCHIVE][0] - ref_file["p_det"].to_numpy()).max()
    diff_eye = np.abs(oof[ARCHIVE][1] - ref_file["p_eye"].to_numpy()).max()
    print(f"\nFolds check: archive-value OOF probabilities vs {Path(args.oof).name}: "
          f"max |diff| P_det {diff_det:.2e}, P_eye {diff_eye:.2e}")

    point = {name: scores(y, *oof[name]) for name in names}
    # sklearn cross-check of the rank AUC on the reference.
    if not np.isclose(point[REFERENCE]["auc1"], roc_auc_score(seen, oof[REFERENCE][0])):
        raise RuntimeError("Rank AUC disagrees with sklearn")

    # Cluster bootstrap of the change from delta = 0, probabilities held fixed.
    rows = rows_by_lunation(groups)
    rng = np.random.default_rng(BOOT_SEED)
    boot = {name: {k: [] for k, _ in SCORES} for name in names}
    for _ in range(args.n_boot):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        b = {name: scores(y[idx], oof[name][0][idx], oof[name][1][idx]) for name in names}
        for name in names:
            for k, _ in SCORES:
                boot[name][k].append(b[name][k] - b[REFERENCE][k])

    print(f"\n2. Out-of-fold scores (grouped 5-fold CV, random_state={CV_SEED}); change from "
          f"delta = 0 with 95% cluster-bootstrap interval\n   ({args.n_boot} lunation resamples, "
          f"default_rng({BOOT_SEED}), out-of-fold probabilities held fixed)")
    print(f"Stage 1: N = {len(y):,} ({int(seen.sum()):,} seen); Stage 2: N = {int(seen.sum()):,} "
          f"({int((y == 2).sum()):,} naked eye); accuracy at the 0.50 cutoff")
    print(f"{'Delta':>8s} " + " ".join(f"{lab:>36s}" for _, lab in SCORES))
    for name in names:
        cells = []
        for k, _ in SCORES:
            lo, hi = np.percentile(boot[name][k], [2.5, 97.5])
            change = point[name][k] - point[REFERENCE][k]
            cells.append(f"{point[name][k]:.4f} {change:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        print(f"{name:>8s} " + " ".join(f"{c:>36s}" for c in cells))

    # ---- Section 3 ----------------------------------------------------------
    fits = {}
    for name, X in variants.items():
        m1 = fit_logistic(X, seen.astype(int))
        m2 = fit_logistic(X[seen], (y[seen] == 2).astype(int))
        fits[name] = (m1, m2)
    print("\n3. Coefficients of the fits on all records (Stage 1: 4,470 records; "
          "Stage 2: 2,151 sightings)")
    head = ["Intercept", "ARCV", "ARCL", "lag"]
    head_raw = ["Intercept", "ARCV/deg", "ARCL/deg", "lag/min"]
    for s, stage in enumerate(["Stage 1 (seen vs not seen)", "Stage 2 (naked eye vs aided)"]):
        print(f"{stage}: standardised | per degree / per minute (raw-scale intercept)")
        print(f"{'Delta':>8s} " + " ".join(f"{h:>9s}" for h in head) + "  | "
              + " ".join(f"{h:>9s}" for h in head_raw))
        for name in names:
            std, raw = coefficients(fits[name][s])
            print(f"{name:>8s} " + " ".join(f"{v:9.4f}" for v in std) + "  | "
                  + " ".join(f"{v:9.4f}" for v in raw))

    # ---- Section 4 ----------------------------------------------------------
    models_dir = Path(args.models_dir)
    final1 = joblib.load(models_dir / "stage1_detectability_model.joblib")
    final2 = joblib.load(models_dir / "stage2_naked_eye_model.joblib")
    check = max(np.abs(final1.predict_proba(variants[ARCHIVE])[:, 1]
                       - fits[ARCHIVE][0].predict_proba(variants[ARCHIVE])[:, 1]).max(),
                np.abs(final2.predict_proba(variants[ARCHIVE])[:, 1]
                       - fits[ARCHIVE][1].predict_proba(variants[ARCHIVE])[:, 1]).max())
    print(f"\n4. Final model (fitted at sunset on all records) applied to the shifted geometry "
          f"of the same records\n   Saved model vs a refit on the archive values: "
          f"max |P difference| {check:.2e}")
    fixed = {name: (final1.predict_proba(X)[:, 1], final2.predict_proba(X)[:, 1])
             for name, X in variants.items()}
    ref_band, ref_cls = bands(*fixed[REFERENCE])
    print("Change from delta = 0 (PyEphem): median [quartiles]; share of the 4,470 records "
          "whose band (Eq. 11) or three-class prediction changes")
    print(f"{'Delta':>8s} {'P_det change':>30s} {'P_eye change':>30s} {'|P_det change|':>15s} "
          f"{'band changed':>13s} {'class changed':>14s}")
    shift = {}
    for name in names:
        dp_det = fixed[name][0] - fixed[REFERENCE][0]
        dp_eye = fixed[name][1] - fixed[REFERENCE][1]
        band, cls = bands(*fixed[name])
        shift[name] = (np.median(dp_det), np.median(np.abs(dp_det)), np.median(dp_eye),
                       np.mean(band != ref_band), np.mean(cls != ref_cls))
        print(f"{name:>8s} {quartiles(dp_det):>30s} {quartiles(dp_eye):>30s} "
              f"{shift[name][1]:15.4f} {100 * shift[name][3]:12.1f}% {100 * shift[name][4]:13.1f}%")

    # ---- Section 5 ----------------------------------------------------------
    print("\n5. Summary: observation time = local sunset + delta (minutes). Negative deltas "
          "(before sunset)\n   are a symmetric check only. '0 arch.' = archive geometry "
          "(published model).")
    summary = []
    for name in names:
        med_arcv = np.median(variants[name][ARCV] - variants[REFERENCE][ARCV])
        med_arcl = np.median(variants[name][ARCL] - variants[REFERENCE][ARCL])
        s1_std, s1_raw = coefficients(fits[name][0])
        s2_std, s2_raw = coefficients(fits[name][1])
        summary.append([med_arcv, med_arcl]
                       + [point[name][k] for k, _ in SCORES]
                       + list(s1_std[1:]) + list(s1_raw[1:])
                       + list(s2_std[1:]) + list(s2_raw[1:])
                       + [shift[name][0], shift[name][2], 100 * shift[name][3],
                          100 * shift[name][4]])
    rows_label = (["Median ARCV change (deg)", "Median ARCL change (deg)"]
                  + [lab for _, lab in SCORES]
                  + [f"Stage 1 {f} (standardised)" for f in ["ARCV", "ARCL", "lag"]]
                  + ["Stage 1 ARCV (per deg)", "Stage 1 ARCL (per deg)", "Stage 1 lag (per min)"]
                  + [f"Stage 2 {f} (standardised)" for f in ["ARCV", "ARCL", "lag"]]
                  + ["Stage 2 ARCV (per deg)", "Stage 2 ARCL (per deg)", "Stage 2 lag (per min)"]
                  + ["Final model: median P_det change", "Final model: median P_eye change",
                     "Final model: band changed (%)", "Final model: class changed (%)"])
    table = np.array(summary).T
    print(f"{'':34s} " + " ".join(f"{n:>9s}" for n in names))
    for lab, vals in zip(rows_label, table):
        fmt = "{:9.1f}" if lab.endswith("(%)") else "{:9.4f}"
        print(f"{lab:34s} " + " ".join(fmt.format(v) for v in vals))


if __name__ == "__main__":
    main()
