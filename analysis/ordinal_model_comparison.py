#!/usr/bin/env python3
"""Compare ordered and two-step three-class models
(second review round, comments 5 and 6).

This script fits the multinomial, ordered and two-step models of the three
classes on the same folds and compares their scores. It also shows the
accuracy of the multinomial model under grouped and random splits.

Input: data/icop/crescent_features.csv,
filtered exactly as in 04_train_two_stage_crescent_criterion.py. A lunation
is one ICOP monthly report page (page_url; 202 lunations). Every
cross-validation below is grouped by lunation and stratified on the
three-class label (StratifiedGroupKFold), so reports from one new moon are
never in both the training and the test rows. Only the '0.763' section also
shows random (ungrouped) splits, and labels them as such.
All models use ARCV, ARCL and lag time, with median imputation and
standardisation fitted on the training rows only:
  M1  multinomial logit (the direct three-class model in the manuscript)
  M2  proportional-odds cumulative logit
  M3  the two-step model, i.e. a sequential (continuation-ratio) logit with
      stage-specific slopes: P0 = 1-Pdet, P1 = Pdet(1-Peye), P2 = Pdet*Peye
      (the two pipelines of script 04, Stage 2 standardised on the seen rows)
  M4  sequential logit with common slopes and stage-specific intercepts
  B0  majority class
Sections:
  (a-c) out-of-fold predictions of the main split, StratifiedGroupKFold
        (5 folds, shuffle=True, random_state=42), as in script 04
  (d)   repeated grouped 5-fold CV (seeds 0-9, 50 folds), paired differences
  (e-f) full-data likelihood fits: AIC/BIC, LR test, Brant test, coefficients
  (g)   M1 under grouped and random set-ups (where 0.763 came from)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel

FEATURES = ["relative_altitude_deg", "elongation_deg", "moon_lag_minutes"]
NAMES = ["ARCV", "ARCL", "lag"]
TARGET = "vis_class_empirical"
GROUP = "page_url"
LABELS = [0, 1, 2]
DEFAULT_INPUT = (
    Path(__file__).resolve().parent.parent
    / "data" / "icop" / "crescent_features.csv"
)
DEFAULT_OOF = DEFAULT_INPUT.with_name("two_stage_oof_predictions.csv")
CV_METRICS = ["acc", "macro_f1", "aided_r", "logloss", "rps"]
LOWER_IS_BETTER = {"logloss", "rps"}
N_FOLDS = 5
MAIN_SEED = 42          # the main split of script 04
CV_SEEDS = range(10)    # repeated grouped CV


# ---------------------------------------------------------------- splits
def grouped_folds(X, y, groups, seed, n_splits=N_FOLDS):
    """Folds that keep each lunation whole, stratified on the three-class label."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(X, y, groups=groups))
    for tr, te in folds:
        if set(groups[tr]) & set(groups[te]):
            raise RuntimeError("A lunation is in both training and test rows")
    return folds


def predict_folds(X, y, folds, fit):
    """Class probabilities for the test rows of each fold, pooled over folds."""
    prob = np.full((len(y), 3), np.nan)
    for tr, te in folds:
        Ztr, Zte = preprocess(X[tr], X[te])
        prob[te] = fit(Ztr, y[tr])(Zte)
    test = ~np.isnan(prob[:, 0])
    return prob[test], y[test]


# ---------------------------------------------------------------- models
def preprocess(X_train, X_test):
    """Median imputation and standardisation fitted on the training rows only."""
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    return prep.fit_transform(X_train), prep.transform(X_test)


def fit_b0(Z, y):
    # Training class shares for every test row; argmax gives the majority class.
    prior = np.bincount(y, minlength=3) / len(y)
    return lambda Znew: np.tile(prior, (len(Znew), 1))


def sequential_probs(pdet, peye):
    return np.column_stack([1 - pdet, pdet * (1 - peye), pdet * peye])


def check(res, label):
    if not res.mle_retvals.get("converged", True):
        raise RuntimeError(f"{label} did not converge")
    return res


def fit_m1(Z, y):
    # Same estimator and settings as the published binary stages (L2, C=1).
    return LogisticRegression(max_iter=4000, solver="lbfgs").fit(Z, y).predict_proba


def fit_m2(Z, y):
    # No constant column: the two thresholds act as the intercepts.
    # Newton steps reach the exact maximum-likelihood estimate (BFGS stops slightly short).
    res = check(OrderedModel(y, Z, distr="logit").fit(
        method="newton", maxiter=100, disp=False), "M2")
    return lambda Znew: res.model.predict(res.params, exog=Znew)


def fit_m3(Z, y):
    # The paper's two binary pipelines, trained on the same rows as M1, M2, M4.
    # Stage 2 is standardised again on the seen rows, as in script 04; z-scoring
    # Z[seen] equals z-scoring the raw seen rows (the three features have no gaps).
    seen = y > 0
    s1 = LogisticRegression(max_iter=4000, solver="lbfgs").fit(Z, seen.astype(int))
    s2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, solver="lbfgs")).fit(
        Z[seen], (y[seen] == 2).astype(int))
    return lambda Znew: sequential_probs(
        s1.predict_proba(Znew)[:, 1], s2.predict_proba(Znew)[:, 1])


def expand_sequential(Z, y):
    """One stage-1 record per row and one stage-2 record per seen row."""
    seen = y > 0
    Zx = np.vstack([Z, Z[seen]])
    stage = np.r_[np.zeros(len(y)), np.ones(seen.sum())]
    resp = np.r_[seen.astype(int), (y[seen] == 2).astype(int)]
    return np.column_stack([np.ones(len(resp)), stage, Zx]), resp


def fit_m4_result(Z, y):
    D, r = expand_sequential(Z, y)
    return check(sm.Logit(r, D).fit(disp=False, maxiter=200), "M4")


def fit_m4(Z, y):
    b = fit_m4_result(Z, y).params
    return lambda Znew: sequential_probs(
        expit(b[0] + Znew @ b[2:]), expit(b[0] + b[1] + Znew @ b[2:]))


MODELS = [("M1 multinomial", fit_m1), ("M2 prop. odds", fit_m2),
          ("M3 two-step", fit_m3), ("M4 seq. common", fit_m4)]


# ---------------------------------------------------------------- metrics
def argmax_rule(prob):
    return prob.argmax(axis=1)


def operational_rule(prob):
    """Paper's thresholds: Pdet<0.50 not seen; Peye>=0.60 naked eye; else aided."""
    pdet = prob[:, 1] + prob[:, 2]
    peye = prob[:, 2] / pdet
    return np.where(pdet < 0.50, 0, np.where(peye >= 0.60, 2, 1))


def rps(y, prob):
    """Mean ranked probability score, averaged over the two cut-points."""
    cum_p = np.cumsum(prob, axis=1)[:, :2]
    cum_o = (y[:, None] <= np.array([0, 1])).astype(float)
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / 2))


def scores(y, prob, pred):
    p, r, f, _ = precision_recall_fscore_support(
        y, pred, labels=LABELS, zero_division=0)
    return {
        "acc": accuracy_score(y, pred),
        "bal_acc": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=LABELS, average="macro", zero_division=0),
        "aided_p": p[1], "aided_r": r[1], "aided_f1": f[1],
        "prf": np.column_stack([p, r, f]).ravel(),
        "logloss": log_loss(y, prob, labels=LABELS),
        "rps": rps(y, prob),
        "cm": confusion_matrix(y, pred, labels=LABELS),
    }


# ---------------------------------------------------------------- (a)-(c)
def compare_saved_oof(prob, path):
    """M3 on the main split should reproduce the out-of-fold file of script 04."""
    if not Path(path).is_file():
        print(f"(Saved out-of-fold file not found, check skipped: {path})")
        return
    saved = pd.read_csv(path)
    pdet = prob[:, 1] + prob[:, 2]
    print(f"Check: M3 out-of-fold probabilities against {Path(path).name}: "
          f"max |P_det diff| = {np.max(np.abs(pdet - saved['p_det'])):.1e}, "
          f"max |P_eye diff| = {np.max(np.abs(prob[:, 2] / pdet - saved['p_eye'])):.1e}")


def main_split(X, y, groups, oof_path):
    folds = grouped_folds(X, y, groups, MAIN_SEED)
    print(f"\n(a-c) Out-of-fold predictions, main split: StratifiedGroupKFold("
          f"n_splits={N_FOLDS}, shuffle=True, random_state={MAIN_SEED}), grouped by "
          f"lunation, stratified on the three-class label")
    print("Each row is predicted by models fitted on the other four folds, which "
          "contain no report from its lunation.")
    print(f"{'Fold':>4s} {'Train':>6s} {'Test':>6s} {'Test lunations':>15s} "
          f"{'Test counts 0/1/2':>18s}")
    for i, (tr, te) in enumerate(folds, 1):
        print(f"{i:4d} {len(tr):6d} {len(te):6d} {len(set(groups[te])):15d} "
              f"{str(np.bincount(y[te], minlength=3).tolist()):>18s}")

    rows = []
    prob, yte = predict_folds(X, y, folds, fit_b0)
    rows.append(("B0 majority", "class 0", scores(yte, prob, argmax_rule(prob))))
    for name, fit in MODELS:
        prob, yte = predict_folds(X, y, folds, fit)
        rows.append((name, "argmax", scores(yte, prob, argmax_rule(prob))))
        rows.append((name, "operational", scores(yte, prob, operational_rule(prob))))
        if fit is fit_m3:
            compare_saved_oof(prob, oof_path)

    print(f"\nScores on the {len(yte)} pooled out-of-fold predictions. B0 probabilities "
          "are the training-fold class shares; log-loss and RPS do not depend on the "
          "decision rule.")
    print(f"{'Model':16s} {'Rule':12s} {'Acc':>7s} {'BalAcc':>7s} {'MacF1':>7s} "
          f"{'AidP':>7s} {'AidR':>7s} {'AidF1':>7s} {'LogLoss':>8s} {'RPS':>7s}")
    for name, rule, s in rows:
        print(f"{name:16s} {rule:12s} {s['acc']:7.4f} {s['bal_acc']:7.4f} "
              f"{s['macro_f1']:7.4f} {s['aided_p']:7.4f} {s['aided_r']:7.4f} "
              f"{s['aided_f1']:7.4f} {s['logloss']:8.4f} {s['rps']:7.4f}")

    print("\nPer-class precision / recall / F1 (0 = not seen, 1 = aided, 2 = naked eye)")
    print(f"{'Model':16s} {'Rule':12s} " + " ".join(
        f"{m + str(c):>6s}" for c in LABELS for m in ("P", "R", "F")))
    for name, rule, s in rows:
        print(f"{name:16s} {rule:12s} " + " ".join(f"{v:6.4f}" for v in s["prf"]))

    print("\nConfusion matrices (rows true 0/1/2, columns predicted 0/1/2)")
    for name, rule, s in rows:
        print(f"{name:16s} {rule:12s} {s['cm'].tolist()}")

    n1 = int(np.sum(yte == 1))
    print(f"\nCeiling for any rule that never predicts 'aided': "
          f"1 - {n1}/{len(yte)} = {1 - n1 / len(yte):.4f}")


def reproduce_0763(X, y, groups):
    print("\n(g) M1 multinomial (argmax) under alternative evaluation set-ups "
          "(earlier reported accuracy: 0.763)")
    print("Grouped = yes: no lunation has reports in both training and test rows. "
          "Grouped = no: random rows, so reports from one new moon fall on both sides.")
    idx = np.arange(len(y))

    def holdout(strat):
        # Single random 75/25 hold-outs, as used before the grouped evaluation.
        return [train_test_split(idx, test_size=0.25, random_state=42, stratify=strat)]

    random_5fold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=MAIN_SEED)
    setups = [
        ("Random 75/25, stratified on 3-class, rs=42", "no", holdout(y)),
        ("Random 75/25, unstratified, rs=42", "no", holdout(None)),
        ("Random 75/25, stratified on y>0, rs=42 (earlier script 04 split)", "no",
         holdout((y > 0).astype(int))),
        ("Random 5-fold out-of-fold, StratifiedKFold rs=42", "no",
         list(random_5fold.split(X, y))),
        ("Grouped 75/25, first fold of StratifiedGroupKFold(4) rs=42", "yes",
         grouped_folds(X, y, groups, MAIN_SEED, n_splits=4)[:1]),
        ("Grouped 5-fold out-of-fold, main split rs=42", "yes",
         grouped_folds(X, y, groups, MAIN_SEED)),
        ("In-sample, all 4,470 rows", "-", [(idx, idx)]),
    ]
    print(f"{'Set-up':64s} {'Grouped':>7s} {'Ntest':>6s} {'Acc':>7s} {'AidP':>7s} "
          f"{'AidR':>7s} {'AidF1':>7s} {'#pred aided':>12s}")
    for label, grouped, folds in setups:
        prob, yte = predict_folds(X, y, folds, fit_m1)
        s = scores(yte, prob, argmax_rule(prob))
        print(f"{label:64s} {grouped:>7s} {len(yte):6d} {s['acc']:7.4f} "
              f"{s['aided_p']:7.4f} {s['aided_r']:7.4f} {s['aided_f1']:7.4f} "
              f"{int(s['cm'][:, 1].sum()):12d}")


# ---------------------------------------------------------------- (d)
def cross_validate(X, y, groups):
    keys = [name for name, _ in MODELS] + [f"{name[:2]} operational" for name, _ in MODELS]
    res = {k: {m: [] for m in CV_METRICS} for k in keys}
    test_sizes = []
    for seed in CV_SEEDS:
        for tr, te in grouped_folds(X, y, groups, seed):
            test_sizes.append(len(te))
            Ztr, Zte = preprocess(X[tr], X[te])
            for name, fit in MODELS:
                prob = fit(Ztr, y[tr])(Zte)
                # The same two decision rules for every model.
                rules = [(name, argmax_rule), (f"{name[:2]} operational", operational_rule)]
                for key, rule in rules:
                    s = scores(y[te], prob, rule(prob))
                    for m in CV_METRICS:
                        res[key][m].append(s[m])
    res = {k: {m: np.array(v) for m, v in d.items()} for k, d in res.items()}

    nf = len(res[keys[0]]["acc"])
    print(f"\n(d) Repeated lunation-grouped 5-fold CV: StratifiedGroupKFold(shuffle=True), "
          f"seeds {min(CV_SEEDS)}-{max(CV_SEEDS)}, groups = lunation, stratified on the "
          f"three-class label; {nf} folds, test fold sizes {min(test_sizes)}-"
          f"{max(test_sizes)} rows: mean (SD) over folds")
    print("M1-M4 use argmax; 'operational' rows apply the paper's 0.50/0.60 thresholds to the "
          "same probabilities, with P_det = P1 + P2 and P_eye = P2 / P_det.")
    print(f"{'Model':16s} " + " ".join(f"{m:>16s}" for m in CV_METRICS))
    for k in keys:
        print(f"{k:16s} " + " ".join(
            f"{res[k][m].mean():7.4f} ({res[k][m].std(ddof=1):6.4f})" for m in CV_METRICS))

    print("\nPaired fold differences (M3 two-step argmax minus comparator): "
          "mean, SD, folds where M3 better / tied / worse")
    print(f"{'Comparison':10s} {'Metric':9s} {'Mean':>8s} {'SD':>7s} {'Better/Tie/Worse':>18s}")
    for other in ["M2 prop. odds", "M1 multinomial", "M4 seq. common"]:
        for m in CV_METRICS:
            d = res["M3 two-step"][m] - res[other][m]
            tie = np.isclose(d, 0, atol=1e-12)
            better = (d < 0 if m in LOWER_IS_BETTER else d > 0) & ~tie
            worse = ~better & ~tie
            print(f"M3-{other[:2]:7s} {m:9s} {d.mean():8.4f} {d.std(ddof=1):7.4f} "
                  f"{f'{better.sum()}/{tie.sum()}/{worse.sum()}':>18s}")


# ---------------------------------------------------------------- (e)-(f)
def brant_test(X1, y, b1, b2):
    """Brant (1990) Wald test of equal slopes across the two cumulative logits."""
    p1, p2 = expit(X1 @ b1), expit(X1 @ b2)
    v11 = np.linalg.inv(X1.T @ (X1 * (p1 * (1 - p1))[:, None]))
    v22 = np.linalg.inv(X1.T @ (X1 * (p2 * (1 - p2))[:, None]))
    v12 = v11 @ (X1.T @ (X1 * (p2 - p1 * p2)[:, None])) @ v22
    d = b1[1:] - b2[1:]
    vd = v11[1:, 1:] + v22[1:, 1:] - v12[1:, 1:] - v12[1:, 1:].T
    overall = float(d @ np.linalg.solve(vd, d))
    single = d ** 2 / np.diag(vd)
    return overall, single


def full_data(X, y):
    n = len(y)
    # One common scaling (all 4,470 rows) so that coefficients are comparable.
    Z, _ = preprocess(X, X)
    X1 = sm.add_constant(Z)
    seen = y > 0

    m1 = check(sm.MNLogit(y, X1).fit(method="newton", maxiter=100, disp=False), "M1")
    m2 = check(OrderedModel(y, Z, distr="logit").fit(
        method="newton", maxiter=100, disp=False), "M2")
    s1 = check(sm.Logit(seen.astype(int), X1).fit(disp=False), "M3 stage 1")
    s2 = check(sm.Logit((y[seen] == 2).astype(int), X1[seen]).fit(disp=False), "M3 stage 2")
    m4 = fit_m4_result(Z, y)
    c2 = check(sm.Logit((y == 2).astype(int), X1).fit(disp=False), "cut 2")

    fits = [("M1 multinomial", m1.llf, m1.params.size),
            ("M2 prop. odds", m2.llf, m2.params.size),
            ("M3 two-step", s1.llf + s2.llf, s1.params.size + s2.params.size),
            ("M4 seq. common", m4.llf, m4.params.size)]
    print(f"\n(e) Full-data unpenalised maximum-likelihood fits (n={n}; "
          f"AIC = 2k - 2LL, BIC = k ln(n) - 2LL)")
    print(f"{'Model':16s} {'LogLik':>11s} {'k':>3s} {'AIC':>10s} {'BIC':>10s}")
    for name, ll, k in fits:
        print(f"{name:16s} {ll:11.4f} {k:3d} {2 * k - 2 * ll:10.4f} "
              f"{k * np.log(n) - 2 * ll:10.4f}")

    lr = 2 * (fits[2][1] - fits[3][1])
    print(f"\nLR test, common slopes (M4) vs stage-specific slopes (M3): "
          f"LR = {lr:.4f}, df = 3, p = {stats.chi2.sf(lr, 3):.4g}")

    # Proportional-odds check for M2.
    b1, b2 = s1.params, c2.params
    w, w1 = brant_test(X1, y, b1, b2)
    print("\nProportional-odds check for M2 (binary logits at cut-points y>=1 and y>=2)")
    print(f"{'Predictor':10s} {'b(y>=1)':>9s} {'b(y>=2)':>9s} {'diff':>9s} "
          f"{'Wald chi2':>10s} {'p (df=1)':>10s}")
    for i, nm in enumerate(NAMES):
        print(f"{nm:10s} {b1[i + 1]:9.4f} {b2[i + 1]:9.4f} {b2[i + 1] - b1[i + 1]:9.4f} "
              f"{w1[i]:10.4f} {stats.chi2.sf(w1[i], 1):10.4g}")
    print(f"Brant Wald test (all slopes equal across cut-points): chi2 = {w:.4f}, "
          f"df = 3, p = {stats.chi2.sf(w, 3):.4g}")

    # Standardised coefficients on the common full-data scaling.
    th = m2.model.transform_threshold_params(m2.params)[1:-1]
    m4b = m4.params
    print("\n(f) Standardised coefficients (full-data unpenalised fits; one common "
          "scaling: all 4,470 rows, median-imputed, z-scored); slope SE in brackets")
    print("Int. 1 / Int. 2 = linear predictor at the mean for the first / second "
          "split (y>=1, then y=2 given y>=1 for M3/M4, or y>=2 for M2 and 'Cut y>=2').")
    print("M2 intercepts are minus its thresholds; M3 stage 1 is identical to the "
          "y>=1 binary logit used in the proportional-odds check.")
    cols = [
        ("M2 PO", [-th[0], -th[1]], m2.params[:3], m2.bse[:3]),
        ("M3 stage 1", [s1.params[0], np.nan], s1.params[1:], s1.bse[1:]),
        ("M3 stage 2", [np.nan, s2.params[0]], s2.params[1:], s2.bse[1:]),
        ("M4 common", [m4b[0], m4b[0] + m4b[1]], m4b[2:], m4.bse[2:]),
        ("Cut y>=2", [np.nan, c2.params[0]], c2.params[1:], c2.bse[1:]),
    ]
    print(f"{'':8s} " + " ".join(f"{c:>17s}" for c, *_ in cols))
    for j, r in enumerate(["Int. 1", "Int. 2"]):
        print(f"{r:8s} " + " ".join(
            f"{'-':>17s}" if np.isnan(a[j]) else f"{a[j]:8.4f}{'':9s}" for _, a, _, _ in cols))
    for i, r in enumerate(NAMES):
        print(f"{r:8s} " + " ".join(f"{b[i]:8.4f} ({se[i]:6.4f})" for _, _, b, se in cols))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="Path to data/icop/crescent_features.csv")
    parser.add_argument("--oof", default=str(DEFAULT_OOF),
                        help="Out-of-fold predictions of script 04, used only as a check")
    args = parser.parse_args()
    df = pd.read_csv(args.input, low_memory=False)
    df = df[
        (df["section_type"] == "first_crescent")
        & (df["obs_relative_to_sunset"] != "before_sunset")
        & df[TARGET].notna()
    ].copy()
    df[TARGET] = df[TARGET].astype(int)
    if len(df) != 4470 or tuple(df[TARGET].value_counts().sort_index()) != (2319, 505, 1646):
        raise ValueError("Source data have changed; verify cohort before comparison")
    # Unique index preserves the source order, as in the published training script.
    df = df.reset_index(drop=True)
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy()
    groups = df[GROUP].to_numpy()
    print(f"Rows: {len(y)}; classes 0/1/2 = {np.bincount(y).tolist()}; "
          f"lunations: {len(set(groups))}; features: {', '.join(NAMES)}")

    main_split(X, y, groups, args.oof)
    reproduce_0763(X, y, groups)
    cross_validate(X, y, groups)
    full_data(X, y)


if __name__ == "__main__":
    main()
