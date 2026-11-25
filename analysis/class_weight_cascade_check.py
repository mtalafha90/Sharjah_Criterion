#!/usr/bin/env python3
"""Class weights, an ordered model and the whole cascade
(second review round, comment 7).

The script compares a multinomial logistic model and the two-step cascade,
each with and without balanced class weights, and a proportional-odds model,
by cross-validated log-loss, Brier score, accuracy and related scores. It also
gives the full three-class confusion matrices of the whole cascade and shows
how Stage 1 errors pass into Stage 2.

Input: data/icop/crescent_features.csv,
filtered as in 04_train_two_stage_crescent_criterion.py (4,470 rows, 202
lunations; a lunation is one ICOP monthly report page, page_url). All models
use ARCV, ARCL and lag time, with median imputation and standardisation
fitted on the training rows only. Each model gives three-class probabilities
P0 (not seen), P1 (aided), P2 (naked eye):
  MN      multinomial logistic regression (lbfgs, L2, C=1, as the two stages)
  MN-bal  the same with class_weight='balanced' (class k weighted n/(3 n_k))
  PO      proportional-odds cumulative logit (statsmodels OrderedModel, logit
          link, no constant: the two thresholds act as intercepts; unpenalised)
  2S      the two-step model (cascade) of script 04: Stage 1 fitted on all
          training rows, Stage 2 on the seen training rows, each with its own
          imputation, standardisation and logistic regression;
          P0 = 1 - P_det, P1 = P_det (1 - P_eye), P2 = P_det P_eye. This is a
          continuation-ratio (sequential) logit with stage-specific slopes.
  2S-bal  the same with class_weight='balanced' in each stage
  B0      training-fold class shares (reference only)
Decision rules, applied to the same probabilities:
  argmax       the most probable class
  operational  the thresholds of Eq. 11: P_det = P1 + P2, P_eye = P2 / P_det;
               P_det < 0.50 not seen; else P_eye >= 0.60 naked eye; else aided
Sections:
  (a) repeated lunation-grouped 5-fold CV (StratifiedGroupKFold, shuffle=True,
      seeds 0-9, stratified on the three-class label): mean (SD) over 50 folds
  (b) paired fold differences: effect of balanced weights, and model contrasts
  (c) main split of script 04 (random_state=42), out-of-fold: full 3x3
      confusion matrices of the cascade under both rules, per-class precision
      and recall with 95% cluster-bootstrap intervals over lunations, and how
      Stage 1 errors pass into Stage 2.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel

FEATURES = ["relative_altitude_deg", "elongation_deg", "moon_lag_minutes"]
TARGET = "vis_class_empirical"
GROUP = "page_url"
LABELS = [0, 1, 2]
CLASS_NAMES = ["not seen", "aided", "naked eye"]
DEFAULT_INPUT = (
    Path(__file__).resolve().parent.parent
    / "data" / "icop" / "crescent_features.csv"
)
DEFAULT_OOF = DEFAULT_INPUT.with_name("two_stage_oof_predictions.csv")
N_FOLDS = 5
CV_SEEDS = range(10)    # repeated grouped CV
MAIN_SEED = 42          # the main split of script 04
PDET_CUT = 0.50         # Eq. 11: detectable
PEYE_CUT = 0.60         # Eq. 11: naked eye
N_BOOT = 2000
BOOT_SEED = 42
PROB_METRICS = [("logloss", "LogLoss"), ("brier", "Brier"), ("mean_p1", "Mean P(aided)")]
DEC_METRICS = [("acc", "Acc"), ("bal_acc", "BalAcc"), ("macro_f1", "MacroF1"),
               ("aided_p", "AidedP"), ("aided_r", "AidedR")]


# ---------------------------------------------------------------- splits
def grouped_folds(y, groups, seed):
    """Five folds that keep each lunation whole, stratified on the three-class label."""
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    folds = list(cv.split(np.zeros(len(y)), y, groups=groups))
    for tr, te in folds:
        if set(groups[tr]) & set(groups[te]):
            raise RuntimeError("A lunation is in both training and test rows")
    return folds


# ---------------------------------------------------------------- models
def logistic(class_weight=None):
    # The pipeline of script 04, with optional class weights.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, solver="lbfgs",
                                   class_weight=class_weight)),
    ])


def cascade_probs(pdet, peye):
    return np.column_stack([1 - pdet, pdet * (1 - peye), pdet * peye])


def fit_b0(X, y):
    prior = np.bincount(y, minlength=3) / len(y)
    return lambda Xnew: np.tile(prior, (len(Xnew), 1))


def fit_mn(X, y, class_weight=None):
    # With three classes and lbfgs, scikit-learn fits the multinomial model.
    return logistic(class_weight).fit(X, y).predict_proba


def fit_mn_bal(X, y):
    return fit_mn(X, y, "balanced")


def fit_po(X, y):
    prep = Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler())])
    Z = prep.fit_transform(X)
    res = OrderedModel(y, Z, distr="logit").fit(method="bfgs", maxiter=5000, disp=False)
    if not res.mle_retvals.get("converged", True):
        raise RuntimeError("Proportional-odds model did not converge")
    return lambda Xnew: res.model.predict(res.params, exog=prep.transform(Xnew))


def fit_stages(X, y, class_weight=None):
    """Stage 1 on all rows (seen or not); Stage 2 on seen rows (naked eye or aided)."""
    seen = y > 0
    s1 = logistic(class_weight).fit(X, seen.astype(int))
    s2 = logistic(class_weight).fit(X[seen], (y[seen] == 2).astype(int))
    return lambda Xnew: (s1.predict_proba(Xnew)[:, 1], s2.predict_proba(Xnew)[:, 1])


def fit_2s(X, y, class_weight=None):
    stages = fit_stages(X, y, class_weight)
    return lambda Xnew: cascade_probs(*stages(Xnew))


def fit_2s_bal(X, y):
    return fit_2s(X, y, "balanced")


MODELS = [("MN", fit_mn), ("MN-bal", fit_mn_bal), ("PO", fit_po),
          ("2S", fit_2s), ("2S-bal", fit_2s_bal)]


# ---------------------------------------------------------------- scores
def argmax_rule(prob):
    return prob.argmax(axis=1)


def operational_rule(prob):
    pdet = prob[:, 1] + prob[:, 2]
    peye = prob[:, 2] / pdet
    return np.where(pdet < PDET_CUT, 0, np.where(peye >= PEYE_CUT, 2, 1))


RULES = [("argmax", argmax_rule), ("operational", operational_rule)]


def prob_scores(y, prob):
    """Log-loss; multiclass Brier = mean over rows of sum_k (P_k - O_k)^2 (range 0-2)."""
    onehot = np.eye(3)[y]
    return {
        "logloss": log_loss(y, prob, labels=LABELS),
        "brier": float(np.mean(np.sum((prob - onehot) ** 2, axis=1))),
        "mean_p1": float(prob[:, 1].mean()),
    }


def decision_scores(y, pred):
    # Aided precision is undefined (NaN) when no row is predicted aided.
    p, r, _, _ = precision_recall_fscore_support(
        y, pred, labels=LABELS, zero_division=np.nan)
    return {
        "acc": accuracy_score(y, pred),
        "bal_acc": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=LABELS, average="macro", zero_division=0),
        "aided_p": p[1],
        "aided_r": r[1],
    }


def mean_sd(v):
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return "n/a"
    return f"{v.mean():.4f} ({v.std(ddof=1):.4f})"


# ---------------------------------------------------------------- (a)-(b)
def cross_validate(X, y, groups):
    prob_res = {name: {m: [] for m, _ in PROB_METRICS} for name in ["B0"] + [n for n, _ in MODELS]}
    keys = [("B0", "argmax")] + [(n, r) for n, _ in MODELS for r, _ in RULES]
    dec_res = {k: {m: [] for m, _ in DEC_METRICS} for k in keys}
    obs_p1, sizes = [], []
    for seed in CV_SEEDS:
        for tr, te in grouped_folds(y, groups, seed):
            sizes.append(len(te))
            obs_p1.append(np.mean(y[te] == 1))
            for name, fit in [("B0", fit_b0)] + MODELS:
                prob = fit(X[tr], y[tr])(X[te])
                for m, v in prob_scores(y[te], prob).items():
                    prob_res[name][m].append(v)
                rules = RULES[:1] if name == "B0" else RULES
                for rule, apply_rule in rules:
                    for m, v in decision_scores(y[te], apply_rule(prob)).items():
                        dec_res[(name, rule)][m].append(v)
    prob_res = {k: {m: np.array(v) for m, v in d.items()} for k, d in prob_res.items()}
    dec_res = {k: {m: np.array(v, dtype=float) for m, v in d.items()} for k, d in dec_res.items()}

    print(f"\n(a) Repeated lunation-grouped 5-fold CV: StratifiedGroupKFold(shuffle=True), "
          f"seeds {min(CV_SEEDS)}-{max(CV_SEEDS)}, stratified on the three-class label; "
          f"{len(sizes)} folds, test fold sizes {min(sizes)}-{max(sizes)} rows. "
          f"Mean (SD) over folds.")
    print("\nProbability scores (lower is better; they do not depend on the decision rule). "
          "Brier = sum over the three classes, range 0-2.")
    print(f"{'Model':8s} " + " ".join(f"{label:>16s}" for _, label in PROB_METRICS))
    for name in prob_res:
        print(f"{name:8s} " + " ".join(f"{mean_sd(prob_res[name][m]):>16s}"
                                        for m, _ in PROB_METRICS))
    print(f"Observed share of aided rows in the test folds: {mean_sd(np.array(obs_p1))}")
    # Weights on the full data, for illustration; each fold uses its own training rows.
    seen = y > 0
    w = len(y) / (3 * np.bincount(y))
    w1 = len(y) / (2 * np.bincount(seen.astype(int)))
    w2 = seen.sum() / (2 * np.bincount((y[seen] == 2).astype(int)))
    print("Balanced weights on the full data, for illustration: MN-bal (n / (3 n_k)) "
          + ", ".join(f"{c} {v:.4f}" for c, v in zip(CLASS_NAMES, w))
          + f"; 2S-bal Stage 1 not seen {w1[0]:.4f}, seen {w1[1]:.4f}; "
          f"Stage 2 aided {w2[0]:.4f}, naked eye {w2[1]:.4f}")

    print("\nDecision scores. AidedP is averaged over the folds where at least one row "
          "is predicted aided; 'No aided' counts the folds where none is.")
    print(f"{'Model':8s} {'Rule':12s} " + " ".join(f"{label:>16s}" for _, label in DEC_METRICS)
          + f" {'No aided':>9s}")
    for name, rule in keys:
        r = dec_res[(name, rule)]
        print(f"{name:8s} {rule:12s} " + " ".join(f"{mean_sd(r[m]):>16s}" for m, _ in DEC_METRICS)
              + f" {int(np.isnan(r['aided_p']).sum()):9d}")
    print("The 0.50/0.60 cut-offs of the operational rule were chosen for the unweighted "
          "two-step probabilities; on the -bal models they act on shifted probabilities.")

    print("\n(b) Paired fold differences (first model minus second; same 50 folds): "
          "mean, SD, folds higher / tied / lower. AidedP uses only the folds in which "
          "both models predict at least one aided row.")
    print(f"{'Contrast':16s} {'Rule':12s} {'Metric':14s} {'Mean':>8s} {'SD':>7s} "
          f"{'Higher/Tie/Lower':>17s}")

    def show(label, rule, metric, a, b):
        ok = ~np.isnan(a) & ~np.isnan(b)
        d = a[ok] - b[ok]
        if len(d) < 2:
            print(f"{label:16s} {rule:12s} {metric:14s} {'n/a':>8s} {'n/a':>7s} "
                  f"{f'({len(d)} usable folds)':>17s}")
            return
        tie = np.isclose(d, 0, atol=1e-12)
        hi, lo = (d > 0) & ~tie, (d < 0) & ~tie
        print(f"{label:16s} {rule:12s} {metric:14s} {d.mean():8.4f} {d.std(ddof=1):7.4f} "
              f"{f'{hi.sum()}/{tie.sum()}/{lo.sum()}':>17s}")

    for weighted, plain in [("MN-bal", "MN"), ("2S-bal", "2S")]:
        label = f"{weighted} - {plain}"
        for m, metric in PROB_METRICS:
            show(label, "-", metric, prob_res[weighted][m], prob_res[plain][m])
        for rule, _ in RULES:
            for m, metric in DEC_METRICS:
                show(label, rule, metric, dec_res[(weighted, rule)][m],
                     dec_res[(plain, rule)][m])
    for first, second in [("2S", "MN"), ("2S", "PO"), ("PO", "MN")]:
        for m, metric in PROB_METRICS[:2]:
            show(f"{first} - {second}", "-", metric, prob_res[first][m], prob_res[second][m])

    print("\nNote on class weights: class_weight='balanced' re-weights the likelihood so "
          "that every class carries the same total weight. The fitted probabilities then "
          "describe a population with equal class shares, not the observed one, so they "
          "are no longer calibrated: compare 'Mean P(aided)' with the observed share. "
          "Weighting assigns a larger part of the feature space to the small aided class, "
          "which is what raises aided recall; it does so at the cost of aided precision "
          "and overall accuracy, and log-loss and Brier show the cost in probability "
          "quality.")


# ---------------------------------------------------------------- (c)
def lunation_counts(y, pred, groups):
    """3x3 confusion counts per lunation, flattened: one row per lunation."""
    _, inv = np.unique(groups, return_inverse=True)
    counts = np.zeros((inv.max() + 1, 9))
    np.add.at(counts, (inv, y * 3 + pred), 1)
    return counts


def bootstrap_matrices(counts):
    """Confusion matrices of 2,000 cluster-bootstrap samples of lunations."""
    rng = np.random.default_rng(BOOT_SEED)
    n = len(counts)
    draws = rng.integers(0, n, size=(N_BOOT, n))
    mult = np.stack([np.bincount(d, minlength=n) for d in draws])
    return (mult @ counts).reshape(N_BOOT, 3, 3)


def matrix_scores(cm):
    """Per-class precision, recall, F1 and overall scores from (..., 3, 3) counts."""
    diag = np.diagonal(cm, axis1=-2, axis2=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = diag / cm.sum(axis=-2)
        rec = diag / cm.sum(axis=-1)
        f1 = 2 * prec * rec / (prec + rec)
    f1 = np.nan_to_num(f1)
    return {
        "precision": prec, "recall": rec, "f1": f1,
        "acc": diag.sum(axis=-1) / cm.sum(axis=(-2, -1)),
        "bal_acc": rec.mean(axis=-1),
        "macro_f1": f1.mean(axis=-1),
    }


def with_ci(value, boot):
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return f"{value:.4f} [{lo:.4f}, {hi:.4f}]"


def show_confusion(y, pred, groups, title):
    cm = confusion_matrix(y, pred, labels=LABELS)
    s = matrix_scores(cm.astype(float))
    b = matrix_scores(bootstrap_matrices(lunation_counts(y, pred, groups)))
    print(f"\n{title}")
    print(f"{'Rows: true; columns: predicted':30s} " + " ".join(f"{c:>10s}" for c in CLASS_NAMES)
          + f" {'Total':>7s}")
    for i, c in enumerate(CLASS_NAMES):
        print(f"{c:30s} " + " ".join(f"{v:10d}" for v in cm[i]) + f" {cm[i].sum():7d}")
    print(f"{'Total':30s} " + " ".join(f"{v:10d}" for v in cm.sum(axis=0))
          + f" {cm.sum():7d}")
    print(f"{'Class':10s} {'Precision [95% CI]':>26s} {'Recall [95% CI]':>26s} "
          f"{'F1':>7s} {'Support':>8s}")
    for i, c in enumerate(CLASS_NAMES):
        print(f"{c:10s} {with_ci(s['precision'][i], b['precision'][:, i]):>26s} "
              f"{with_ci(s['recall'][i], b['recall'][:, i]):>26s} {s['f1'][i]:7.4f} "
              f"{cm[i].sum():8d}")
    undefined = np.isnan(b["precision"]).any(axis=0)
    for i in np.flatnonzero(undefined):
        print(f"({CLASS_NAMES[i]} precision undefined in "
              f"{int(np.isnan(b['precision'][:, i]).sum())} resamples with no such "
              f"prediction; they are left out of its interval)")
    print("Accuracy " + with_ci(s["acc"], b["acc"])
          + "; balanced accuracy " + with_ci(s["bal_acc"], b["bal_acc"])
          + "; macro-F1 " + with_ci(s["macro_f1"], b["macro_f1"]))
    return cm


def propagation(y, pdet, peye):
    """How Stage 1 errors pass into Stage 2 under the operational rule."""
    seen = y > 0
    passed = pdet >= PDET_CUT
    eye = peye >= PEYE_CUT
    n0, n1, n2 = (np.sum(y == k) for k in LABELS)
    print("\nHow Stage 1 errors pass into Stage 2 (unweighted cascade, operational rule, "
          "main split, out-of-fold)")
    print(f"Stage 1 alone, all {len(y)} rows, P_det >= {PDET_CUT:.2f}: accuracy "
          f"{np.mean(passed == seen):.4f}")
    print(f"Stage 2 alone, truly seen rows only ({seen.sum()}), as evaluated in the paper: "
          f"accuracy {np.mean((peye[seen] >= 0.50) == (y[seen] == 2)):.4f} at P_eye >= 0.50, "
          f"{np.mean(eye[seen] == (y[seen] == 2)):.4f} at P_eye >= {PEYE_CUT:.2f}")
    a1 = np.sum((y == 1) & passed & ~eye)
    print(f"  aided recall of Stage 2 alone (P_eye < {PEYE_CUT:.2f} among the {n1} aided "
          f"rows): {np.sum((y == 1) & ~eye) / n1:.4f}; aided precision among truly seen "
          f"rows: {np.sum((y == 1) & ~eye) / np.sum(seen & ~eye):.4f}")
    print("Within the cascade:")
    print(f"  Stage 1 misses (truly seen, P_det < {PDET_CUT:.2f}, so labelled not seen "
          f"whatever Stage 2 says): aided {np.sum((y == 1) & ~passed)}/{n1}, "
          f"naked eye {np.sum((y == 2) & ~passed)}/{n2}")
    fp = (y == 0) & passed
    print(f"  Stage 1 false passes (not seen, P_det >= {PDET_CUT:.2f}): {fp.sum()}/{n0}, "
          f"which Stage 2 then labels aided {np.sum(fp & ~eye)} and naked eye "
          f"{np.sum(fp & eye)}")
    sp = seen & passed
    s2_err = np.sum(sp & (eye != (y == 2)))
    print(f"  Truly seen rows passed by Stage 1: {sp.sum()}; Stage 2 wrong on {s2_err} of them "
          f"(accuracy {1 - s2_err / sp.sum():.4f})")
    s1_err = np.sum(seen & ~passed) + fp.sum()
    print(f"  Three-class errors: {s1_err + s2_err} = {s1_err} from Stage 1 + {s2_err} from "
          f"Stage 2; cascade accuracy {1 - (s1_err + s2_err) / len(y):.4f}")
    a2 = np.sum((y == 1) & ~eye)
    pred1 = passed & ~eye
    print(f"  Aided recall: {a2 / n1:.4f} for Stage 2 alone, {a1 / n1:.4f} in the cascade; "
          f"{np.sum((y == 1) & ~eye & ~passed)} of the {a2} aided rows that Stage 2 alone "
          f"labels aided are Stage 1 misses")
    print(f"  Aided precision: {a2 / np.sum(seen & ~eye):.4f} for Stage 2 alone (truly seen "
          f"rows), {a1 / pred1.sum():.4f} in the cascade; of the {pred1.sum()} rows the "
          f"cascade labels aided, {a1} are aided, {np.sum(pred1 & (y == 0))} not seen "
          f"(Stage 1 false passes) and {np.sum(pred1 & (y == 2))} naked eye")
    print(f"  Mean P_det of truly aided rows {pdet[y == 1].mean():.4f} (median "
          f"{np.median(pdet[y == 1]):.4f}); of naked-eye rows {pdet[y == 2].mean():.4f} "
          f"(median {np.median(pdet[y == 2]):.4f})")


def main_split(X, y, groups, oof_path):
    folds = grouped_folds(y, groups, MAIN_SEED)
    print(f"\n(c) Main split of script 04: StratifiedGroupKFold(n_splits={N_FOLDS}, "
          f"shuffle=True, random_state={MAIN_SEED}), grouped by lunation; each row is "
          f"predicted by models fitted on the other four folds.")
    out = {}
    for label, weight in [("2S", None), ("2S-bal", "balanced")]:
        pdet, peye = np.full(len(y), np.nan), np.full(len(y), np.nan)
        for tr, te in folds:
            pdet[te], peye[te] = fit_stages(X[tr], y[tr], weight)(X[te])
        out[label] = (pdet, peye)

    pdet, peye = out["2S"]
    if Path(oof_path).is_file():
        saved = pd.read_csv(oof_path)
        print(f"Check against {Path(oof_path).name}: max |P_det diff| = "
              f"{np.max(np.abs(pdet - saved['p_det'])):.1e}, max |P_eye diff| = "
              f"{np.max(np.abs(peye - saved['p_eye'])):.1e}")
    else:
        print(f"(Saved out-of-fold file not found, check skipped: {oof_path})")
    print("Under argmax a row is 'not seen' when P_det < 1 / (1 + max(P_eye, 1 - P_eye)), "
          "a cut-off between 0.50 and 0.67, and 'naked eye' rather than 'aided' when "
          "P_eye >= 0.50.")
    print(f"Intervals: 95% percentile cluster bootstrap over lunations ({N_BOOT} resamples "
          f"of the {len(set(groups))} lunations, default_rng({BOOT_SEED})), out-of-fold "
          f"predictions held fixed.")

    for label, title in [("2S", "Two-step cascade (unweighted, as published)"),
                         ("2S-bal", "Two-step cascade with class_weight='balanced' in each stage")]:
        prob = cascade_probs(*out[label])
        for rule, apply_rule in [RULES[1], RULES[0]]:
            show_confusion(y, apply_rule(prob), groups, f"{title}, {rule} rule")
    propagation(y, pdet, peye)


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
    print(f"Rows: {len(y)}; classes 0/1/2 (not seen / aided / naked eye) = "
          f"{np.bincount(y).tolist()}; lunations: {len(set(groups))}; "
          f"features: ARCV, ARCL, lag")

    cross_validate(X, y, groups)
    main_split(X, y, groups, args.oof)


if __name__ == "__main__":
    main()
