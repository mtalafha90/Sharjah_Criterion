#!/usr/bin/env python3
"""Reproduce Table 1 (tab:criterion_comparison): the Sharjah criterion against
classical crescent-visibility criteria, two simple rules and a majority-class
baseline, with confidence intervals from a bootstrap over lunations.

Reviewer comment 6 asked for confidence intervals or a significance test for
the small differences in Table 1, for a split by lunation instead of a random
split, for simple baselines, and for a Fig. 10 y-axis that does not start at
0.55. This script answers all four.

Input: data/icop/crescent_features.csv.
Truth is vis_class_empirical (0 = not seen, 1 = aided only, 2 = naked eye).
Every criterion uses the sunset geometry columns of the feature table:
ARCV = relative_altitude_deg, ARCL = elongation_deg, DAZ = relative_azimuth_deg,
W = crescent_width_arcmin (arcmin) and lag = moon_lag_minutes.
A lunation is one ICOP monthly report page (page_url).

Metrics:
  3-class    mean(pred == y)
  Detect     mean((pred > 0) == (y > 0))
  Eye|seen   on rows with y > 0, mean((pred == 2) == (y == 2)); a 'not seen'
             prediction counts as 'not naked eye'
  Macro-F1   unweighted mean of the three per-class F1 scores (0 for a class
             that is never predicted)

Samples (--sample):
  all    all 6,092 rows, no filtering; reproduces the published Table 1 for the
         classical rows (Sharjah: P_det < 0.30 -> not seen, as originally coded;
         the Sharjah row no longer matches, because the saved stages have been
         refitted on all rows with the corrected lags)
  model  revised Table 1: the 4,470 first-crescent, after-sunset rows used to
         fit the model. Sharjah is scored out-of-fold by lunation: 5 folds
         (StratifiedGroupKFold, stratified on the three-class label, groups =
         page_url, shuffle, random_state=42), with both stages refitted on the
         other four folds. No report is therefore scored by a model trained on
         any report of the same new moon. The probabilities are checked against
         data/icop/two_stage_oof_predictions.csv (written by script 04).
         Sharjah P_det < 0.50 -> not seen (the operational threshold), and the
         degenerate Cond-DAZ>=4-W>=10 rule is dropped.

Majority class: always 'not seen', the largest class in both samples.

Uncertainty (--sample model): a cluster bootstrap over lunations. Each of the
2,000 resamples draws the 202 lunations with replacement (numpy default_rng(42))
and keeps every report of each drawn lunation. Intervals are the 2.5 and 97.5
percentiles. For the paired difference (Sharjah minus another criterion, on the
same resample) the two-sided p-value is 2 x min(share of differences <= 0,
share >= 0). McNemar's test is not used: reports from the same lunation are not
independent. The predictions are held fixed in the resamples, so the intervals
show how the scores vary across lunations, not the (small) variation from
refitting the models (repeated grouped cross-validation SD 0.0007 for Stage 1).

Figure (--figure, --sample model only; replaces Fig. 10, which showed only the
multirange criteria): every Table 1 row as a dot with its 95 % cluster-bootstrap
interval, in four panels side by side that share the criterion axis: (a) 3-class
accuracy and (b) detection accuracy on all 4,470 reports, (c) naked-eye
accuracy on the 2,151 seen reports, (d) macro-F1 on all 4,470 reports. The rows
are grouped by approach (single-boundary, multirange, simple rule, baseline),
and the Sharjah row is tinted blue. Every panel runs from 0 to 1, so that small
differences are not enlarged; the value is written beside each point. The
alternative models of --include-alternatives are not drawn. The script prints
the drawn values. Reviewer point 4 (round 2) asked for the approaches, the
metrics and their evaluation bases to be clearly distinguished, with intervals.

Run (writes criteria_comparison.png, 300 dpi, 6.5 in wide):
  python3 -W ignore -I -B table1_criteria_comparison.py --sample model \\
      --include-alternatives --figure ../figures/criteria_comparison.png
"""
import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

ARCV = "relative_altitude_deg"
ARCL = "elongation_deg"
DAZ = "relative_azimuth_deg"
WIDTH = "crescent_width_arcmin"
LAG = "moon_lag_minutes"
FEATURES = [ARCV, ARCL, LAG]
TARGET = "vis_class_empirical"
LUNATION = "page_url"
REPO = Path(__file__).resolve().parent.parent  # the repository root
DEFAULT_INPUT = REPO / "data" / "icop" / "crescent_features.csv"
DEFAULT_MODELS = REPO / "models"
DEFAULT_OOF = REPO / "data" / "icop" / "two_stage_oof_predictions.csv"
# Operational thresholds of the published model (two_stage_model_meta.json).
PDET_LOW, PDET_HIGH, PEYE_HIGH = 0.30, 0.50, 0.60
# Lunation folds and cluster bootstrap.
N_FOLDS, FOLD_SEED = 5, 42
N_BOOT, BOOT_SEED = 2000, 42
METRICS = ["3-class", "Detect", "Eye|seen", "Macro-F1"]
PAIRED = [0, 1, 3]  # metrics of the paired-difference table

# Published Table 1, in table order: group and (3-class, detectability, naked-eye on seen).
TABLE1 = {
    "Maunder": ("Single-boundary", (0.6628, 0.7022, 0.7220)),
    "Indian": ("Single-boundary", (0.6604, 0.7029, 0.7247)),
    "Fotheringham": ("Single-boundary", (0.6474, 0.6733, 0.6670)),
    "Alrefay-Unaided": ("Single-boundary", (0.6353, 0.7078, 0.6948)),
    "Sharjah": ("Multirange", (0.6436, 0.7050, 0.7223)),
    "Yallop": ("Multirange", (0.6408, 0.7017, 0.7235)),
    "Qureshi": ("Multirange", (0.6318, 0.6958, 0.7183)),
    "Alrefay-TwoBoundary": ("Multirange", (0.6064, 0.6825, 0.6948)),
    "Odeh": ("Multirange", (0.5647, 0.6280, 0.7204)),
    "Cond-ARCV>=3-ARCL>=6.4": ("Simple rule", (0.4966, 0.6154, 0.6163)),
    "Cond-DAZ>=4-W>=10": ("Simple rule", (0.4686, 0.4686, 0.3027)),
}


# ---------------------------------------------------------------- criteria
def classical_predictions(df):
    """Three-class predictions of every Table 1 criterion except Sharjah.

    Single-boundary curves and simple rules give 2 (naked eye) when satisfied
    and 0 otherwise; they never predict 'aided'. A missing input compares as
    False and so gives class 0 (the sunset columns have no missing values).
    """
    arcv = df[ARCV].to_numpy(float)
    arcl = df[ARCL].to_numpy(float)
    daz = np.abs(df[DAZ].to_numpy(float))  # the column is already |A_Moon - A_Sun|
    w = df[WIDTH].to_numpy(float)
    yallop_w = -6.3226 * w + 0.7319 * w**2 - 0.1018 * w**3
    eq5 = 8.7 - 0.06 * daz - 0.0012 * daz**2  # Alrefay et al. (2018) eq. 5, naked eye
    eq6 = 7.4 - 0.063 * daz - 0.0013 * daz**2  # Alrefay et al. (2018) eq. 6, optical aid
    q = (arcv - (11.8371 + yallop_w)) / 10
    s = (arcv - (10.4341 - 5.422643 * w + 2.222075 * w**2 - 0.351964 * w**3)) / 10
    v = arcv - (7.1651 + yallop_w)
    return {
        # Maunder (1911), as tabulated by Yallop (1997): ARCV >= 11 - DAZ/20 - DAZ^2/100; unchanged.
        "Maunder": np.where(arcv >= 11 - daz / 20 - daz**2 / 100, 2, 0),
        # Indian Astronomical Ephemeris (1996), via Yallop (1997): ARCV >= 10.3743 - 0.0137 DAZ - 0.0097 DAZ^2; unchanged.
        "Indian": np.where(arcv >= 10.3743 - 0.0137 * daz - 0.0097 * daz**2, 2, 0),
        # Fotheringham (1910), via Yallop (1997): ARCV >= 12 - 0.008 DAZ^2; unchanged.
        "Fotheringham": np.where(arcv >= 12 - 0.008 * daz**2, 2, 0),
        # Alrefay et al. (2018) eq. 5 as a DAZ-ARCV quadratic; coefficients recovered from their Table IV, W not used.
        "Alrefay-Unaided": np.where(arcv >= eq5, 2, 0),
        # Yallop (1997) q-test, coefficients and zones as published, but at sunset, not best time; A,B->2, C,D->1, E,F->0.
        "Yallop": np.where(q > -0.014, 2, np.where(q > -0.232, 1, 0)),
        # Qureshi (2010) s-test with his own W cubic (constant 10.4341 from secondary sources), at sunset; A,B->2, C,D->1, E->0.
        "Qureshi": np.where(s > 0.05, 2, np.where(s > -0.16, 1, 0)),
        # Alrefay et al. (2018) eqs. 5 and 6 (Tables IV and V): above eq. 5 -> 2, between eqs. 6 and 5 -> 1, below eq. 6 -> 0.
        "Alrefay-TwoBoundary": np.where(arcv >= eq5, 2, np.where(arcv >= eq6, 1, 0)),
        # Odeh (2004) V-test, coefficients and zones as published, but at sunset, not best time; A->2, B,C->1, D->0.
        "Odeh": np.where(v >= 5.65, 2, np.where(v >= -0.96, 1, 0)),
        # Simple rule on ARCV, not lunar altitude as in MABIMS 2021: satisfied -> 2, otherwise 0.
        "Cond-ARCV>=3-ARCL>=6.4": np.where((arcv >= 3) & (arcl >= 6.4), 2, 0),
        # Simple rule with W in arcmin; W never reaches 10' (maximum 2.3'), so every row is predicted 'not seen'.
        "Cond-DAZ>=4-W>=10": np.where((daz >= 4) & (w >= 10), 2, 0),
    }


def sharjah_classes(p_det, p_eye, not_seen_below):
    # Sharjah two-step logistic (this paper): P_det < cut -> 0; P_det >= 0.50 and P_eye >= 0.60 -> 2; otherwise 1.
    # The published Table 1 used cut = 0.30; the revised table uses the operational 0.50.
    return np.where(p_det < not_seen_below, 0,
                    np.where((p_det >= PDET_HIGH) & (p_eye >= PEYE_HIGH), 2, 1))


def stage_probabilities(stage1, stage2, X):
    return stage1.predict_proba(X)[:, 1], stage2.predict_proba(X)[:, 1]


# ---------------------------------------------------------------- Sharjah stages
def load_saved_models(models_dir):
    stages = []
    # The pipelines were pickled with an older scikit-learn; the patches below
    # make them usable, so the version warnings are not shown.
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    for name in ("stage1_detectability_model.joblib", "stage2_naked_eye_model.joblib"):
        model = joblib.load(Path(models_dir) / name)
        # Compatibility patch, as in compare_twostep_with_alrefay_rowwise.py.
        clf = model.named_steps["clf"]
        if not hasattr(clf, "multi_class"):
            clf.multi_class = "ovr"
        # Newer SimpleImputer versions also need _fill_dtype, which equals the stored _fit_dtype.
        imputer = model.named_steps["imputer"]
        if not hasattr(imputer, "_fill_dtype"):
            imputer._fill_dtype = imputer._fit_dtype
        stages.append(model)
    return stages


def fit_binary_logistic(X, y):
    # Same pipeline and settings as 04_train_two_stage_crescent_criterion.py.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, solver="lbfgs")),
    ]).fit(X, y)


def refit_as_script04(model_df):
    """Both stages on the training parts of the original script 04's 75/25 splits."""
    seen = (model_df[TARGET] > 0).astype(int)
    X1_train, _, y1_train, _ = train_test_split(
        model_df[FEATURES], seen, test_size=0.25, random_state=42, stratify=seen)
    df2 = model_df[model_df[TARGET] > 0]
    eye = (df2[TARGET] == 2).astype(int)
    X2_train, _, y2_train, _ = train_test_split(
        df2[FEATURES], eye, test_size=0.25, random_state=42, stratify=eye)
    return fit_binary_logistic(X1_train, y1_train), fit_binary_logistic(X2_train, y2_train)


def fit_on_rows(train):
    """Stage 1 on all training rows; Stage 2 on the seen training rows."""
    seen = train[TARGET] > 0
    stage1 = fit_binary_logistic(train[FEATURES], seen.astype(int))
    stage2 = fit_binary_logistic(
        train.loc[seen, FEATURES], (train.loc[seen, TARGET] == 2).astype(int))
    return stage1, stage2


def lunation_folds(model_df):
    """Five folds that never split a lunation, stratified on the three-class label (as script 04)."""
    folds = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    return folds.split(model_df, model_df[TARGET].to_numpy(), groups=model_df[LUNATION])


def out_of_fold(model_df, include_alternatives):
    """Out-of-fold Sharjah probabilities (and alternative predictions) for every row.

    The folds are by lunation (lunation_folds), so every row is scored by
    models fitted on other new moons only.
    """
    p_det = np.full(len(model_df), np.nan)
    p_eye = np.full(len(model_df), np.nan)
    alternatives = {}
    for train_idx, test_idx in lunation_folds(model_df):
        train, test = model_df.iloc[train_idx], model_df.iloc[test_idx]
        p_det[test_idx], p_eye[test_idx] = stage_probabilities(*fit_on_rows(train), test[FEATURES])
        if include_alternatives:
            for name, pred in alternative_predictions(train, test).items():
                alternatives.setdefault(name, np.zeros(len(model_df), dtype=int))[test_idx] = pred
    return p_det, p_eye, alternatives


def check_against_saved_oof(model_df, p_det, p_eye, path):
    """Stop unless the probabilities equal script 04's saved out-of-fold ones (to 1e-9)."""
    saved = pd.read_csv(path, low_memory=False)
    same_rows = (len(saved) == len(model_df)
                 and (saved[LUNATION].to_numpy() == model_df[LUNATION].to_numpy()).all()
                 and (saved[TARGET].to_numpy() == model_df[TARGET].to_numpy()).all()
                 and np.allclose(saved[FEATURES], model_df[FEATURES], rtol=0, atol=1e-12,
                                 equal_nan=True))
    if not same_rows:
        raise ValueError(f"{path} does not hold the modelling rows in the same order")
    gap = max(np.max(np.abs(saved["p_det"].to_numpy() - p_det)),
              np.max(np.abs(saved["p_eye"].to_numpy() - p_eye)))
    if not gap <= 1e-9:
        raise ValueError(f"Out-of-fold probabilities differ from {path} by up to {gap:.3g}")
    return gap


# ---------------------------------------------------------------- alternatives
def alternative_predictions(train, test):
    """Direct three-class models on ARCV, ARCL and lag, decided by argmax."""
    # Imported here so that the Table 1 reproduction does not need statsmodels.
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    # Median imputation and standardisation fitted on the training rows only.
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    Z_train = prep.fit_transform(train[FEATURES])
    Z_test = prep.transform(test[FEATURES])
    y_train = train[TARGET].to_numpy()
    mnl = LogisticRegression(max_iter=4000, solver="lbfgs").fit(Z_train, y_train)
    # No constant column: the two thresholds act as the intercepts.
    po = OrderedModel(y_train, Z_train, distr="logit").fit(method="newton", maxiter=100, disp=False)
    if not po.mle_retvals.get("converged", True):
        raise RuntimeError("Proportional-odds model did not converge")
    return {
        "Multinomial logit": mnl.predict_proba(Z_test).argmax(axis=1),
        "Proportional odds": po.model.predict(po.params, exog=Z_test).argmax(axis=1),
    }


# ---------------------------------------------------------------- scores
def scores(y, pred):
    seen = y > 0
    return (np.mean(pred == y),
            np.mean((pred > 0) == seen),
            np.mean((pred[seen] == 2) == (y[seen] == 2)),
            f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0))


def lunation_resamples(groups, n_boot=N_BOOT, seed=BOOT_SEED):
    """Each row's lunation code and, per resample, how often each lunation is drawn."""
    labels, codes = np.unique(np.asarray(groups), return_inverse=True)
    draws = np.random.default_rng(seed).integers(0, len(labels), size=(n_boot, len(labels)))
    counts = np.zeros((n_boot, len(labels)))
    np.add.at(counts, (np.arange(n_boot)[:, None], draws), 1)
    return codes, counts


def resampled_scores(y, pred, codes, counts):
    """The four scores in every resample (n_boot x 4).

    A resample repeats each lunation as often as it was drawn, so every count
    it needs is the per-lunation count weighted by the draw counts. This gives
    exactly the scores of the stacked resampled rows.
    """
    def total(mask):
        return counts @ np.bincount(codes, weights=np.asarray(mask, float), minlength=counts.shape[1])

    seen = y > 0
    n = total(np.ones(len(y)))
    f1 = []
    for c in (0, 1, 2):
        tp = total((pred == c) & (y == c))
        wrong = total((pred == c) & (y != c)) + total((pred != c) & (y == c))
        f1.append(np.divide(2 * tp, 2 * tp + wrong, out=np.zeros_like(tp), where=2 * tp + wrong > 0))
    return np.column_stack([
        total(pred == y) / n,
        total((pred > 0) == seen) / n,
        total(seen & ((pred == 2) == (y == 2))) / total(seen),
        np.mean(f1, axis=0),
    ])


def interval(values):
    """95 % percentile interval along the resample axis."""
    return np.percentile(values, [2.5, 97.5], axis=0)


def bootstrap_p(diff):
    """Two-sided bootstrap p-value: 2 x min(share <= 0, share >= 0), at most 1."""
    return min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0)))


# ---------------------------------------------------------------- reporting
# Figure colours, as in the other figure scripts.
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE = "#2a78d6"  # the Sharjah row
INTERVAL = "#8f8e88"  # intervals of the other rows
GROUP_SHADE, SHARJAH_SHADE = "#f4f3f0", "#e3edfa"
# Group headings of the figure: what kind of prediction each group gives.
GROUP_HEADINGS = {
    "Single-boundary": "Single-boundary (no aided class)",
    "Multirange": "Multirange (three classes)",
    "Simple rule": "Simple rule (no aided class)",
    "Baseline": "Baseline (always 'not seen')",
}
# Plain names for the figure; the other criteria keep their Table 1 names.
FIGURE_NAMES = {
    "Sharjah": "Sharjah (this work)",
    "Cond-ARCV>=3-ARCL>=6.4": "ARCV \u2265 3\u00b0, ARCL \u2265 6.4\u00b0",
}


def plot_criteria(table_rows, intervals, path, n_rows, n_seen, n_lunations):
    """Dot-and-interval figure of every Table 1 row: four panels, one per metric.

    table_rows: (name, group, four scores) in table order. Each score is a short
    vertical tick on a bar that spans its 95 % cluster-bootstrap interval (from
    print_uncertainty). The value is written beside each interval, on the side
    with room for it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import blended_transform_factory

    panels = [
        ("(a) 3-class\naccuracy", f"(all {n_rows:,} reports)"),
        ("(b) Detectability\naccuracy", f"(all {n_rows:,} reports)"),
        ("(c) Naked-eye\naccuracy", f"({n_seen:,} seen reports)"),
        ("(d) Macro-F1\nscore", f"(all {n_rows:,} reports)"),
    ]
    # One heading row per group, then one row per criterion, from the top down.
    groups = list(dict.fromkeys(group for _, group, _ in table_rows))
    row_of, heading_rows, spans, row = {}, {}, {}, 0
    for group in groups:
        heading_rows[group] = row
        members = [name for name, g, _ in table_rows if g == group]
        for k, name in enumerate(members):
            row_of[name] = row + 1 + k
        spans[group] = (row - 0.5, row + len(members) + 0.5)
        row += len(members) + 1

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 4.8), sharey=True)
    fig.subplots_adjust(left=0.235, right=0.985, top=0.885, bottom=0.12, wspace=0.15)
    # Bands, headings and names span the whole width: x in figure units, y in rows.
    across = blended_transform_factory(fig.transFigure, axes[0].transData)

    # Every other group is shaded, and the Sharjah row is tinted blue.
    for k, group in enumerate(groups):
        shade = GROUP_SHADE if k % 2 == 0 else "white"
        lo, hi = spans[group]
        fig.add_artist(Rectangle((0.005, lo), 0.99, hi - lo, transform=across,
                                 facecolor=shade, edgecolor="none", zorder=-2))
        fig.text(0.012, heading_rows[group], GROUP_HEADINGS.get(group, group), transform=across,
                 ha="left", va="center", fontsize=8, fontweight="bold", color=INK,
                 bbox=dict(facecolor=shade, edgecolor="none", pad=1.0))
    if "Sharjah" in row_of:
        y0 = row_of["Sharjah"]
        fig.add_artist(Rectangle((0.005, y0 - 0.5), 0.99, 1.0, transform=across,
                                 facecolor=SHARJAH_SHADE, edgecolor="none", zorder=-1))
    for name, _, _ in table_rows:
        fig.text(0.03, row_of[name], FIGURE_NAMES.get(name, name), transform=across,
                 ha="left", va="center", fontsize=8, color=INK,
                 fontweight="bold" if name == "Sharjah" else "normal")

    def interval_bar(colour):
        return dict(color=colour, linewidth=6, solid_capstyle="butt")

    tick = dict(marker="|", markersize=9, markeredgewidth=1.1, color=INK, linestyle="none")
    for k, (ax, (title, basis)) in enumerate(zip(axes, panels)):
        ax.set_facecolor("none")
        for name, _, s in table_rows:
            y0 = row_of[name]
            lo, hi = intervals[name][:, k]
            mine = name == "Sharjah"
            ax.plot([lo, hi], [y0, y0], zorder=3, **interval_bar(BLUE if mine else INTERVAL))
            ax.plot([s[k]], [y0], zorder=4, **tick)
            # The value on the left of the interval, unless there is no room there.
            left = lo > 0.36
            ax.text(lo - 0.035 if left else hi + 0.035, y0, f"{s[k]:.3f}",
                    ha="right" if left else "left", va="center", fontsize=8,
                    color=INK if mine else MUTED, fontweight="bold" if mine else "normal",
                    zorder=5)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.set_xticks([0.25, 0.75], minor=True)
        ax.tick_params(axis="x", which="both", length=0, pad=3, labelsize=8)
        # End labels inside the panel, so that '1' and the next panel's '0' stay apart.
        labels = ax.get_xticklabels()
        labels[0].set_horizontalalignment("left")
        labels[-1].set_horizontalalignment("right")
        ax.grid(axis="x", which="both", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(row - 0.5, -0.5)
        ax.tick_params(axis="y", left=False, labelleft=False)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.text(0, 1.05, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
                fontweight="bold", color=INK, linespacing=1.15)
        ax.text(0, 1.012, basis, transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
                color=MUTED)

    key_tick = Line2D([], [], **tick)
    handles = [(Line2D([], [], **interval_bar(INTERVAL)), key_tick),
               (Line2D([], [], **interval_bar(BLUE)), key_tick)]
    labels = [f"Score (tick) and 95 % interval (bar), bootstrap over {n_lunations} lunations",
              "Sharjah criterion"]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2,
               frameon=False, fontsize=8, handlelength=2.0, columnspacing=1.8)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def print_figure_values(table_rows, intervals, path):
    """The drawn values (three decimals) with their 95 % intervals (four decimals, as above)."""
    print(f"\nDrawn in {Path(path).name}: value as written beside each interval "
          f"[95 % interval as drawn]")
    print(f"{'Criterion':24s} {'Group':16s} " + " ".join(f"{m:>22s}" for m in METRICS))
    for name, group, s in table_rows:
        cells = [f"{s[k]:.3f} [{intervals[name][0, k]:.4f}, {intervals[name][1, k]:.4f}]"
                 for k in range(len(METRICS))]
        print(f"{name:24s} {group:16s} " + " ".join(f"{c:>22s}" for c in cells))


def check_cohort(df, n, counts, label):
    if len(df) != n or tuple(df[TARGET].value_counts().sort_index()) != counts:
        raise ValueError(f"Source data have changed: expected {n:,} {label} rows "
                         f"with classes 0/1/2 = {counts}")


def print_uncertainty(rows, y, groups):
    """Cluster-bootstrap intervals for every row and paired differences against Sharjah."""
    codes, counts = lunation_resamples(groups)
    # Guard: one draw of every lunation must give the ordinary scores.
    for _, _, pred, _ in rows:
        if not np.allclose(resampled_scores(y, pred, codes, np.ones((1, counts.shape[1])))[0],
                           scores(y, pred), rtol=0, atol=1e-12):
            raise RuntimeError("Weighted scores do not reproduce the ordinary scores")
    boot = {name: resampled_scores(y, pred, codes, counts) for name, _, pred, _ in rows}
    point = {name: np.array(scores(y, pred)) for name, _, pred, _ in rows}
    intervals = {name: interval(b) for name, b in boot.items()}

    print(f"\n95 % confidence intervals: cluster bootstrap over the {counts.shape[1]} lunations "
          f"({len(counts):,} resamples, default_rng({BOOT_SEED}); percentile intervals)")
    print(f"{'Criterion':24s} " + " ".join(f"{m:>18s}" for m in METRICS))
    for name, _, _, _ in rows:
        lo, hi = intervals[name]
        print(f"{name:24s} " + " ".join(f"{f'[{a:.4f}, {b:.4f}]':>18s}" for a, b in zip(lo, hi)))

    print(f"\nPaired differences, Sharjah minus criterion, on the same {len(counts):,} resamples "
          f"(95 % interval; two-sided bootstrap p)")
    print(f"{'Criterion':24s}" + "".join(f" {'d ' + METRICS[k]:>11s} {'95 % CI':>18s} {'p':>7s}"
                                         for k in PAIRED))
    table1_p = {}
    for name, group, _, _ in rows:
        if name == "Sharjah":
            continue
        line = f"{name:24s}"
        for k in PAIRED:
            diff = boot["Sharjah"][:, k] - boot[name][:, k]
            lo, hi = interval(diff)
            p = bootstrap_p(diff)
            if group != "Alternative":
                table1_p[(name, METRICS[k])] = p
            line += (f" {point['Sharjah'][k] - point[name][k]:+11.4f} "
                     f"{f'[{lo:+.4f}, {hi:+.4f}]':>18s} {f'{p:.4f}' if p > 0 else '<0.001':>7s}")
        print(line)
    print(f"p = '<0.001': no resample gave a difference at or beyond zero "
          f"(the smallest non-zero p with {len(counts):,} resamples is {2 / len(counts):.3f}).")
    print_holm(table1_p, 2 / len(counts))
    return intervals


def print_holm(pvalues, p_floor, alpha=0.05):
    """Holm step-down correction over all Table 1 comparisons (alternatives excluded).

    A bootstrap p of 0 is replaced by the smallest non-zero value, p_floor, which
    is conservative.
    """
    items = sorted(pvalues.items(), key=lambda kv: max(kv[1], p_floor))
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * max(p, p_floor)))
        adjusted[key] = running
    kept = [key for key, q in adjusted.items() if q < alpha]
    print(f"\nHolm correction over the {m} Table 1 comparisons (Sharjah minus each criterion, "
          f"{len(PAIRED)} metrics): {len(kept)} remain significant at the 5 % level.")
    print("Not significant after the correction:")
    for key, q in adjusted.items():
        if q >= alpha:
            print(f"  {key[0]:24s} {key[1]:9s} p = {pvalues[key]:.4f}, Holm-adjusted {q:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="Path to data/icop/crescent_features.csv")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS),
                        help="Folder with the saved Stage 1 and Stage 2 joblib models")
    parser.add_argument("--oof", default=str(DEFAULT_OOF),
                        help="Script 04's out-of-fold probabilities, checked in --sample model")
    parser.add_argument("--sample", choices=["all", "model"], default="all",
                        help="Evaluation rows (default: all, as in the published Table 1)")
    parser.add_argument("--refit", action="store_true",
                        help="With --sample all: refit both Sharjah stages on the original "
                             "script 04's 75 %% training splits instead of loading the saved models")
    parser.add_argument("--include-alternatives", action="store_true",
                        help="Add multinomial and proportional-odds models (--sample model only)")
    parser.add_argument("--figure", metavar="PNG",
                        help="Save the dot-and-interval figure of every Table 1 row (four "
                             "metrics, bootstrap intervals) to this file (--sample model only)")
    args = parser.parse_args()
    if args.sample == "all" and (args.include_alternatives or args.figure):
        parser.error("--include-alternatives and --figure need --sample model")
    if args.sample == "model" and args.refit:
        parser.error("--refit applies to --sample all only")

    df = pd.read_csv(args.input, low_memory=False)
    check_cohort(df, 6092, (2855, 980, 2257), "feature-table")
    df[TARGET] = df[TARGET].astype(int)
    # Same filter as the training script; reset_index keeps the source order.
    model_df = df[
        (df["section_type"] == "first_crescent")
        & (df["obs_relative_to_sunset"] != "before_sunset")
    ].reset_index(drop=True)
    check_cohort(model_df, 4470, (2319, 505, 1646), "modelling")

    alternatives = {}
    if args.sample == "model":
        test = model_df
        p_det, p_eye, alternatives = out_of_fold(model_df, args.include_alternatives)
        gap = check_against_saved_oof(model_df, p_det, p_eye, args.oof)
        sharjah_source = (f"out-of-fold by lunation: {N_FOLDS} folds (StratifiedGroupKFold, "
                          f"groups = {LUNATION}, random_state={FOLD_SEED}), both stages refitted "
                          f"on the other four folds; equal to {Path(args.oof).name} "
                          f"(largest difference {gap:.1e})")
    else:
        test = df
        if args.refit:
            stage1, stage2 = refit_as_script04(model_df)
            sharjah_source = "refitted on the 75 % training splits of the original script 04"
        else:
            stage1, stage2 = load_saved_models(args.models_dir)
            sharjah_source = f"saved models in {args.models_dir}"
        p_det, p_eye = stage_probabilities(stage1, stage2, test[FEATURES])

    revised = args.sample == "model"
    y = test[TARGET].to_numpy()
    preds = classical_predictions(test)
    preds["Sharjah"] = sharjah_classes(p_det, p_eye, PDET_HIGH if revised else PDET_LOW)
    rows = [(name, group, preds[name], published) for name, (group, published) in TABLE1.items()
            if not (revised and name == "Cond-DAZ>=4-W>=10")]
    majority = np.bincount(y).argmax()
    rows.append(("Majority class", "Baseline", np.full(len(y), majority), None))
    rows += [(name, "Alternative", pred, None) for name, pred in alternatives.items()]

    print(f"Sample: {args.sample}; rows = {len(y):,}; lunations = {test[LUNATION].nunique()}; "
          f"classes 0/1/2 = {np.bincount(y, minlength=3).tolist()}")
    print(f"Sharjah stages: {sharjah_source}")
    print(f"Sharjah 'not seen' below P_det = {PDET_HIGH if revised else PDET_LOW:.2f}")
    print(f"Majority class: always {majority} ('not seen'), the largest class.")
    if alternatives:
        print("'Alternative' rows are not Table 1 criteria: direct three-class models on "
              "ARCV, ARCL and lag, decided by argmax and scored out-of-fold on the same "
              "lunation folds as Sharjah.")

    show_published = not revised
    header = (f"\n{'Criterion':24s} {'Group':16s} {'3-class':>8s} {'Detect':>8s} "
              f"{'Eye|seen':>9s} {'Macro-F1':>9s} {'Predicted 0/1/2':>20s}")
    if show_published:
        header += f"   {'Table 1':>22s}  Match"
    print(header)
    mismatches = []
    for name, group, pred, published in rows:
        got = scores(y, pred)
        line = (f"{name:24s} {group:16s} {got[0]:8.4f} {got[1]:8.4f} {got[2]:9.4f} {got[3]:9.4f} "
                f"{str(np.bincount(pred, minlength=3).tolist()):>20s}")
        if show_published:
            if published is None:
                line += f"   {'-':>22s}  n/a"
            else:
                match = [f"{a:.4f}" for a in got[:3]] == [f"{b:.4f}" for b in published]
                if not match:
                    mismatches.append(name)
                line += (f"   {' '.join(f'{b:.4f}' for b in published):>22s}  "
                         f"{'yes' if match else 'NO'}")
        print(line)

    if show_published:
        print(f"\nPublished Table 1 rows reproduced exactly to 4 decimals: "
              f"{len(TABLE1) - len(mismatches)}/{len(TABLE1)}"
              + (f"; mismatches: {', '.join(mismatches)}" if mismatches else ""))
        return

    intervals = print_uncertainty(rows, y, test[LUNATION])
    if args.figure:
        # Every Table 1 row; the alternative models are not drawn.
        table_rows = [(name, group, scores(y, pred)) for name, group, pred, _ in rows
                      if group != "Alternative"]
        plot_criteria(table_rows, intervals, args.figure, len(y), int((y > 0).sum()),
                      test[LUNATION].nunique())
        print(f"\nSaved {args.figure}")
        print_figure_values(table_rows, intervals, args.figure)


if __name__ == "__main__":
    main()
