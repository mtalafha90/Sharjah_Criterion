#!/usr/bin/env python3
"""Confusion matrices of both stages and of the full cascade (reviewer point 3).

Reviewer: "Consider presenting the confusion matrices as a two-panel heatmap.
The manuscript already reports the confusion matrices for both stages in the
text. Presenting them visually, with consistent class labels and the
corresponding sample counts, would make the types of classification errors
easier to identify ... help readers assess how the model performs in
distinguishing non-detection, aided detection, and naked-eye visibility."

What the figure shows (confusion_matrices.png, 300 dpi, 6.5 in wide, no
title). Rows are the observed classes, columns the predicted classes, and the
classes always appear in the same order: not seen, aided only, naked eye.
Each cell gives the number of reports and the share of its row (the share of
the observed class that received that prediction); the colour shows the same
row share on one blue scale, 0-100 %, shared by all panels. Correct
predictions lie on the diagonal. Row totals are on the right.
  (a) Stage 1 on all 4,470 reports: seen (aided only or naked eye) when
      P_det >= 0.50.
  (b) Stage 2 on the 2,151 sightings only: naked eye when P_eye >= 0.50, the
      cut-off of the Stage 2 results in the manuscript.
  (c) The full cascade on all 4,470 reports with the operational rule:
      P_det < 0.50 not seen; P_det >= 0.50 and P_eye >= 0.60 naked eye;
      otherwise aided only. The margins give the recall of each observed
      class (right), and the number of reports and the precision of each
      predicted class (below).
Before the figure is drawn, the counts, recall and precision are checked
against the Stage 1 and Stage 2 matrices in the manuscript text and the
cascade table (tab:cascade). Accuracy (correct predictions over all rows) is
given under each panel title.

Input (default path in this repository):
  data/icop/two_stage_oof_predictions.csv  out-of-fold P_det and P_eye of the
      main evaluation for the 4,470 modelling rows (2,319 not seen, 505 aided
      only, 1,646 naked eye): StratifiedGroupKFold(5, shuffle=True,
      random_state=42), stratified on the three-class label, groups = page_url
      (one ICOP monthly page per lunation)

The operational rule is sharjah_classes() of table1_criteria_comparison.py,
and precision and recall come from matrix_scores() of
class_weight_cascade_check.py.

Run: python3 -W ignore -I -B analysis/confusion_matrices.py   (a few seconds)
"""
import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

TARGET = "vis_class_empirical"
HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # the repository root
DEFAULT_OOF = REPO / "data" / "icop" / "two_stage_oof_predictions.csv"
DEFAULT_OUTPUT = REPO / "figures" / "confusion_matrices.png"

STAGE2_CUT = 0.50  # Stage 2 results in the manuscript (Section 3.3)
CLASS_COUNTS = (2319, 505, 1646)  # not seen, aided only, naked eye
# Matrices quoted in the manuscript (rows observed, columns predicted).
EXPECTED = {
    "stage1": [[1890, 429], [492, 1659]],
    "stage2": [[371, 134], [90, 1556]],
    "cascade": [[1890, 43, 386], [377, 26, 102], [115, 32, 1499]],
}
EXPECTED_RECALL = (0.815, 0.051, 0.911)
EXPECTED_PRECISION = (0.793, 0.257, 0.754)

# Class names, in the same order and wording in every panel.
NOT_SEEN, AIDED, EYE = "Not seen", "Aided only", "Naked eye"
SEEN = "Seen"
SEEN_NOTE = "(aided only or\nnaked eye)"
# Column headers on two lines, so that neighbouring names never touch.
HEADER = {NOT_SEEN: "Not\nseen", SEEN: "Seen", AIDED: "Aided\nonly", EYE: "Naked\neye"}

# Reference palette: dark ink, muted secondary ink, and the single-hue blue
# ramp (steps 100 to 700) for the row shares.
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
WHITE = "#ffffff"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("row_share", BLUE_RAMP)
NORM = Normalize(0.0, 1.0)

# Layout in inches (figure 6.5 in wide; every text is at least 8 pt).
FIG_W = 6.5
CELL = 0.68         # side of one matrix cell
LABEL_W = 1.08      # room for the row labels left of each matrix
GAP = 0.10          # between a matrix and its margins
MARGIN_W = 0.62     # one margin column (total, recall)
MARGIN_H = 0.34     # one margin row (predicted total, precision)
FS_COUNT, FS_SHARE, FS_LABEL, FS_TITLE = 9.5, 8.5, 9, 9.5


def load_sibling(name):
    """A script in this folder, loaded by path so that 'python3 -I' also works."""
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- matrices
def load_oof(path):
    """Out-of-fold P_det and P_eye of the 4,470 modelling rows."""
    oof = pd.read_csv(path)
    y = oof[TARGET].astype(int).to_numpy()
    counts = tuple(np.bincount(y, minlength=3).tolist())
    if len(oof) != 4470 or counts != CLASS_COUNTS:
        raise ValueError(f"Unexpected cohort: {len(oof)} rows, classes {counts}")
    return y, oof["p_det"].to_numpy(), oof["p_eye"].to_numpy()


def all_matrices(y, p_det, p_eye, pdet_cut, sharjah_classes):
    """Stage 1 (all rows), Stage 2 (sightings) and three-class cascade matrices."""
    seen = y > 0
    stage1 = confusion_matrix(seen.astype(int), (p_det >= pdet_cut).astype(int), labels=[0, 1])
    eye_true = (y[seen] == 2).astype(int)
    eye_pred = (p_eye[seen] >= STAGE2_CUT).astype(int)
    stage2 = confusion_matrix(eye_true, eye_pred, labels=[0, 1])
    # sharjah_classes: P_det < cut -> 0; P_det >= 0.50 and P_eye >= 0.60 -> 2; else 1.
    pred3 = sharjah_classes(p_det, p_eye, pdet_cut)
    cascade = confusion_matrix(y, pred3, labels=[0, 1, 2])
    return {"stage1": stage1, "stage2": stage2, "cascade": cascade}


def check(matrices, scores):
    for key, expected in EXPECTED.items():
        if matrices[key].tolist() != expected:
            raise AssertionError(f"{key}: {matrices[key].tolist()} differs from {expected}")
    recall = tuple(np.round(scores["recall"], 3))
    precision = tuple(np.round(scores["precision"], 3))
    if recall != EXPECTED_RECALL or precision != EXPECTED_PRECISION:
        raise AssertionError(f"Cascade recall {recall} / precision {precision} differ from the table")


def print_matrix(title, cm, names, scores=None):
    rows = cm.sum(axis=1)
    print(f"\n{title}")
    print(f"  {'Observed':<12}" + "".join(f"{n:>20}" for n in names) + f"{'Total':>8}"
          + (f"{'Recall':>9}" if scores else ""))
    for i, name in enumerate(names):
        cells = "".join(f"{cm[i, j]:>10,} ({100 * cm[i, j] / rows[i]:5.1f} %)" for j in range(len(names)))
        line = f"  {name:<12}{cells}{rows[i]:>8,}"
        if scores:
            line += f"{scores['recall'][i]:>9.3f}"
        print(line)
    if scores:
        cols = cm.sum(axis=0)
        print(f"  {'Pred. total':<12}" + "".join(f"{c:>20,}" for c in cols))
        print(f"  {'Precision':<12}" + "".join(f"{p:>20.3f}" for p in scores["precision"]))
    accuracy = np.trace(cm) / cm.sum()
    print(f"  Accuracy {accuracy:.4f} ({np.trace(cm):,} of {cm.sum():,} correct)")


# ---------------------------------------------------------------- figure
def text_colour(share):
    """Ink or white, whichever contrasts more with the cell colour."""
    def lum(rgb):
        c = np.array(rgb)
        c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    cell = lum(CMAP(NORM(share))[:3])
    on_white = (1.05) / (cell + 0.05)
    on_ink = (cell + 0.05) / (lum(to_rgb(INK)) + 0.05)
    return WHITE if on_white > on_ink else INK


def place(fig, left, top, width, height, fig_h):
    """Axes from inches, measured from the top-left corner of the figure."""
    ax = fig.add_axes([left / FIG_W, 1 - (top + height) / fig_h, width / FIG_W, height / fig_h])
    ax.set_axis_off()
    return ax


def draw_panel(ax, cm, rows, cols, scores=None):
    """One heatmap in cell units: x = column, y = row (row 0 at the top)."""
    k = len(rows)
    totals = cm.sum(axis=1)
    gap, mw, mh = GAP / CELL, MARGIN_W / CELL, MARGIN_H / CELL
    n_margin_cols = 2 if scores else 1
    n_margin_rows = 2 if scores else 0
    ax.set_xlim(0, k + gap + n_margin_cols * mw)
    ax.set_ylim(k + (gap + n_margin_rows * mh if scores else 0), 0)
    for i in range(k):
        for j in range(k):
            share = cm[i, j] / totals[i]
            # A white gap between cells (2 px at screen size) keeps them apart.
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=CMAP(NORM(share)),
                                   edgecolor=WHITE, linewidth=1.5))
            colour = text_colour(share)
            ax.text(j + 0.5, i + 0.42, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=FS_COUNT, fontweight="bold", color=colour)
            ax.text(j + 0.5, i + 0.70, f"{100 * share:.1f} %", ha="center", va="center",
                    fontsize=FS_SHARE, color=colour)
    # Column headers: 'Predicted' over the class names.
    for j, name in enumerate(cols):
        ax.text(j + 0.5, -0.08, HEADER[name], ha="center", va="bottom", fontsize=FS_LABEL,
                color=INK, linespacing=1.05)
    ax.text(k / 2, -0.74, "Predicted", ha="center", va="bottom", fontsize=FS_LABEL,
            color=MUTED, style="italic")
    ax.plot([0.08, k - 0.08], [-0.68, -0.68], color=GRID, lw=1.0, clip_on=False,
            solid_capstyle="butt")
    # Row labels and 'Observed'.
    for i, name in enumerate(rows):
        if isinstance(name, tuple):
            ax.text(-0.10, i + 0.36, name[0], ha="right", va="center", fontsize=FS_LABEL, color=INK)
            ax.text(-0.10, i + 0.68, name[1], ha="right", va="center", fontsize=8, color=MUTED,
                    linespacing=1.0)
        else:
            ax.text(-0.10, i + 0.5, name, ha="right", va="center", fontsize=FS_LABEL, color=INK)
    ax.text(-LABEL_W / CELL + 0.05, k / 2, "Observed", ha="left", va="center", rotation=90,
            fontsize=FS_LABEL, color=MUTED, style="italic")
    # Margin columns: row totals, and recall for the cascade.
    x_total = k + gap + mw / 2
    ax.text(x_total, -0.08, "Total", ha="center", va="bottom", fontsize=FS_LABEL, color=INK)
    for i in range(k):
        ax.text(x_total, i + 0.5, f"{totals[i]:,}", ha="center", va="center",
                fontsize=FS_COUNT, color=INK)
    # A thin rule sets the margins apart from the heatmap.
    ax.plot([k + gap / 2] * 2, [0.04, k - 0.04], color=GRID, lw=1.0, clip_on=False)
    if not scores:
        return
    x_recall = k + gap + 1.5 * mw
    ax.text(x_recall, -0.08, "Recall", ha="center", va="bottom", fontsize=FS_LABEL, color=INK)
    for i in range(k):
        ax.text(x_recall, i + 0.5, f"{scores['recall'][i]:.3f}", ha="center", va="center",
                fontsize=FS_COUNT, color=INK)
    # Margin rows: predicted totals and precision.
    col_totals = cm.sum(axis=0)
    y_total, y_prec = k + gap + mh / 2, k + gap + 1.5 * mh
    ax.text(-0.10, y_total, "Predicted total", ha="right", va="center", fontsize=FS_LABEL, color=INK)
    ax.text(-0.10, y_prec, "Precision", ha="right", va="center", fontsize=FS_LABEL, color=INK)
    for j in range(k):
        ax.text(j + 0.5, y_total, f"{col_totals[j]:,}", ha="center", va="center",
                fontsize=FS_COUNT, color=INK)
        ax.text(j + 0.5, y_prec, f"{scores['precision'][j]:.3f}", ha="center", va="center",
                fontsize=FS_COUNT, color=INK)
    ax.plot([0.04, k - 0.04], [k + gap / 2] * 2, color=GRID, lw=1.0, clip_on=False)


def title(ax_fig, x, y, bold, detail):
    ax_fig.text(x, y, bold, ha="left", va="top", fontsize=FS_TITLE, fontweight="bold", color=INK)
    ax_fig.text(x, y + 0.22, detail, ha="left", va="top", fontsize=8.5, color=MUTED,
                linespacing=1.3)


def plot(matrices, scores, pdet_cut, peye_cut, out):
    plt.rcParams.update({"font.size": FS_LABEL, "text.color": INK, "mathtext.default": "regular"})
    acc = {k: np.trace(m) / m.sum() for k, m in matrices.items()}
    head = 0.50      # title block (two lines)
    header = 0.70    # 'Predicted' and the two-line column names
    top_a = 0.05 + head + header
    mid = top_a + 2 * CELL + 0.30
    top_c = mid + head + 0.16 + header
    fig_h = top_c + 3 * CELL + GAP + 2 * MARGIN_H + 0.08
    fig = plt.figure(figsize=(FIG_W, fig_h), facecolor=WHITE)
    # A full-figure axes in inches for the titles.
    base = fig.add_axes([0, 0, 1, 1])
    base.set_axis_off()
    base.set_xlim(0, FIG_W)
    base.set_ylim(fig_h, 0)

    width2 = 2 * CELL + GAP + MARGIN_W
    left_a = LABEL_W
    left_b = FIG_W - width2 - 0.06
    ax_a = place(fig, left_a, top_a, width2, 2 * CELL, fig_h)
    ax_b = place(fig, left_b, top_a, width2, 2 * CELL, fig_h)
    draw_panel(ax_a, matrices["stage1"], [NOT_SEEN, (SEEN, SEEN_NOTE)], [f"{NOT_SEEN}", SEEN])
    draw_panel(ax_b, matrices["stage2"], [AIDED, EYE], [AIDED, EYE])
    ge = "\u2265"
    title(base, 0.02, 0.05, "(a) Stage 1, all 4,470 reports",
          f"Seen if $P_\\mathrm{{det}}$ {ge} {pdet_cut:.2f}; accuracy {acc['stage1']:.3f}")
    title(base, left_b - LABEL_W + 0.02, 0.05, "(b) Stage 2, the 2,151 sightings only",
          f"Naked eye if $P_\\mathrm{{eye}}$ {ge} {STAGE2_CUT:.2f}; accuracy {acc['stage2']:.3f}")

    width3 = 3 * CELL + GAP + 2 * MARGIN_W
    height3 = 3 * CELL + GAP + 2 * MARGIN_H
    ax_c = place(fig, left_a, top_c, width3, height3, fig_h)
    draw_panel(ax_c, matrices["cascade"], [NOT_SEEN, AIDED, EYE], [NOT_SEEN, AIDED, EYE],
               scores=scores)
    title(base, 0.02, mid, "(c) Full cascade with the operational rule, all 4,470 reports",
          f"Not seen if $P_\\mathrm{{det}}$ < {pdet_cut:.2f}; naked eye if $P_\\mathrm{{det}}$ {ge} "
          f"{pdet_cut:.2f} and $P_\\mathrm{{eye}}$ {ge} {peye_cut:.2f}; otherwise aided only\n"
          f"Accuracy {acc['cascade']:.3f}")

    # One colour scale for all panels.
    cb_left = left_a + width3 + 0.36
    cax = fig.add_axes([cb_left / FIG_W, 1 - (top_c + 3 * CELL) / fig_h, 0.13 / FIG_W,
                        3 * CELL / fig_h])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"])
    cb.ax.tick_params(labelsize=8.5, colors=MUTED, length=3, width=0.6)
    cb.outline.set_visible(False)
    cb.set_label("Share of the observed class\n(row percentage), all panels", fontsize=8.5,
                 color=MUTED, labelpad=6)
    fig.savefig(out, dpi=300, facecolor=WHITE)
    plt.close(fig)
    return fig_h


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--oof", default=str(DEFAULT_OOF),
                        help="data/icop/two_stage_oof_predictions.csv")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Figure path (PNG, 300 dpi)")
    args = parser.parse_args()

    table1 = load_sibling("table1_criteria_comparison")
    cascade = load_sibling("class_weight_cascade_check")
    pdet_cut, peye_cut = table1.PDET_HIGH, table1.PEYE_HIGH
    if (pdet_cut, peye_cut) != (0.50, 0.60):
        raise ValueError(f"Operational cut-offs {pdet_cut}, {peye_cut} differ from Eq. 11")
    y, p_det, p_eye = load_oof(args.oof)
    matrices = all_matrices(y, p_det, p_eye, pdet_cut, table1.sharjah_classes)
    scores = cascade.matrix_scores(matrices["cascade"])
    check(matrices, scores)

    print(f"Out-of-fold predictions: {len(y):,} reports; observed not seen / aided only / naked eye "
          f"= {' / '.join(f'{c:,}' for c in CLASS_COUNTS)}")
    print_matrix(f"(a) Stage 1, all reports, seen if P_det >= {pdet_cut:.2f} "
                 "(rows observed, columns predicted; row share in brackets)",
                 matrices["stage1"], [NOT_SEEN, SEEN])
    print_matrix(f"(b) Stage 2, sightings only, naked eye if P_eye >= {STAGE2_CUT:.2f}",
                 matrices["stage2"], [AIDED, EYE])
    print_matrix(f"(c) Cascade, operational rule (P_det < {pdet_cut:.2f} not seen; P_det >= "
                 f"{pdet_cut:.2f} and P_eye >= {peye_cut:.2f} naked eye; otherwise aided only)",
                 matrices["cascade"], [NOT_SEEN, AIDED, EYE], scores=scores)
    print("\nAll counts, recall and precision agree with the manuscript.")
    fig_h = plot(matrices, scores, pdet_cut, peye_cut, args.output)
    print(f"Figure: {args.output} ({FIG_W:.1f} x {fig_h:.2f} in, 300 dpi)")


if __name__ == "__main__":
    main()
