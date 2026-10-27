"""Choose probability thresholds and plot ROC, threshold and calibration curves.

The script reads the out-of-fold probabilities written by
04_train_two_stage_crescent_criterion.py (5-fold cross-validation grouped by
lunation, so no record is scored by a model that saw its own new moon) and:

  1. prints, for each stage, the ROC AUC and the scores at five candidate
     thresholds: the maximum of Youden's J (J = TPR - FPR), the maximum F1
     score on a grid 0.050, 0.055, ..., 0.950, and the fixed values 0.30,
     0.50 and 0.70;
  2. saves those candidates in ICOP_Data/chosen_thresholds.json;
  3. draws three figures. The first two are 6.5 in wide at 300 dpi, made to
     be included at the full text width, so that no text prints smaller
     than 8 pt:

     roc_two_stage.png         (a) Stage 1, seen by any means against not
                               seen (all reports); (b) Stage 2, naked eye
                               against aided only (sightings only). Each
                               panel gives the AUC, marks the operating
                               points of the adopted thresholds with filled
                               dots (P_det = 0.30 and 0.50; P_eye = 0.30 and
                               0.60) and marks the threshold of maximum
                               Youden's J with an open diamond.
     thresholds_two_stage.png  precision, recall and F1 score against the
                               threshold for the same two stages, on the
                               grid 0.050, 0.055, ..., 0.950 (so the curves
                               stop short of 0 and 1). Vertical lines mark
                               the adopted thresholds; the strip above each
                               panel names the operational bands they
                               separate, with the letters of the band
                               equation (Stage 1: E not detectable,
                               D marginal, then C, B or A; Stage 2, whose
                               bands apply only when P_det >= 0.50:
                               C aid-assisted likely, B borderline naked-eye,
                               A naked-eye likely). A dot marks the
                               threshold with the highest F1 score.
     calibration_two_stage.png observed frequency against mean predicted
                               probability (10 quantile bins per stage).
                               This figure is unchanged: 6 x 5 in at
                               160 dpi, in the default matplotlib style.

A record is called positive when its probability is at or above the
threshold.

Inputs (in ../ICOP_Data):
  two_stage_stage1_predictions.csv  y_true (1 = seen by any means) and
                                    p_positive (out-of-fold P_det), 4,470 rows
  two_stage_stage2_predictions.csv  y_true (1 = naked eye) and p_positive
                                    (out-of-fold P_eye), the 2,151 sightings

Outputs: the three PNG files above, next to this script, and
ICOP_Data/chosen_thresholds.json.

Run from the repository root:
  python3 Sharjah_Criterion/05_choose_thresholds_and_plot_roc.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory, offset_copy
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

_ICOP_DATA_DIR = Path(__file__).parent.parent / "ICOP_Data"
_OUT_DIR = Path(__file__).parent

STAGE1_CSV = _ICOP_DATA_DIR / "two_stage_stage1_predictions.csv"
STAGE2_CSV = _ICOP_DATA_DIR / "two_stage_stage2_predictions.csv"
OUT_JSON = _ICOP_DATA_DIR / "chosen_thresholds.json"

ROC_PNG = _OUT_DIR / "roc_two_stage.png"
THRESHOLDS_PNG = _OUT_DIR / "thresholds_two_stage.png"
CALIBRATION_PNG = _OUT_DIR / "calibration_two_stage.png"

# Adopted thresholds of the operational bands (Stage 1 on P_det, Stage 2 on P_eye)
# and the bands on either side of each threshold: (letter in the band equation, name).
STAGE1_CUTS = (0.30, 0.50)
STAGE2_CUTS = (0.30, 0.60)
STAGE1_BANDS = (("E", "not\ndetectable"), ("D", "marginal"),
                ("C, B or A", "detectable\n(see Stage 2)"))
STAGE2_BANDS = (("C", "aid-assisted\nlikely"), ("B", "borderline\nnaked-eye"),
                ("A", "naked-eye\nlikely"))
STAGE1_BAND_NOTE = r"Bands on $P_\mathrm{det}$ apply to all reports"
STAGE2_BAND_NOTE = r"Bands on $P_\mathrm{eye}$ apply only when $P_\mathrm{det} \geq 0.50$"

# Figure style, as in the manuscript-repository figures.
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
STYLE = {
    "font.size": 8.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "text.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.default": "regular",
}
# ROC markers: adopted thresholds (filled dot) and highest Youden's J (open diamond).
CUT_MARKER = dict(marker="o", markersize=5.5, color=INK, markeredgecolor="white",
                  markeredgewidth=0.8)
J_MARKER = dict(marker="D", markersize=7, markerfacecolor="white", markeredgecolor=INK,
                markeredgewidth=1.0)
F1_MARKER = dict(marker="o", markersize=6, color=ORANGE, markeredgecolor=INK,
                 markeredgewidth=0.9)


def evaluate_threshold(y_true, p, thr):
    y_pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
    }


def choose_thresholds(y_true, p, label):
    fpr, tpr, roc_thr = roc_curve(y_true, p)
    auc = roc_auc_score(y_true, p)

    # Youden J
    j = tpr - fpr
    i_best_j = int(np.argmax(j))
    thr_j = roc_thr[i_best_j]

    # Best F1 on a dense grid
    grid = np.linspace(0.05, 0.95, 181)
    rows = [evaluate_threshold(y_true, p, thr) for thr in grid]
    df = pd.DataFrame(rows)

    i_best_f1 = int(df["f1"].idxmax())
    best_f1_row = df.loc[i_best_f1].to_dict()

    # Practical low / medium / high confidence thresholds
    # You can tune these later, but these are useful for reporting
    candidates = {
        "youden_j": evaluate_threshold(y_true, p, thr_j),
        "best_f1": best_f1_row,
        "low_confidence": evaluate_threshold(y_true, p, 0.30),
        "balanced": evaluate_threshold(y_true, p, 0.50),
        "high_confidence": evaluate_threshold(y_true, p, 0.70),
    }

    print(f"\n=== {label} ===")
    print(f"ROC AUC = {auc:.4f}")
    for k, v in candidates.items():
        print(
            f"{k:15s} thr={v['threshold']:.3f} "
            f"acc={v['accuracy']:.3f} "
            f"prec={v['precision']:.3f} "
            f"rec={v['recall']:.3f} "
            f"f1={v['f1']:.3f} "
            f"spec={v['specificity']:.3f}"
        )

    return auc, fpr, tpr, candidates, df


def stage_panels(stage1, stage2):
    """What each panel of the two-panel figures needs, Stage 1 first.

    stage1 and stage2 are (y, p, (auc, fpr, tpr, candidates, grid table)), the last
    item as returned by choose_thresholds.
    """
    panels = []
    for y, p, chosen, letter, title, prob, name, cuts, bands, note in (
        (*stage1, "a", "Stage 1: seen by any means (all reports)", r"$P_\mathrm{det}$",
         "P_det", STAGE1_CUTS, STAGE1_BANDS, STAGE1_BAND_NOTE),
        (*stage2, "b", "Stage 2: naked eye among sightings", r"$P_\mathrm{eye}$",
         "P_eye", STAGE2_CUTS, STAGE2_BANDS, STAGE2_BAND_NOTE),
    ):
        auc, fpr, tpr, cand, df = chosen
        panels.append(dict(letter=letter, title=title, y=y, p=p, auc=auc, fpr=fpr, tpr=tpr,
                           cand=cand, df=df, prob=prob, name=name, cuts=cuts, bands=bands,
                           note=note))
    return panels


def style_axes(ax):
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.6)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)


def plot_roc_two_stage(panels, outfile):
    """Two ROC curves side by side, with the adopted thresholds marked on each."""
    print("\nROC figure: operating points (TPR = recall, FPR = 1 - specificity)")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.55))
        j_labels = []
        for ax, s in zip(axes, panels):
            ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
            # va="top": the label hangs below the diagonal instead of growing into it.
            ax.text(0.62, 0.58, "no skill", rotation=45, rotation_mode="anchor",
                    fontsize=8, color=MUTED, ha="center", va="top")
            # One ink colour for both curves: the panels already separate the stages.
            ax.plot(s["fpr"], s["tpr"], color=INK, linewidth=1.5, zorder=2)
            print(f"  ({s['letter']}) AUC = {s['auc']:.4f}")

            # Highest Youden's J: open diamond, drawn under the threshold dots so that a
            # dot lying close to it (Stage 2, P_eye = 0.60) stays visible.
            yj = s["cand"]["youden_j"]
            xj_, yj_ = 1 - yj["specificity"], yj["recall"]
            ax.plot(xj_, yj_, **J_MARKER, zorder=4)
            print(f"      maximum Youden J at {yj['threshold']:.4f}: TPR = {yj_:.4f}, "
                  f"FPR = {xj_:.4f}, J = {yj_ - xj_:.4f}")

            # Operating points of the adopted thresholds: filled dots.
            for cut in s["cuts"]:
                row = evaluate_threshold(s["y"], s["p"], cut)
                x, y = 1 - row["specificity"], row["recall"]
                ax.plot(x, y, **CUT_MARKER, zorder=5)
                ax.annotate(f"{s['prob']} = {cut:.2f}", (x, y), xytext=(7, -12),
                            textcoords="offset points", fontsize=8, color=INK,
                            ha="left", va="center")
                print(f"      {s['name']} >= {cut:.2f}: "
                      f"TPR = {y:.4f}, FPR = {x:.4f}")

            ax.text(0.97, 0.155, f"AUC = {s['auc']:.3f}", transform=ax.transAxes,
                    ha="right", va="center", fontsize=8.5, color=INK)
            j_labels.append(ax.text(0.97, 0.06, f"highest $J$ at {s['prob']} = {yj['threshold']:.2f}",
                                    transform=ax.transAxes, ha="right", va="center",
                                    fontsize=8.5, color=INK))

            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.01, 1.01)
            ax.set_aspect("equal")
            ticks = np.linspace(0, 1, 6)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xlabel("False-positive rate (1 − specificity)")
            ax.set_ylabel("True-positive rate (sensitivity)")
            ax.set_title(f"({s['letter']}) {s['title']}", loc="left", fontsize=9,
                         fontweight="bold")
            style_axes(ax)

        handles = [
            Line2D([], [], linestyle="none", label="Adopted threshold", **CUT_MARKER),
            Line2D([], [], linestyle="none", label="Highest Youden's $J$ (TPR $-$ FPR)",
                   **J_MARKER),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8,
                   bbox_to_anchor=(0.5, 0.0), handletextpad=0.4, columnspacing=1.6)
        fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=2.0)

        # A diamond in front of the 'highest J' line ties the text to its marker.
        # Placed after the layout is fixed, because it needs the width of the text.
        fig.canvas.draw()
        for ax, label in zip(axes, j_labels):
            box = label.get_window_extent().transformed(ax.transAxes.inverted())
            ax.plot(box.x0 - 0.045, 0.06, transform=ax.transAxes, clip_on=False, **J_MARKER)
        fig.savefig(outfile, dpi=300, facecolor="white")
        plt.close(fig)


def plot_thresholds_two_stage(panels, outfile):
    """Precision, recall and F1 against the threshold, with the adopted thresholds marked."""
    metrics = [("precision", "Precision", BLUE, (0, (5, 2))),
               ("recall", "Recall", GREEN, (0, (1, 1.2))),
               ("f1", "F1 score", ORANGE, "-")]
    # Heights above the top of each panel, in points: band names, band letters, the rule
    # that closes the strip, and the note that says which reports the bands apply to.
    name_y, letter_y, rule_y, note_y, title_pad = 13, 29.5, 37, 40, 54
    print("\nThreshold figure: scores at the adopted thresholds and the highest F1")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.0), sharey=True)
        for ax, s in zip(axes, panels):
            df = s["df"]
            for col, _, colour, ls in metrics:
                ax.plot(df["threshold"], df[col], color=colour, linestyle=ls, linewidth=1.6,
                        zorder=3)

            # Adopted thresholds: vertical lines that continue into the band strip above.
            top = blended_transform_factory(ax.transData, ax.transAxes)
            for cut in s["cuts"]:
                ax.axvline(cut, color=INK, linewidth=0.9, zorder=2)
                ax.annotate("", (cut, 1.0), xycoords=top, xytext=(0, rule_y),
                            textcoords="offset points", annotation_clip=False,
                            arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.9,
                                            shrinkA=0, shrinkB=0))
                ax.text(cut + 0.012, 0.03, f"{cut:.2f}", transform=top, fontsize=8,
                        color=INK, ha="left", va="bottom")
                row = evaluate_threshold(s["y"], s["p"], cut)
                print(f"  ({s['letter']}) threshold {cut:.2f}: precision = {row['precision']:.4f}, "
                      f"recall = {row['recall']:.4f}, F1 = {row['f1']:.4f}")
            edges = (0.0,) + tuple(s["cuts"]) + (1.0,)
            for left, right, (letter, name) in zip(edges[:-1], edges[1:], s["bands"]):
                centre = (left + right) / 2
                ax.annotate(letter, (centre, 1.0), xycoords=top, xytext=(0, letter_y),
                            textcoords="offset points", fontsize=8, fontweight="bold",
                            color=INK, ha="center", va="center", annotation_clip=False)
                ax.annotate(name, (centre, 1.0), xycoords=top, xytext=(0, name_y),
                            textcoords="offset points", fontsize=8, color=INK, ha="center",
                            va="center", linespacing=1.0, annotation_clip=False)
            rule = offset_copy(ax.transAxes, fig=fig, y=rule_y, units="points")
            ax.plot([0, 1], [1, 1], transform=rule, color=MUTED, linewidth=0.6, clip_on=False)
            ax.annotate(s["note"], (0, 1.0), xycoords="axes fraction", xytext=(0, note_y),
                        textcoords="offset points", fontsize=8, color=MUTED, ha="left",
                        va="bottom", annotation_clip=False)

            # Highest F1 score on the grid. The label sits in the empty lower right, with no
            # leader line, so that nothing crosses an adopted-threshold line.
            best = s["cand"]["best_f1"]
            ax.plot(best["threshold"], best["f1"], **F1_MARKER, zorder=5)
            ax.text(s["cuts"][1] + 0.04, 0.12,
                    f"highest F1\n= {best['f1']:.3f}\nat {s['prob']} = {best['threshold']:.3f}",
                    fontsize=8, color=INK, ha="left", va="bottom", linespacing=1.15)
            print(f"  ({s['letter']}) highest F1 = {best['f1']:.4f} at threshold "
                  f"{best['threshold']:.4f}")

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.0)
            ticks = np.linspace(0, 1, 6)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xlabel(f"Threshold on {s['prob']}")
            ax.set_title(f"({s['letter']}) {s['title']}", loc="left", fontsize=9,
                         fontweight="bold", pad=title_pad)
            style_axes(ax)
        axes[0].set_ylabel("Score")

        handles = [Line2D([], [], color=c, linestyle=ls, linewidth=1.6, label=name)
                   for _, name, c, ls in metrics]
        handles += [
            Line2D([], [], color=INK, linewidth=0.9, label="Adopted threshold"),
            Line2D([], [], linestyle="none", label="Highest F1", **F1_MARKER),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8,
                   bbox_to_anchor=(0.5, 0.0), handletextpad=0.4, columnspacing=1.4,
                   handlelength=2.4)
        fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=2.0)
        fig.savefig(outfile, dpi=300, facecolor="white")
        plt.close(fig)


def plot_calibration(stages, outfile):
    """Reliability diagram: observed frequency against mean predicted probability."""
    plt.figure(figsize=(6, 5))
    for label, y, p in stages:
        frac, mean_p = calibration_curve(y, p, n_bins=10, strategy="quantile")
        plt.plot(mean_p, frac, marker="o", label=f"{label} (Brier = {brier_score_loss(y, p):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration of out-of-fold probabilities")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()


def main():
    s1 = pd.read_csv(STAGE1_CSV)
    s2 = pd.read_csv(STAGE2_CSV)

    y1 = s1["y_true"].astype(int).to_numpy()
    p1 = s1["p_positive"].astype(float).to_numpy()

    y2 = s2["y_true"].astype(int).to_numpy()
    p2 = s2["p_positive"].astype(float).to_numpy()

    chosen1 = choose_thresholds(y1, p1, "Stage 1: Detectability")
    chosen2 = choose_thresholds(y2, p2, "Stage 2: Naked-eye")
    cand1, cand2 = chosen1[3], chosen2[3]

    panels = stage_panels((y1, p1, chosen1), (y2, p2, chosen2))
    plot_roc_two_stage(panels, ROC_PNG)
    plot_thresholds_two_stage(panels, THRESHOLDS_PNG)
    plot_calibration([("Stage 1: detectability", y1, p1), ("Stage 2: naked eye", y2, p2)],
                     CALIBRATION_PNG)

    out = {
        "stage1_detectability": cand1,
        "stage2_naked_eye": cand2,
    }
    Path(OUT_JSON).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(f"  {ROC_PNG.name}")
    print(f"  {THRESHOLDS_PNG.name}")
    print(f"  {CALIBRATION_PNG.name}")
    print("  chosen_thresholds.json")


if __name__ == "__main__":
    main()
