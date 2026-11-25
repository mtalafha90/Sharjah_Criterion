#!/usr/bin/env python3
"""Probability contours of the final model over ARCV and ARCL (reviewer point 5).

Reviewer: "Consider adding a visual representation of the model's probability
boundaries. A probability-contour plot or a small set of decision-region plots
illustrating the combined influence of relative altitude (ARCV), elongation
(ARCL), and Moon lag time ... the predicted detectability and naked-eye
probabilities could be illustrated across ARCV and ARCL for selected lag-time
values, with the adopted probability thresholds indicated."

What the figure shows (probability_contours.png, 300 dpi, 6.5 in wide, no
title). Nine panels with ARCL (0-25 deg) across and ARCV (-2 to 22 deg) up,
on the same scale. Each column holds the lag time fixed. The three values are
the 20th, 50th and 80th percentiles of the lag time in the 4,470 modelling
rows (24.1, 47.0 and 71.2 min), rounded to 5 min: 25, 45 and 70 min.
  Top row (a-c): P_det of the final Stage 1 model, in steps of 0.1, with the
      lines P_det = 0.30 and 0.50.
  Middle row (d-f): P_eye of the final Stage 2 model, with the lines
      P_eye = 0.30 and 0.60. P_eye is the probability of a naked-eye sighting
      if the crescent is detected, so it matters only where P_det is high.
  Bottom row (g-i): the five operational bands of the manuscript's band
      equation, in the colours of the visibility maps:
        E  P_det < 0.30                         not detectable
        D  0.30 <= P_det < 0.50                 marginal detectability
        C  P_det >= 0.50 and P_eye < 0.30       aid-assisted likely
        B  P_det >= 0.50, 0.30 <= P_eye < 0.60  borderline naked-eye
        A  P_det >= 0.50 and P_eye >= 0.60      naked-eye likely
Both stages are linear in the log-odds, so every line of equal probability is
straight, the lines of one stage are parallel, and a longer lag time moves
them towards lower ARCV and ARCL. ARCV cannot be larger than ARCL, and ARCV
cannot be below -ARCL (light grey). The lag time is closely tied to ARCV and
ARCL (Pearson r = 0.94 with ARCV), so a given lag time occurs only in part of
each panel. The dots are the ICOP reports whose lag time lies within
+/-7.5 min of the column's value (the count is in the column heading): they
show where the model is supported by data, and away from them the panels are
an extrapolation.

Printed:
  [1] the lag-time percentiles and the chosen values;
  [2] for each lag time and each threshold line, the ARCV on the line at
      ARCL = 10 deg, the lowest ARCL on the line (where ARCV = ARCL) and the
      ARCL on the line at the median ARCV of the reports in that column, and
      how far the lines move for 10 min more lag time;
  [3] the reports in each lag window: observed outcomes, median geometry,
      mean final-model and out-of-fold P_det and P_eye, and the number of
      reports in each band (final model, at each report's own lag time).

Inputs (default paths in this repository):
  data/icop/crescent_features.csv  ICOP feature table; the modelling subset of
      4,470 rows (2,319 not seen, 505 aided only, 1,646 naked eye) is taken
      with load_icop() of hilalpy_independent_check.py
  data/icop/two_stage_oof_predictions.csv  out-of-fold P_det and P_eye of the
      main evaluation (5-fold, grouped by lunation), used only in [3]
  models/stage1_detectability_model.joblib and
      stage2_naked_eye_model.joblib  the final models, fitted on all 4,470
      rows, loaded with load_saved_models() of table1_criteria_comparison.py
The bands come from eq16_class() and the band colours from CLASSES in
visibility_maps.py.

Run: python3 -W ignore -I -B analysis/probability_contours.py   (about 15 s)
"""
import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

ARCV = "relative_altitude_deg"
ARCL = "elongation_deg"
LAG = "moon_lag_minutes"
FEATURES = [ARCV, ARCL, LAG]
GROUP = "page_url"
HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # the repository root
DEFAULT_ICOP = REPO / "data" / "icop" / "crescent_features.csv"
DEFAULT_OOF = REPO / "data" / "icop" / "two_stage_oof_predictions.csv"
DEFAULT_MODELS = REPO / "models"
DEFAULT_OUTPUT = REPO / "figures" / "probability_contours.png"

LAG_PERCENTILES = (20, 50, 80)
LAG_ROUND = 5.0       # min
HALF_WINDOW = 7.5     # min: reports shown in a column
ARCL_RANGE = (0.0, 25.0)
ARCV_RANGE = (-2.0, 22.0)
STEP = 0.025          # deg, grid spacing
ARCL_FIXED = 10.0     # deg, for the printed summary [2]
LETTERS = "abcdefghi"

# Reference palette: dark ink, muted ink, the single-hue blue ramp (steps 100
# to 700) for the probabilities, light grey for the impossible region, and
# categorical orange for the ICOP reports, which stays visible on light and
# dark blue alike.
INK, MUTED = "#0b0b0b", "#52514e"
WHITE, IMPOSSIBLE, SPINE = "#ffffff", "#f0efec", "#c3c2b7"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
P_LEVELS = np.linspace(0, 1, 11)  # filled steps of 0.1
P_COLOURS = [LinearSegmentedColormap.from_list("probability", BLUE_RAMP)(v)
             for v in np.linspace(0, 1, len(P_LEVELS) - 1)]
DOT = dict(s=3.2, color=INK, alpha=0.7, edgecolors="white", linewidths=0.25, zorder=4)
HALO = [pe.withStroke(linewidth=2.4, foreground=WHITE)]

# Layout in inches (figure 6.5 in wide; every text is at least 8 pt).
FIG_W = 6.5
LEFT = 0.50           # y tick labels and y label
GAP_X = 0.10          # between panels of a row
RIGHT = 1.14          # colour bars and band key
PANEL_W = (FIG_W - LEFT - 2 * GAP_X - RIGHT) / 3
PANEL_H = PANEL_W * (ARCV_RANGE[1] - ARCV_RANGE[0]) / (ARCL_RANGE[1] - ARCL_RANGE[0])
COL_HEAD = 0.46       # column headings
ROW_HEAD = 0.27       # row heading above each row of panels
GAP_Y = 0.08          # between rows
BOTTOM = 0.72         # x tick labels, x label and the key to dots and grey
FS, FS_SMALL, FS_HEAD = 8.5, 8, 9


def load_sibling(name):
    """A script in this folder, loaded by path so that 'python3 -I' also works."""
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- model
def proba(model, arcv, arcl, lag):
    """Final-model probability for arrays of ARCV and ARCL at one lag time."""
    arcv, arcl = np.broadcast_arrays(np.asarray(arcv, float), np.asarray(arcl, float))
    X = pd.DataFrame({ARCV: arcv.ravel(), ARCL: arcl.ravel(), LAG: float(lag)})[FEATURES]
    return model.predict_proba(X)[:, 1].reshape(arcv.shape)


def arcv_on_line(model, p, arcl, lag):
    """ARCV at which the model gives probability p, for a given ARCL and lag."""
    return brentq(lambda v: proba(model, v, arcl, lag)[()] - p, -200, 200, xtol=1e-9)


def arcl_on_line(model, p, arcv, lag):
    """ARCL at which the model gives probability p, for a given ARCV and lag."""
    return brentq(lambda x: proba(model, arcv, x, lag)[()] - p, -200, 200, xtol=1e-9)


def diagonal_on_line(model, p, lag):
    """Lowest ARCL on the line of probability p: the point where ARCV = ARCL."""
    return brentq(lambda x: proba(model, x, x, lag)[()] - p, -200, 200, xtol=1e-9)


def choose_lags(lag):
    pct = np.percentile(lag, LAG_PERCENTILES)
    return pct, [LAG_ROUND * round(v / LAG_ROUND) for v in pct]


# ---------------------------------------------------------------- figure
def place(fig, left, top, width, height, fig_h):
    """Axes from inches, measured from the top-left corner of the figure."""
    return fig.add_axes([left / FIG_W, 1 - (top + height) / fig_h, width / FIG_W, height / fig_h])


def text_colour(hex_colour):
    """Ink or white, whichever contrasts more with the fill."""
    rgb = np.array(matplotlib.colors.to_rgb(hex_colour))
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return WHITE if 1.05 / (lum + 0.05) > (lum + 0.05) / 0.05 else INK


def style_axes(ax, show_x, show_y):
    ax.set_xlim(*ARCL_RANGE)
    ax.set_ylim(*ARCV_RANGE)
    ax.set_aspect("equal")
    # Labels every 10 deg, so that '25' and '0' of neighbouring panels never touch.
    ax.set_xticks([0, 10, 20])
    ax.set_xticks([5, 15, 25], minor=True)
    ax.set_yticks([0, 10, 20])
    ax.set_yticks([5, 15], minor=True)
    ax.tick_params(labelsize=FS_SMALL, colors=MUTED, length=2.5, width=0.6, pad=2,
                   labelbottom=show_x, labelleft=show_y)
    ax.tick_params(which="minor", colors=MUTED, length=1.8, width=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.6)


def shade_impossible(ax):
    """|ARCV| > ARCL cannot occur (exact edges, drawn over the fills)."""
    # Drawn a little beyond the axes (clipped there), so no fill shows at the edges.
    x = np.linspace(ARCL_RANGE[0] - 1, ARCL_RANGE[1] + 1, 601)
    ax.fill_between(x, x, ARCV_RANGE[1] + 1, color=IMPOSSIBLE, lw=0, zorder=3)
    ax.fill_between(x, ARCV_RANGE[0] - 1, -x, where=-x > ARCV_RANGE[0] - 1, color=IMPOSSIBLE,
                    lw=0, zorder=3)


def label_point(model, p, lag, kind):
    """A point on the line of probability p, inside the possible region, for its label."""
    if kind == "det":   # steep lines: label low in the panel
        v = 3.0
        x = arcl_on_line(model, p, v, lag)
        if x > 22.5:
            x = 22.5
            v = arcv_on_line(model, p, x, lag)
    else:               # shallow lines: label towards the right
        x = 21.0
        v = arcv_on_line(model, p, x, lag)
        if v < 0:
            v = 0.0
            x = arcl_on_line(model, p, v, lag)
    return x, v


def probability_panel(ax, model, gx, gy, lag, cuts, kind):
    p = proba(model, gy, gx, lag)
    ax.contourf(gx, gy, p, levels=P_LEVELS, colors=P_COLOURS, zorder=1)
    # Lines stop just past the diagonal, under the grey shading.
    p_lines = np.ma.masked_where(np.abs(gy) > gx + 0.3, p)
    cs = ax.contour(gx, gy, p_lines, levels=list(cuts), colors=[INK, INK],
                    linewidths=[0.9, 1.4], zorder=2)
    cs.set(path_effects=[pe.withStroke(linewidth=2.6, foreground=WHITE)])
    texts = ax.clabel(cs, fmt={c: f"{c:.2f}" for c in cuts}, fontsize=FS_SMALL, inline=True,
                      inline_spacing=2, manual=[label_point(model, c, lag, kind) for c in cuts])
    for t in texts:
        t.set(color=INK, path_effects=HALO, zorder=6)
    return p


def band_panel(ax, cls, gx, gy, classes):
    colours = [c[2] for c in classes]
    ax.contourf(gx, gy, cls, levels=np.arange(-0.5, len(classes)), colors=colours, zorder=1)
    ax.contour(gx, gy, cls, levels=np.arange(0.5, len(classes) - 1), colors=WHITE,
               linewidths=0.8, zorder=2)
    # Each band's letter at the point of its region farthest from any edge.
    inside = (np.abs(gy) <= gx) & (gx <= ARCL_RANGE[1]) & (gy <= ARCV_RANGE[1])
    for k, (letter, _, colour) in enumerate(classes):
        region = inside & (cls == k)
        if not region.any():
            continue
        dist = distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1] * STEP
        i, j = np.unravel_index(np.argmax(dist), dist.shape)
        if dist[i, j] >= 1.0:  # deg: room for the letter
            ax.text(gx[i, j], gy[i, j], letter, ha="center", va="center", fontsize=FS_HEAD,
                    fontweight="bold", color=text_colour(colour), zorder=6)


def colour_bar(fig, ax_left, top, height, fig_h, cuts, symbol):
    cax = place(fig, ax_left, top + 0.08, 0.11, height - 0.16, fig_h)
    norm = BoundaryNorm(P_LEVELS, len(P_COLOURS))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=ListedColormap(P_COLOURS)), cax=cax)
    ticks = [0.0, *cuts, 1.0]
    cb.set_ticks(ticks, labels=["0" if t == 0 else "1" if t == 1 else f"{t:.2f}" for t in ticks])
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=FS_SMALL, colors=MUTED, length=2.5, width=0.6, pad=2)
    cb.outline.set_visible(False)
    for c, lw in zip(cuts, (0.9, 1.4)):
        cb.ax.axhline(c, color=INK, lw=lw)
    cb.set_label(symbol, fontsize=FS, color=INK, labelpad=3)


def band_key(fig, left, top, height, fig_h, classes):
    """Swatches with the band letter and its name, beside the bottom row."""
    ax = place(fig, left, top, RIGHT - 0.14, height, fig_h)
    ax.set_axis_off()
    ax.set_xlim(0, RIGHT - 0.14)
    ax.set_ylim(height, 0)
    row = height / len(classes)
    sw = 0.17
    for k, (letter, name, colour) in enumerate(classes):
        y = (k + 0.5) * row
        ax.add_patch(Rectangle((0, y - sw / 2), sw, sw, color=colour, lw=0))
        ax.text(sw / 2, y, letter, ha="center", va="center", fontsize=FS_SMALL,
                fontweight="bold", color=text_colour(colour))
        first, rest = name.split(" ", 1)
        ax.text(sw + 0.06, y, f"{first}\n{rest}", ha="left", va="center", fontsize=FS_SMALL,
                color=INK, linespacing=1.05)


def plot(models, lags, windows, classes, eq16_class, out):
    plt.rcParams.update({"font.size": FS, "text.color": INK, "mathtext.default": "regular"})
    stage1, stage2 = models
    gx, gy = np.meshgrid(np.arange(ARCL_RANGE[0], ARCL_RANGE[1] + STEP / 2, STEP),
                         np.arange(ARCV_RANGE[0], ARCV_RANGE[1] + STEP / 2, STEP))
    row_block = ROW_HEAD + PANEL_H
    fig_h = COL_HEAD + 3 * row_block + 2 * GAP_Y + BOTTOM
    fig = plt.figure(figsize=(FIG_W, fig_h), facecolor=WHITE)
    base = place(fig, 0, 0, FIG_W, fig_h, fig_h)
    base.set_axis_off()
    base.set_xlim(0, FIG_W)
    base.set_ylim(fig_h, 0)

    row_titles = [
        r"Stage 1: probability that the crescent is detected, $P_\mathrm{det}$",
        r"Stage 2: probability of a naked-eye sighting if the crescent is detected, $P_\mathrm{eye}$",
        "Operational bands (both stages combined)",
    ]
    lefts = [LEFT + j * (PANEL_W + GAP_X) for j in range(3)]
    right_col = lefts[2] + PANEL_W + 0.12
    for j, (lag, w) in enumerate(zip(lags, windows)):
        cx = lefts[j] + PANEL_W / 2
        base.text(cx, 0.05, rf"Lag time $t_\mathrm{{lag}}$ = {lag:.0f} min", ha="center",
                  va="top", fontsize=FS_HEAD, fontweight="bold", color=INK)
        base.text(cx, 0.25, f"{len(w):,} reports, {lag - HALF_WINDOW:.1f}–{lag + HALF_WINDOW:.1f} min",
                  ha="center", va="top", fontsize=FS_SMALL, color=MUTED)

    for r in range(3):
        top = COL_HEAD + r * (row_block + GAP_Y)
        base.text(0.04, top + 0.04, row_titles[r], ha="left", va="top", fontsize=FS_HEAD,
                  fontweight="bold", color=INK)
        panel_top = top + ROW_HEAD
        for j, (lag, w) in enumerate(zip(lags, windows)):
            ax = place(fig, lefts[j], panel_top, PANEL_W, PANEL_H, fig_h)
            ax.set_facecolor(WHITE)
            style_axes(ax, show_x=(r == 2), show_y=(j == 0))
            if r == 0:
                probability_panel(ax, stage1, gx, gy, lag, (0.30, 0.50), "det")
            elif r == 1:
                probability_panel(ax, stage2, gx, gy, lag, (0.30, 0.60), "eye")
            else:
                p_det, p_eye = proba(stage1, gy, gx, lag), proba(stage2, gy, gx, lag)
                band_panel(ax, eq16_class(p_det, p_eye), gx, gy, classes)
            shade_impossible(ax)
            ax.scatter(w[ARCL], w[ARCV], **DOT)
            ax.text(0.8, ARCV_RANGE[1] - 0.8, f"({LETTERS[3 * r + j]})", ha="left", va="top",
                    fontsize=FS, fontweight="bold", color=INK, zorder=7)
            if j == 0:
                ax.set_ylabel("ARCV (deg)", fontsize=FS, color=INK, labelpad=2)
            if r == 2 and j == 1:
                ax.set_xlabel("ARCL (deg)", fontsize=FS, color=INK, labelpad=2)
        if r == 0:
            colour_bar(fig, right_col, panel_top, PANEL_H, fig_h, (0.30, 0.50), r"$P_\mathrm{det}$")
        elif r == 1:
            colour_bar(fig, right_col, panel_top, PANEL_H, fig_h, (0.30, 0.60), r"$P_\mathrm{eye}$")
        else:
            band_key(fig, right_col, panel_top + 0.02, PANEL_H - 0.04, fig_h, classes)

    handles = [
        Line2D([], [], marker="o", ls="", color=INK, markeredgecolor="white", alpha=0.8, ms=3.4,
               label=r"ICOP reports with $t_\mathrm{lag}$ within ±7.5 min of the column's value"),
        Patch(facecolor=IMPOSSIBLE, edgecolor=SPINE, lw=0.5, label="|ARCV| > ARCL: not possible"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.06 / fig_h), ncol=2,
               fontsize=FS_SMALL, frameon=False, handletextpad=0.4, columnspacing=1.6,
               handlelength=1.2)
    fig.savefig(out, dpi=300, facecolor=WHITE)
    plt.close(fig)
    return fig_h


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--icop", default=str(DEFAULT_ICOP),
                        help="Path to data/icop/crescent_features.csv")
    parser.add_argument("--oof", default=str(DEFAULT_OOF),
                        help="Path to data/icop/two_stage_oof_predictions.csv")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS),
                        help="Folder with the final Stage 1 and Stage 2 joblib models")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Figure path (PNG, 300 dpi)")
    args = parser.parse_args()

    hc = load_sibling("hilalpy_independent_check")
    t1 = load_sibling("table1_criteria_comparison")
    vm = load_sibling("visibility_maps")
    stage1, stage2 = t1.load_saved_models(args.models_dir)
    _, model_df = hc.load_icop(args.icop)
    model_df["oof_det"], model_df["oof_eye"] = hc.load_oof(args.oof, model_df)
    model_df["p_det"], model_df["p_eye"] = vm.probabilities(stage1, stage2, model_df)
    model_df["band"] = vm.eq16_class(model_df["p_det"], model_df["p_eye"])
    cuts = {"det": (vm.PDET_LOW, vm.PDET_HIGH), "eye": (vm.PEYE_LOW, vm.PEYE_HIGH)}
    if cuts != {"det": (0.30, 0.50), "eye": (0.30, 0.60)} or (t1.PDET_HIGH, t1.PEYE_HIGH) != (0.50, 0.60):
        raise ValueError("The operational thresholds have changed")

    # ------------------------------------------------ [1] lag values
    pct, lags = choose_lags(model_df[LAG])
    print(f"[1] Lag time of the {len(model_df):,} modelling rows (min): "
          + ", ".join(f"P{p} = {v:.2f}" for p, v in zip(LAG_PERCENTILES, pct))
          + f"; median {model_df[LAG].median():.2f}, range {model_df[LAG].min():.2f} to "
          f"{model_df[LAG].max():.2f}")
    print("    Columns (rounded to 5 min): " + ", ".join(f"{v:.0f} min" for v in lags))
    windows = [model_df[(model_df[LAG] - v).abs() <= HALF_WINDOW] for v in lags]

    # ------------------------------------------------ [2] threshold lines
    print("\n[2] Lines of equal probability (final models). 'ARCV at ARCL 10': the ARCV on the line"
          f"\n    at ARCL = {ARCL_FIXED:.0f} deg; 'lowest ARCL': where the line meets ARCV = ARCL;"
          "\n    'ARCL there': the ARCL on the line at the median ARCV of the reports in that column."
          "\n    * = the point has |ARCV| > ARCL and cannot occur (the line misses that ARCL or ARCV).")
    print(f"{'lag (min)':>9s}  {'line':14s} {'ARCV at ARCL 10':>16s} {'lowest ARCL':>12s} "
          f"{'median ARCV':>12s} {'ARCL there':>11s}")
    for lag, w in zip(lags, windows):
        med_v = w[ARCV].median()
        for name, model, stage in (("P_det", stage1, "det"), ("P_eye", stage2, "eye")):
            for c in cuts[stage]:
                v10 = arcv_on_line(model, c, ARCL_FIXED, lag)
                l_med = arcl_on_line(model, c, med_v, lag)
                flag10 = "*" if abs(v10) > ARCL_FIXED else " "
                flag_med = "*" if abs(med_v) > l_med else " "
                print(f"{lag:9.0f}  {name} = {c:.2f}   {v10:15.2f}{flag10} "
                      f"{diagonal_on_line(model, c, lag):12.2f} {med_v:12.2f} {l_med:11.2f}{flag_med}")
    # The lines are straight and parallel (both stages are linear in the log-odds),
    # so these shifts and slopes are the same for every line and lag time.
    mid = lags[1]
    for name, model in (("P_det", stage1), ("P_eye", stage2)):
        dv = arcv_on_line(model, 0.5, ARCL_FIXED, mid + 10) - arcv_on_line(model, 0.5, ARCL_FIXED, mid)
        dl = arcl_on_line(model, 0.5, 5.0, mid + 10) - arcl_on_line(model, 0.5, 5.0, mid)
        slope = arcv_on_line(model, 0.5, 11.0, mid) - arcv_on_line(model, 0.5, 10.0, mid)
        print(f"    {name}: 10 min more lag moves every line by {dv:+.2f} deg in ARCV (ARCL fixed) or "
              f"{dl:+.2f} deg in ARCL (ARCV fixed); slope of the lines {slope:+.3f} deg ARCV per deg ARCL")

    # ------------------------------------------------ [3] reports in each column
    print(f"\n[3] ICOP reports with a lag time within +/-{HALF_WINDOW} min of each column's value "
          "(the dots). Probabilities: final model at each report's own lag time; OOF = out-of-fold.")
    letters = [c[0] for c in vm.CLASSES]
    print(f"{'lag':>4s} {'window (min)':>13s} {'n':>5s} {'lunations':>9s} {'not seen':>9s} {'aided':>6s} "
          f"{'naked':>6s} {'seen %':>7s} {'P_det':>6s} {'OOF':>6s} {'eye % of seen':>13s} {'P_eye':>6s} "
          f"{'OOF':>6s}  {'median ARCV, ARCL':>17s}  bands " + " ".join(letters))
    for lag, w in zip(lags, windows):
        cls = w["cls"].to_numpy()
        seen = cls > 0
        n0, n1, n2 = (int((cls == k).sum()) for k in (0, 1, 2))
        bands = np.bincount(w["band"], minlength=len(letters))
        print(f"{lag:4.0f} {lag - HALF_WINDOW:6.1f}-{lag + HALF_WINDOW:5.1f} {len(w):5,d} "
              f"{w[GROUP].nunique():9d} {n0:9d} {n1:6d} {n2:6d} {100 * seen.mean():7.1f} "
              f"{w['p_det'].mean():6.3f} {w['oof_det'].mean():6.3f} {100 * np.mean(cls[seen] == 2):13.1f} "
              f"{w.loc[seen, 'p_eye'].mean():6.3f} {w.loc[seen, 'oof_eye'].mean():6.3f}  "
              f"{w[ARCV].median():8.2f}, {w[ARCL].median():6.2f}  "
              + " ".join(f"{b:d}" for b in bands))
    print("    (P_eye and 'eye % of seen' are means over the sightings only.)")

    fig_h = plot((stage1, stage2), lags, windows, vm.CLASSES, vm.eq16_class, args.output)
    print(f"\nFigure: {args.output} ({FIG_W} x {fig_h:.2f} in, 300 dpi); panel {PANEL_W:.2f} x "
          f"{PANEL_H:.2f} in; grid step {STEP} deg")


if __name__ == "__main__":
    main()
