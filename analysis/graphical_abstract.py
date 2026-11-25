#!/usr/bin/env python3
"""Graphical abstract from the real outputs (second review round, comment 6).

The script draws the three-panel graphical abstract and saves it as a PNG file
at 300 dpi.

Panels (Elsevier: at least 1328 x 531 px, about 2.5:1, readable when small):
  left    the pipeline: ICOP reports -> ARCV, ARCL and lag at local sunset ->
          Stage 1 P_det -> Stage 2 P_eye -> the five classes of Eq. 11;
  middle  a real map from visibility_maps.py (final model, each grid point at
          its own sunset; default 17 May 2026, the first evening after the
          new moon of 16 May 2026);
  right   ROC AUC of the two stages: out-of-fold, 5-fold cross-validation
          grouped by lunation (data/icop/two_stage_oof_predictions.csv), and
          the independent HilalPy check (headline sample of
          hilalpy_independent_check.py: 1,365 de-duplicated non-ICOP evening
          reports without the Alrefay tables; final models). Without --hilalpy
          the HilalPy values are left blank and a warning is printed.
"""
import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # the repository root
DEFAULT_ICOP = REPO / "data" / "icop" / "crescent_features.csv"
DEFAULT_OOF = REPO / "data" / "icop" / "two_stage_oof_predictions.csv"
DEFAULT_MODELS = REPO / "models"
DEFAULT_OUTPUT = REPO / "figures" / "graphical_abstract.png"
GROUP = "page_url"
TARGET = "vis_class_empirical"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, PANEL, LINE = "#ffffff", "#f6f6f4", "#c3c2b7"
FIG_W, FIG_H = 8.0, 3.2  # inches; 2,400 x 960 px at 300 dpi (2.5:1)


def load_sibling(name):
    """A script in this folder, loaded by path so that 'python3 -I' also works."""
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- numbers
def cv_results(oof_path):
    """Out-of-fold ROC AUC of both stages (lunation-grouped 5-fold CV)."""
    oof = pd.read_csv(oof_path, low_memory=False)
    y = oof[TARGET].astype(int).to_numpy()
    seen = y > 0
    return {
        "n": len(oof), "lunations": oof[GROUP].nunique(),
        "auc_det": roc_auc_score(seen, oof["p_det"]),
        "auc_eye": roc_auc_score(y[seen] == 2, oof.loc[seen, "p_eye"]),
    }


def hilalpy_results(hilalpy_path, icop_path, stage1, stage2):
    """ROC AUC of the final model on the HilalPy headline sample."""
    hc = load_sibling("hilalpy_independent_check")
    h = hc.load_hilalpy(hilalpy_path)
    archive, _ = hc.load_icop(icop_path)
    _, primary, steps, _ = hc.independent_sample(h, archive)
    if steps != hc.EXPECTED_STEPS:
        raise ValueError(f"HilalPy filter counts {steps} differ from {hc.EXPECTED_STEPS}")
    head = primary[primary["src"] != "AlR"]
    hc.check_counts(head, hc.EXPECTED_HEADLINE, "Headline sample")
    X = head[hc.FEATURES]
    p_det, p_eye = stage1.predict_proba(X)[:, 1], stage2.predict_proba(X)[:, 1]
    y = head["cls"].to_numpy()
    seen = y > 0
    return {"n": len(head), "auc_det": roc_auc_score(seen, p_det),
            "auc_eye": roc_auc_score(y[seen] == 2, p_eye[seen])}


# ---------------------------------------------------------------- drawing helpers
def box(fig, x, y, w, h, fc=SURFACE, ec=LINE, lw=0.8):
    """Rounded box in figure inches."""
    p = FancyBboxPatch((x / FIG_W, y / FIG_H), w / FIG_W, h / FIG_H,
                       boxstyle="round,pad=0,rounding_size=0.012", transform=fig.transFigure,
                       fc=fc, ec=ec, lw=lw)
    fig.patches.append(p)


def text(fig, x, y, s, **kw):
    kw.setdefault("color", INK)
    kw.setdefault("fontsize", 8)
    fig.text(x / FIG_W, y / FIG_H, s, **kw)


def arrow(fig, x, y0, y1):
    fig.patches.append(FancyArrowPatch(
        (x / FIG_W, y0 / FIG_H), (x / FIG_W, y1 / FIG_H), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=8, color=INK2, lw=1.0))


def pipeline(fig, vm, cv):
    """Left panel: the method as a vertical flow."""
    x0, w = 0.12, 2.16
    text(fig, x0, 2.98, "From reports to bands", fontsize=9.5, weight="bold")
    steps = [
        ("ICOP crescent reports", f"{cv['n']:,} reports, {cv['lunations']} lunations"),
        ("Geometry at local sunset", "ARCV, ARCL and Moon lag"),
        ("Stage 1: P_det", "seen by any means?"),
        ("Stage 2: P_eye", "seen by naked eye, if seen?"),
    ]
    h, gap, top, last = 0.45, 0.1, 2.86, 0.5
    for i, (head, sub) in enumerate(steps):
        y = top - h - i * (h + gap)
        box(fig, x0, y, w, h)
        text(fig, x0 + 0.1, y + h - 0.07, head, fontsize=8.5, weight="bold", va="top")
        text(fig, x0 + 0.1, y + 0.06, sub, fontsize=7.5, color=INK2, va="bottom")
        arrow(fig, x0 + w / 2, y, y - gap)
    # Final step: the five classes, with the colours of the map.
    y = top - 4 * (h + gap) - last
    box(fig, x0, y, w, last)
    text(fig, x0 + 0.1, y + last - 0.07, "Five operational bands", fontsize=8.5, weight="bold",
         va="top")
    cw = (w - 0.2) / 5
    for i, (k, _, colour) in enumerate(vm.CLASSES):
        cx = x0 + 0.1 + i * cw
        fig.patches.append(Rectangle((cx / FIG_W, (y + 0.07) / FIG_H), (cw - 0.04) / FIG_W,
                                     0.16 / FIG_H, transform=fig.transFigure, fc=colour,
                                     ec=LINE, lw=0.4))
        text(fig, cx + (cw - 0.04) / 2, y + 0.15, k, fontsize=7.5, weight="bold", ha="center",
             va="center", color="#fcfcfb" if i < 2 else INK)


def map_panel(fig, vm, grid, point, conj):
    """Middle panel: the real class map with a short legend."""
    left, top, width = 2.48, 2.7, 3.34
    height = width * (grid_span(vm)[1] / grid_span(vm)[0])
    ax = fig.add_axes([left / FIG_W, (top - height) / FIG_H, width / FIG_W, height / FIG_H],
                      projection=ccrs.PlateCarree())
    vm.draw_map(ax, grid, point, compact=True)
    for s in ax.spines.values():
        s.set_edgecolor(LINE)
        s.set_linewidth(0.6)
    day = vm.long_date(grid["day"])
    text(fig, left, 2.98, f"Map at local sunset, {day}", fontsize=9.5, weight="bold")
    text(fig, left, 2.8, f"Each point at its own sunset; new moon {vm.ut_text(conj)}",
         fontsize=7.5, color=INK2)
    # Legend: detectable classes in the first column, the others in the second.
    for i, (k, label, colour) in enumerate(vm.CLASSES):
        x = left + (i // 3) * 1.7
        y = top - height - 0.27 - (i % 3) * 0.18
        fig.patches.append(Rectangle((x / FIG_W, y / FIG_H), 0.13 / FIG_W, 0.12 / FIG_H,
                                     transform=fig.transFigure, fc=colour, ec=LINE, lw=0.4))
        text(fig, x + 0.18, y + 0.06, f"{k}: {label}", fontsize=7, va="center")


def grid_span(vm):
    return (vm.LONS[-1] - vm.LONS[0] + 1, vm.LATS[-1] - vm.LATS[0] + 1)


def results_panel(fig, cv, hp):
    """Right panel: ROC AUC table, cross-validation and the independent check."""
    x0, w = 6.02, 1.86
    text(fig, x0, 2.98, "Discrimination (ROC AUC)", fontsize=9.5, weight="bold")
    blocks = [
        ("ICOP, 5-fold CV", "grouped by lunation", cv),
        ("Independent: HilalPy", f"{hp['n']:,} non-ICOP reports" if hp else "non-ICOP reports", hp),
    ]
    h, gap, top = 1.12, 0.14, 2.86
    for i, (head, sub, r) in enumerate(blocks):
        y = top - h - i * (h + gap)
        box(fig, x0, y, w, h, fc=PANEL, ec=PANEL)
        text(fig, x0 + 0.1, y + h - 0.1, head, fontsize=8.5, weight="bold", va="top")
        text(fig, x0 + 0.1, y + h - 0.29, sub, fontsize=7.2, color=INK2, va="top")
        for j, (key, label) in enumerate([("auc_det", "detection"), ("auc_eye", "naked eye")]):
            cx = x0 + 0.1 + j * (w / 2)
            value = f"{r[key]:.3f}" if r else "–"
            text(fig, cx, y + 0.28, value, fontsize=17, weight="bold", va="bottom")
            text(fig, cx, y + 0.1, label, fontsize=7.5, color=INK2, va="bottom")
    text(fig, x0, top - 2 * h - gap - 0.12, "Out-of-fold for ICOP; final model for HilalPy.",
         fontsize=6.5, color=MUTED, va="top")


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hilalpy", default=None,
                        help="HilalPy 'Final 30.6.2023.csv' (Mendeley Data, DOI 10.17632/rs994bxgc4.1)")
    parser.add_argument("--icop", default=str(DEFAULT_ICOP),
                        help="Path to data/icop/crescent_features.csv")
    parser.add_argument("--oof", default=str(DEFAULT_OOF),
                        help="Path to data/icop/two_stage_oof_predictions.csv")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS),
                        help="Folder with the final Stage 1 and Stage 2 joblib models")
    parser.add_argument("--date", default="2026-05-17", help="Evening of the map (YYYY-MM-DD)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Figure path (PNG, 300 dpi)")
    args = parser.parse_args()

    vm = load_sibling("visibility_maps")
    stage1, stage2 = vm.load_models(args.models_dir)

    cv = cv_results(args.oof)
    print(f"Lunation-grouped 5-fold CV (out-of-fold): {cv['n']:,} reports, {cv['lunations']} "
          f"lunations; ROC AUC detection {cv['auc_det']:.4f}, naked eye {cv['auc_eye']:.4f}")
    if args.hilalpy:
        hp = hilalpy_results(args.hilalpy, args.icop, stage1, stage2)
        print(f"HilalPy headline sample (final model): {hp['n']:,} reports; ROC AUC detection "
              f"{hp['auc_det']:.4f}, naked eye {hp['auc_eye']:.4f}")
    else:
        hp = None
        print("WARNING: --hilalpy not given; the HilalPy values are left blank in the figure")

    grid = vm.compute_grid(args.date, stage1, stage2)
    point = vm.site_point(args.date, stage1, stage2)
    conj = vm.nearest_new_moon(point["sunset"])
    print(f"Map {args.date}: {point['name']} P_det {point['p_det']:.4f}, P_eye {point['p_eye']:.4f}, "
          f"band {vm.CLASSES[point['cls']][0]}")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(SURFACE)
    pipeline(fig, vm, cv)
    map_panel(fig, vm, grid, point, conj)
    results_panel(fig, cv, hp)
    fig.savefig(args.output, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    w, h = int(round(FIG_W * 300)), int(round(FIG_H * 300))
    print(f"Saved {args.output} ({w} x {h} px, ratio {w / h:.2f}:1)")


if __name__ == "__main__":
    main()
