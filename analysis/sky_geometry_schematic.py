#!/usr/bin/env python3
"""Schematic of the sky geometry at local sunset
(second review round, comment 6).

This script makes a labelled drawing of the quantities used in the
paper, all taken at local sunset: the western horizon, the Sun just below it
(its centre at -0.83 deg), the Moon above, ARCV (vertical), DAZ (along the
horizon), ARCL (the Sun-Moon arc) and the crescent width W on an enlarged
crescent. The drawing is a schematic, not to scale, and is labelled so.
"""
import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE.parent / "figures" / "sky_geometry_schematic.png"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SKY_TOP, SKY_LOW, GROUND = "#d6e5f7", "#f8efe0", "#e6e4dc"
SUN, SUNLIT, DARK_SIDE = "#eda100", "#f7d774", "#c3c2b7"
ACCENT = "#1c5cab"  # the three angles

# Schematic positions (arbitrary units; the angles are exaggerated for clarity).
SUN_XY, SUN_R = (0.0, -0.75), 0.5
MOON_XY, MOON_R = (-4.2, 5.8), 0.42
INSET_XY, INSET_R = (4.6, 5.3), 1.55
ILLUMINATED = 0.16  # illuminated fraction of the enlarged crescent


def crescent(centre, radius, k, toward):
    """Polygon of a crescent whose sunlit limb faces the direction 'toward' (radians).

    The outer edge is the limb (a half circle); the terminator is a half ellipse
    with semi-axis radius * (1 - 2k) on the same side, so the width at the middle
    is 2 * radius * k = W.
    """
    t = np.linspace(-np.pi / 2, np.pi / 2, 120)
    limb = np.c_[radius * np.cos(t), radius * np.sin(t)]
    term = np.c_[radius * (1 - 2 * k) * np.cos(t[::-1]), radius * np.sin(t[::-1])]
    pts = np.r_[limb, term]
    rot = np.array([[math.cos(toward), -math.sin(toward)], [math.sin(toward), math.cos(toward)]])
    return pts @ rot.T + np.asarray(centre)


def dim_arrow(ax, a, b, colour=ACCENT, lw=1.6, style="<|-|>", rad=0.0, z=6):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11, color=colour, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0,
                                 zorder=z))


def draw(out):
    # Equal scales in x and y, so that the discs stay round: 0.47 in per unit.
    # The quantities are defined in the caption and the text, so the figure
    # holds only the drawing, with a header strip of 0.6 in above it.
    fw, scale = 8.0, 0.47
    w, h = 16 * scale, 10.7 * scale
    fh = h + 0.6 + 0.12
    fig = plt.figure(figsize=(fw, fh))
    ax = fig.add_axes([0.03, 0.12 / fh, w / fw, h / fh])
    ax.axis("off")

    # Sky (twilight gradient) and ground.
    sky = np.linspace(0, 1, 256)[:, None]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("sky", [SKY_LOW, SKY_TOP])
    ax.imshow(sky, extent=(-8.6, 7.4, 0, 8.3), origin="lower", cmap=cmap, zorder=0)
    ax.add_patch(Rectangle((-8.6, -2.4), 16, 2.4, fc=GROUND, ec="none", zorder=1))
    ax.set_xlim(-8.6, 7.4)
    ax.set_ylim(-2.4, 8.3)
    ax.set_aspect("equal")
    ax.plot([-8.6, 7.4], [0, 0], color=INK, lw=1.4, zorder=3)
    ax.text(7.2, 0.12, "Western horizon (altitude 0°)", fontsize=9, color=INK2, ha="right",
            va="bottom")

    # Sun just below the horizon (dashed edge: hidden from view).
    sx, sy = SUN_XY
    mx, my = MOON_XY
    ax.add_patch(Circle(SUN_XY, SUN_R, fc=SUN, ec=INK2, lw=1.0, ls=(0, (3, 2)), alpha=0.9, zorder=4))
    ax.annotate("Sun, just below the horizon\n(centre at −0.83° at sunset)", xy=(sx + 0.45, sy - 0.25),
                xytext=(1.4, -1.85), fontsize=9, color=INK, va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8), zorder=7)

    # Right triangle: DAZ along the horizon level, ARCV vertical, ARCL the Sun-Moon arc.
    foot = (mx, sy)
    ax.plot([sx, mx], [sy, sy], color=ACCENT, lw=0.8, ls=(0, (2, 2)), zorder=5)
    ax.plot([mx, mx], [sy, my], color=ACCENT, lw=0.8, ls=(0, (2, 2)), zorder=5)
    s = 0.32  # right-angle mark
    ax.plot([mx, mx + s, mx + s], [sy + s, sy + s, sy], color=ACCENT, lw=0.9, zorder=5)
    # DAZ, drawn just below the horizon at the Sun's altitude.
    dim_arrow(ax, (mx, sy - 0.75), (sx, sy - 0.75))
    for x in (mx, sx):
        ax.plot([x, x], [sy - 0.95, sy - 0.1], color=ACCENT, lw=0.8, zorder=5)
    ax.text((mx + sx) / 2, sy - 0.95, "DAZ", fontsize=12, weight="bold", color=ACCENT,
            ha="center", va="top", zorder=7)
    # ARCV, to the left of the Moon's vertical.
    dim_arrow(ax, (mx - 0.75, sy), (mx - 0.75, my))
    for y in (sy, my):
        ax.plot([mx - 0.95, mx - 0.1], [y, y], color=ACCENT, lw=0.8, zorder=5)
    ax.text(mx - 0.95, (sy + my) / 2 + 0.3, "ARCV", fontsize=12, weight="bold", color=ACCENT,
            ha="right", va="center", zorder=7)
    ax.text(mx - 0.95, (sy + my) / 2 - 0.25, "Moon altitude\n− Sun altitude", fontsize=8.5,
            color=INK2, ha="right", va="top", zorder=7)
    # ARCL, the Sun-Moon arc.
    dim_arrow(ax, (sx - 0.05, sy + 0.05), (mx + 0.3, my - 0.3), rad=-0.12, lw=1.8)
    ax.text(-1.15, 3.35, "ARCL", fontsize=12, weight="bold", color=ACCENT, ha="left",
            va="center", zorder=7)
    ax.text(-1.15, 2.85, "Sun–Moon arc\n(elongation)", fontsize=8.5, color=INK2, ha="left",
            va="top", zorder=7)

    # Moon: dark disc with a thin sunlit crescent facing the Sun.
    toward = math.atan2(sy - my, sx - mx)
    ax.add_patch(Circle(MOON_XY, MOON_R, fc=DARK_SIDE, ec=INK2, lw=0.8, zorder=6))
    ax.add_patch(Polygon(crescent(MOON_XY, MOON_R, 0.12, toward), fc=SUNLIT, ec=INK, lw=0.6,
                         zorder=7))
    ax.text(mx, my + 0.6, "Moon", fontsize=9.5, color=INK, ha="center", va="bottom", zorder=7)

    # Enlarged crescent with the width W.
    ix, iy = INSET_XY
    ax.add_patch(Circle(INSET_XY, INSET_R + 0.35, fc="#fcfcfb", ec=INK2, lw=0.8, zorder=5))
    for sign in (1, -1):
        # Tangent-like guide lines from the Moon to the enlarged view.
        ang = math.atan2(iy - my, ix - mx) + sign * math.pi / 2
        ax.plot([mx + MOON_R * math.cos(ang), ix + (INSET_R + 0.35) * math.cos(ang)],
                [my + MOON_R * math.sin(ang), iy + (INSET_R + 0.35) * math.sin(ang)],
                color=MUTED, lw=0.7, ls=(0, (3, 3)), zorder=4)
    ax.add_patch(Circle(INSET_XY, INSET_R, fc=DARK_SIDE, ec=INK2, lw=0.8, zorder=6))
    ax.add_patch(Polygon(crescent(INSET_XY, INSET_R, ILLUMINATED, toward), fc=SUNLIT, ec=INK,
                         lw=0.8, zorder=7))
    # W runs from the terminator to the limb, at the middle of the crescent.
    u = np.array([math.cos(toward), math.sin(toward)])
    v = np.array([-u[1], u[0]])
    inner = np.array(INSET_XY) + INSET_R * (1 - 2 * ILLUMINATED) * u
    outer = np.array(INSET_XY) + INSET_R * u
    dim_arrow(ax, tuple(inner), tuple(outer), colour=INK, lw=1.2, z=8)
    ax.annotate("W", xy=tuple((inner + outer) / 2 - 0.12 * v), xytext=(ix - 0.9, iy - 2.75),
                fontsize=12, weight="bold", color=INK, va="center", ha="right",
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8), zorder=8)
    ax.text(ix, iy + INSET_R + 0.5, "Crescent, enlarged", fontsize=9.5, color=INK, ha="center",
            va="bottom", zorder=8)
    ax.text(ix - 0.8, iy - 2.75, "crescent width", fontsize=8.5, color=INK2, va="center",
            zorder=8)

    # Header and the 'not to scale' label, 0.27 in below the top edge.
    top = 1 - 0.27 / fh
    fig.text(0.04, top, "Sky geometry at local sunset, looking west", fontsize=14,
             weight="bold", color=INK, va="top")
    fig.text(0.96, top, "Schematic, not to scale", fontsize=11, weight="bold", color=INK,
             ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.35", fc="#fcfcfb", ec=INK, lw=1.0))
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Figure path (PNG, 300 dpi)")
    args = parser.parse_args()
    draw(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
