#!/usr/bin/env python3
"""The model's output against the minimum-visibility records (reviewer comment 6).

Reviewer: "Faid et al. report aided detections as low as ARCL ~ 4.6 to 4.7 deg
and ARCV ~ 0.45 deg. In Fig. 8, Sharjah has ARCV = 7.5 deg and ARCL = 7.5 deg,
which is above the lowest naked-eye record and above the MABIMS 2021 minima,
yet the model says 'not detectable' (Pdet = 0.20). I would ask the authors to
explain this, ideally by plotting the model's output against the
minimum-visibility records."

Inputs:
  --icop        data/icop/crescent_features.csv;
                modelling subset of 4,470 rows as in script 04
  --oof         data/icop/two_stage_oof_predictions.csv: out-of-fold P_det and
                P_eye from 5-fold cross-validation grouped by lunation
                (StratifiedGroupKFold, groups = page_url, random_state=42)
  --models-dir  the final Stage 1 and Stage 2 models (fitted on all 4,470 rows)
  --hilalpy     HilalPy 'Final 30.6.2023.csv' (Mendeley Data, DOI
                10.17632/rs994bxgc4.1); evening records of all sources (ICOP
                included) after filters 2-5 of hilalpy_independent_check.py

Sections:
  [1] 15 June 2026 at Sharjah (25.3463 N, 55.4209 E): sunset geometry with
      PyEphem in the conventions of the ICOP features (sunset when the Sun's
      centre is at -0.833 deg; airless topocentric positions, elevation 0;
      moonset with PyEphem defaults, i.e. upper limb and standard refraction,
      as in 03b_correct_moon_lag.py; W = 2 x semidiameter x illuminated
      fraction), the final-model P_det and P_eye, and the verdicts of the
      Table 1 and calendar criteria at the same point.
  [2] ICOP reports with similar geometry: how many were seen, by which means,
      and their mean out-of-fold P_det. The 95 % interval of the seen share is
      a cluster bootstrap over lunations (2,000 resamples, default_rng(42)).
  [3] The lowest sightings by ARCL and by ARCV, in the ICOP modelling data and
      in the HilalPy evening records, with the final-model P_det. Identical
      records (same date, geometry and class) are shown once, with a count.
  [4] What a P_det near 0.20 means: shares of sightings below the 0.30 and
      0.50 cut-offs (out-of-fold), the observed seen share among reports with
      P_det near 0.20, and the MABIMS 2021 local screen (logic of
      calendar_criteria_benchmark.py) on the same records.
  [5] Figure (--output): (a) final-model P_det over ARCL and ARCV, with the lag
      predicted from ARCV and ARCL by a least-squares line fitted on the 4,470
      ICOP rows, and the ICOP reports, the 20 lowest-ARCL HilalPy sightings,
      the Sharjah point and the MABIMS 2021 limits overlaid; (b) cumulative
      distributions of the out-of-fold P_det by observed outcome.
"""
import argparse
import importlib.util
import math
from pathlib import Path

import ephem
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

ARCV = "relative_altitude_deg"
ARCL = "elongation_deg"
LAG = "moon_lag_minutes"
DAZ = "relative_azimuth_deg"
WIDTH = "crescent_width_arcmin"
FEATURES = [ARCV, ARCL, LAG]
GROUP = "page_url"
HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # the repository root
DEFAULT_ICOP = REPO / "data" / "icop" / "crescent_features.csv"
DEFAULT_OOF = REPO / "data" / "icop" / "two_stage_oof_predictions.csv"
DEFAULT_MODELS = REPO / "models"
DEFAULT_OUTPUT = REPO / "figures" / "minimum_visibility.png"

SITE = ("Sharjah", 25.3463, 55.4209)
EVENING = "2026/6/15"
SUN_ALT_AT_SUNSET = -0.8333  # Sun's centre at sunset, as sun_alt_deg in the ICOP features
PDET_LOW, PDET_HIGH, PEYE_LOW, PEYE_HIGH = 0.30, 0.50, 0.30, 0.60
CENTRE = (7.5, 7.5, 40.0)  # ARCV, ARCL (deg) and lag (min) quoted by the reviewer, rounded
N_BOOT, BOOT_SEED = 2000, 42
N_LIST, N_PLOT = 10, 20
CLASS_NAME = {0: "not seen", 1: "aided only", 2: "naked eye"}

# Reference palette: categorical slots 2 and 3, a neutral grey for 'not seen' and
# a single-hue blue ramp for P_det (colour-vision separation checked).
C_AID, C_EYE, C_NOT = "#eb6834", "#1baf7a", "#6b6a65"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
BLUES = ["#fcfcfb", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec"]


# ---------------------------------------------------------------- helpers
def load_sibling(name):
    """A script in this folder, loaded by path so that 'python3 -I' also works."""
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def band(p_det, p_eye):
    """Operational band of Eq. 11."""
    if p_det < PDET_LOW:
        return "E, not detectable"
    if p_det < PDET_HIGH:
        return "D, marginal detectability"
    if p_eye < PEYE_LOW:
        return "C, aid-assisted likely"
    if p_eye < PEYE_HIGH:
        return "B, borderline naked-eye"
    return "A, naked-eye likely"


def lunation_interval(seen, groups, seed=BOOT_SEED):
    """95 % cluster-bootstrap interval of the seen share (lunations resampled)."""
    codes, _ = pd.factorize(pd.Series(groups))
    n_g = np.bincount(codes).astype(float)
    k_g = np.bincount(codes, weights=np.asarray(seen, float))
    draws = np.random.default_rng(seed).integers(0, len(n_g), size=(N_BOOT, len(n_g)))
    share = k_g[draws].sum(axis=1) / n_g[draws].sum(axis=1)
    return np.percentile(share, [2.5, 97.5])


# ---------------------------------------------------------------- geometry
def sunset_geometry(lat, lon, day):
    """Sunset geometry in the conventions of the ICOP feature table."""
    obs = ephem.Observer()
    obs.lat, obs.lon, obs.elevation = str(lat), str(lon), 0
    obs.pressure = 0  # airless: geometric topocentric positions
    obs.horizon = str(SUN_ALT_AT_SUNSET)
    obs.date = ephem.Date(day) + (12 - lon / 15) / 24  # local mean noon
    sunset = obs.next_setting(ephem.Sun(), use_center=True)
    obs.date = sunset
    sun, moon = ephem.Sun(obs), ephem.Moon(obs)
    # Moonset as in 03b_correct_moon_lag.py: PyEphem defaults (upper limb, refraction).
    ms_obs = ephem.Observer()
    ms_obs.lat, ms_obs.lon, ms_obs.elevation = str(lat), str(lon), 0
    ms_obs.date = sunset
    m = ephem.Moon()
    set_before, rise_before = ms_obs.previous_setting(m), ms_obs.previous_rising(m)
    moonset = set_before if set_before > rise_before else ms_obs.next_setting(m)
    # Rounded down to the whole minute, as in 03b_correct_moon_lag.py and visibility_maps.py.
    moonset_min = ephem.Date(moonset).datetime().replace(second=0, microsecond=0)
    semidiameter = math.degrees(moon.radius) * 60
    return {
        "sunset_utc": ephem.Date(sunset).datetime(),
        ARCV: math.degrees(moon.alt - sun.alt),
        ARCL: math.degrees(ephem.separation(sun, moon)),
        DAZ: abs(math.degrees(moon.az - sun.az)),
        LAG: (moonset_min - ephem.Date(sunset).datetime()).total_seconds() / 60,
        "moon_alt": math.degrees(moon.alt),
        "sun_alt": math.degrees(sun.alt),
        WIDTH: 2 * semidiameter * moon.moon_phase,
        "age_h": (sunset - ephem.previous_new_moon(sunset)) * 24,
    }


# ---------------------------------------------------------------- tables
def aid_label(row, flags):
    """'naked eye', or the optical aids recorded for an aided-only sighting."""
    if row["cls"] == 2:
        return "naked eye"
    used = [name for col, name in flags if str(row[col]) in ("True", "V")]
    return "aid (" + ", ".join(used) + ")" if used else "aid"


# Reports checked by hand (record and ARCL to 0.01 deg).
KNOWN_ISSUES = {
    ("jut38", 0.54): "eclipse report (partial solar eclipse at sunset), not a crescent sighting",
    ("jua39", 6.72): "contradictory: ICOP text says 'not seen'; HilalPy codes it V = I",
}


def note(row):
    known = KNOWN_ISSUES.get((str(row["record"]), round(float(row[ARCL]), 2)))
    if known:
        return known
    if row[LAG] < 0:
        return "Moon set before the Sun (daytime image)"
    if row[ARCV] < -SUN_ALT_AT_SUNSET:
        return "Moon centre below the horizon at sunset"
    if row[ARCL] < 4:
        return "implausible: ARCL < 4 deg"
    return ""


def collapse(df, key_cols, id_col):
    """One row per identical record (same date, geometry and class); keeps order."""
    g = df.groupby(key_cols, sort=False, dropna=False)
    out = g.first()
    out["n"] = g.size()
    out[id_col] = g[id_col].agg(lambda s: "/".join(dict.fromkeys(s.astype(str))))
    return out.reset_index()


def print_list(df, title, id_col, extra=None):
    print(f"\n{title}")
    head = (f"{'Record':22s} {'Date':10s} {'Lat':>7s} {'Long':>8s} {'Method':26s} {'ARCV':>7s} "
            f"{'ARCL':>6s} {'Lag':>7s} {'P_det':>7s}")
    if extra:
        head += f" {extra[1]:>7s}"
    print(head + f" {'n':>2s}  Note")
    for _, r in df.iterrows():
        line = (f"{str(r[id_col])[:22]:22s} {r['date']:10s} {r['lat']:7.2f} {r['lon']:8.2f} "
                f"{r['method'][:26]:26s} {r[ARCV]:7.2f} {r[ARCL]:6.2f} {r[LAG]:7.1f} {r['pdet']:7.4f}")
        if extra:
            line += f" {r[extra[0]]:7.4f}"
        print(line + f" {int(r['n']):2d}  {note(r)}")


def lowest(df, col, k, key_cols, id_col):
    return collapse(df.sort_values([col, ARCL, ARCV], kind="mergesort"), key_cols, id_col).head(k)


# ---------------------------------------------------------------- figure
def plot(model_df, sight_low, point, reg, stage1, out):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 6.3), gridspec_kw={"width_ratios": [1.25, 1]})
    for a in (ax, bx):
        a.set_facecolor("#fcfcfb")
        for s in a.spines.values():
            s.set_color("#c3c2b7")
        a.tick_params(colors=INK2, labelsize=9)

    # (a) background: final-model P_det with the lag from the regression line.
    x = np.linspace(0, 20, 401)
    y = np.linspace(-5, 16, 421)
    gx, gy = np.meshgrid(x, y)
    lag = reg.predict(pd.DataFrame({ARCV: gy.ravel(), ARCL: gx.ravel()}))
    grid = pd.DataFrame({ARCV: gy.ravel(), ARCL: gx.ravel(), LAG: lag})
    p = stage1.predict_proba(grid[FEATURES])[:, 1].reshape(gx.shape)
    p = np.ma.masked_where(np.abs(gy) > gx, p)  # ARCL >= |ARCV| for any geometry
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)
    cf = ax.contourf(gx, gy, p, levels=np.linspace(0, 1, 11), cmap=cmap, zorder=0)
    cs = ax.contour(gx, gy, p, levels=[PDET_LOW, PDET_HIGH], colors=[INK2, INK],
                    linewidths=[1.2, 1.6], zorder=3)
    ax.clabel(cs, fmt={PDET_LOW: r"$P_\mathrm{det}$ = 0.30", PDET_HIGH: r"$P_\mathrm{det}$ = 0.50"}, fontsize=8,
              inline_spacing=4, manual=[(12.5, 7.4), (14.5, 9.4)])
    ax.fill_between(x, x, 16, color="#f0efec", lw=0, zorder=1)
    ax.fill_between(x, -5, -x, color="#f0efec", lw=0, zorder=1)
    ax.text(2.6, 12.6, "|ARCV| > ARCL:\nnot possible", fontsize=8, color=INK2, ha="center")

    cls = model_df["cls"].to_numpy()
    for c, colour, size, alpha, z in [(0, C_NOT, 4, 0.3, 2), (1, C_AID, 7, 0.75, 4),
                                      (2, C_EYE, 7, 0.6, 4)]:
        m = cls == c
        ax.scatter(model_df.loc[m, ARCL], model_df.loc[m, ARCV], s=size, color=colour,
                   alpha=alpha, lw=0, zorder=z)
    for c, colour in [(1, C_AID), (2, C_EYE)]:
        m = sight_low["cls"] == c
        ax.scatter(sight_low.loc[m, ARCL], sight_low.loc[m, ARCV], s=46, marker="^",
                   color=colour, edgecolor=INK, lw=0.8, zorder=6)

    # MABIMS 2021 local screen: Moon altitude >= 3 deg and ARCL >= 6.4 deg.
    arcv_min = 3 - SUN_ALT_AT_SUNSET
    # The boundary stops at the diagonal: ARCV cannot exceed ARCL.
    ax.plot([6.4, 6.4, 20], [6.4, arcv_min, arcv_min], color=INK, lw=1.3, ls=(0, (5, 3)), zorder=5)
    ax.scatter(point[ARCL], point[ARCV], s=300, marker="*", color=INK, edgecolor="#fcfcfb",
               lw=1.2, zorder=8)
    box = dict(boxstyle="round,pad=0.3", fc="#fcfcfb", ec="#c3c2b7", alpha=0.92)
    ax.annotate(f"Sharjah, 15 June 2026\nARCV {point[ARCV]:.2f}°, ARCL {point[ARCL]:.2f}°\n"
                rf"$P_\mathrm{{det}}$ = {point['pdet']:.2f}", (point[ARCL], point[ARCV]), xytext=(9.6, 0.3),
                fontsize=8, color=INK, bbox=box, zorder=9,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    low = sight_low.iloc[0]  # the lowest-ARCL HilalPy sighting
    how = low["method"].replace("aid (", "aided, ").rstrip(")")
    ax.annotate(f"Lowest ARCL: {low[ARCL]:.2f}°\n{low['record']}, {low['date'][:4]}, {how}\n"
                rf"$P_\mathrm{{det}}$ = {low['pdet']:.2f}", (low[ARCL], low[ARCV]),
                xytext=(0.3, 8.6), fontsize=7.5, color=INK, zorder=9,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))
    b0, (b1, b2) = reg.intercept_, reg.coef_
    ax.text(19.7, -4.6, f"Background lag (min) = {b0:.2f} + {b1:.2f} ARCV + {b2:.2f} ARCL",
            fontsize=7.5, color=INK2, ha="right", va="bottom", bbox=box)
    ax.set_xlim(0, 20)
    ax.set_ylim(-5, 16)
    ax.set_xlabel("ARCL (deg)", color=INK)
    ax.set_ylabel("ARCV (deg)", color=INK)
    ax.set_title(r"(a) Final-model $P_\mathrm{det}$ and the lowest sightings", fontsize=11, color=INK, loc="left")
    cb = fig.colorbar(cf, ax=ax, pad=0.015, fraction=0.045)
    cb.set_label(r"Final-model $P_\mathrm{det}$ (lag from the regression line)", fontsize=8.5, color=INK2)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    n0, n1, n2 = (int((cls == c).sum()) for c in (0, 1, 2))
    handles = [
        Line2D([], [], marker="o", ls="", color=C_NOT, ms=4, label=f"ICOP not seen ({n0:,})"),
        Line2D([], [], marker="o", ls="", color=C_AID, ms=4, label=f"ICOP aided only ({n1:,})"),
        Line2D([], [], marker="o", ls="", color=C_EYE, ms=4, label=f"ICOP naked eye ({n2:,})"),
        Line2D([], [], marker="^", ls="", color=C_AID, mec=INK, ms=7,
               label=f"HilalPy {len(sight_low)} lowest-ARCL sightings: aided"),
        Line2D([], [], marker="^", ls="", color=C_EYE, mec=INK, ms=7,
               label=f"HilalPy {len(sight_low)} lowest-ARCL sightings: naked eye"),
        Line2D([], [], marker="*", ls="", color=INK, ms=11, label="Sharjah, 15 June 2026"),
        Line2D([], [], color=INK, lw=1.3, ls=(0, (5, 3)), label="MABIMS 2021 limits (visible above and right)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3,
              fontsize=7.5, frameon=False, handletextpad=0.3, columnspacing=1.2)

    # (b) cumulative distributions of the out-of-fold P_det.
    pdet = model_df["oof"].to_numpy()
    for c, colour, label in [(2, C_EYE, "Seen by naked eye"), (1, C_AID, "Seen with aid only"),
                             (0, C_NOT, "Not seen")]:
        v = np.sort(pdet[cls == c])
        bx.step(np.r_[0, v, 1], np.r_[0, np.arange(1, len(v) + 1) / len(v), 1], where="post",
                color=colour, lw=2, label=f"{label} ({len(v):,})")
    seen = pdet[cls > 0]
    v = np.sort(seen)
    bx.step(np.r_[0, v, 1], np.r_[0, np.arange(1, len(v) + 1) / len(v), 1], where="post",
            color=INK, lw=1.2, ls=(0, (1, 1.5)), label=f"All sightings ({len(v):,})")
    # Cut-offs of Eq. 11, with the share of all sightings below each.
    for t, dy in ((PDET_LOW, -0.05), (PDET_HIGH, -0.055)):
        bx.axvline(t, color=INK2, lw=1, ls=(0, (5, 3)))
        share = np.mean(seen < t)
        bx.scatter([t], [share], s=22, color=INK, zorder=5)
        bx.text(t + 0.012, share + dy, f"{100 * share:.1f} %", fontsize=8, color=INK, va="center")
        bx.text(t + 0.008, 0.985, rf"$P_\mathrm{{det}}$ = {t:.2f}", fontsize=7.5, color=INK2, va="top")
    bx.axvline(point["pdet"], color=INK, lw=1)
    bx.text(point["pdet"] + 0.01, 0.985, "Sharjah, 15 June 2026", rotation=90, fontsize=8,
            color=INK, ha="left", va="top")
    # Reports with an out-of-fold P_det close to the Sharjah value.
    near = (pdet >= 0.15) & (pdet < 0.25)
    bx.axvspan(0.15, 0.25, color=GRID, alpha=0.6, lw=0, zorder=0)
    bx.text(0.015, 0.975,
            rf"$P_\mathrm{{det}}$ 0.15–0.25:" f"\n{int(near.sum()):,} reports,\n"
            f"{100 * np.mean(cls[near] > 0):.1f} % seen,\n"
            f"{100 * np.mean(cls[near] == 2):.1f} % naked eye",
            fontsize=7.5, color=INK, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fcfcfb", ec="#c3c2b7"))
    bx.grid(color=GRID, lw=0.6)
    bx.set_axisbelow(True)
    bx.set_xlim(0, 1)
    bx.set_ylim(0, 1)
    bx.set_xlabel(r"Out-of-fold $P_\mathrm{det}$ (5-fold, grouped by lunation)", color=INK)
    bx.set_ylabel("Cumulative share of ICOP reports", color=INK)
    bx.set_title(r"(b) What a $P_\mathrm{det}$ near 0.20 means", fontsize=11, color=INK, loc="left")
    handles, _ = bx.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="o", ls="", color=INK, ms=4,
                          label="Share of all sightings below a cut-off"))
    bx.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2,
              fontsize=7.5, frameon=False, handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hilalpy", required=True,
                        help="HilalPy 'Final 30.6.2023.csv' (Mendeley Data, DOI 10.17632/rs994bxgc4.1)")
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
    stage1, stage2 = t1.load_saved_models(args.models_dir)
    _, model_df = hc.load_icop(args.icop)
    oof = pd.read_csv(args.oof, low_memory=False)
    if not (len(oof) == len(model_df) and (oof[GROUP].to_numpy() == model_df[GROUP].to_numpy()).all()
            and np.allclose(oof[FEATURES], model_df[FEATURES], rtol=0, atol=1e-12)):
        raise ValueError(f"{args.oof} does not hold the modelling rows in the same order")
    model_df["oof"] = oof["p_det"].to_numpy()
    model_df["pdet"] = stage1.predict_proba(model_df[FEATURES])[:, 1]
    cls = model_df["cls"].to_numpy()
    seen = cls > 0

    # ------------------------------------------------ 1. the Sharjah evening
    name, lat, lon = SITE
    g = sunset_geometry(lat, lon, EVENING)
    pt = pd.DataFrame([g])
    p_det, p_eye = (s.predict_proba(pt[FEATURES])[0, 1] for s in (stage1, stage2))
    reg = LinearRegression().fit(model_df[[ARCV, ARCL]], model_df[LAG])
    resid = model_df[LAG] - reg.predict(model_df[[ARCV, ARCL]])
    lag_line = float(reg.predict(pt[[ARCV, ARCL]])[0])
    p_det_line = stage1.predict_proba(pt[[ARCV, ARCL]].assign(**{LAG: lag_line})[FEATURES])[0, 1]
    print(f"[1] {name} ({lat} N, {lon} E), evening of {EVENING.replace('/', '-')}")
    print(f"Sunset {g['sunset_utc']:%Y-%m-%d %H:%M:%S} UTC; Sun altitude {g['sun_alt']:.4f} deg; "
          f"Moon altitude {g['moon_alt']:.4f} deg (airless topocentric)")
    print(f"ARCV {g[ARCV]:.4f} deg, ARCL {g[ARCL]:.4f} deg, DAZ {g[DAZ]:.4f} deg, lag {g[LAG]:.4f} min, "
          f"W {g[WIDTH]:.4f} arcmin, Moon age {g['age_h']:.4f} h")
    print(f"Final model: P_det = {p_det:.4f}, P_eye = {p_eye:.4f}; band {band(p_det, p_eye)}")
    print(f"Lag line fitted on the 4,470 ICOP rows (used for the figure background): lag = "
          f"{reg.intercept_:.4f} + {reg.coef_[0]:.4f} ARCV + {reg.coef_[1]:.4f} ARCL (min); "
          f"R^2 = {reg.score(model_df[[ARCV, ARCL]], model_df[LAG]):.4f}; "
          f"residual SD = {resid.std(ddof=3):.4f} min")
    print(f"At this point the line gives lag {lag_line:.4f} min and P_det = {p_det_line:.4f} "
          f"(the background value of the figure)")
    verdict = {0: "not seen", 1: "aided", 2: "naked eye"}
    table1 = t1.classical_predictions(pt)
    print("Other criteria at the same sunset geometry (Table 1 implementations, at sunset):")
    for crit in ["Maunder", "Indian", "Fotheringham", "Alrefay-Unaided", "Yallop", "Qureshi",
                 "Alrefay-TwoBoundary", "Odeh"]:
        print(f"  {crit:22s} {verdict[int(table1[crit][0])]}")
    alt, arcl, age = g["moon_alt"], g[ARCL], g["age_h"]
    for crit, ok in [("MABIMS 1995 (local)", (alt >= 2 and arcl >= 3) or age >= 8),
                     ("MABIMS 2021 (local)", alt >= 3 and arcl >= 6.4),
                     ("Istanbul 2016 (local)", alt >= 5 and arcl >= 8)]:
        print(f"  {crit:22s} {'visible' if ok else 'not visible'}")

    # ------------------------------------------------ 2. similar ICOP geometry
    cv, cl, clag = CENTRE
    d_v, d_l = (model_df[ARCV] - cv).abs(), (model_df[ARCL] - cl).abs()
    d_lag = (model_df[LAG] - clag).abs()
    windows = [
        ("ARCV and ARCL within +/-1.0 deg of 7.5", (d_v <= 1.0) & (d_l <= 1.0)),
        ("ARCV and ARCL within +/-1.5 deg of 7.5", (d_v <= 1.5) & (d_l <= 1.5)),
        ("Lag within +/-10 min of 40", d_lag <= 10),
        ("+/-1.5 deg box and lag +/-10 min", (d_v <= 1.5) & (d_l <= 1.5) & (d_lag <= 10)),
    ]
    print(f"\n[2] ICOP reports with geometry like the {name} evening (modelling subset, 4,470 rows)")
    print(f"{'Window':40s} {'Reports':>7s} {'Lunat.':>6s} {'Seen (share) [95% CI]':>32s} "
          f"{'Naked eye (share)':>18s} {'Aided only (share)':>19s} {'OOF P_det':>9s}")
    for label, m in windows:
        m = m.to_numpy()
        n, k = int(m.sum()), int(seen[m].sum())
        lo, hi = lunation_interval(seen[m], model_df.loc[m, GROUP])
        n2, n1 = int((cls[m] == 2).sum()), int((cls[m] == 1).sum())
        print(f"{label:40s} {n:7,d} {model_df.loc[m, GROUP].nunique():6d} "
              f"{f'{k} ({k / n:.4f}) [{lo:.4f}, {hi:.4f}]':>32s} {f'{n2} ({n2 / n:.4f})':>18s} "
              f"{f'{n1} ({n1 / n:.4f})':>19s} {model_df.loc[m, 'oof'].mean():9.4f}")

    # ------------------------------------------------ 3. lowest sightings
    icop = model_df[seen].copy()
    icop["lat"], icop["lon"] = icop["latitude"], icop["longitude"]
    icop["record"] = icop[GROUP].str.extract(r"/icop/([^/.]+)\.html")[0]
    icop["method"] = icop.apply(aid_label, axis=1, flags=[("seen_binocular", "binoculars"),
                                                         ("seen_telescope", "telescope"),
                                                         ("seen_camera", "camera")])
    h = hc.load_hilalpy(args.hilalpy)
    ev, _ = hc.evening_filters(h)
    hs = ev[ev["cls"] > 0].copy()
    hs["lat"], hs["lon"] = hs["Lat"], hs["Long"]
    hs["record"] = hs["Ref No"].astype(str)
    hs["method"] = hs.apply(aid_label, axis=1, flags=[("B", "binoculars"), ("T", "telescope"),
                                                     ("CCD", "CCD")])
    hs["pdet"] = stage1.predict_proba(hs[FEATURES])[:, 1]
    keys = ["date", ARCV, ARCL, LAG, "cls"]
    print(f"\n[3] Lowest sightings. ICOP: modelling subset ({int(seen.sum()):,} sightings; P_det = "
          f"final model, OOF = out-of-fold). HilalPy: evening records of all sources after filters "
          f"2-5 ({len(ev):,} rows, {len(hs):,} sightings; ARCV = ArcV, ARCL = ArcL, lag = LT). "
          f"n = identical records shown once.")
    print_list(lowest(icop, ARCL, N_LIST, keys, "record"), "ICOP, 10 lowest ARCL", "record",
               extra=("oof", "OOF"))
    print_list(lowest(icop, ARCV, N_LIST, keys, "record"), "ICOP, 10 lowest ARCV", "record",
               extra=("oof", "OOF"))
    print_list(lowest(hs, ARCL, N_LIST, keys, "record"), "HilalPy, 10 lowest ARCL", "record")
    print_list(lowest(hs, ARCV, N_LIST, keys, "record"), "HilalPy, 10 lowest ARCV", "record")
    print_list(lowest(icop[icop["cls"] == 2], ARCL, 5, keys, "record"),
               "ICOP, 5 lowest-ARCL naked-eye sightings", "record", extra=("oof", "OOF"))
    print_list(lowest(hs[hs["cls"] == 2], ARCL, 5, keys, "record"),
               "HilalPy, 5 lowest-ARCL naked-eye sightings", "record")
    sight_low = lowest(hs, ARCL, N_PLOT, keys, "record")
    print(f"\nHilalPy {N_PLOT} lowest-ARCL sightings (plotted): ARCL {sight_low[ARCL].min():.2f} to "
          f"{sight_low[ARCL].max():.2f} deg; aided {int((sight_low['cls'] == 1).sum())}, naked eye "
          f"{int((sight_low['cls'] == 2).sum())}; final-model P_det {sight_low['pdet'].min():.4f} to "
          f"{sight_low['pdet'].max():.4f}; sources: "
          + ", ".join(f"{k} {v}" for k, v in sight_low["src"].value_counts().items()))

    # ------------------------------------------------ 4. what P_det = 0.20 means
    oofp = model_df["oof"].to_numpy()
    print("\n[4] Out-of-fold P_det of the ICOP sightings (4,470 modelling rows)")
    print(f"{'Group':22s} {'n':>6s} {'P_det < 0.30':>18s} {'P_det < 0.50':>18s}")
    for label, m in [("All sightings", seen), ("Naked eye", cls == 2), ("Aided only", cls == 1),
                     ("Not seen", cls == 0)]:
        a, b = int((oofp[m] < PDET_LOW).sum()), int((oofp[m] < PDET_HIGH).sum())
        n = int(m.sum())
        print(f"{label:22s} {n:6,d} {f'{a} ({a / n:.4f})':>18s} {f'{b} ({b / n:.4f})':>18s}")
    near = (oofp >= 0.15) & (oofp < 0.25)
    lo, hi = lunation_interval(seen[near], model_df.loc[near, GROUP])
    print(f"Reports with out-of-fold P_det in [0.15, 0.25): {int(near.sum()):,} "
          f"({model_df.loc[near, GROUP].nunique()} lunations); mean P_det {oofp[near].mean():.4f}; "
          f"seen {int(seen[near].sum())} ({seen[near].mean():.4f}) [95% CI {lo:.4f}, {hi:.4f}]; "
          f"naked eye {int((cls[near] == 2).sum())} ({np.mean(cls[near] == 2):.4f})")

    # MABIMS 2021 local screen, as in calendar_criteria_benchmark.py (4,460 complete rows).
    fields = ["moon_alt_deg", ARCL, "moon_age_hours_at_sunset"]
    comp = model_df[fields].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).to_numpy()
    alt = pd.to_numeric(model_df["moon_alt_deg"], errors="coerce").to_numpy()
    mabims = comp & (alt >= 3) & (model_df[ARCL].to_numpy() >= 6.4)
    if (int((mabims & ~seen).sum()), int((comp & ~mabims & seen).sum())) != (1429, 39):
        raise ValueError("MABIMS 2021 counts differ from calendar_criteria_benchmark.py")
    print(f"\nMABIMS 2021 local screen (Moon altitude >= 3 deg and ARCL >= 6.4 deg at sunset) and "
          f"the out-of-fold Sharjah decisions, on the same {int(comp.sum()):,} complete rows")
    print(f"{'Rule':40s} {'Called visible':>14s} {'Not seen (share)':>18s} "
          f"{'Sightings called not visible':>30s}")
    rules = [("MABIMS 2021 'visible'", mabims),
             ("Sharjah P_det >= 0.50 (detectable)", comp & (oofp >= PDET_HIGH)),
             ("Sharjah P_det >= 0.30 (marginal or above)", comp & (oofp >= PDET_LOW))]
    for label, m in rules:
        n, k = int(m.sum()), int((m & ~seen).sum())
        miss = int((comp & ~m & seen).sum())
        print(f"{label:40s} {n:14,d} {f'{k} ({k / n:.4f})':>18s} "
              f"{f'{miss}/{int((comp & seen).sum()):,} ({miss / (comp & seen).sum():.4f})':>30s}")
    print("Reports that MABIMS 2021 calls visible, by out-of-fold Sharjah band")
    print(f"{'Sharjah band':30s} {'Reports':>7s} {'Lunat.':>6s} {'Seen (share) [95% CI]':>32s} "
          f"{'Naked eye (share)':>18s}")
    for label, m in [("P_det < 0.30 (not detectable)", mabims & (oofp < PDET_LOW)),
                     ("0.30 <= P_det < 0.50", mabims & (oofp >= PDET_LOW) & (oofp < PDET_HIGH)),
                     ("P_det >= 0.50", mabims & (oofp >= PDET_HIGH))]:
        n, k = int(m.sum()), int(seen[m].sum())
        lo, hi = lunation_interval(seen[m], model_df.loc[m, GROUP])
        n2 = int((cls[m] == 2).sum())
        print(f"{label:30s} {n:7,d} {model_df.loc[m, GROUP].nunique():6d} "
              f"{f'{k} ({k / n:.4f}) [{lo:.4f}, {hi:.4f}]':>32s} {f'{n2} ({n2 / n:.4f})':>18s}")

    # ------------------------------------------------ 5. figure
    point = {ARCV: g[ARCV], ARCL: g[ARCL], "pdet": p_det}
    plot(model_df, sight_low, point, reg, stage1, args.output)
    print(f"\n[5] Figure saved: {args.output}")


if __name__ == "__main__":
    main()
