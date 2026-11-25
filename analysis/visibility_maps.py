#!/usr/bin/env python3
"""Crescent visibility maps at local sunset (second review round, comment 6).

The script draws maps of the five operational bands of the final two-stage
model, by default for the evenings of 17 May, 15 June and 15 July 2026.

Method. Every map point is evaluated at its own local sunset on the evening of
the given date, in the conventions of the ICOP feature table
(data/icop/crescent_features.csv, pipeline scripts 03 and 03b):
  - sunset: PyEphem next_setting of the Sun's centre (use_center=True) at
    altitude -0:50, observer at elevation 0 and pressure 0 (airless), searched
    from local mean noon, so that the evening of that local date is found;
  - ARCV = Moon altitude - Sun altitude, ARCL = Sun-Moon separation and
    DAZ = |Moon azimuth - Sun azimuth|, airless topocentric, at that sunset;
  - lag = moonset - sunset (min), the moonset from PyEphem defaults (upper limb,
    standard refraction) and rounded down to the whole minute, as in
    03b_correct_moon_lag.py: if the Moon is already down at sunset, the moonset
    before sunset is used and the lag is negative; otherwise the first moonset
    after sunset (searched from 10 min before sunset, because PyEphem can miss
    a moonset within seconds of the start time);
  - Moon age = time since the nearest new moon (negative before it).
Section [0] recomputes 20 archive rows (16 at random, 4 with the Moon already
down at sunset; default_rng(42)) and stops if ARCV or ARCL differ by more than
0.01 deg or the lag by more than 2 min.
The final model (both stages fitted on all 4,470 modelling rows) gives P_det and
P_eye, and the five operational bands of Eq. 11 are
  E  P_det < 0.30                       not detectable
  D  0.30 <= P_det < 0.50               marginal detectability
  C  P_det >= 0.50 and P_eye < 0.30     aid-assisted likely
  B  P_det >= 0.50, 0.30 <= P_eye < 0.60  borderline naked-eye
  A  P_det >= 0.50 and P_eye >= 0.60    naked-eye likely
Section [1] gives the Sharjah (25.3463 N, 55.4209 E) geometry and probabilities
for 16 and 17 May, 15 June and 15 July 2026, and [2] the new moons in UT.
Maps [3]: 1 deg grid, longitude -100 to 90, latitude -45 to 60 (points at whole
degrees), coastlines and borders from Natural Earth (cartopy), 300 dpi; the
grid points per class and those outside the training range are printed.
"""
import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import ephem
import joblib
import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ARCV = "relative_altitude_deg"
ARCL = "elongation_deg"
LAG = "moon_lag_minutes"
DAZ = "relative_azimuth_deg"
FEATURES = [ARCV, ARCL, LAG]
HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # the repository root
DEFAULT_ICOP = REPO / "data" / "icop" / "crescent_features.csv"
DEFAULT_MODELS = REPO / "models"

SITE = ("Sharjah", 25.3463, 55.4209, "Asia/Dubai")
MAP_DATES = ["2026-05-17", "2026-06-15", "2026-07-15"]
TABLE_DATES = ["2026-05-16", "2026-05-17", "2026-06-15", "2026-07-15"]
LONS = np.arange(-100, 91)
LATS = np.arange(-45, 61)
SUN_HORIZON = "-0:50"  # Sun's centre at sunset, as sun_alt_deg (-0.8333) in the ICOP features
PDET_LOW, PDET_HIGH, PEYE_LOW, PEYE_HIGH = 0.30, 0.50, 0.30, 0.60
N_RANDOM, N_NEGATIVE, CHECK_SEED = 16, 4, 42

# Classes of Eq. 11, best first. One-hue ordinal ramp (light = less visible),
# so the order survives colour-vision deficiency and greyscale printing. The
# steps are evenly spaced in lightness (OKLab L 0.34, 0.48, 0.62, 0.76, 0.91),
# so that neighbouring classes stay distinct when the figure is small.
CLASSES = [("A", "Naked-eye likely", "#0d366b"),
           ("B", "Borderline naked-eye", "#1c5cab"),
           ("C", "Aid-assisted likely", "#3987e5"),
           ("D", "Marginal detectability", "#86b6ef"),
           ("E", "Not detectable", "#cde2fb")]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"


# ---------------------------------------------------------------- geometry
def _observer(lat, lon):
    obs = ephem.Observer()
    obs.lat, obs.lon, obs.elevation = str(lat), str(lon), 0
    return obs


def nearest_new_moon(when):
    """The new moon (geocentric conjunction in longitude) closest to 'when'."""
    prev, nxt = ephem.previous_new_moon(when), ephem.next_new_moon(when)
    return prev if when - prev <= nxt - when else nxt


def sunset_geometry(lat, lon, day):
    """Geometry at local sunset on the evening of 'day' (YYYY-MM-DD)."""
    obs = _observer(lat, lon)
    obs.pressure = 0  # airless: geometric topocentric positions
    obs.horizon = SUN_HORIZON
    obs.date = ephem.Date(day.replace("-", "/")) + (12 - lon / 15) / 24  # local mean noon
    sunset = obs.next_setting(ephem.Sun(), use_center=True)
    obs.date = sunset
    sun, moon = ephem.Sun(obs), ephem.Moon(obs)

    # Moonset as in 03b_correct_moon_lag.py (PyEphem defaults: upper limb, refraction).
    ms_obs = _observer(lat, lon)
    ms_obs.date = sunset
    m = ephem.Moon()
    try:
        set_before, rise_before = ms_obs.previous_setting(m), ms_obs.previous_rising(m)
        # The Moon is down at sunset if it last set after it last rose.
        if set_before > rise_before:
            moonset = set_before
        else:
            # PyEphem can miss a moonset within seconds of the start time and return the next
            # day's (lag 1,513 min at 1 of the 60,738 points of the three maps), so search
            # from 10 min before sunset: the first moonset after that is the one at sunset.
            ms_obs.date = sunset - 10 * ephem.minute
            moonset = ms_obs.next_setting(m)
        moonset_min = ephem.Date(moonset).datetime().replace(second=0, microsecond=0)
        lag = (moonset_min - ephem.Date(sunset).datetime()).total_seconds() / 60
    except (ephem.AlwaysUpError, ephem.NeverUpError):
        lag = np.nan
    return {
        "sunset": sunset,
        ARCV: math.degrees(moon.alt - sun.alt),
        ARCL: math.degrees(ephem.separation(sun, moon)),
        DAZ: abs(math.degrees(moon.az - sun.az)),
        LAG: lag,
        "moon_alt": math.degrees(moon.alt),
        "age_h": (sunset - nearest_new_moon(sunset)) * 24,
    }


# ---------------------------------------------------------------- model
def load_models(models_dir):
    return (joblib.load(Path(models_dir) / "stage1_detectability_model.joblib"),
            joblib.load(Path(models_dir) / "stage2_naked_eye_model.joblib"))


def probabilities(stage1, stage2, df):
    X = df[FEATURES]
    return stage1.predict_proba(X)[:, 1], stage2.predict_proba(X)[:, 1]


def eq16_class(p_det, p_eye):
    """Index into CLASSES (0 = A ... 4 = E) for each point."""
    p_det, p_eye = np.asarray(p_det), np.asarray(p_eye)
    return np.select(
        [p_det < PDET_LOW, p_det < PDET_HIGH, p_eye < PEYE_LOW, p_eye < PEYE_HIGH],
        [4, 3, 2, 1], default=0)


def compute_grid(day, stage1, stage2):
    """Sunset geometry, probabilities and classes on the map grid."""
    rows = [sunset_geometry(lat, lon, day) for lat in LATS for lon in LONS]
    g = pd.DataFrame(rows)
    shape = (len(LATS), len(LONS))
    ok = g[FEATURES].notna().all(axis=1).to_numpy()
    p_det, p_eye = np.full(len(g), np.nan), np.full(len(g), np.nan)
    p_det[ok], p_eye[ok] = probabilities(stage1, stage2, g[ok])
    cls = np.where(ok, eq16_class(np.nan_to_num(p_det), np.nan_to_num(p_eye)), -1)
    return {"day": day, "geom": g, "p_det": p_det.reshape(shape),
            "p_eye": p_eye.reshape(shape), "cls": cls.reshape(shape), "ok": ok.reshape(shape)}


def site_point(day, stage1, stage2, site=SITE):
    name, lat, lon, tz = site
    g = sunset_geometry(lat, lon, day)
    p_det, p_eye = probabilities(stage1, stage2, pd.DataFrame([g]))
    g.update(p_det=float(p_det[0]), p_eye=float(p_eye[0]),
             cls=int(eq16_class(p_det, p_eye)[0]), name=name, lat=lat, lon=lon, tz=tz)
    return g


# ---------------------------------------------------------------- map
def long_date(day):
    d = datetime.strptime(day, "%Y-%m-%d")
    return f"{d.day} {d:%B %Y}"


def ut_text(when):
    """An ephem date as '16 May 2026, 20:01 UT' (nearest minute)."""
    d = ephem.Date(when + 30 * ephem.second).datetime()
    return f"{d.day} {d:%B %Y, %H:%M} UT"


def draw_map(ax, grid, point, compact=False):
    """Class map on a cartopy PlateCarree axis; returns the legend handles."""
    cmap = ListedColormap([c for _, _, c in CLASSES])
    norm = BoundaryNorm(np.arange(-0.5, len(CLASSES)), cmap.N)
    cls = np.ma.masked_less(grid["cls"], 0)
    ax.pcolormesh(LONS, LATS, cls, cmap=cmap, norm=norm, shading="nearest",
                  transform=ccrs.PlateCarree(), zorder=0, rasterized=True)
    halo = [pe.withStroke(linewidth=1.4 if not compact else 1.0, foreground="#fcfcfb")]
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.55 if not compact else 0.35,
                   edgecolor=INK, zorder=2, path_effects=halo)
    # Borders get a thin light edge too, so that they stay visible on the dark classes.
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3 if not compact else 0.2,
                   edgecolor=INK2, zorder=2,
                   path_effects=[pe.withStroke(linewidth=0.8 if not compact else 0.5,
                                               foreground="#fcfcfb", alpha=0.6)])
    ax.set_extent([LONS[0] - 0.5, LONS[-1] + 0.5, LATS[0] - 0.5, LATS[-1] + 0.5],
                  crs=ccrs.PlateCarree())
    if not compact:
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#c3c2b7", alpha=0.8,
                          xlocs=range(-90, 91, 30), ylocs=range(-40, 61, 20))
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 8, "color": INK2}

    ring = [pe.withStroke(linewidth=2.5, foreground="#fcfcfb")]
    ax.plot(point["lon"], point["lat"], marker="*", ms=13 if not compact else 9, color="#eda100",
            mec=INK, mew=0.8, transform=ccrs.PlateCarree(), zorder=5)
    letter = CLASSES[point["cls"]][0]
    if compact:
        text = f"{point['name']}: band {letter}\nP_det {point['p_det']:.2f}, P_eye {point['p_eye']:.2f}"
        xytext = (point["lon"] - 2, point["lat"] - 40)
    else:
        text = (f"{point['name']}\nP_det = {point['p_det']:.2f}\nP_eye = {point['p_eye']:.2f}\n"
                f"Band {letter}")
        xytext = (point["lon"] + 14, point["lat"] - 33)
    box = dict(boxstyle="round,pad=0.3", fc="#fcfcfb", ec="#c3c2b7", alpha=0.95)
    # The axis is PlateCarree, so data coordinates are longitude and latitude.
    ax.annotate(text, xy=(point["lon"], point["lat"]), xytext=xytext,
                fontsize=8.5 if not compact else 6.5, color=INK, bbox=box, zorder=6,
                ha="left" if not compact else "center",
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8, path_effects=ring))
    return [Patch(facecolor=c, edgecolor="#c3c2b7", lw=0.5, label=f"{k}: {label}")
            for k, label, c in CLASSES]


def plot_map(grid, point, conj, out):
    fig = plt.figure(figsize=(10, 6.6))
    ax = fig.add_axes([0.06, 0.2, 0.9, 0.68], projection=ccrs.PlateCarree())
    handles = draw_map(ax, grid, point)
    fig.text(0.06, 0.955, f"Sharjah criterion at local sunset, {long_date(grid['day'])}",
             fontsize=14, color=INK, weight="bold", va="top")
    fig.text(0.06, 0.915,
             f"Final two-stage model (operational bands), each 1° grid point at its own sunset; "
             f"new moon {ut_text(conj)}",
             fontsize=9.5, color=INK2, va="top")
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.51, 0.06), ncol=5,
               fontsize=9, frameon=False, handlelength=1.6, handleheight=1.1, columnspacing=1.4)
    fig.text(0.51, 0.035,
             "A–C: P_det ≥ 0.50 with P_eye ≥ 0.60 (A), 0.30–0.60 (B) or < 0.30 (C);   "
             "D: 0.30 ≤ P_det < 0.50;   E: P_det < 0.30",
             fontsize=8, color=INK2, ha="center")
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- checks and tables
def archive_check(icop_path):
    """Recompute 20 archive rows and compare with the stored sunset features."""
    a = pd.read_csv(icop_path, low_memory=False)
    a = a[(a["section_type"] == "first_crescent") & (a["obs_relative_to_sunset"] != "before_sunset")
          & a["vis_class_empirical"].notna()].reset_index(drop=True)
    if len(a) != 4470:
        raise ValueError("ICOP modelling subset has changed; expected 4,470 rows")
    rng = np.random.default_rng(CHECK_SEED)
    neg = rng.choice(np.flatnonzero(a[LAG] < 0), N_NEGATIVE, replace=False)
    rest = rng.choice(np.setdiff1d(np.arange(len(a)), neg), N_RANDOM, replace=False)
    rows = []
    for i in np.r_[rest, neg]:
        r = a.iloc[i]
        sunset_local = datetime.fromisoformat(r["sunset_local"])
        g = sunset_geometry(r["latitude"], r["longitude"], sunset_local.strftime("%Y-%m-%d"))
        stored = sunset_local.astimezone(timezone.utc).replace(tzinfo=None)
        rows.append({
            "row": int(i), "date": r["sunset_local"][:10],
            "d_sunset_s": (ephem.Date(g["sunset"]).datetime() - stored).total_seconds(),
            "ARCV": r[ARCV], "d_ARCV": g[ARCV] - r[ARCV],
            "d_ARCL": g[ARCL] - r[ARCL], "d_DAZ": g[DAZ] - r[DAZ],
            "lag": r[LAG], "d_lag": g[LAG] - r[LAG],
            "d_moon_alt": g["moon_alt"] - r["moon_alt_deg"],
            "d_age_h": g["age_h"] - r["moon_age_hours_at_sunset"],
        })
    t = pd.DataFrame(rows)
    print(f"[0] Archive check: {N_RANDOM} random modelling rows and {N_NEGATIVE} with the Moon "
          f"down at sunset (default_rng({CHECK_SEED})); differences = this script minus the archive")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    diffs = ["d_sunset_s", "d_ARCV", "d_ARCL", "d_DAZ", "d_lag", "d_moon_alt", "d_age_h"]
    print("Maximum absolute difference: " + ", ".join(
        f"{c[2:]} {t[c].abs().max():.4f}" for c in diffs)
        + " (sunset in s, lag in min, age in h, angles in deg)")
    if t[["d_ARCV", "d_ARCL"]].abs().max().max() > 0.01 or t["d_lag"].abs().max() > 2:
        raise ValueError("The recomputed geometry does not match the archive")
    return a


def print_site_table(points):
    name, lat, lon, tz = SITE
    print(f"\n[1] {name} ({lat} N, {lon} E), local sunset ({tz}); airless topocentric "
          f"angles in deg, lag in min, Moon age in h from the nearest new moon")
    print(f"{'Evening':10s} {'Sunset (local)':>15s} {'ARCV':>8s} {'ARCL':>8s} {'DAZ':>8s} "
          f"{'Lag':>9s} {'Age':>8s} {'Moon alt':>8s} {'P_det':>7s} {'P_eye':>7s}  Class")
    for p in points:
        local = ephem.Date(p["sunset"]).datetime().replace(tzinfo=timezone.utc).astimezone(ZoneInfo(tz))
        k, label, _ = CLASSES[p["cls"]]
        print(f"{p['day']:10s} {local:%H:%M:%S %Z}".ljust(26)
              + f" {p[ARCV]:8.4f} {p[ARCL]:8.4f} {p[DAZ]:8.4f} {p[LAG]:9.4f} {p['age_h']:8.4f} "
              f"{p['moon_alt']:8.4f} {p['p_det']:7.4f} {p['p_eye']:7.4f}  {k}: {label}")


def print_grid_summary(grid, train):
    g = grid["geom"]
    cls = grid["cls"].ravel()
    n = len(cls)
    counts = ", ".join(f"{k} {int((cls == i).sum()):,} ({(cls == i).mean():.1%})"
                       for i, (k, _, _) in enumerate(CLASSES))
    print(f"{grid['day']}: {n:,} grid points; by class {counts}; no moonset found "
          f"{int((cls < 0).sum())}")
    out = []
    for col, name in [(ARCV, "ARCV"), (ARCL, "ARCL"), (LAG, "lag")]:
        lo, hi = train[col].min(), train[col].max()
        k = int(((g[col] < lo) | (g[col] > hi)).sum())
        out.append(f"{name} outside [{lo:.2f}, {hi:.2f}] {k:,} ({k / n:.4f})")
    print("  points outside the training range: " + "; ".join(out))


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--icop", default=str(DEFAULT_ICOP),
                        help="Path to data/icop/crescent_features.csv")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS),
                        help="Folder with the final Stage 1 and Stage 2 joblib models")
    parser.add_argument("--outdir", default=str(REPO / "figures"),
                        help="Folder for map_YYYYMMDD.png (300 dpi)")
    parser.add_argument("--dates", nargs="+", default=MAP_DATES,
                        help="Evenings to map (YYYY-MM-DD)")
    args = parser.parse_args()

    train = archive_check(args.icop)
    stage1, stage2 = load_models(args.models_dir)

    points = []
    for day in TABLE_DATES:
        p = site_point(day, stage1, stage2)
        p["day"] = day
        points.append(p)
    print_site_table(points)

    print("\n[2] New moons (geocentric conjunction in ecliptic longitude, PyEphem), UT")
    for start in ("2026/5/1", "2026/6/1", "2026/7/1"):
        conj = ephem.next_new_moon(start)
        print(f"  {ephem.Date(conj).datetime():%Y-%m-%d %H:%M:%S} ({ut_text(conj)})")

    print("\n[3] Maps")
    for day in args.dates:
        grid = compute_grid(day, stage1, stage2)
        point = site_point(day, stage1, stage2)
        conj = nearest_new_moon(point["sunset"])
        out = Path(args.outdir) / f"map_{day.replace('-', '')}.png"
        plot_map(grid, point, conj, out)
        print_grid_summary(grid, train)
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
