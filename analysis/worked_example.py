#!/usr/bin/env python3
"""Record counts, final model parameters and a worked example (reviewer comment 7).

Reviewer: "please give the number of records in every stage and split, the
final model coefficients and scaling values, the software versions, and a
frozen copy of the data. A worked numerical example would be very useful here
too."

Inputs: data/icop and models/ (final models from
pipeline/04_train_two_stage_crescent_criterion.py). Sections:
 [0] software versions (the same pins are in requirements.txt) and the
     scikit-learn version stored in the saved models;
 [1] records in every file of the pipeline and at every filter of the modelling
     subset, with SHA-256 checksums of the files read here;
 [2] records, lunations and classes in every fold of the main grouped split
     (StratifiedGroupKFold, 5 folds, shuffle=True, random_state=42, stratified
     on the three-class label, groups = page_url). Stage 1 uses all rows of a
     fold and Stage 2 its detected rows. The fold models are refitted and must
     reproduce data/icop/two_stage_oof_predictions.csv;
 [3] final model (both stages fitted on all 4,470 records): imputation medians,
     standardisation means and SDs, coefficients and intercepts, also on the
     original scale of the predictors. Refitting must reproduce the saved models;
 [4] worked example for Sharjah (25.3463 N, 55.4209 E) at local sunset on
     17 May and 15 July 2026, from the geometry to the operational class.

Geometry follows the ICOP feature table (scripts 03 and 03b), with PyEphem:
  - sunset: Sun's centre (use_center=True) at altitude -0:50, observer at
    elevation 0 and pressure 0 (airless), searched from local mean noon;
  - ARCV = Moon altitude - Sun altitude and ARCL = ephem.separation(Sun, Moon),
    airless topocentric, at that sunset;
  - lag = moonset - sunset in minutes, with the moonset from PyEphem defaults
    (upper limb, standard refraction) rounded down to the whole minute. If the
    Moon is already down at sunset, the moonset before sunset is used (lag < 0).
Model: P = 1 / (1 + exp(-z)), z = b0 + sum_i b_i * x~_i and x~_i = (x_i - mean_i) / sd_i
(Eqs. for z and P in the Methods). Operational classes (Eq. 11):
  E  P_det < 0.30                         not detectable
  D  0.30 <= P_det < 0.50                 marginal detectability
  C  P_det >= 0.50 and P_eye < 0.30       aid-assisted likely
  B  P_det >= 0.50, 0.30 <= P_eye < 0.60  borderline naked-eye
  A  P_det >= 0.50 and P_eye >= 0.60      naked-eye likely
"""
import argparse
import hashlib
import importlib.metadata
import math
import platform
import warnings
from pathlib import Path
from zoneinfo import ZoneInfo

import ephem
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ARCV = "relative_altitude_deg"
ARCL = "elongation_deg"
LAG = "moon_lag_minutes"
FEATURES = [ARCV, ARCL, LAG]
NAMES = {ARCV: "ARCV (deg)", ARCL: "ARCL (deg)", LAG: "lag (min)"}
GROUP = "page_url"
CLASS = "vis_class_empirical"
HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # the repository root

PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "ephem",
            "matplotlib", "cartopy", "timezonefinder", "joblib", "threadpoolctl",
            "shapely", "pyproj", "pyshp"]  # the last three are installed with cartopy
PIPELINE_FILES = [
    ("01 scrape", "crescent_observations_raw.csv"),
    ("02 clean: kept", "crescent_observations_clean.csv"),
    ("02 clean: rejected", "crescent_observations_rejected.csv"),
    ("03 features (+03b lag)", "crescent_features.csv"),
    ("04 out-of-fold", "two_stage_oof_predictions.csv"),
]
MODEL_FILES = ["stage1_detectability_model.joblib", "stage2_naked_eye_model.joblib"]
N_FOLDS, SEED = 5, 42

SITE = ("Sharjah", 25.3463, 55.4209, "Asia/Dubai")
EXAMPLE_DATES = ["2026-05-17", "2026-07-15"]
SUN_HORIZON = "-0:50"
PDET_LOW, PDET_HIGH, PEYE_LOW, PEYE_HIGH = 0.30, 0.50, 0.30, 0.60
BANDS = {"A": "naked-eye likely", "B": "borderline naked-eye", "C": "aid-assisted likely",
         "D": "marginal detectability", "E": "not detectable"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fit_logistic(X, y):
    """The pipeline of 04_train_two_stage_crescent_criterion.py."""
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, solver="lbfgs")),
    ])
    return model.fit(X, y)


def load_model(path):
    """Load a saved pipeline and report the scikit-learn version it was saved with."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = joblib.load(path)
    saved = [w.message.original_sklearn_version for w in caught
             if issubclass(w.category, InconsistentVersionWarning)]
    return model, (saved[0] if saved else importlib.metadata.version("scikit-learn"))


def load_modelling_subset(features_csv):
    df = pd.read_csv(features_csv, low_memory=False)
    steps = [("crescent_features.csv", len(df), df[GROUP].nunique())]
    df = df[df["section_type"] == "first_crescent"]
    steps.append(("first crescent (evening)", len(df), df[GROUP].nunique()))
    df = df[df["obs_relative_to_sunset"] != "before_sunset"]
    steps.append(("not before sunset", len(df), df[GROUP].nunique()))
    df = df[df[CLASS].notna()].copy()
    steps.append(("with a visibility class", len(df), df[GROUP].nunique()))
    df[CLASS] = df[CLASS].astype(int)
    if len(df) != 4470 or tuple(df[CLASS].value_counts().sort_index()) != (2319, 505, 1646):
        raise ValueError("Source data have changed; expected 4,470 rows (2,319 / 505 / 1,646)")
    # Unique index preserves the source order, as in the training script.
    return df.reset_index(drop=True), steps


# ---------------------------------------------------------------- [0] versions
def print_versions(models_dir):
    print(f"[0] Software: Python {platform.python_version()} ({platform.python_implementation()}), "
          f"{platform.system()} {platform.machine()}")
    for name in PACKAGES:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        print(f"  {name:15s} {version}")
    for name in MODEL_FILES:
        _, saved = load_model(models_dir / name)
        print(f"  {name} saved with scikit-learn {saved}")


# ---------------------------------------------------------------- [1] records
def print_records(data_dir, models_dir, steps, df):
    print("\n[1] Records in each file of the pipeline (data/icop); lunations = distinct page_url")
    print(f"{'Step':24s} {'File':36s} {'Records':>8s} {'Lunations':>9s} "
          f"{'First cr.':>9s} {'Last cr.':>8s}")
    for step, name in PIPELINE_FILES:
        t = pd.read_csv(data_dir / name, low_memory=False)
        first = last = ""
        if "section_type" in t:
            first = f"{int((t['section_type'] == 'first_crescent').sum()):,}"
            last = f"{int((t['section_type'] == 'last_crescent').sum()):,}"
        print(f"{step:24s} {name:36s} {len(t):8,} {t[GROUP].nunique():9,} {first:>9s} {last:>8s}")

    print("Modelling subset (filters of 04_train_two_stage_crescent_criterion.py)")
    print(f"{'Filter':36s} {'Records':>8s} {'Lunations':>9s}")
    for label, n, lun in steps:
        print(f"{label:36s} {n:8,} {lun:9,}")
    seen = df[df[CLASS] > 0]
    counts = df[CLASS].value_counts().sort_index()
    print(f"{'Stage 1 (all rows)':36s} {len(df):8,} {df[GROUP].nunique():9,}"
          f"   not seen {counts[0]:,}, seen {counts[1] + counts[2]:,}")
    print(f"{'Stage 2 (seen rows)':36s} {len(seen):8,} {seen[GROUP].nunique():9,}"
          f"   aided {counts[1]:,}, naked eye {counts[2]:,}")
    missing = df[FEATURES].isna().sum()
    print("Missing predictor values in the modelling subset: "
          + ", ".join(f"{NAMES[c]} {int(missing[c])}" for c in FEATURES))
    print_negative_lags(df)

    print("SHA-256 of the files read here")
    for _, name in PIPELINE_FILES:
        print(f"  {sha256(data_dir / name)}  data/icop/{name}")
    for name in MODEL_FILES:
        print(f"  {sha256(models_dir / name)}  models/{name}")


def print_negative_lags(df):
    """Negative lags: the Moon set before sunset, or within a minute after it.

    Moonset times are rounded down to the whole minute (03b_correct_moon_lag.py), so
    a lag between -1 and 0 min can also mean a moonset just after sunset. For those
    records the exact moonset is recomputed (PyEphem defaults, elevation 0) from
    10 min before the stored sunset.
    """
    neg = df[df[LAG] < 0]
    small = neg[neg[LAG] > -1]
    after = 0
    for _, r in small.iterrows():
        sunset = pd.Timestamp(r["sunset_local"]).tz_convert("UTC").tz_localize(None)
        obs = ephem.Observer()
        obs.lat, obs.lon, obs.elevation = str(r["latitude"]), str(r["longitude"]), 0
        obs.date = ephem.Date(sunset.to_pydatetime()) - 10 / 1440
        moonset = obs.next_setting(ephem.Moon())
        after += int(moonset > ephem.Date(sunset.to_pydatetime()))
    print(f"Negative lags: {len(neg)} records; {len(small)} between -1 and 0 min, of which "
          f"{after} have the exact moonset after sunset (negative only after rounding down) "
          f"and {len(small) - after} before it; Moon set before sunset in {len(neg) - after}")


# ---------------------------------------------------------------- [2] folds
def print_folds(df, oof_csv):
    y = df[CLASS].to_numpy()
    groups = df[GROUP].to_numpy()
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(cv.split(df, y, groups=groups))

    tested = np.concatenate([test for _, test in folds])
    if sorted(tested) != list(range(len(df))):
        raise RuntimeError("The test folds do not cover every row exactly once")
    for train, test in folds:
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("A lunation is in both training and test rows")

    def counts(idx, stage):
        part = df.iloc[idx]
        if stage == 2:
            part = part[part[CLASS] > 0]
            return len(part), part[GROUP].nunique(), int((part[CLASS] == 1).sum()), int((part[CLASS] == 2).sum())
        return len(part), part[GROUP].nunique(), int((part[CLASS] == 0).sum()), int((part[CLASS] > 0).sum())

    print(f"\n[2] Main grouped split: StratifiedGroupKFold({N_FOLDS}, shuffle=True, "
          f"random_state={SEED}), stratified on the three-class label, groups = {GROUP}")
    print("Test folds: records, lunations and classes (0 not seen, 1 aided, 2 naked eye)")
    print(f"{'Fold':>4s} {'Records':>8s} {'Lunations':>9s} {'Class 0':>8s} {'Class 1':>8s} "
          f"{'Class 2':>8s} {'Share 0/1/2':>17s}")
    for k, (_, test) in enumerate(folds, 1):
        c = np.bincount(y[test], minlength=3)
        print(f"{k:4d} {len(test):8,} {len(set(groups[test])):9,} {c[0]:8,} {c[1]:8,} {c[2]:8,} "
              f"{'/'.join(f'{v / len(test):.3f}' for v in c):>17s}")
    c = np.bincount(y, minlength=3)
    print(f"{'All':>4s} {len(df):8,} {df[GROUP].nunique():9,} {c[0]:8,} {c[1]:8,} {c[2]:8,} "
          f"{'/'.join(f'{v / len(df):.3f}' for v in c):>17s}")

    for stage, label, neg, pos in [(1, "Stage 1 (all rows): 0 not seen, 1 seen", "not seen", "seen"),
                                   (2, "Stage 2 (seen rows): 0 aided, 1 naked eye", "aided", "naked eye")]:
        print(label)
        print(f"{'Fold':>4s} | {'Train':>6s} {'Lun.':>5s} {neg:>9s} {pos:>9s} | "
              f"{'Test':>6s} {'Lun.':>5s} {neg:>9s} {pos:>9s}")
        for k, (train, test) in enumerate(folds, 1):
            a, b = counts(train, stage), counts(test, stage)
            print(f"{k:4d} | {a[0]:6,} {a[1]:5,} {a[2]:9,} {a[3]:9,} | "
                  f"{b[0]:6,} {b[1]:5,} {b[2]:9,} {b[3]:9,}")

    # Refit the fold models and compare with the published out-of-fold file.
    oof = pd.read_csv(oof_csv)
    if len(oof) != len(df) or not (oof[GROUP].to_numpy() == groups).all() \
            or not np.allclose(oof[FEATURES].to_numpy(), df[FEATURES].to_numpy()):
        raise RuntimeError("two_stage_oof_predictions.csv is not in the modelling-subset order")
    p_det, p_eye = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for train, test in folds:
        tr = df.iloc[train]
        seen = tr[CLASS] > 0
        m1 = fit_logistic(tr[FEATURES], seen.astype(int))
        m2 = fit_logistic(tr.loc[seen, FEATURES], (tr.loc[seen, CLASS] == 2).astype(int))
        p_det[test] = m1.predict_proba(df.iloc[test][FEATURES])[:, 1]
        p_eye[test] = m2.predict_proba(df.iloc[test][FEATURES])[:, 1]
    d1 = np.abs(p_det - oof["p_det"].to_numpy()).max()
    d2 = np.abs(p_eye - oof["p_eye"].to_numpy()).max()
    print(f"Refitted fold models vs two_stage_oof_predictions.csv: maximum |difference| "
          f"P_det {d1:.2e}, P_eye {d2:.2e}")
    if max(d1, d2) > 1e-6:
        raise RuntimeError("The out-of-fold predictions are not reproduced")


# ---------------------------------------------------------------- [3] model
def original_scale(model):
    """Coefficients for the unstandardised predictors: z = c0 + sum_i c_i * x_i."""
    sc, clf = model.named_steps["scaler"], model.named_steps["clf"]
    c = clf.coef_[0] / sc.scale_
    return clf.intercept_[0] - float(np.sum(c * sc.mean_)), c


def print_model(stages):
    print("\n[3] Final model: both stages fitted on all modelling records "
          "(median imputation, standardisation, logistic regression, L2 with C = 1)")
    for label, model, X, y in stages:
        imp, sc, clf = (model.named_steps[k] for k in ("imputer", "scaler", "clf"))
        print(f"{label}: {len(X):,} records, {int(y.sum()):,} positive; "
              f"intercept b0 = {clf.intercept_[0]:.4f}; lbfgs iterations {clf.n_iter_[0]}")
        print(f"  {'Predictor':11s} {'Median':>9s} {'Mean':>9s} {'SD':>9s} {'b (std.)':>9s} "
              f"{'c (orig.)':>11s} {'Odds ratio per SD':>18s}")
        c0, c = original_scale(model)
        for j, f in enumerate(FEATURES):
            b = clf.coef_[0][j]
            print(f"  {NAMES[f]:11s} {imp.statistics_[j]:9.4f} {sc.mean_[j]:9.4f} {sc.scale_[j]:9.4f} "
                  f"{b:9.4f} {c[j]:11.6f} {math.exp(b):18.4f}")
        print(f"  {'intercept':11s} {'':9s} {'':9s} {'':9s} {clf.intercept_[0]:9.4f} {c0:11.6f}")
        print("  On the original scale: z = " + f"{c0:.6f} "
              + " ".join(f"{'+' if c[j] >= 0 else '-'} {abs(c[j]):.6f} {n}"
                         for j, n in enumerate(["ARCV", "ARCL", "lag"])))

        # The means and SDs are those of the training records (SD with ddof = 0).
        dm = np.abs(sc.mean_ - X.mean().to_numpy()).max()
        ds = np.abs(sc.scale_ - X.std(ddof=0).to_numpy()).max()
        refit = fit_logistic(X, y).named_steps["clf"]
        dc = max(np.abs(refit.coef_ - clf.coef_).max(), np.abs(refit.intercept_ - clf.intercept_).max())
        print(f"  Checks: |mean - sample mean| {dm:.1e}, |SD - sample SD (ddof 0)| {ds:.1e}, "
              f"refitted vs saved coefficients {dc:.1e}")
        if max(dm, ds, dc) > 1e-6:
            raise RuntimeError(f"{label}: the saved model is not reproduced from the data")


# ---------------------------------------------------------------- [4] example
def sunset_geometry(lat, lon, day):
    """Geometry at local sunset on the evening of 'day' (YYYY-MM-DD), as in scripts 03/03b."""
    obs = ephem.Observer()
    obs.lat, obs.lon, obs.elevation = str(lat), str(lon), 0
    obs.pressure = 0  # airless: geometric topocentric positions
    obs.horizon = SUN_HORIZON
    obs.date = ephem.Date(day.replace("-", "/")) + (12 - lon / 15) / 24  # local mean noon
    sunset = obs.next_setting(ephem.Sun(), use_center=True)
    obs.date = sunset
    sun, moon = ephem.Sun(obs), ephem.Moon(obs)

    # Moonset with PyEphem defaults (upper limb, standard refraction), as in 03b.
    ms = ephem.Observer()
    ms.lat, ms.lon, ms.elevation = str(lat), str(lon), 0
    ms.date = sunset
    set_before, rise_before = ms.previous_setting(ephem.Moon()), ms.previous_rising(ephem.Moon())
    moon_down = set_before > rise_before  # it last set after it last rose
    if moon_down:
        moonset = set_before
    else:
        # PyEphem can miss a moonset within seconds of the start time: search from 10 min before.
        ms.date = sunset - 10 * ephem.minute
        moonset = ms.next_setting(ephem.Moon())
    moonset_min = ephem.Date(moonset).datetime().replace(second=0, microsecond=0)
    return {
        "sunset": ephem.Date(sunset).datetime(),
        "sun_alt": math.degrees(sun.alt),
        "moon_alt": math.degrees(moon.alt),
        ARCV: math.degrees(moon.alt - sun.alt),
        ARCL: math.degrees(ephem.separation(sun, moon)),
        "moon_down": moon_down,
        "moonset": ephem.Date(moonset).datetime(),
        "moonset_min": moonset_min,
        LAG: (moonset_min - ephem.Date(sunset).datetime()).total_seconds() / 60,
        "new_moon": ephem.Date(ephem.previous_new_moon(sunset)).datetime(),
    }


def hand_calculation(label, symbol, model, x):
    """Print z and P step by step and check them against predict_proba."""
    imp, sc, clf = (model.named_steps[k] for k in ("imputer", "scaler", "clf"))
    xv = np.where(np.isnan(x), imp.statistics_, x)  # no value is missing in this example
    xt = (xv - sc.mean_) / sc.scale_
    terms = clf.coef_[0] * xt
    z = clf.intercept_[0] + terms.sum()
    p = 1 / (1 + math.exp(-z))
    p_model = model.predict_proba(pd.DataFrame([x], columns=FEATURES))[0, 1]

    print(f"  {label}")
    print(f"    {'Predictor':11s} {'x':>9s} {'Mean':>9s} {'SD':>9s} {'x~=(x-mean)/SD':>15s} "
          f"{'b':>8s} {'b*x~':>9s}")
    for j, f in enumerate(FEATURES):
        print(f"    {NAMES[f]:11s} {xv[j]:9.4f} {sc.mean_[j]:9.4f} {sc.scale_[j]:9.4f} "
              f"{xt[j]:15.4f} {clf.coef_[0][j]:8.4f} {terms[j]:9.4f}")
    print(f"    z = b0 + sum b*x~ = {clf.intercept_[0]:.4f} "
          + " ".join(f"{'+' if t >= 0 else '-'} {abs(t):.4f}" for t in terms) + f" = {z:.4f}")
    print(f"    {symbol} = 1 / (1 + exp(-z)) = 1 / (1 + exp({-z:.4f})) = {p:.4f}")
    c0, c = original_scale(model)
    z_orig = c0 + float(np.sum(c * xv))
    # The same with the parameters rounded to 4 decimals, as printed above.
    xt4 = (xv - sc.mean_.round(4)) / sc.scale_.round(4)
    p4 = 1 / (1 + math.exp(-(round(clf.intercept_[0], 4) + float(np.sum(clf.coef_[0].round(4) * xt4)))))
    print(f"    Checks: predict_proba {p_model:.4f} (|difference| {abs(p - p_model):.1e}); "
          f"original-scale z {z_orig:.4f}; with the 4-decimal parameters {symbol} = {p4:.4f} "
          f"(|difference| {abs(p4 - p):.1e})")
    if abs(p - p_model) > 1e-12:
        raise RuntimeError("The hand calculation does not reproduce predict_proba")
    return p


def band(p_det, p_eye):
    if p_det < PDET_LOW:
        return "E"
    if p_det < PDET_HIGH:
        return "D"
    if p_eye < PEYE_LOW:
        return "C"
    return "B" if p_eye < PEYE_HIGH else "A"


def cascade(p_det, p_eye):
    """Three-class rule: 0 not seen, 1 aided only, 2 naked eye."""
    if p_det < PDET_HIGH:
        return 0
    return 2 if p_eye >= PEYE_HIGH else 1


def worked_example(stage1, stage2):
    name, lat, lon, tz = SITE
    zone = ZoneInfo(tz)

    def local(t):
        return t.replace(tzinfo=ZoneInfo("UTC")).astimezone(zone)

    print(f"\n[4] Worked example: {name} ({lat} N, {lon} E, {tz}), local sunset")
    for day in EXAMPLE_DATES:
        g = sunset_geometry(lat, lon, day)
        print(f"\n{day}")
        print(f"  Step 1, geometry (PyEphem {ephem.__version__})")
        print(f"    New moon        {g['new_moon']:%Y-%m-%d %H:%M:%S} UT; Moon age at sunset "
              f"{(g['sunset'] - g['new_moon']).total_seconds() / 3600:.4f} h")
        print(f"    Sunset          {g['sunset']:%Y-%m-%d %H:%M:%S} UT = {local(g['sunset']):%H:%M:%S} "
              f"local (Sun's centre at -0:50, airless)")
        print(f"    Sun altitude    {g['sun_alt']:9.4f} deg")
        print(f"    Moon altitude   {g['moon_alt']:9.4f} deg")
        print(f"    ARCV            {g[ARCV]:9.4f} deg  (Moon altitude - Sun altitude)")
        print(f"    ARCL            {g[ARCL]:9.4f} deg  (Sun-Moon separation)")
        when = "before" if g["moon_down"] else "after"
        print(f"    Moonset         {g['moonset']:%H:%M:%S} UT = {local(g['moonset']):%H:%M:%S} local "
              f"({when} sunset), rounded down to {local(g['moonset_min']):%H:%M} local")
        print(f"    Lag             {g[LAG]:9.4f} min  (rounded moonset - sunset)")
        x = np.array([g[ARCV], g[ARCL], g[LAG]])
        print("  Step 2, standardisation, linear score and probability")
        p_det = hand_calculation("Stage 1, detectability (seen by any means)", "P_det", stage1, x)
        p_eye = hand_calculation("Stage 2, naked eye given detection", "P_eye", stage2, x)
        k = band(p_det, p_eye)
        print("  Step 3, operational class (Eq. 11)")
        if p_det < PDET_HIGH:
            rule = (f"P_det = {p_det:.4f} < {PDET_LOW:.2f}" if k == "E"
                    else f"{PDET_LOW:.2f} <= P_det = {p_det:.4f} < {PDET_HIGH:.2f}")
            print(f"    {rule}, so P_eye ({p_eye:.4f}) is not used: class {k}, {BANDS[k]}")
        else:
            print(f"    P_det = {p_det:.4f} >= {PDET_HIGH:.2f} and P_eye = {p_eye:.4f}: "
                  f"class {k}, {BANDS[k]}")
        print(f"    Three-class rule: {['not seen', 'aided only', 'naked eye'][cascade(p_det, p_eye)]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data" / "icop",
                        help="data/icop")
    parser.add_argument("--models-dir", type=Path, default=REPO / "models",
                        help="folder with the saved stage 1 and stage 2 models")
    args = parser.parse_args()

    print_versions(args.models_dir)
    df, steps = load_modelling_subset(args.data_dir / "crescent_features.csv")
    print_records(args.data_dir, args.models_dir, steps, df)
    print_folds(df, args.data_dir / "two_stage_oof_predictions.csv")

    stage1, _ = load_model(args.models_dir / MODEL_FILES[0])
    stage2, _ = load_model(args.models_dir / MODEL_FILES[1])
    seen = df[CLASS] > 0
    print_model([
        ("Stage 1, detectability (P_det)", stage1, df[FEATURES], seen.astype(int)),
        ("Stage 2, naked eye given detection (P_eye)", stage2, df.loc[seen, FEATURES],
         (df.loc[seen, CLASS] == 2).astype(int)),
    ])
    worked_example(stage1, stage2)


if __name__ == "__main__":
    main()
