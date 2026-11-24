#!/usr/bin/env python3
"""Checks for the minor statistical points of the third review round (Reviewer 1).

  (A) Lunations: where the three lunations with only Monday first-crescent reports
      leave the record counts of Fig. 1 (205 lunations after feature computation,
      202 in the modelling subset).
  (B) Alrefay et al. (2018): the 425 records of Section 3.8 against HilalPy's 405
      Alrefay records of Table 5, and the Stage 2 ROC AUC on the 265 sightings of
      the 425 records.
  (C) Optical aids tried by the 2,319 non-sightings: binoculars, telescope and
      camera are not mutually exclusive.
  (D) The 90 modelling records with a negative lag time (82 with moonset before
      sunset, 8 with moonset less than a minute after sunset): class distribution,
      observing means, and a sensitivity check that refits both stages without them
      (lunation-grouped five-fold cross-validation, main split and ten repeats).

Data: data/icop and data/alrefay in this repository, and the HilalPy table (--hilalpy; see
hilalpy_independent_check.py for where to obtain it).
Run:  python3 -W ignore minor_statistical_checks.py --hilalpy "Final 30.6.2023.csv"
"""
import argparse
import importlib.util
from pathlib import Path

import ephem
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # the repository root
DATA = REPO / "data" / "icop"
ALR = REPO / "data" / "alrefay"
FEATURES = ["relative_altitude_deg", "elongation_deg", "moon_lag_minutes"]
TARGET = "vis_class_empirical"
GROUP = "page_url"
PDET, PEYE = 0.50, 0.60
MAIN_SEED, CV_SEEDS, N_FOLDS = 42, range(10), 5
N_BOOT, BOOT_SEED = 2000, 42
MONDAY = "الإثنين"


def modelling_rows(feats):
    keep = ((feats["section_type"] == "first_crescent")
            & (feats["obs_relative_to_sunset"] != "before_sunset") & feats[TARGET].notna())
    return keep


# ---------------------------------------------------------------- (A)
def part_a(feats, clean):
    print("=" * 90 + "\n(A) Lunations lost through the Monday defect\n" + "=" * 90)
    model = feats[modelling_rows(feats)]
    print(f"Clean reports: {clean[GROUP].nunique()} lunations; feature table (6,092 rows): "
          f"{feats[GROUP].nunique()}; modelling subset: {model[GROUP].nunique()}")
    gone = sorted(set(feats[GROUP]) - set(model[GROUP]))
    n_first = n_left = 0
    for url in gone:
        f = feats[feats[GROUP] == url]
        c = clean[clean[GROUP] == url]
        elig = c[c["section_type"] == "first_crescent"]  # all first-crescent reports
        mondays = elig["gregorian_date"].astype(str).str.startswith(MONDAY).sum()
        n_first += len(elig)
        n_left += len(f)
        print(f"  {url.rsplit('/', 1)[-1]}: {len(elig)} first-crescent reports, "
              f"{mondays} dated on a Monday ({elig['gregorian_date'].iloc[0]}; "
              f"{int((elig['obs_relative_to_sunset'] == 'before_sunset').sum())} before sunset); left in the "
              f"feature table: {len(f)} rows, sections {f['section_type'].value_counts().to_dict()}")
    print(f"Together: {n_first} first-crescent reports lost at the date-parsing step; their "
          f"{n_left} last-crescent (morning) reports stay in the feature table and leave at the "
          f"modelling step, where Fig. 1 shows the lunation count fall from 205 to 202.")


# ---------------------------------------------------------------- (B)
def load_module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cluster_auc_interval(y, score, groups):
    rng = np.random.default_rng(BOOT_SEED)
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(N_BOOT):
        pick = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[pick])) == 2:
            vals.append(roc_auc_score(y[pick], score[pick]))
    return np.percentile(vals, [2.5, 97.5])


def part_b(hilalpy_path):
    print("\n" + "=" * 90 + "\n(B) Alrefay et al. (2018): the 425 table records and HilalPy's 405\n" + "=" * 90)
    s1 = pd.read_csv(ALR / "alrefay_stage1_rowwise_comparison.csv")
    s2 = pd.read_csv(ALR / "alrefay_stage2_rowwise_comparison.csv")
    y1 = s1["target_detectability"].astype(int).to_numpy()
    g1 = s1["Date_UT"].astype(str).to_numpy()
    print(f"Section 3.8 sample: {len(s1)} records ({int((y1 == 0).sum())} non-sightings, "
          f"{int(y1.sum())} sightings) on {len(set(g1))} evenings; Stage 2 rows: {len(s2)}")
    auc1 = roc_auc_score(y1, s1["pdet_twostep"])
    y2 = s2["target_naked_eye"].astype(int).to_numpy()
    auc2 = roc_auc_score(y2, s2["peye_twostep"])
    lo, hi = cluster_auc_interval(y2, s2["peye_twostep"].to_numpy(), s2["Date_UT"].astype(str).to_numpy())
    # Misread values that still parse as numbers break cos(ARCL) = cos(ARCV) cos(DAZ).
    tabs = pd.concat([pd.read_csv(ALR / "Table_I_positive_observations.csv"),
                      pd.read_csv(ALR / "Table_II_negative_observations.csv")], ignore_index=True)
    num = {c: pd.to_numeric(tabs[c], errors="coerce") for c in ["ARCV_deg", "ARCL_deg", "DAZ_deg", "Lag_min"]}
    used = num["ARCV_deg"].notna() & num["ARCL_deg"].notna() & num["Lag_min"].notna()
    implied = np.degrees(np.arccos(np.clip(np.cos(np.radians(num["ARCV_deg"])) * np.cos(np.radians(num["DAZ_deg"])), -1, 1)))
    off = used & num["DAZ_deg"].notna() & ((implied - num["ARCL_deg"]).abs() > 0.5)
    print(f"Table rows: {len(tabs)}; with readable lag, ARCV and ARCL: {int(used.sum())}; of these, rows whose "
          f"ARCL differs by more than 0.5 deg from the value implied by ARCV and DAZ: {int(off.sum())} "
          f"(eight misread values were corrected in Nov 2026; see data/alrefay/README.txt)")
    print(f"Stage 1 ROC AUC (425 records): {auc1:.4f}")
    print(f"Stage 2 ROC AUC (265 sightings: {int((y2 == 0).sum())} aided only, {int(y2.sum())} "
          f"naked eye): {auc2:.4f} (95% interval {lo:.3f}-{hi:.3f}, resampling evenings)")

    hc = load_module("hilalpy_independent_check")
    h = hc.load_hilalpy(hilalpy_path)
    archive, _ = hc.load_icop(DATA / "crescent_features.csv")
    alr_all = h[h["src"] == "AlR"]
    s = alr_all.copy()
    steps = [("HilalPy rows from Alrefay et al.", len(s))]
    s, more = hc.evening_filters(s)
    names = ["evening (first crescent)", "ArcV, ArcL and LT present", "ArcL <= 30 deg",
             "lunar age >= 0 or missing"]
    steps += list(zip(names, more))
    rows, primary, _, in_archive = hc.independent_sample(h, archive)
    alr_rows = rows[rows["src"] == "AlR"]
    alr_prim = primary[primary["src"] == "AlR"]
    alr_icop = in_archive[in_archive["src"] == "AlR"]
    print("HilalPy's Alrefay rows through the filters of hilalpy_independent_check.py:")
    for name, n in steps:
        print(f"  {name:40s} {n:5d}")
    print(f"  {'not in our ICOP archive':40s} {len(alr_rows):5d}  (removed: {len(alr_icop)})")
    print(f"  {'one row per event and outcome':40s} {len(alr_prim):5d}  classes 0/1/2 = "
          f"{list(np.bincount(alr_prim['cls'], minlength=3))}")
    print(f"  merged rows (same evening and outcome, sites within 0.5 deg): {len(alr_rows) - len(alr_prim)}")

    # Match the 425 table records with HilalPy's Alrefay rows (same date, within 0.5 deg).
    ours = s1.assign(date=pd.to_datetime(s1["Date_UT"]).dt.strftime("%Y-%m-%d"),
                     cls=np.where(s1["target_detectability"] == 0, 0,
                                  np.where(s1["target_naked_eye"] == 1, 2, 1)),
                     oid=np.arange(len(s1)))
    hp = alr_all[["row", "date", "Lat", "Long", "cls", "ArcV", "ArcL", "LT"]]
    m = ours[["oid", "date", "Lat_deg", "Long_deg", "cls", "ARCV_deg", "ARCL_deg", "Lag_min"]].merge(
        hp, on="date", suffixes=("_ours", "_hp"))
    m = m[((m["Lat_deg"] - m["Lat"]).abs() <= 0.5) & ((m["Long_deg"] - m["Long"]).abs() <= 0.5)]
    pair = m.sort_values(["oid", "row"]).drop_duplicates("oid")
    print(f"Our 425 records with a HilalPy Alrefay row on the same date within 0.5 deg: "
          f"{pair['oid'].nunique()}; HilalPy Alrefay rows matched: {m['row'].nunique()} of {len(alr_all)}")
    same = (pair["cls_ours"] == pair["cls_hp"]).mean()
    print(f"  same outcome class: {same:.1%}; class pairs (ours -> HilalPy):")
    print(pd.crosstab(pair["cls_ours"], pair["cls_hp"]).to_string())
    for a, b, lab in [("ARCV_deg", "ArcV", "ARCV"), ("ARCL_deg", "ArcL", "ARCL"), ("Lag_min", "LT", "lag")]:
        d = pd.to_numeric(pair[a], errors="coerce") - pd.to_numeric(pair[b], errors="coerce")
        print(f"  {lab}: median |difference| {d.abs().median():.2f}, largest {d.abs().max():.2f}")


# ---------------------------------------------------------------- (C)
def part_c(model):
    print("\n" + "=" * 90 + "\n(C) Optical aids tried by the non-sightings\n" + "=" * 90)
    neg = model[model[TARGET] == 0]
    aids = neg[["attempted_binocular", "attempted_telescope", "attempted_camera"]].fillna(False).astype(bool)
    n_aid = aids.sum(axis=1)
    print(f"Non-sightings: {len(neg)}; with an optical aid tried: {(n_aid > 0).sum()} "
          f"({(n_aid > 0).mean():.1%}); binoculars {aids.iloc[:, 0].sum()}, telescope "
          f"{aids.iloc[:, 1].sum()}, camera {aids.iloc[:, 2].sum()} (sum {aids.values.sum()})")
    print(f"Reports with one aid: {(n_aid == 1).sum()}, two: {(n_aid == 2).sum()}, three: {(n_aid == 3).sum()}")


# ---------------------------------------------------------------- (D)
def logistic():
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=4000, solver="lbfgs"))])


def oof(df, seed):
    X = df[FEATURES].to_numpy(float)
    y = df[TARGET].to_numpy(int)
    g = df[GROUP].to_numpy()
    pdet, peye = np.full(len(y), np.nan), np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, groups=g):
        seen = y[tr] > 0
        s1 = logistic().fit(X[tr], seen.astype(int))
        s2 = logistic().fit(X[tr][seen], (y[tr][seen] == 2).astype(int))
        pdet[te], peye[te] = s1.predict_proba(X[te])[:, 1], s2.predict_proba(X[te])[:, 1]
    return pdet, peye


def scores(y, pdet, peye):
    seen = y > 0
    pred = np.where(pdet < PDET, 0, np.where(peye >= PEYE, 2, 1))
    return {"S1 acc": np.mean((pdet >= PDET) == seen), "S1 AUC": roc_auc_score(seen, pdet),
            "S2 acc": np.mean((peye[seen] >= PEYE) == (y[seen] == 2)),
            "S2 AUC": roc_auc_score(y[seen] == 2, peye[seen]), "3-class": np.mean(pred == y)}


def exact_lag(r):
    """Unrounded lag (min) between the moonset around sunset and sunset, as in script 03b."""
    sunset = pd.Timestamp(r["sunset_local"]).tz_convert("UTC")
    obs = ephem.Observer()
    obs.lat, obs.lon, obs.elevation = str(r["latitude"]), str(r["longitude"]), 0
    obs.date = ephem.Date(sunset.tz_localize(None).to_pydatetime())
    moon = ephem.Moon()
    set_before, rise_before = obs.previous_setting(moon), obs.previous_rising(moon)
    moonset = set_before if set_before > rise_before else obs.next_setting(moon)
    return (pd.Timestamp(ephem.Date(moonset).datetime(), tz="UTC") - sunset).total_seconds() / 60


def part_d(model):
    print("\n" + "=" * 90 + "\n(D) Records with a negative lag time\n" + "=" * 90)
    model = model.reset_index(drop=True)
    neg = model["moon_lag_minutes"] < 0
    # Stored moonsets are rounded down to the minute, so the side of sunset on which the
    # Moon set is decided with the unrounded moonset (computed as in script 03b).
    exact = pd.Series(np.nan, index=model.index)
    exact[neg] = [exact_lag(r) for _, r in model[neg].iterrows()]
    before = exact < 0
    groups = {"moonset before sunset": neg & before, "moonset < 1 min after sunset": neg & ~before}
    names = ["not seen", "aided only", "naked eye"]
    for lab, mask in [("all negative lags", neg)] + list(groups.items()):
        cls = np.bincount(model.loc[mask, TARGET].astype(int), minlength=3)
        print(f"{lab:30s} {int(mask.sum()):3d} records: " + ", ".join(f"{n} {c}" for n, c in zip(names, cls))
              + f"; lag {model.loc[mask, 'moon_lag_minutes'].min():.0f} to "
              f"{model.loc[mask, 'moon_lag_minutes'].max():.0f} min; lunations {model.loc[mask, GROUP].nunique()}")
    sightings = model[neg & (model[TARGET] > 0)].assign(unrounded_lag=exact[neg & (model[TARGET] > 0)].round(2))
    cols = ["gregorian_date", "country", "city", "obs_relative_to_sunset", "unrounded_lag",
            "relative_altitude_deg", "seen_naked_eye", "seen_binocular", "seen_telescope", "seen_camera",
            "comment_text"]
    print(f"\nSightings among them ({len(sightings)}):")
    with pd.option_context("display.width", 250, "display.max_colwidth", 90):
        print(sightings[cols[:-1]].to_string())
        for _, r in sightings.iterrows():
            print(f"  {r['city']}: {str(r['comment_text'])[:160]}")
    nons = model[neg & (model[TARGET] == 0)]
    aids = nons[["attempted_binocular", "attempted_telescope", "attempted_camera"]].fillna(False).astype(bool)
    print(f"\nNon-sightings among them: {len(nons)}; an optical aid tried in {int(aids.any(axis=1).sum())}")

    print("\nSensitivity: both stages refitted without these records (out-of-fold, same pipeline).")
    print(f"{'Variant':58s} {'n':>5s} {'S1 acc':>7s} {'S1 AUC':>7s} {'S2 acc':>7s} {'S2 AUC':>7s} {'3-class':>8s}")
    full = oof(model, MAIN_SEED)
    saved = pd.read_csv(DATA / "two_stage_oof_predictions.csv")
    print(f"(check: max |P_det - saved| = {np.max(np.abs(full[0] - saved['p_det'])):.1e})")
    y = model[TARGET].to_numpy(int)

    def row(label, sc, n):
        print(f"{label:58s} {n:5d} " + " ".join(f"{sc[k]:7.3f}" for k in ["S1 acc", "S1 AUC", "S2 acc", "S2 AUC"])
              + f" {sc['3-class']:8.3f}")

    row("All records, main split", scores(y, *full), len(y))
    for lab, mask in [("without the 82 (moonset before sunset)", groups["moonset before sunset"]),
                      ("without all 90 negative lags", neg)]:
        keep = ~mask.to_numpy()
        row(f"  model with them, scored on the rest", scores(y[keep], full[0][keep], full[1][keep]), keep.sum())
        sub = model[keep].reset_index(drop=True)
        row(f"  model {lab}", scores(sub[TARGET].to_numpy(int), *oof(sub, MAIN_SEED)), len(sub))
        rep_full, rep_sub = [], []
        for seed in CV_SEEDS:
            pf = oof(model, seed)
            rep_full.append(scores(y[keep], pf[0][keep], pf[1][keep]))
            rep_sub.append(scores(sub[TARGET].to_numpy(int), *oof(sub, seed)))
        for name, rep in [("    10 repeats, with them (scored on the rest)", rep_full),
                          ("    10 repeats, without them", rep_sub)]:
            row(name, {k: np.mean([r[k] for r in rep]) for k in rep[0]}, keep.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hilalpy", required=True, help="HilalPy table (Final 30.6.2023.csv)")
    args = p.parse_args()
    feats = pd.read_csv(DATA / "crescent_features.csv", low_memory=False)
    clean = pd.read_csv(DATA / "crescent_observations_clean.csv", low_memory=False)
    model = feats[modelling_rows(feats)].copy()
    model[TARGET] = model[TARGET].astype(int)
    if len(model) != 4470:
        raise ValueError("Expected the 4,470 modelling records")
    part_a(feats, clean)
    part_b(args.hilalpy)
    part_c(model)
    part_d(model)


if __name__ == "__main__":
    main()
