#!/usr/bin/env python3
"""Performance of the two-stage criterion within subgroups (reviewer comment 7).

Comment 7 (statistical review):
  "It would also be informative to report performance by region or latitude
  band, by season, by observer type or aid, and before versus after a chosen
  year."

Leave-one-region-out refitting is reported separately (regional_check.py).
Here no model is refitted: the out-of-fold predictions of the main evaluation
are scored within subgroups of the records.

Inputs (default paths in this repository):
  data/icop/crescent_features.csv  the modelling subset (4,470 first-crescent
      reports: 2,319 not seen, 505 aided, 1,646 naked eye; 202 lunations, one
      ICOP monthly page each), filtered as in
      04_train_two_stage_crescent_criterion.py
  data/icop/two_stage_oof_predictions.csv  out-of-fold P_det and P_eye (same
      rows, same order; checked row by row): StratifiedGroupKFold(5,
      shuffle=True, random_state=42), stratified on the three-class label,
      groups = page_url; Stage 1 fitted on all training rows, Stage 2 on the
      seen training rows

Metrics, for each subgroup
  Stage 1      seen by any means against not seen, all records: accuracy at
               P_det >= 0.50 and ROC AUC of P_det
  Stage 2      naked eye against aided, sighted records only: accuracy at
               P_eye >= 0.50 and ROC AUC of P_eye
  Three-class  the operational cascade (Eq. 11): P_det < 0.50 -> not seen;
               P_det >= 0.50 and P_eye >= 0.60 -> naked eye; otherwise aided
  Majority     the accuracy of always predicting the subgroup's most common
               class (for Stage 2, among its sighted records)
Accuracy depends on the class mix, which differs a lot between subgroups, so
it should be read against the majority-class accuracy of the same subgroup.
The AUC does not depend on the class mix. An AUC needs both classes and is
shown as n/a when a subgroup has only one.

Intervals: cluster bootstrap within each subgroup. The lunations (page_url)
with records in the subgroup are resampled with replacement, keeping only the
subgroup's records of each lunation (1,000 resamples, numpy default_rng(42),
restarted for every subgroup), with the out-of-fold probabilities held fixed;
the 2.5th and 97.5th percentiles are reported. Stage 2 uses the sighted
records of the same resamples. Resamples in which an AUC is undefined (one
class only) are left out, and the number used is then stated. Subgroups with
fewer than 100 records (for Stage 2, fewer than 100 sighted records) are
flagged '<100', and those with fewer than 10 lunations 'few lun.', because a
cluster bootstrap over so few lunations gives unreliable intervals.

Subgroups
  Latitude     absolute latitude of the report: < 15, 15-30, 30-45, >= 45 deg;
               hemisphere from the sign of the latitude. These are the
               coordinates used for the crescent geometry.
  Season       meteorological season of the local date of the observation
               evening (local date of sunset_local): Dec-Feb winter, Mar-May
               spring, Jun-Aug summer, Sep-Nov autumn, swapped (winter <->
               summer, spring <-> autumn) in the southern hemisphere, so that
               'summer' is always the local summer. Near the equator the
               seasons are weak.
  Means tried  from the attempted_* columns (naked eye, binoculars,
               telescope, camera), taken from the report's fixed ICOP
               sentence: naked eye only; any optical aid (binoculars,
               telescope or camera, with or without the naked eye); none
               recorded (every means marked 'not sought'). Any optical aid is
               then split by whether the naked eye was also tried, and shown
               by aid (these three groups overlap, as a report can try
               several aids). The archive does not record the type of
               observer, so the means tried is the only proxy for it. The
               means tried is not independent of the outcome: by
               construction, naked-eye-only reports contain no aided
               sightings and aid-only reports no naked-eye sightings.
  Period       Gregorian year of the local observation date: before 2018
               against 2018 onwards, and each year.
  Hijri month  the archive's hijri_month_name column is empty, so the month is
               read from the ICOP page name of the lunation (for example
               jut46 = Jumada al-Thani 1446), i.e. the month that the crescent
               begins. It is checked against the tabular Islamic calendar.

Contrasts: for the two-way splits (hemisphere, naked eye only against any
optical aid, and the two periods), the difference between the subgroups has a
95% interval from a joint cluster bootstrap: all 202 lunations are resampled
together (1,000 resamples, default_rng(42)), since a lunation can contribute
records to both subgroups. The difference in majority-class accuracy is given
alongside, to show how much of an accuracy difference comes from the class mix.

Run: python3 -W ignore analysis/subgroup_performance.py   (about ten seconds)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ARCV = "relative_altitude_deg"
ARCL = "elongation_deg"
LAG = "moon_lag_minutes"
FEATURES = [ARCV, ARCL, LAG]
TARGET = "vis_class_empirical"
LUNATION = "page_url"
CUT = 0.50            # accuracy cutoff of both stages, as in the main evaluation
PDET, PEYE = 0.50, 0.60   # cascade cutoffs of Eq. 11
N_BOOT = 1000
BOOT_SEED = 42
SMALL_N = 100
FEW_LUNATIONS = 10
SPLIT_YEAR = 2018
METRICS = ["s1_acc", "s1_auc", "s2_acc", "s2_auc", "c3_acc"]
MAJORITY = {"s1_acc": "s1_maj", "s2_acc": "s2_maj", "c3_acc": "c3_maj"}
METRIC_NAMES = {"s1_acc": "Stage 1 accuracy", "s1_auc": "Stage 1 AUC",
                "s2_acc": "Stage 2 accuracy", "s2_auc": "Stage 2 AUC",
                "c3_acc": "Three-class accuracy"}

# ICOP page names: three-letter month code and the last two digits of the Hijri year (14xx).
HIJRI_MONTHS = [("muh", "Muharram"), ("saf", "Safar"), ("raa", "Rabi al-Awwal"),
                ("rat", "Rabi al-Thani"), ("jua", "Jumada al-Ula"), ("jut", "Jumada al-Thani"),
                ("raj", "Rajab"), ("sha", "Shaban"), ("ram", "Ramadan"), ("shw", "Shawwal"),
                ("kea", "Dhu al-Qadah"), ("hej", "Dhu al-Hijjah")]
SEASONS_NORTH = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
                 6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
SWAP = {"Winter": "Summer", "Summer": "Winter", "Spring": "Autumn", "Autumn": "Spring"}
AIDS = [("attempted_binocular", "binoculars"), ("attempted_telescope", "telescope"),
        ("attempted_camera", "camera")]

HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # the repository root
DEFAULT_INPUT = REPO / "data" / "icop" / "crescent_features.csv"
DEFAULT_OOF = REPO / "data" / "icop" / "two_stage_oof_predictions.csv"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_subset(path):
    """The 4,470-row modelling subset, in source order."""
    df = pd.read_csv(path, low_memory=False)
    df = df[
        (df["section_type"] == "first_crescent")
        & (df["obs_relative_to_sunset"] != "before_sunset")
        & df[TARGET].notna()
    ].copy()
    df[TARGET] = df[TARGET].astype(int)
    if len(df) != 4470 or tuple(df[TARGET].value_counts().sort_index()) != (2319, 505, 1646):
        raise ValueError("Source data have changed; verify the cohort before comparison")
    return df.reset_index(drop=True)


def load_oof(path, df):
    oof = pd.read_csv(path)
    if (len(oof) != len(df) or not (oof[LUNATION].to_numpy() == df[LUNATION].to_numpy()).all()
            or not (oof[TARGET].to_numpy() == df[TARGET].to_numpy()).all()
            or not np.allclose(oof[FEATURES].to_numpy(), df[FEATURES].to_numpy())):
        raise ValueError("Out-of-fold file does not match the modelling subset row by row")
    if oof[["p_det", "p_eye"]].isna().any().any():
        raise ValueError("Missing out-of-fold probabilities")
    return oof


def islamic_from_date(d):
    """Tabular (civil) Islamic calendar date of a Gregorian date (Reingold and Dershowitz)."""
    epoch = 227015  # fixed day number of 1 Muharram AH 1 (16 July 622, Julian)

    def fixed(y, m, day):
        return day + 29 * (m - 1) + (6 * m - 1) // 11 + (y - 1) * 354 + (3 + 11 * y) // 30 + epoch - 1

    rd = d.toordinal()
    year = (30 * (rd - epoch) + 10646) // 10631
    month = (11 * (rd - fixed(year, 1, 1)) + 330) // 325
    return year, month, rd - fixed(year, month, 1) + 1


def add_subgroup_columns(df):
    local = pd.to_datetime(df["sunset_local"].str[:10])  # local date of the observation evening
    df["year"] = local.dt.year
    df["abs_lat"] = df["latitude"].abs()
    south = df["latitude"] < 0
    season = local.dt.month.map(SEASONS_NORTH)
    df["season"] = np.where(south, season.map(SWAP), season)

    eye = df["attempted_naked_eye"].astype(bool)
    aid = df[[c for c, _ in AIDS]].astype(bool).any(axis=1)
    df["means"] = np.select([aid, eye], ["Any optical aid", "Naked eye only"], "None recorded")
    df["aid_split"] = np.where(aid & eye, "Aid and naked eye", np.where(aid, "Aid only", ""))

    slug = df[LUNATION].str.extract(r"/icop/([a-z]{3})(\d{2})\.html$")
    if slug.isna().any().any():
        raise ValueError("Unexpected ICOP page name; cannot read the Hijri month")
    codes = {code: k + 1 for k, (code, _) in enumerate(HIJRI_MONTHS)}
    df["hijri_month"] = slug[0].map(codes)
    df["hijri_year"] = 1400 + slug[1].astype(int)
    if df["hijri_month"].isna().any():
        raise ValueError("Unknown Hijri month code in the ICOP page names")

    # Check: the day after the observation evening should be day 1-2 of the page's month
    # in the tabular calendar, or day 29-30 of the month before (it can differ by a day or two).
    tab = [islamic_from_date((d + pd.Timedelta(days=1)).date()) for d in local]
    step = np.array([12 * y + m for y, m, _ in tab]) - (12 * df["hijri_year"] + df["hijri_month"]).to_numpy()
    day = np.array([d for _, _, d in tab])
    agree = (step == 0) | ((step == -1) & (day >= 29))
    print(f"Hijri month from the ICOP page name: the day after the observation evening falls at the "
          f"start of that month in the tabular Islamic calendar (or on day 29-30 of the month before) "
          f"for {agree.sum():,} of {len(df):,} records; the others are kept under their page's month.")
    return df


def groupings(df):
    """(title, [(label, row mask), ...]) for every subgroup analysis."""
    lat = df["abs_lat"]
    early = df["year"] < SPLIT_YEAR
    y0, y1 = df["year"].min(), df["year"].max()
    out = [
        ("Absolute latitude", [
            ("|lat| < 15 deg", lat < 15),
            ("15 <= |lat| < 30 deg", (lat >= 15) & (lat < 30)),
            ("30 <= |lat| < 45 deg", (lat >= 30) & (lat < 45)),
            ("|lat| >= 45 deg", lat >= 45)]),
        ("Hemisphere", [
            ("Northern", df["latitude"] >= 0),
            ("Southern", df["latitude"] < 0)]),
        ("Season (local meteorological season; swapped in the southern hemisphere)", [
            (s, df["season"] == s) for s in ["Winter", "Spring", "Summer", "Autumn"]]),
        ("Means tried (attempted_* columns)", [
            (m, df["means"] == m) for m in ["Naked eye only", "Any optical aid", "None recorded"]]),
        ("Any optical aid, by whether the naked eye was also tried", [
            (m, df["aid_split"] == m) for m in ["Aid and naked eye", "Aid only"]]),
        ("Optical aid tried, by aid (overlapping: a report can try several aids)", [
            (f"{name} tried", df[col].astype(bool)) for col, name in AIDS]),
        ("Period (Gregorian year of the observation)", [
            (f"Before {SPLIT_YEAR} ({y0}-{SPLIT_YEAR - 1})", early),
            (f"{SPLIT_YEAR} onwards ({SPLIT_YEAR}-{y1})", ~early)]),
        ("Gregorian year", [(str(y), df["year"] == y) for y in sorted(df["year"].unique())]),
        ("Hijri month (from the ICOP page name; the month the crescent begins)", [
            (name, df["hijri_month"] == k + 1) for k, (_, name) in enumerate(HIJRI_MONTHS)]),
    ]
    return out


# ----------------------------------------------------------------------------
# Metrics with row weights (a bootstrap resample = integer weight per row)
# ----------------------------------------------------------------------------
def weighted_auc(rank, y, w):
    """ROC AUC as the Mann-Whitney statistic (ties count one half), with row weights.

    rank is the index of each row's score among the distinct scores (ascending)."""
    if len(rank) == 0:
        return np.nan
    size = rank.max() + 1
    pos = np.bincount(rank, weights=w * y, minlength=size)
    neg = np.bincount(rank, weights=w * ~y, minlength=size)
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    below = np.cumsum(neg) - neg
    return float((pos * (below + 0.5 * neg)).sum() / (n_pos * n_neg))


def prepare(df, oof):
    """Per-row arrays needed by the metrics."""
    y = df[TARGET].to_numpy()
    p_det, p_eye = oof["p_det"].to_numpy(), oof["p_eye"].to_numpy()
    seen, eye = y > 0, y == 2
    cascade = np.where(p_det < PDET, 0, np.where(p_eye >= PEYE, 2, 1))
    return {"y": y, "p_det": p_det, "p_eye": p_eye, "seen": seen, "eye": eye,
            "s1_hit": (p_det >= CUT) == seen, "s2_hit": (p_eye >= CUT) == eye,
            "c3_hit": cascade == y}


def subgroup_arrays(base, rows):
    d = {k: v[rows] for k, v in base.items()}
    d["rank_det"] = np.unique(d["p_det"], return_inverse=True)[1]
    d["rank_eye"] = np.unique(d["p_eye"][d["seen"]], return_inverse=True)[1]
    return d


def scores(d, w):
    """All metrics of one subgroup with row weights w."""
    total = w.sum()
    if total == 0:
        return {k: np.nan for k in METRICS + list(MAJORITY.values())}
    seen = d["seen"]
    ws = w[seen]
    counts = np.bincount(d["y"], weights=w, minlength=3)
    n_seen = counts[1] + counts[2]
    return {
        "s1_acc": (w * d["s1_hit"]).sum() / total,
        "s1_auc": weighted_auc(d["rank_det"], seen, w),
        "s2_acc": (ws * d["s2_hit"][seen]).sum() / n_seen if n_seen > 0 else np.nan,
        "s2_auc": weighted_auc(d["rank_eye"], d["eye"][seen], ws),
        "c3_acc": (w * d["c3_hit"]).sum() / total,
        "s1_maj": max(counts[0], n_seen) / total,
        "s2_maj": max(counts[1], counts[2]) / n_seen if n_seen > 0 else np.nan,
        "c3_maj": counts.max() / total,
    }


def resample_weights(codes, n_boot, seed):
    """Row weights of n_boot resamples of whole lunations (codes 0..G-1), with replacement."""
    n_groups = codes.max() + 1
    draws = np.random.default_rng(seed).integers(0, n_groups, size=(n_boot, n_groups))
    for pick in draws:
        yield np.bincount(pick, minlength=n_groups)[codes].astype(float)


def interval(values):
    values = np.asarray(values, dtype=float)
    ok = ~np.isnan(values)
    if not ok.any():
        return np.nan, np.nan, 0
    lo, hi = np.percentile(values[ok], [2.5, 97.5])
    return lo, hi, int(ok.sum())


def evaluate(base, groups, mask, n_boot):
    """Sizes, point estimates and within-subgroup cluster-bootstrap intervals."""
    rows = np.flatnonzero(mask)
    d = subgroup_arrays(base, rows)
    codes = pd.factorize(groups[rows])[0]
    point = scores(d, np.ones(len(rows)))
    boot = pd.DataFrame([scores(d, w) for w in resample_weights(codes, n_boot, BOOT_SEED)])
    res = {"n": len(rows), "lunations": codes.max() + 1,
           "counts": np.bincount(d["y"], minlength=3),
           "seen_lunations": len(set(groups[rows][d["seen"]])),
           "point": point, "ci": {}}
    for m in METRICS:
        res["ci"][m] = interval(boot[m])
    return res


# ----------------------------------------------------------------------------
# Printing
# ----------------------------------------------------------------------------
def fmt(point, ci, signed=False):
    width = 7 if signed else 6
    if np.isnan(point):
        return f"{'n/a (one class)':<{3 * width + 5}s}"
    lo, hi, _ = ci
    return f"{point:{width}.4f} [{lo:{width}.4f}, {hi:{width}.4f}]"


def note(res, metric, n_boot, n, lunations):
    flags = []
    if n < SMALL_N:
        flags.append(f"<{SMALL_N}")
    if lunations < FEW_LUNATIONS:
        flags.append("few lun.")
    used = res["ci"][metric][2]
    if not np.isnan(res["point"][metric]) and used < n_boot:
        flags.append(f"AUC CI from {used} resamples")
    return "; ".join(flags)


def print_grouping(title, labelled, n_boot):
    w = max(22, max(len(label) for label, _ in labelled) + 1)
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    print("Sizes and three-class (cascade) accuracy")
    print(f"{'Subgroup':{w}s} {'N':>5s} {'Lun.':>5s} {'Not seen':>9s} {'Aided':>6s} "
          f"{'Naked eye':>10s}  {'Accuracy [95% CI]':23s} {'Majority':>8s}  Note")
    for label, r in labelled:
        n0, n1, n2 = r["counts"]
        print(f"{label:{w}s} {r['n']:5,d} {r['lunations']:5d} {n0:9,d} {n1:6,d} {n2:10,d}  "
              f"{fmt(r['point']['c3_acc'], r['ci']['c3_acc'])} {r['point']['c3_maj']:8.4f}  "
              f"{note(r, 'c3_acc', n_boot, r['n'], r['lunations'])}")

    print(f"\nStage 1: seen by any means against not seen (all records; accuracy at P_det >= {CUT:.2f})")
    print(f"{'Subgroup':{w}s} {'N':>5s}  {'Accuracy [95% CI]':23s} {'Majority':>8s}  "
          f"{'AUC [95% CI]':23s}  Note")
    for label, r in labelled:
        print(f"{label:{w}s} {r['n']:5,d}  {fmt(r['point']['s1_acc'], r['ci']['s1_acc'])} "
              f"{r['point']['s1_maj']:8.4f}  {fmt(r['point']['s1_auc'], r['ci']['s1_auc'])}  "
              f"{note(r, 's1_auc', n_boot, r['n'], r['lunations'])}")

    print(f"\nStage 2: naked eye against aided (sighted records; accuracy at P_eye >= {CUT:.2f})")
    print(f"{'Subgroup':{w}s} {'Seen':>5s}  {'Accuracy [95% CI]':23s} {'Majority':>8s}  "
          f"{'AUC [95% CI]':23s}  Note")
    for label, r in labelled:
        n_seen = int(r["counts"][1:].sum())
        if n_seen == 0:
            print(f"{label:{w}s} {0:5d}  no sighted records")
            continue
        print(f"{label:{w}s} {n_seen:5,d}  {fmt(r['point']['s2_acc'], r['ci']['s2_acc'])} "
              f"{r['point']['s2_maj']:8.4f}  {fmt(r['point']['s2_auc'], r['ci']['s2_auc'])}  "
              f"{note(r, 's2_auc', n_boot, n_seen, r['seen_lunations'])}")


def print_contrast(title, base, groups, mask_a, mask_b, n_boot):
    """Subgroup A minus subgroup B, with a joint cluster bootstrap over all lunations."""
    rows_a, rows_b = np.flatnonzero(mask_a), np.flatnonzero(mask_b)
    da, db = subgroup_arrays(base, rows_a), subgroup_arrays(base, rows_b)
    codes = pd.factorize(groups)[0]
    pa, pb = scores(da, np.ones(len(rows_a))), scores(db, np.ones(len(rows_b)))
    diffs = []
    for w in resample_weights(codes, n_boot, BOOT_SEED):
        sa, sb = scores(da, w[rows_a]), scores(db, w[rows_b])
        diffs.append({m: sa[m] - sb[m] for m in METRICS})
    diffs = pd.DataFrame(diffs)
    print(f"\n{title}")
    print(f"{'Metric':22s} {'Difference [95% CI]':26s} {'Majority difference':>20s}")
    for m in METRICS:
        d = pa[m] - pb[m]
        lo, hi, used = interval(diffs[m])
        maj = f"{pa[MAJORITY[m]] - pb[MAJORITY[m]]:20.4f}" if m in MAJORITY else f"{'':20s}"
        extra = f"  (from {used} resamples)" if not np.isnan(d) and used < n_boot else ""
        print(f"{METRIC_NAMES[m]:22s} {fmt(d, (lo, hi, used), signed=True)} {maj}{extra}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default=str(DEFAULT_INPUT),
                        help="Path to data/icop/crescent_features.csv")
    parser.add_argument("--oof", default=str(DEFAULT_OOF),
                        help="Path to data/icop/two_stage_oof_predictions.csv")
    parser.add_argument("--n-boot", type=int, default=N_BOOT, help="Lunation resamples per subgroup")
    parser.add_argument("--csv", default=None, help="Optional path to save the subgroup table as CSV")
    args = parser.parse_args()

    df = load_subset(args.features)
    oof = load_oof(args.oof, df)
    print(f"Modelling subset: {len(df):,} records, {df[LUNATION].nunique()} lunations; "
          f"out-of-fold predictions matched row by row.")
    df = add_subgroup_columns(df)
    base = prepare(df, oof)
    groups = df[LUNATION].to_numpy()

    print(f"\nIntervals: cluster bootstrap within each subgroup ({args.n_boot:,} resamples of its "
          f"lunations, default_rng({BOOT_SEED}) restarted per subgroup; out-of-fold probabilities "
          f"held fixed; percentile 95% intervals). Majority = accuracy of always predicting the "
          f"subgroup's most common class. Note: <{SMALL_N} = fewer than {SMALL_N} records (Stage 2: "
          f"sighted records); few lun. = fewer than {FEW_LUNATIONS} lunations.")

    everything = evaluate(base, groups, np.ones(len(df), bool), args.n_boot)
    print_grouping("All records (reference)", [("All records", everything)], args.n_boot)

    table = []
    for title, subgroups in groupings(df):
        labelled = [(label, evaluate(base, groups, mask.to_numpy(), args.n_boot))
                    for label, mask in subgroups]
        print_grouping(title, labelled, args.n_boot)
        for label, r in labelled:
            row = {"grouping": title, "subgroup": label, "n": r["n"], "lunations": r["lunations"],
                   "not_seen": r["counts"][0], "aided": r["counts"][1], "naked_eye": r["counts"][2]}
            for m in METRICS:
                row[m], row[f"{m}_lo"], row[f"{m}_hi"] = r["point"][m], *r["ci"][m][:2]
            for m in MAJORITY.values():
                row[m] = r["point"][m]
            table.append(row)

    print("\n" + "=" * 100)
    print(f"Differences between two subgroups (joint cluster bootstrap over all "
          f"{df[LUNATION].nunique()} lunations, {args.n_boot:,} resamples, default_rng({BOOT_SEED}))")
    print("=" * 100)
    early = (df["year"] < SPLIT_YEAR).to_numpy()
    south = (df["latitude"] < 0).to_numpy()
    print_contrast(f"{SPLIT_YEAR} onwards minus before {SPLIT_YEAR}", base, groups, ~early, early,
                   args.n_boot)
    print_contrast("Southern minus northern hemisphere", base, groups, south, ~south, args.n_boot)
    print_contrast("Any optical aid minus naked eye only (Stage 2 is degenerate: naked-eye-only "
                   "reports have no aided sightings)", base, groups,
                   (df["means"] == "Any optical aid").to_numpy(),
                   (df["means"] == "Naked eye only").to_numpy(), args.n_boot)

    if args.csv:
        pd.DataFrame(table).to_csv(args.csv, index=False)
        print(f"\nSubgroup table saved to {args.csv}")


if __name__ == "__main__":
    main()
