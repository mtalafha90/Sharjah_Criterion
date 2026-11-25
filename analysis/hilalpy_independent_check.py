#!/usr/bin/env python3
"""Independent check of the Sharjah criterion on the HilalPy sighting database.

This script applies the saved Sharjah criterion (both stages) to the
non-ICOP evening reports in the HilalPy sighting database. It also scores
other criteria and models on the same reports.

Data (--input): the authors' Mendeley Data deposit "Hilalpy: Database of Lunar
Crescent Sighting" (Faid et al.), DOI 10.17632/rs994bxgc4.1 (version 1,
22 August 2023, CC BY 4.0), file "Final 30.6.2023.csv" (any local name can be
passed to --input), 8,026 rows.
Secondary note: the same table was in the public git history of
https://github.com/msyazwanfaid/hilalpy as Final.csv at commit 00bb144
(3 July 2023; deleted from main in commit e991e7f, 31 July 2023):
    git show 00bb144:Final.csv > Final.csv
That file has two extra ICOP rows (ICOP34KEA 2013-09-06, ICOP34SHW 2013-08-07);
every non-ICOP row is identical, so either file gives the same independent sample.
--icop is data/icop/crescent_features.csv
(training data, and filter 6 below); --models-dir holds the saved joblib models
(the final criterion, both stages fitted on all 4,470 ICOP rows); --oof is
data/icop/two_stage_oof_predictions.csv, the out-of-fold probabilities of script 04
(five folds grouped by lunation), used for every ICOP reference row.

Feature mapping (HilalPy geometry is at the local, elevation-corrected sunset):
  relative_altitude_deg <- ArcV (deg), elongation_deg <- ArcL (deg),
  moon_lag_minutes <- LT (min); for the Table 1 criteria also
  relative_azimuth_deg <- DAZ (deg) and crescent_width_arcmin <- W / 60 (W in arcsec).
Labels: 0 if V == 'I'; 1 if V == 'V' and M == 'OA'; 2 if V == 'V' and M == 'NE';
  ICOP rows only: V == 'V' and NE == 'V' gives 2 (as in our archive).

Independent sample, filters in order: (1) source (Ref No prefix) is not ICOP;
(2) evening, O.upper() == 'E'; (3) ArcV, ArcL and LT present (LT may be
negative); (4) ArcL <= 30 deg; (5) MA >= 0 or missing; (6) no report in our ICOP archive
on the same date and section within 0.5 deg in latitude and longitude.
De-duplicated sample: one row per (event, class), where an event joins reports
on the same date and O within 0.5 deg (union-find); the retained row follows the
source priority Yallop > Fatoohi > Krauss > Odeh > Qur > Amir > Ho > AlR > TK,
then the lowest Ref No.
Headline sample: the de-duplicated sample without the AlR rows, which are the
Alrefay et al. (2018) tables already used for external validation in the
manuscript; AlR is reported separately. Most Odeh rows (1998-2005) are probably
reports from the ICOP programme; they predate our archive (from October 2009),
so they cannot be the same reports, but they come from the same programme.

Operational rule: P_det < 0.50 -> 0; P_eye >= 0.60 -> 2; otherwise 1. For the
direct three-class alternatives the same rule is applied with P_det = P1 + P2
and P_eye = P2 / (P1 + P2).
Contradiction rates follow hilalpy 0.0.4 equa.py (recall type, complementary
regions): positive = share of sightings that the decision calls not visible,
negative = share of non-sightings that the decision calls visible. They are given
for the detectability decision (P_det >= 0.50) and, as an adaptation of the same
definitions, for the naked-eye decision (P_det >= 0.50 and P_eye >= 0.60;
positives = naked-eye sightings, negatives = all other rows). Outside ICOP, most
non-sightings are 'not seen by naked eye' (M == 'NE'); whether optical aid was
tried is not recorded.

Calendar criteria (sections 8 and 9) use HilalPy's own columns, as given in the
file: MAlt (Moon altitude at sunset, deg), ArcL (deg) and MA (Moon age, h):
MABIMS 1995 (MAlt >= 2 and ArcL >= 3, or MA >= 8), MABIMS 2021 (MAlt >= 3 and
ArcL >= 6.4) and Istanbul 2016 as a local screen (MAlt >= 5 and ArcL >= 8). They
are compared with the Sharjah detection decision (final model) on the same rows.
Teluk Kemang (Ref No prefix TK; 2.45 N, 101.86 E, Malaysia) is reported as a
separate regional subset; the Malaysian MyRHK database itself is not public.
Section 10 compares the adopted thresholds (0.50 and 0.60) with the Youden-J and
max-F1 thresholds that each sample would choose for itself.
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ARCV = "relative_altitude_deg"
ARCL = "elongation_deg"
DAZ = "relative_azimuth_deg"
WIDTH = "crescent_width_arcmin"
LAG = "moon_lag_minutes"
FEATURES = [ARCV, ARCL, LAG]
TARGET = "vis_class_empirical"
LABELS = [0, 1, 2]
PDET, PEYE = 0.50, 0.60
HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # the repository root
DEFAULT_ICOP = REPO / "data" / "icop" / "crescent_features.csv"
DEFAULT_MODELS = REPO / "models"
DEFAULT_OOF = REPO / "data" / "icop" / "two_stage_oof_predictions.csv"
PRIORITY = {"Yallop": 0, "Fatoohi": 1, "Krauss": 2, "Odeh": 3, "Qur": 4,
            "Amir": 5, "Ho": 6, "AlR": 7, "TK": 8}
# Expected counts of the evaluation specification (the same for both file versions).
EXPECTED_STEPS = [3784, 3389, 3246, 3219, 3209, 3142]
EXPECTED_ROW_LEVEL = (3142, (831, 390, 1921))
EXPECTED_PRIMARY = (1770, (517, 228, 1025))
EXPECTED_HEADLINE = (1365, (355, 157, 853))


# ---------------------------------------------------------------- data
def load_sibling(name):
    """A script in this folder, e.g. table1_criteria_comparison (criterion formulas)."""
    # Loaded by path, so the import also works under 'python3 -I'.
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_hilalpy(path):
    h = pd.read_csv(path, low_memory=False)
    need = {"Ref No", "Year", "Month", "Day", "Lat", "Long", "M", "V", "O", "NE",
            "LT", "MA", "ArcV", "ArcL", "DAZ", "W"}
    missing = need - set(h.columns)
    if missing:
        raise ValueError(f"Missing HilalPy columns: {sorted(missing)}")
    # Ref No is a report code, not a row identifier, so add a positional one.
    h["row"] = np.arange(len(h))
    h["src"] = h["Ref No"].astype(str).str.extract(r"^([A-Za-z]+)")[0]
    h["O2"] = h["O"].astype(str).str.upper()  # one row has a lower-case 'e'
    seen = h["V"] == "V"
    h["cls"] = np.select([h["V"] == "I", seen & (h["M"] == "OA"), seen & (h["M"] == "NE")],
                         [0, 1, 2], default=-1)
    h.loc[(h["src"] == "ICOP") & seen & (h["NE"] == "V"), "cls"] = 2
    if (h["cls"] < 0).any():
        raise ValueError("Rows with V or M outside the expected codes")
    h["date"] = (h["Year"].astype(str).str.zfill(4) + "-" + h["Month"].astype(str).str.zfill(2)
                 + "-" + h["Day"].astype(str).str.zfill(2))
    # Model features; no unit conversion is needed (deg, deg, min).
    h[ARCV], h[ARCL], h[LAG] = h["ArcV"], h["ArcL"], h["LT"]
    h[DAZ], h[WIDTH] = h["DAZ"], h["W"] / 60.0  # W is in arcseconds
    return h


def load_icop(path):
    a = pd.read_csv(path, low_memory=False)
    a["aid"] = np.arange(len(a))
    a["date"] = a["sunset_local"].astype(str).str[:10]
    a["O2"] = np.where(a["section_type"] == "first_crescent", "E", "M")
    a["insub"] = ((a["section_type"] == "first_crescent")
                  & (a["obs_relative_to_sunset"] != "before_sunset") & a[TARGET].notna())
    # Same filter and order as the training script.
    model_df = a[a["insub"]].reset_index(drop=True).copy()
    model_df["cls"] = model_df[TARGET].astype(int)
    if len(model_df) != 4470 or tuple(np.bincount(model_df["cls"])) != (2319, 505, 1646):
        raise ValueError("ICOP training data have changed; expected 4,470 rows, 2319/505/1646")
    return a, model_df


def load_oof(path, model_df):
    """Script 04's out-of-fold P_det and P_eye for the 4,470 rows (same order)."""
    oof = pd.read_csv(path, low_memory=False)
    if (len(oof) != len(model_df)
            or not (oof["page_url"].to_numpy() == model_df["page_url"].to_numpy()).all()
            or not (oof[TARGET].to_numpy() == model_df["cls"].to_numpy()).all()):
        raise ValueError(f"{path} does not hold the modelling rows in the same order")
    return oof["p_det"].to_numpy(), oof["p_eye"].to_numpy()


def evening_filters(df):
    """Filters 2-5; returns the filtered rows and the count after each step."""
    counts = []
    df = df[df["O2"] == "E"]
    counts.append(len(df))
    # Negative LT (moonset before sunset) is kept: the training data contain such lags.
    df = df[df[["ArcV", "ArcL", "LT"]].notna().all(axis=1)]
    counts.append(len(df))
    df = df[df["ArcL"] <= 30]  # training maximum is 30.05 deg
    counts.append(len(df))
    df = df[~(df["MA"] < 0)]  # keeps missing MA; removes attempts before conjunction
    counts.append(len(df))
    return df, counts


def independent_sample(h, archive):
    s = h[h["src"] != "ICOP"]
    counts = [len(s)]
    s, more = evening_filters(s)
    counts += more
    # Filter 6: reports that are also in our ICOP archive (same date, section, 0.5 deg box).
    cand = s[["row", "date", "O2", "Lat", "Long"]].merge(
        archive[["date", "O2", "latitude", "longitude"]], on=["date", "O2"])
    near = cand[((cand["Lat"] - cand["latitude"]).abs() <= 0.5)
                & ((cand["Long"] - cand["longitude"]).abs() <= 0.5)]
    in_archive = s[s["row"].isin(near["row"])]
    s = s[~s["row"].isin(near["row"])].reset_index(drop=True).copy()
    counts.append(len(s))

    # Events: same date and O, within 0.5 deg in latitude and longitude (union-find).
    parent = list(range(len(s)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    lat, lon = s["Lat"].to_numpy(), s["Long"].to_numpy()
    for idx in s.groupby(["Year", "Month", "Day", "O2"]).indices.values():
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                p, q = idx[i], idx[j]
                if abs(lat[p] - lat[q]) <= 0.5 and abs(lon[p] - lon[q]) <= 0.5:
                    parent[find(p)] = find(q)
    s["event"] = [find(i) for i in range(len(s))]
    s["ev_ncls"] = s.groupby("event")["cls"].transform("nunique")
    s["prio"] = s["src"].map(PRIORITY)
    if s["prio"].isna().any():
        raise ValueError(f"Unexpected sources: {sorted(set(s['src']) - set(PRIORITY))}")
    # Within one source, the lowest Ref No (as text) is kept, as in the Ref No-sorted
    # GitHub file; this makes the choice independent of the row order of the file.
    s = s.sort_values(["event", "cls", "prio", "Ref No", "row"], kind="mergesort")
    primary = s[~s.duplicated(["event", "cls"])].copy()
    return s, primary, counts, in_archive


def class_counts(y):
    return tuple(int(c) for c in np.bincount(y, minlength=3))


def check_counts(df, expected, label):
    n, cls = expected
    got = (len(df), class_counts(df["cls"].to_numpy()))
    if got != (n, cls):
        raise ValueError(f"{label}: expected {n:,} rows {cls}, got {got[0]:,} rows {got[1]}")


# ---------------------------------------------------------------- metrics
def operational(pdet, peye):
    return np.where(pdet < PDET, 0, np.where(peye >= PEYE, 2, 1))


def safe_auc(y, score):
    return roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan


def rate(mask_num, mask_den):
    den = int(mask_den.sum())
    num = int((mask_num & mask_den).sum())
    return (num / den if den else np.nan), num, den


def evaluate(y, pdet, peye):
    seen = y > 0
    pred = operational(pdet, peye)
    det = pdet >= PDET
    r = {"n": len(y), "counts": class_counts(y), "pred_counts": class_counts(pred)}
    r["s1_acc"] = accuracy_score(seen, det)
    r["s1_auc"] = safe_auc(seen.astype(int), pdet)
    r["s1_cm"] = confusion_matrix(seen, det, labels=[False, True])
    y2, eye = y[seen] == 2, peye[seen] >= PEYE
    r["s2_n"] = int(seen.sum())
    r["s2_acc"] = accuracy_score(y2, eye) if len(y2) else np.nan
    r["s2_auc"] = safe_auc(y2.astype(int), peye[seen]) if len(y2) else np.nan
    r["s2_cm"] = confusion_matrix(y2, eye, labels=[False, True])
    r["acc3"] = accuracy_score(y, pred)
    r["recall"] = [np.mean(pred[y == c] == c) if np.any(y == c) else np.nan for c in LABELS]
    r["cm3"] = confusion_matrix(y, pred, labels=LABELS)
    # hilalpy equa.py: positive = sightings outside the 'visible' region,
    # negative = non-sightings inside it; the two regions are complements.
    r["det_pos"] = rate(~det, seen)
    r["det_neg"] = rate(det, ~seen)
    r["eye_pos"] = rate(pred != 2, y == 2)
    r["eye_neg"] = rate(pred == 2, y != 2)
    return r


def calendar_flags(df):
    """Local MABIMS and Istanbul screens from HilalPy's MAlt, ArcL and MA columns."""
    alt, arcl, age = df["MAlt"], df["ArcL"], df["MA"]
    return {
        "MABIMS 1995": (((alt >= 2) & (arcl >= 3)) | (age >= 8)).to_numpy(),
        "MABIMS 2021": ((alt >= 3) & (arcl >= 6.4)).to_numpy(),
        "Istanbul 2016 (local)": ((alt >= 5) & (arcl >= 8)).to_numpy(),
    }


LAYOUT_HEAD = (f"{'Decision':30s} {'Acc':>7s} {'Whole+':>7s} {'Whole-':>7s} {'NE+':>7s} "
               f"{'NE-':>7s} {'OA+':>7s} {'OA-':>7s}")


def layout_line(label, y, visible, method):
    """Detection accuracy and hilalpy contradiction rates: whole sample, M == 'NE', M == 'OA'."""
    seen = y > 0
    out = [np.mean(visible == seen)]
    for m in (np.ones(len(y), bool), method == "NE", method == "OA"):
        out += [rate(~visible & m, seen & m)[0], rate(visible & m, ~seen & m)[0]]
    return f"{label:30s} " + " ".join(fmt(v) for v in out)


def print_calendar_comparison(df, pdet):
    """MABIMS and Istanbul against the Sharjah detection decision on the same rows."""
    y = df["cls"].to_numpy()
    method = df["M"].to_numpy()
    complete = df[["MAlt", "MA"]].notna().all(axis=1).to_numpy()
    print(f"Rows: {len(df):,}; missing MAlt: {int(df['MAlt'].isna().sum())}, missing MA: "
          f"{int(df['MA'].isna().sum())}; rows with both: {int(complete.sum()):,}, "
          f"classes 0/1/2 = {list(class_counts(y[complete]))}; "
          f"by M: NE {int((method[complete] == 'NE').sum())}, OA {int((method[complete] == 'OA').sum())}")
    print(LAYOUT_HEAD)
    flags = calendar_flags(df[complete])
    flags["Sharjah, P_det >= 0.50"] = pdet[complete] >= PDET
    flags["Sharjah, P_det >= 0.30"] = pdet[complete] >= 0.30
    for name, visible in flags.items():
        print(layout_line(name, y[complete], visible, method[complete]))
    print("Whole+/NE+/OA+: share of sightings called not visible (all, M == 'NE', M == 'OA'); "
          "Whole-/NE-/OA-: share of non-sightings called visible.")


def fmt(x, w=7):
    return f"{x:{w}.4f}" if np.isfinite(x) else f"{'-':>{w}s}"


def print_detail(title, r):
    n0, n1, n2 = r["counts"]
    print(f"\n{title}")
    print(f"  Stage 1 (seen vs not seen): n = {r['n']:,} (not seen {n0:,}, seen {n1 + n2:,}); "
          f"accuracy at P_det >= 0.50 = {r['s1_acc']:.4f}; ROC AUC = {fmt(r['s1_auc'], 0).strip()}")
    print(f"    confusion (rows true not seen / seen, columns predicted not seen / seen): "
          f"{r['s1_cm'].tolist()}")
    print(f"  Stage 2 (naked eye vs aided, seen rows): n = {r['s2_n']:,} (aided {n1:,}, "
          f"naked eye {n2:,}); accuracy at P_eye >= 0.60 = {fmt(r['s2_acc'], 0).strip()}; "
          f"ROC AUC = {fmt(r['s2_auc'], 0).strip()}")
    print(f"    confusion (rows true aided / naked eye, columns predicted aided / naked eye): "
          f"{r['s2_cm'].tolist()}")
    rec = " / ".join(fmt(v, 0).strip() for v in r["recall"])
    print(f"  Three-class (operational rule): accuracy = {r['acc3']:.4f}; recall 0/1/2 = {rec}; "
          f"predicted 0/1/2 = {list(r['pred_counts'])}")
    print(f"    confusion (rows true 0/1/2, columns predicted 0/1/2): {r['cm3'].tolist()}")
    print("  Contradiction rates (hilalpy equa.py definitions; naked-eye line adapted):")
    for label, pos, neg, pos_txt, neg_txt in [
            ("Detectability (P_det >= 0.50)", r["det_pos"], r["det_neg"],
             "sightings called not visible", "non-sightings called visible"),
            ("Naked eye (P_det >= 0.50 and P_eye >= 0.60)", r["eye_pos"], r["eye_neg"],
             "naked-eye sightings not called naked eye", "other rows called naked eye")]:
        print(f"    {label:44s} positive = {fmt(pos[0], 0).strip()} ({pos[1]:,}/{pos[2]:,} "
              f"{pos_txt}); negative = {fmt(neg[0], 0).strip()} ({neg[1]:,}/{neg[2]:,} {neg_txt})")


SUMMARY_HEAD_A = (f"{'Sample':40s} {'n':>5s} {'0/1/2':>18s} {'S1 acc':>7s} {'S1 AUC':>7s} "
                  f"{'S2 n':>5s} {'S2 acc':>7s} {'S2 AUC':>7s} {'Acc3':>7s}")
SUMMARY_HEAD_B = (f"{'Sample':40s} {'Rec0':>7s} {'Rec1':>7s} {'Rec2':>7s} "
                  f"{'Det+':>7s} {'Det-':>7s} {'Eye+':>7s} {'Eye-':>7s}")


def summary_line_a(label, r):
    return (f"{label:40s} {r['n']:5d} {str(list(r['counts'])):>18s} {fmt(r['s1_acc'])} "
            f"{fmt(r['s1_auc'])} {r['s2_n']:5d} {fmt(r['s2_acc'])} {fmt(r['s2_auc'])} "
            f"{fmt(r['acc3'])}")


def summary_line_b(label, r):
    return (f"{label:40s} " + " ".join(fmt(v) for v in r["recall"]) + " "
            + " ".join(fmt(r[k][0]) for k in ("det_pos", "det_neg", "eye_pos", "eye_neg")))


def print_summary(rows):
    print(SUMMARY_HEAD_A)
    for label, r in rows:
        print(summary_line_a(label, r))
    print()
    print(SUMMARY_HEAD_B)
    for label, r in rows:
        print(summary_line_b(label, r))


def print_by_source(df, pdet, peye):
    print(f"{'Source':8s} {'n':>5s} {'0/1/2':>16s} {'S1 acc':>7s} {'S1 AUC':>7s} {'S2 acc':>7s} "
          f"{'S2 AUC':>7s} {'Acc3':>7s} {'Det+':>7s} {'Det-':>7s} {'Eye+':>7s} {'Eye-':>7s}")
    src = df["src"].to_numpy()
    y = df["cls"].to_numpy()
    for name in sorted(np.unique(src), key=lambda s: PRIORITY.get(s, 99)):
        m = src == name
        r = evaluate(y[m], pdet[m], peye[m])
        print(f"{name:8s} {r['n']:5d} {str(list(r['counts'])):>16s} {fmt(r['s1_acc'])} "
              f"{fmt(r['s1_auc'])} {fmt(r['s2_acc'])} {fmt(r['s2_auc'])} {fmt(r['acc3'])} "
              + " ".join(fmt(r[k][0]) for k in ("det_pos", "det_neg", "eye_pos", "eye_neg")))


def rps(y, prob):
    """Mean ranked probability score, averaged over the two cut-points."""
    cum_p = np.cumsum(prob, axis=1)[:, :2]
    cum_o = (y[:, None] <= np.array([0, 1])).astype(float)
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / 2))


def sequential_probs(pdet, peye):
    return np.column_stack([1 - pdet, pdet * (1 - peye), pdet * peye])


# ---------------------------------------------------------------- secondary: ICOP
def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    x = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371 * np.arcsin(np.sqrt(x))


def match_icop(hic, archive):
    """One-to-one greedy matching, nearest first, same date and section."""
    c = hic[["row", "Ref No", "witness", "date", "O2", "Lat", "Long"]].merge(
        archive[["aid", "date", "O2", "latitude", "longitude", "insub", TARGET]],
        on=["date", "O2"])
    c["km"] = haversine_km(c["Lat"], c["Long"], c["latitude"], c["longitude"])
    # Ties (several reports at one site) are broken by content, not by file order;
    # the pairing among such reports is still arbitrary.
    c = c.sort_values(["km", "Ref No", "witness", "row", "aid"], kind="mergesort")
    used_h, used_a, keep = set(), set(), []
    for i, hr, ar in zip(c.index, c["row"].to_numpy(), c["aid"].to_numpy()):
        if hr in used_h or ar in used_a:
            continue
        used_h.add(hr)
        used_a.add(ar)
        keep.append(i)
    m = c.loc[keep].copy()
    m["tierA"] = ((m["Lat"] - m["latitude"]).abs() <= 0.1) & ((m["Long"] - m["longitude"]).abs() <= 0.1)
    m["within50"] = m["km"] <= 50
    return m


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True,
                        help="HilalPy 'Final 30.6.2023.csv' (Mendeley Data, DOI 10.17632/rs994bxgc4.1)")
    parser.add_argument("--icop", default=str(DEFAULT_ICOP),
                        help="Path to data/icop/crescent_features.csv")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS),
                        help="Folder with the saved Stage 1 and Stage 2 joblib models")
    parser.add_argument("--oof", default=str(DEFAULT_OOF),
                        help="Path to data/icop/two_stage_oof_predictions.csv")
    args = parser.parse_args()

    t1 = load_sibling("table1_criteria_comparison")
    h = load_hilalpy(args.input)
    archive, model_df = load_icop(args.icop)
    stage1, stage2 = t1.load_saved_models(args.models_dir)
    pdet_m, peye_m = load_oof(args.oof, model_df)

    def probs(df):
        X = df[FEATURES]
        return stage1.predict_proba(X)[:, 1], stage2.predict_proba(X)[:, 1]

    src_counts = h["src"].value_counts()
    print(f"HilalPy file: {args.input}")
    print(f"Rows: {len(h):,}; ICOP {int(src_counts.get('ICOP', 0)):,}; "
          f"non-ICOP {int((h['src'] != 'ICOP').sum()):,}")
    print("Sources: " + ", ".join(f"{k} {v:,}" for k, v in src_counts.items()))

    # ------------------------------------------------ 1. independent sample
    rows, primary, steps, in_archive = independent_sample(h, archive)
    names = ["1 Source is not ICOP", "2 Evening (O upper-cased == 'E')",
             "3 ArcV, ArcL and LT present", "4 ArcL <= 30 deg",
             "5 MA >= 0 or missing", "6 Not in our ICOP archive (same date, 0.5 deg)"]
    print("\n[1] Independent sample (non-ICOP evening first-crescent reports)")
    print(f"{'Filter':48s} {'Rows':>6s}")
    for name, n in zip(names, steps):
        print(f"{name:48s} {n:6,d}")
    if steps != EXPECTED_STEPS:
        raise ValueError(f"Filter counts {steps} differ from the specification {EXPECTED_STEPS}")
    print("Removed by filter 6: " + ", ".join(
        f"{k} {v}" for k, v in in_archive["src"].value_counts().items()))
    check_counts(rows, EXPECTED_ROW_LEVEL, "Row-level sample")
    check_counts(primary, EXPECTED_PRIMARY, "Primary sample")
    n_events = rows["event"].nunique()
    conflict = rows[rows["ev_ncls"] > 1]
    print(f"Row-level sample: {len(rows):,} rows, classes 0/1/2 = {list(class_counts(rows['cls']))}")
    print(f"Events: {n_events:,}; events with conflicting labels: {conflict['event'].nunique()} "
          f"({len(conflict)} rows)")
    print(f"De-duplicated sample (one row per event and class): {len(primary):,} rows, "
          f"classes 0/1/2 = {list(class_counts(primary['cls']))}")
    head = primary[primary["src"] != "AlR"].copy()
    check_counts(head, EXPECTED_HEADLINE, "Headline sample")
    print(f"Headline sample (de-duplicated, without AlR): {len(head):,} rows, "
          f"classes 0/1/2 = {list(class_counts(head['cls']))}")
    print(f"\n{'Source':8s} {'Row-level 0/1/2':>18s} {'n':>5s} {'De-dup. 0/1/2':>18s} {'n':>5s}")
    for name in sorted(PRIORITY, key=PRIORITY.get):
        a_ = rows[rows["src"] == name]
        b_ = primary[primary["src"] == name]
        print(f"{name:8s} {str(list(class_counts(a_['cls']))):>18s} {len(a_):5d} "
              f"{str(list(class_counts(b_['cls']))):>18s} {len(b_):5d}")
    neg = head[head["cls"] == 0]
    print(f"Headline non-sightings by M: NE {int((neg['M'] == 'NE').sum())}, "
          f"OA {int((neg['M'] == 'OA').sum())} (non-ICOP 'I' with M = 'NE' means not seen by "
          f"naked eye; whether optical aid was tried is not recorded)")
    tr_max = model_df[[ARCV, ARCL]].max()
    print(f"Headline rows outside the training range: ArcV > {tr_max[ARCV]:.2f} deg: "
          f"{int((head[ARCV] > tr_max[ARCV]).sum())}; ArcL > {tr_max[ARCL]:.2f} deg: "
          f"{int((head[ARCL] > tr_max[ARCL]).sum())}; negative ArcV, ArcL or LT "
          f"(where hilalpy's abs() would act): "
          f"{int((head[['ArcV', 'ArcL', 'LT']] < 0).any(axis=1).sum())}")

    # ------------------------------------------------ 2. Sharjah, detailed
    pdet_h, peye_h = probs(head)
    y_h = head["cls"].to_numpy()
    res_head = evaluate(y_h, pdet_h, peye_h)
    pdet_p, peye_p = probs(primary)
    y_p = primary["cls"].to_numpy()
    y_m = model_df["cls"].to_numpy()
    res_model = evaluate(y_m, pdet_m, peye_m)
    print(f"\n[2] Sharjah two-step criterion, saved final models ({args.models_dir})")
    print_detail("Headline independent sample (de-duplicated, non-ICOP, without AlR)", res_head)
    print_detail("ICOP reference: 4,470 training rows, out-of-fold probabilities "
                 "(script 04, five folds grouped by lunation)", res_model)

    # Wheel layout: Whole / Naked Eye (M == 'NE') / Optical Aided (M == 'OA').
    print("\nhilalpy equa.py layout, detectability decision (P_det >= 0.50), headline sample")
    print(f"{'Parameter':14s} {'Positive':>9s} {'(k/n)':>11s} {'Negative':>9s} {'(k/n)':>11s}")
    det = pdet_h >= PDET
    seen = y_h > 0
    for label, m in [("Whole", np.ones(len(y_h), bool)), ("Naked Eye", (head["M"] == "NE").to_numpy()),
                     ("Optical Aided", (head["M"] == "OA").to_numpy())]:
        pos, neg = rate(~det & m, seen & m), rate(det & m, ~seen & m)
        print(f"{label:14s} {fmt(pos[0], 9)} {f'{pos[1]}/{pos[2]}':>11s} "
              f"{fmt(neg[0], 9)} {f'{neg[1]}/{neg[2]}':>11s}")
    print(f"Non-ICOP negatives are mostly 'not seen by naked eye' (M = 'NE': "
          f"{int(((y_h == 0) & (head['M'] == 'NE').to_numpy()).sum())} of {int((y_h == 0).sum())}); "
          f"ICOP negatives are 'not seen by any means'.")

    # ------------------------------------------------ 3. sensitivity samples
    py = primary["Year"].to_numpy()
    pre_tel = py < 1610
    no_alr = (primary["src"] != "AlR").to_numpy()
    sel = {
        "Headline (de-duplicated, without AlR)": no_alr,
        "De-duplicated, with AlR": np.ones(len(primary), bool),
        "Row-level (not de-duplicated, with AlR)": None,
        "Headline, modern (Year >= 1610)": no_alr & ~pre_tel,
        "Headline, Year >= 1900": no_alr & (py >= 1900),
        "Headline, no conflicting-label events": no_alr & (primary["ev_ncls"] == 1).to_numpy(),
        "AlR only (Alrefay et al. 2018 tables)": ~no_alr,
    }
    pdet_r, peye_r = probs(rows)
    summary = []
    for label, m in sel.items():
        if m is None:
            summary.append((label, evaluate(rows["cls"].to_numpy(), pdet_r, peye_r)))
        else:
            summary.append((label, evaluate(y_p[m], pdet_p[m], peye_p[m])))
    summary.append(("ICOP out-of-fold reference (4,470)", res_model))
    print("\n[3] Sharjah (saved models) across samples; Det+/Det- and Eye+/Eye- are the "
          "contradiction rates")
    print_summary(summary)

    y_pt, pred_pt = y_p[pre_tel], operational(pdet_p[pre_tel], peye_p[pre_tel])
    eye_pt = y_pt == 2
    print(f"\nPre-telescopic (Year < 1610) de-duplicated rows: {int(pre_tel.sum())}, classes 0/1/2 = "
          f"{list(class_counts(y_pt))} (sightings only, so only recall is meaningful)")
    print(f"  Naked-eye sightings: {int(eye_pt.sum())}; called seen (P_det >= 0.50): "
          f"{np.mean(pdet_p[pre_tel][eye_pt] >= PDET):.4f}; called naked eye (operational rule): "
          f"{np.mean(pred_pt[eye_pt] == 2):.4f}")
    print(f"  Non-sightings: {int((y_pt == 0).sum())}; called seen: "
          f"{int((pdet_p[pre_tel][y_pt == 0] >= PDET).sum())}")

    # ------------------------------------------------ 4. by source
    print("\n[4] By source (Ref No prefix), de-duplicated sample with AlR")
    print_by_source(primary, pdet_p, peye_p)
    keep = py >= 1900
    print(f"\nBy source, de-duplicated sample excluding pre-1900 records (Year >= 1900; "
          f"{int(keep.sum()):,} rows, {int((~keep).sum())} removed)")
    print_by_source(primary[keep], pdet_p[keep], peye_p[keep])

    # ------------------------------------------------ 5. classical criteria
    # Revised Table 1 settings: Sharjah P_det < 0.50 -> 0, degenerate W >= 10 rule dropped;
    # on the ICOP rows Sharjah is scored out-of-fold, as in --sample model.
    preds_h = t1.classical_predictions(head)
    preds_h["Sharjah"] = t1.sharjah_classes(pdet_h, peye_h, t1.PDET_HIGH)
    preds_m = t1.classical_predictions(model_df)
    preds_m["Sharjah"] = t1.sharjah_classes(pdet_m, peye_m, t1.PDET_HIGH)
    print("\n[5] Revised Table 1 criteria (formulas imported from table1_criteria_comparison.py; "
          "Sharjah: P_det < 0.50 -> 0)")
    print("HilalPy inputs: ARCV = ArcV, DAZ = DAZ, ARCL = ArcL, W = W / 60 arcmin, lag = LT. "
          "ICOP columns are the sunset values of crescent_features.csv; Sharjah is out-of-fold there.")
    print(f"{'Criterion':24s} {'Group':16s} {'Headline: 3-class':>17s} {'Detect':>7s} {'Eye|seen':>8s} "
          f"{'MacF1':>6s} {'Predicted 0/1/2':>17s}   {'ICOP 4,470: 3-class':>19s} {'Detect':>7s} "
          f"{'Eye|seen':>8s} {'MacF1':>6s}")
    for name, (group, _) in t1.TABLE1.items():
        if name == "Cond-DAZ>=4-W>=10":
            continue
        sp, sm_ = t1.scores(y_h, preds_h[name]), t1.scores(y_m, preds_m[name])
        print(f"{name:24s} {group:16s} {sp[0]:17.4f} {sp[1]:7.4f} {sp[2]:8.4f} {sp[3]:6.4f} "
              f"{str(list(class_counts(preds_h[name]))):>17s}   {sm_[0]:19.4f} {sm_[1]:7.4f} "
              f"{sm_[2]:8.4f} {sm_[3]:6.4f}")
    print("Yallop, Odeh, Qureshi and Alrefay were calibrated on compilations that overlap this "
          "sample, so their scores here are not fully independent.")

    # ------------------------------------------------ 6. alternatives
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    # Median imputation and standardisation fitted on the 4,470 training rows.
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(model_df[FEATURES])
    Z_m, Z_h = prep.transform(model_df[FEATURES]), prep.transform(head[FEATURES])
    mnl = LogisticRegression(max_iter=4000, solver="lbfgs").fit(Z_m, y_m)
    # No constant column: the two thresholds act as the intercepts.
    po = OrderedModel(y_m, Z_m, distr="logit").fit(method="newton", maxiter=100, disp=False)
    if not po.mle_retvals.get("converged", True):
        raise RuntimeError("Proportional-odds model did not converge")
    # Two-step refit: script 04 pipelines, Stage 1 on all rows, Stage 2 on the seen rows.
    # This is the saved final model; stop if it is not.
    r1, r2 = t1.fit_on_rows(model_df)
    gap = max(np.max(np.abs(r1.predict_proba(head[FEATURES])[:, 1] - pdet_h)),
              np.max(np.abs(r2.predict_proba(head[FEATURES])[:, 1] - peye_h)))
    if not gap <= 1e-9:
        raise ValueError(f"The refitted two-step model differs from the saved one by up to {gap:.3g}")
    # Sequential model with common slopes, as in ordinal_model_comparison.py.
    m4 = load_sibling("ordinal_model_comparison").fit_m4(Z_m, y_m)

    def alt_probs(df, Z):
        X = df[FEATURES]
        return {
            "M1 multinomial": mnl.predict_proba(Z),
            "M2 prop. odds": po.model.predict(po.params, exog=Z),
            "M3 two-step refit": sequential_probs(r1.predict_proba(X)[:, 1], r2.predict_proba(X)[:, 1]),
            "M4 seq. common": m4(Z),
        }

    print("\n[6] Alternatives fitted on all 4,470 ICOP rows (ARCV, ARCL, lag; median-imputed, "
          "standardised)")
    print("M1 = multinomial logit; M2 = proportional-odds cumulative logit (statsmodels "
          "OrderedModel, logit, no constant); M3 = the two script 04 pipelines refitted on all 4,470 rows "
          f"(identical to the saved final model; largest difference {gap:.1e}); "
          "M4 = sequential model with common slopes.")
    header = (f"{'Model':20s} {'Rule':12s} {'Acc3':>7s} {'MacF1':>7s} {'AidR':>7s} {'LogLoss':>8s} "
              f"{'RPS':>7s} {'Predicted 0/1/2':>17s}")
    for title, df, Z, y in [(f"Headline independent sample (n = {len(head):,})", head, Z_h, y_h),
                            ("In-sample: 4,470 ICOP training rows", model_df, Z_m, y_m)]:
        print(f"\n{title}")
        print(header)
        for name, prob in alt_probs(df, Z).items():
            # The same two decision rules for every model.
            pdet = prob[:, 1] + prob[:, 2]
            rules = [("argmax", prob.argmax(axis=1)),
                     ("operational", operational(pdet, prob[:, 2] / pdet))]
            for rule, pred in rules:
                print(f"{name:20s} {rule:12s} {accuracy_score(y, pred):7.4f} "
                      f"{f1_score(y, pred, labels=LABELS, average='macro', zero_division=0):7.4f} "
                      f"{recall_score(y, pred, labels=[1], average='macro', zero_division=0):7.4f} "
                      f"{log_loss(y, prob, labels=LABELS):8.4f} {rps(y, prob):7.4f} "
                      f"{str(list(class_counts(pred))):>17s}")

    # ------------------------------------------------ 7. secondary: HilalPy ICOP
    hic = h[h["src"] == "ICOP"]
    m = match_icop(hic, archive)
    print("\n[7] Secondary: HilalPy ICOP rows (the same observing programme as our training "
          "data, so NOT independent)")
    print(f"HilalPy ICOP rows: {len(hic):,} (E {int((hic['O2'] == 'E').sum()):,}, "
          f"M {int((hic['O2'] == 'M').sum()):,}); dates {hic['date'].min()} to {hic['date'].max()}")
    print("Matching to our 6,092-row archive: same date and section, one-to-one, nearest first "
          "(pairing among reports at identical coordinates is arbitrary).")
    print(f"{'Tier':38s} {'Pairs':>6s} {'first/last crescent':>20s} {'In 4,470 subset':>16s}")
    for label, mk in [("A: |dLat| and |dLong| <= 0.1 deg", m["tierA"]), ("Within 50 km", m["within50"]),
                      ("Any distance, same date", np.ones(len(m), bool))]:
        mm = m[mk]
        fc = int((mm["O2"] == "E").sum())
        print(f"{label:38s} {len(mm):6,d} {f'{fc}/{len(mm) - fc}':>20s} {int(mm['insub'].sum()):16,d}")
    sub = archive[archive["insub"]]
    span = sub[(sub["date"] >= hic["date"].min()) & (sub["date"] <= hic["date"].max())]
    print(f"Our 4,470 subset rows inside the HilalPy ICOP date span: {len(span):,}; matched in tier A: "
          f"{int(m[m['tierA']]['insub'].sum()):,}; within 50 km: {int(m[m['within50']]['insub'].sum()):,}")
    # Geometry agreement: evening reports on the same date within 1 km.
    close = m[(m["O2"] == "E") & (m["km"] <= 1)].merge(
        h[["row", "ArcV", "ArcL", "LT"]], on="row").merge(
        archive[["aid", ARCV, ARCL, LAG]], on="aid")
    print(f"Geometry, evening pairs within 1 km (n = {len(close):,}): RMS / median absolute "
          f"difference, HilalPy minus ours")
    for label, a, b in [("ArcV (deg)", "ArcV", ARCV), ("ArcL (deg)", "ArcL", ARCL),
                        ("LT (min)", "LT", LAG)]:
        d = (close[a] - close[b]).dropna()
        print(f"  {label:11s} {np.sqrt(np.mean(d ** 2)):.3f} / {np.median(np.abs(d)):.3f} (n = {len(d):,})")
    ta = m[m["tierA"] & m["insub"]].merge(h[["row", "V", "M", "cls"]], on="row")
    ours = ta[TARGET].astype(int).to_numpy()
    vm = np.where(ta["V"] == "I", 0, np.where(ta["M"] == "OA", 1, 2))
    print(f"Label agreement with our labels on tier-A pairs in the subset (n = {len(ta):,}): "
          f"V/M rule {np.mean(vm == ours):.4f}; V/M rule with the ICOP NE-flag override "
          f"{np.mean(ta['cls'].to_numpy() == ours):.4f}; seen vs not seen "
          f"{np.mean((vm > 0) == (ours > 0)):.4f}")

    ice, _ = evening_filters(hic)
    near50 = ice["row"].isin(m.loc[m["within50"], "row"])
    near50_sub = ice["row"].isin(m.loc[m["within50"] & m["insub"], "row"])
    print(f"\nHilalPy ICOP evening rows after filters 2-5: {len(ice):,}; with a matched archive "
          f"partner within 50 km: {int(near50.sum()):,} (in the 4,470 subset: {int(near50_sub.sum()):,}); "
          f"without: {int((~near50).sum()):,}")
    sec = []
    for label, df in [("HilalPy ICOP evening, all", ice),
                      ("HilalPy ICOP evening, no matched partner", ice[~near50.to_numpy()])]:
        pd_, pe_ = probs(df)
        sec.append((label, evaluate(df["cls"].to_numpy(), pd_, pe_)))
    sec.append(("ICOP out-of-fold reference (4,470)", res_model))
    print_summary(sec)

    # ------------------------------------------------ 8. calendar criteria on HilalPy
    print("\n[8] Calendar criteria on the headline independent sample (HilalPy MAlt, ArcL, MA) "
          "and the Sharjah detection decision (saved final model, fully independent here)")
    print_calendar_comparison(head, pdet_h)

    # ------------------------------------------------ 9. Malaysian subset
    tk = (head["src"] == "TK").to_numpy()
    print(f"\n[9] Malaysian subset: Teluk Kemang (Ref No prefix TK; {head.loc[tk, 'Lat'].iloc[0]:.2f} N, "
          f"{head.loc[tk, 'Long'].iloc[0]:.2f} E; years {int(head.loc[tk, 'Year'].min())}-"
          f"{int(head.loc[tk, 'Year'].max())}), headline sample")
    print_detail(f"Sharjah, saved final model, {int(tk.sum())} Teluk Kemang records",
                 evaluate(y_h[tk], pdet_h[tk], peye_h[tk]))
    print_calendar_comparison(head[tk], pdet_h[tk])

    # ------------------------------------------------ 10. thresholds on other samples
    alr = (primary["src"] == "AlR").to_numpy()
    print_thresholds([
        ("ICOP out-of-fold (4,470)", y_m, pdet_m, peye_m, model_df["page_url"].to_numpy()),
        ("Headline (1,365)", y_h, pdet_h, peye_h, head["date"].to_numpy()),
        ("AlR only (405)", y_p[alr], pdet_p[alr], peye_p[alr], primary.loc[alr, "date"].to_numpy()),
    ])


def print_thresholds(samples, n_boot=2000):
    """Thresholds that each sample would choose for itself, against the adopted ones.

    The Youden-J and max-F1 rules of threshold_and_calibration_checks.py are applied to
    each sample's own probabilities (final model; out-of-fold for ICOP), with a 95 %
    cluster-bootstrap interval over lunations (ICOP) or evenings (HilalPy: all reports
    with the same date, which share almost the same geometry). For ICOP this
    reproduces section 2 of that script.
    """
    tc = load_sibling("threshold_and_calibration_checks")
    print("\n[10] Thresholds chosen on each sample itself (Youden J, max F1) against the adopted "
          f"0.50 (Stage 1) and 0.60 (Stage 2); value [95% interval], {n_boot} resamples of\nlunations (ICOP) or evenings (HilalPy)")
    print(f"{'Sample':26s} {'Stage':8s} {'n':>5s} {'Pos.':>6s} {'Mean P':>7s} "
          f"{'Youden threshold':>26s} {'max-F1 threshold':>26s} {'Acc adopted':>11s} "
          f"{'Acc Youden':>10s} {'J adopted':>9s} {'J max':>6s}")
    for name, y, pdet, peye, groups in samples:
        point = tc.threshold_summary(tc.stage_arrays(y, pdet, peye))
        boot = [tc.threshold_summary(tc.stage_arrays(y, pdet, peye, idx))
                for idx in tc.lunation_resamples(tc.rows_by_lunation(groups), n_boot, tc.BOOT_SEED)]
        for stage, (t, p) in tc.stage_arrays(y, pdet, peye).items():
            cut = {"Stage 1": PDET, "Stage 2": PEYE}[stage]
            thr = point[(stage, "youden_thr")]
            cell = {k: tc.fmt_ci(point[(stage, k)], [b[(stage, k)] for b in boot])
                    for k in ["youden_thr", "f1_thr"]}
            print(f"{name:26s} {stage:8s} {len(t):5d} {t.mean():6.3f} {p.mean():7.3f} "
                  f"{cell['youden_thr']:>26s} {cell['f1_thr']:>26s} {np.mean((p >= cut) == t):11.4f} "
                  f"{np.mean((p >= thr) == t):10.4f} {tc.j_at(t, p, cut):9.4f} "
                  f"{point[(stage, 'j_max')]:6.4f}")
    print("Pos. = share of positives (Stage 1: sightings; Stage 2: naked-eye sightings among them); "
          "Mean P = mean predicted probability.\nAcc Youden uses the sample's own Youden threshold, "
          "so it is optimistic. AlR Stage 1 is perfectly separated (J max = 1), so its Youden "
          "threshold is only one of many equivalent cuts.")


if __name__ == "__main__":
    main()
