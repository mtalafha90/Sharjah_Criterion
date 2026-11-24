#!/usr/bin/env python3
"""Data attrition and the meaning of negative reports (reviewer comment 7).

Comment 7 asks: "The loss from about 7,100 reports to about 6,100 and then
4,470 should also be shown step by step, with class counts at each step, and
it would help to check whether the dropped records differ from those kept,
for example by region, year or class." It also asks, after noting that in
Faid et al. (2024) almost all negative records were naked-eye-only attempts:
"Please report how many negatives had aids attempted, and exclude or adjust
for poor-sky reports."

Inputs, from data/icop: the scraped reports
(01), the clean and rejected reports (02), crescent_features.csv (03, then
03b, which changes lags but removes no rows) and the out-of-fold predictions
of 04 (used only as a check). The pipeline scripts are not imported; the few
rules needed are restated here.
  Part 1  records left after each step, with the class counts
          (0 not seen, 1 seen with an optical aid only, 2 seen by naked eye).
  Part 2  first-crescent reports lost in 03 against the 4,470 kept, by class,
          year and region. Chi-squared tests treat the records as
          independent; the 95% intervals come from a cluster bootstrap over
          lunations (page_url; 2,000 resamples, numpy default_rng(42)).
  Part 3  the means tried in the negative reports, read from the fixed ICOP
          sentence (each means: seen / not seen / not sought), and the
          reported sky and weather by class.
  Part 4  the two-stage model refitted after (A1) removing negatives under a
          totally cloudy sky or very dusty air / heavy fog, (A2) also removing
          negatives under a partly cloudy sky, (B0) keeping only negatives
          where at least one means was tried (some reports mark every means
          'not sought'), and (B) keeping only negatives where an optical aid
          was tried. Positives are never removed, so
          Stage 2 is unchanged. Removing negatives raises the share seen, so
          every model is also scored on the same rows, with the folds of the
          main evaluation (StratifiedGroupKFold, 5 folds, shuffle,
          random_state=42, groups = page_url).
"""
import argparse
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # the repository root
DEFAULT_DATA = REPO / "data" / "icop"
DEFAULT_REPORT = REPO / "models" / "two_stage_crescent_criterion_report.json"

FEATURES = ["relative_altitude_deg", "elongation_deg", "moon_lag_minutes"]
FEATURE_NAMES = ["ARCV", "ARCL", "lag"]
GROUP = "page_url"
LABEL = "vis_class_empirical"
CLASSES = [(0, "not seen"), (1, "aided"), (2, "naked eye")]
# Reference geometry: Sharjah, 15 June 2026 (second example map).
REFERENCE = np.array([[7.48, 7.52, 40.0]])
N_FOLDS = 5
SEED = 42

# Duplicate key of 02; it also identifies each clean report in crescent_features.csv.
KEY = ["page_url", "gregorian_date", "country", "city", "province", "obs_time_text", "raw_text"]
SEEN_COLS = ["seen_naked_eye", "seen_binocular", "seen_telescope", "seen_camera"]

# Date rules of 03: weekday names it strips, and the month names it knows.
WEEKDAYS_03 = {"الاثنين", "الثلاثاء", "الأربعاء", "الاربعاء", "الخميس", "الجمعة",
               "السبت", "الأحد", "الاحد"}
MONTHS_03 = {"يناير": 1, "فبراير": 2, "مارس": 3, "أبريل": 4, "ابريل": 4, "مايو": 5,
             "يونيو": 6, "يوليو": 7, "أغسطس": 8, "اغسطس": 8, "سبتمبر": 9, "أكتوبر": 10,
             "اكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12}
WEEKDAY_EN = {"السبت": "Saturday", "الأحد": "Sunday", "الإثنين": "Monday",
              "الاثنين": "Monday", "الثلاثاء": "Tuesday", "الأربعاء": "Wednesday",
              "الخميس": "Thursday", "الجمعة": "Friday"}

# ICOP categories, in English.
SKY_EN = {"صافية": "clear", "غائمة جزئيا": "partly cloudy", "غائمة كليا": "totally cloudy"}
WEATHER_EN = {"صافية جدا": "very clear", "صافية": "clear", "مغبرة أو ضباب": "dusty or foggy",
              "مغبرة جدا أو ضباب كثيف": "very dusty or heavy fog"}
HEADER_EN = {"لم تتم رؤية الهلال": "crescent not seen", "تمت رؤية الهلال": "crescent seen",
             "تم تصوير الهلال": "crescent imaged", "تمت الرؤية من قبل آخرين": "seen by others"}
# Means of observation, as worded in the fixed ICOP sentence.
MEANS = [("naked eye", "بالعين المجردة"), ("binoculars", "بالمنظار"),
         ("telescope", "بالمرقب"), ("camera", "باستخدام الكاميرا")]

ARABIA, MIDEAST, N_AFRICA, SS_AFRICA = ("Arabian Peninsula", "Levant, Iraq, Iran, Turkey",
                                        "North Africa", "Sub-Saharan Africa")
EUROPE, S_ASIA, SE_ASIA, AMERICAS = ("Europe", "South Asia", "SE and E Asia, Oceania",
                                     "Americas")
REGIONS = [ARABIA, MIDEAST, N_AFRICA, SS_AFRICA, EUROPE, S_ASIA, SE_ASIA, AMERICAS]
COUNTRIES = {
    "المملكة العربية السعودية": ("Saudi Arabia", ARABIA), "الإمارات": ("UAE", ARABIA),
    "عُمان": ("Oman", ARABIA), "البحرين": ("Bahrain", ARABIA), "الكويت": ("Kuwait", ARABIA),
    "قطر": ("Qatar", ARABIA), "اليمن": ("Yemen", ARABIA),
    "الأردن": ("Jordan", MIDEAST), "فلسطين": ("Palestine", MIDEAST),
    "العراق": ("Iraq", MIDEAST), "سوريا": ("Syria", MIDEAST), "لبنان": ("Lebanon", MIDEAST),
    "إيران": ("Iran", MIDEAST), "تركيا": ("Turkey", MIDEAST),
    "أذربيجان": ("Azerbaijan", MIDEAST),
    "الجزائر": ("Algeria", N_AFRICA), "المغرب": ("Morocco", N_AFRICA),
    "ليبيا": ("Libya", N_AFRICA), "تونس": ("Tunisia", N_AFRICA), "مصر": ("Egypt", N_AFRICA),
    "موريتانيا": ("Mauritania", N_AFRICA), "السودان": ("Sudan", N_AFRICA),
    "نيجيريا": ("Nigeria", SS_AFRICA), "غانا": ("Ghana", SS_AFRICA),
    "جنوب أفريقيا": ("South Africa", SS_AFRICA), "تنزانيا": ("Tanzania", SS_AFRICA),
    "السنغال": ("Senegal", SS_AFRICA), "كاميرون": ("Cameroon", SS_AFRICA),
    "الصومال": ("Somalia", SS_AFRICA), "غامبيا": ("Gambia", SS_AFRICA),
    "موريشيوس": ("Mauritius", SS_AFRICA), "كينيا": ("Kenya", SS_AFRICA),
    "مالي": ("Mali", SS_AFRICA), "أوغندا": ("Uganda", SS_AFRICA), "تشاد": ("Chad", SS_AFRICA),
    "المملكة المتحدة": ("United Kingdom", EUROPE), "ألمانيا": ("Germany", EUROPE),
    "المجر": ("Hungary", EUROPE), "البوسنة و الهرسك": ("Bosnia and Herzegovina", EUROPE),
    "فرنسا": ("France", EUROPE), "إيطاليا": ("Italy", EUROPE), "ألبانيا": ("Albania", EUROPE),
    "إسبانيا": ("Spain", EUROPE), "النرويج": ("Norway", EUROPE),
    "هولندا": ("Netherlands", EUROPE), "النمسا": ("Austria", EUROPE),
    "كرواتيا": ("Croatia", EUROPE), "بلجيكا": ("Belgium", EUROPE),
    "البرتغال": ("Portugal", EUROPE),
    "بنغلادش": ("Bangladesh", S_ASIA), "سريلانكا": ("Sri Lanka", S_ASIA),
    "باكستان": ("Pakistan", S_ASIA), "نيبال": ("Nepal", S_ASIA), "الهند": ("India", S_ASIA),
    "المالديف": ("Maldives", S_ASIA),
    "أندونيسيا": ("Indonesia", SE_ASIA), "ماليزيا": ("Malaysia", SE_ASIA),
    "بروناي": ("Brunei", SE_ASIA), "الفليبين": ("Philippines", SE_ASIA),
    "اليابان": ("Japan", SE_ASIA), "هونغ كونغ": ("Hong Kong", SE_ASIA),
    "أستراليا": ("Australia", SE_ASIA), "نيوزيلندا": ("New Zealand", SE_ASIA),
    "بولينيزيا الفرنسية": ("French Polynesia", SE_ASIA),
    "الولايات المتحدة": ("United States", AMERICAS), "كندا": ("Canada", AMERICAS),
    "ترينيداد وتوباغو": ("Trinidad and Tobago", AMERICAS),
    "فنزويلا": ("Venezuela", AMERICAS), "كولومبيا": ("Colombia", AMERICAS),
}
YEAR_BINS = [(2009, 2012), (2013, 2016), (2017, 2020), (2021, 2023), (2024, 2026)]
LAT_BINS = [(0, 15), (15, 30), (30, 45), (45, 90)]


# ---------------------------------------------------------------- helpers
def flag(s, values):
    return s.astype(str).str.strip().str.lower().isin(values)


def class_02(df):
    """Class by the rule of 02: NaN where 02 rejects the report."""
    seen = {c: flag(df[c], {"true", "1", "yes"}) for c in SEEN_COLS}
    aid = seen["seen_binocular"] | seen["seen_telescope"] | seen["seen_camera"]
    not_seen = flag(df["seen_any"], {"false", "0", "no"})
    return pd.Series(np.select([seen["seen_naked_eye"], aid, not_seen], [2, 1, 0], np.nan),
                     index=df.index)


def date_03(text):
    """Gregorian date as parsed by 03 (None where 03 skips the report)."""
    s = re.sub(r"\s+", " ", str(text)).strip()
    parts = [p for p in s.split() if p not in WEEKDAYS_03]
    if len(parts) < 3:
        m = re.search(r"(\d{1,2})\s+([^\d]+?)\s+(\d{4})", s)
        if not m:
            return None
        day, month, year = int(m.group(1)), m.group(2).strip(), int(m.group(3))
    else:
        try:
            day, month, year = int(parts[0]), parts[1], int(parts[2])
        except ValueError:
            return None
    try:
        return date(year, MONTHS_03[month], day)
    except (KeyError, ValueError):
        return None


def keys(df):
    return list(zip(*[df[c].fillna("").astype(str) for c in KEY]))


def means_status(text, word):
    """seen / not seen / not sought for one means, from the fixed ICOP sentence."""
    sentence = str(text).split(" وقال ")[0]  # the observer's own comment starts here
    found = [status for status, phrase in [("not seen", f"لم تتم رؤية الهلال {word}"),
                                           ("not sought", f"لم يتم تحري الهلال {word}"),
                                           ("seen", f"تمت رؤية الهلال {word}")]
             if phrase in sentence]
    return found[0] if len(found) == 1 else ("not stated" if not found else "conflicting")


def fit_logistic(X, y):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, solver="lbfgs")),
    ]).fit(X, y)


def out_of_fold(X, y3, folds, use):
    """Out-of-fold P_det and P_eye for every row; models see only training rows with use=True."""
    p_det, p_eye = np.full(len(y3), np.nan), np.full(len(y3), np.nan)
    for train_idx, test_idx in folds:
        train_idx = train_idx[use[train_idx]]
        seen = y3[train_idx] > 0
        stage1 = fit_logistic(X[train_idx], seen.astype(int))
        stage2 = fit_logistic(X[train_idx][seen], (y3[train_idx][seen] == 2).astype(int))
        p_det[test_idx] = stage1.predict_proba(X[test_idx])[:, 1]
        p_eye[test_idx] = stage2.predict_proba(X[test_idx])[:, 1]
    return p_det, p_eye


def coefficients(model):
    """Standardised coefficients, and the same per degree / per minute of the raw inputs."""
    clf, scaler = model.named_steps["clf"], model.named_steps["scaler"]
    std = np.r_[clf.intercept_, clf.coef_.ravel()]
    per_unit = clf.coef_.ravel() / scaler.scale_
    raw = np.r_[clf.intercept_ - (per_unit * scaler.mean_).sum(), per_unit]
    return std, raw


def band(p_det, p_eye):
    """Operational band of Eq. 11."""
    if p_det < 0.30:
        return "E"
    if p_det < 0.50:
        return "D"
    return "C" if p_eye < 0.30 else ("B" if p_eye < 0.60 else "A")


def resample_weights(n_groups, n_boot, seed=SEED):
    """Times each lunation is drawn in each cluster-bootstrap resample."""
    draws = np.random.default_rng(seed).integers(0, n_groups, size=(n_boot, n_groups))
    return np.array([np.bincount(d, minlength=n_groups) for d in draws])


def ci(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return f"[{lo:.4f}, {hi:.4f}]"


def class_counts(label, groups):
    c = label.value_counts()
    return [len(label)] + [int(c.get(k, 0)) for k, _ in CLASSES] + [int(label.isna().sum()),
                                                                     groups.nunique()]


def print_flow(lines):
    print(f"{'Step':60s} {'Records':>8s} {'Not seen':>9s} {'Aided':>7s} {'Naked eye':>10s} "
          f"{'Missing':>8s} {'Lunations':>10s}")
    for step, (n, c0, c1, c2, miss, lun) in lines:
        print(f"{step:60s} {n:8,d} {c0:9,d} {c1:7,d} {c2:10,d} {miss:8,d} {lun:10,d}")


def compare_lost_kept(title, category, lost, order=None):
    """Kept and lost counts per category, share lost, and a chi-squared test."""
    table = pd.crosstab(category, lost).reindex(columns=[False, True], fill_value=0)
    if order is not None:
        table = table.reindex([c for c in order if c in table.index])
    chi2, p, dof, _ = chi2_contingency(table.to_numpy())
    print(f"\n{title}")
    print(f"{'':28s} {'Kept':>6s} {'Lost':>6s} {'% lost':>7s} {'Kept %':>7s} {'Lost %':>7s}")
    for cat, (k, l) in table.iterrows():
        print(f"{str(cat):28s} {k:6,d} {l:6,d} {100 * l / (k + l):7.1f} "
              f"{100 * k / table[False].sum():7.1f} {100 * l / table[True].sum():7.1f}")
    print(f"Chi-squared (records independent) = {chi2:.1f}, df = {dof}, p = {p:.2g}")


# ------------------------------------------------------------------ parts
def part1_attrition(raw, clean, rejected, feats):
    print("=" * 100)
    print("PART 1. Records left after each step (class from the rule of 02; "
          "lunation = ICOP report page)")
    print("=" * 100)

    raw = raw.copy()
    raw["label"] = class_02(raw)
    no_statement = raw["label"].isna() & ~flag(raw["seen_any"], {"true", "false"})
    seen_no_means = raw["label"].isna() & flag(raw["seen_any"], {"true"})
    not_entry = ~(raw["raw_text"].fillna("").str.contains("وقت الرصد")
                  | raw["obs_time_text"].fillna("").str.strip().ne(""))
    rejected_mask = raw["label"].isna() | not_entry
    if rejected_mask.sum() != len(rejected) or len(raw) - rejected_mask.sum() < len(clean):
        raise ValueError("The rejection rules of 02 do not reproduce the rejected file")

    # Match every feature row to its clean report (the features keep the clean order).
    position = {k: i for i, k in enumerate(keys(clean))}
    matched = [position.get(k) for k in keys(feats)]
    if None in matched or any(a >= b for a, b in zip(matched, matched[1:])):
        raise ValueError("crescent_features.csv rows do not map one-to-one onto the clean file")
    clean = clean.copy()
    clean["kept_03"] = False
    clean.loc[clean.index[matched], "kept_03"] = True
    clean["date_ok"] = clean["gregorian_date"].map(date_03).notna()
    # 03 keeps no log, so a lost report whose date parses is put down to its place when no
    # report from the same place survived (03 skips a place it cannot geocode or time-zone).
    place = list(zip(clean["city"].fillna(""), clean["province"].fillna(""),
                     clean["country"].fillna("")))
    geocoded = set(p for p, k in zip(place, clean["kept_03"]) if k)
    clean["place_ok"] = [p in geocoded for p in place]
    clean["reason_03"] = np.select(
        [clean["kept_03"], ~clean["date_ok"], ~clean["place_ok"]],
        ["kept", "date not parsed", "place not geocoded"], "other")

    first = feats["section_type"] == "first_crescent"
    not_before = feats["obs_relative_to_sunset"] != "before_sunset"
    labelled = feats[LABEL].notna()
    model_rows = feats[first & not_before & labelled]

    def counts(df, label=None):
        return class_counts(df[LABEL] if label is None else label, df[GROUP])

    kept_02 = raw[~rejected_mask]
    lost_03 = clean[~clean["kept_03"]]
    lines = [
        ("Scraped reports (01)", counts(raw, raw["label"])),
        ("  removed by 02: no statement whether the crescent was seen",
         counts(raw[no_statement], raw.loc[no_statement, "label"])),
        ("  removed by 02: reported seen, but no means marked as seen",
         counts(raw[seen_no_means], raw.loc[seen_no_means, "label"])),
        ("  removed by 02: text not an observation entry",
         counts(raw[not_entry & ~raw["label"].isna()], raw.loc[not_entry & ~raw["label"].isna(), "label"])),
        ("After the rejection rules of 02", counts(kept_02, kept_02["label"])),
        ("Clean reports, duplicates removed (02)", counts(clean)),
        ("  removed by 03: date not parsed",
         counts(lost_03[lost_03["reason_03"] == "date not parsed"])),
        ("  removed by 03: place not geocoded",
         counts(lost_03[lost_03["reason_03"] == "place not geocoded"])),
        ("  removed by 03: other (time zone, ephemeris)",
         counts(lost_03[lost_03["reason_03"] == "other"])),
        ("Feature table (03; 03b removes no rows)", counts(feats)),
        ("  removed: last-crescent (morning) section", counts(feats[~first])),
        ("First-crescent section", counts(feats[first])),
        ("  removed: observed before sunset", counts(feats[first & ~not_before])),
        ("First crescent, not before sunset", counts(feats[first & not_before])),
        ("  removed: class label missing", counts(feats[first & not_before & ~labelled])),
        ("Modelling subset", counts(model_rows)),
    ]
    print_flow(lines)
    n_dup = len(kept_02) - len(clean)
    print(f"Duplicates removed by 02: {n_dup}. Reports rejected by 02 have no class label "
          f"(listed as missing).")

    weekday = clean["gregorian_date"].str.split().str[0].map(WEEKDAY_EN).fillna("other")
    print("\nReports removed by 03, by weekday of the Gregorian date "
          "(03 strips the weekday 'الاثنين', but ICOP spells Monday 'الإثنين')")
    tab = pd.crosstab(weekday.rename("Weekday"), clean["reason_03"]).reindex(
        index=["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        columns=["kept", "date not parsed", "place not geocoded"], fill_value=0)
    tab.columns.name = None
    print(tab.to_string())
    lost_places = lost_03[lost_03["reason_03"] == "place not geocoded"]
    per_place = lost_places.groupby(["city", "province", "country"], dropna=False).size()
    print(f"Places not geocoded (inferred: the date parsed, but no report from the place reached "
          f"the feature table): {len(per_place)} distinct (city, province, country) entries, "
          f"{len(lost_places)} reports (largest single place: {per_place.max()} reports); by country: "
          + ", ".join(f"{COUNTRIES[c][0]} {n}" for c, n in lost_places["country"].value_counts().items()))

    # Same steps for the reports that the design admits (first crescent, not before sunset).
    print("\nThe same steps for first-crescent reports not made before sunset "
          "(the losses that are not by design)")
    elig_raw = (raw["section_type"] == "first_crescent") & (raw["obs_relative_to_sunset"] != "before_sunset")
    elig_clean = (clean["section_type"] == "first_crescent") & (clean["obs_relative_to_sunset"] != "before_sunset")
    lines = [
        ("Scraped reports (01)", counts(raw[elig_raw], raw.loc[elig_raw, "label"])),
        ("  removed by 02 (no usable class)",
         counts(raw[elig_raw & rejected_mask], raw.loc[elig_raw & rejected_mask, "label"])),
        ("Clean reports (02)", counts(clean[elig_clean])),
        ("  removed by 03: date not parsed (Monday)",
         counts(clean[elig_clean & (clean["reason_03"] == "date not parsed")])),
        ("  removed by 03: place not geocoded",
         counts(clean[elig_clean & (clean["reason_03"] == "place not geocoded")])),
        ("Modelling subset", counts(clean[elig_clean & clean["kept_03"]])),
    ]
    print_flow(lines)
    return clean


def part2_lost_vs_kept(clean, feats, n_boot):
    print("\n" + "=" * 100)
    print("PART 2. First-crescent reports (not before sunset) lost in 03 against those kept")
    print("=" * 100)
    elig = clean[(clean["section_type"] == "first_crescent")
                 & (clean["obs_relative_to_sunset"] != "before_sunset")
                 & clean[LABEL].notna()].copy()
    lost = ~elig["kept_03"]
    if (~lost).sum() != 4470:
        raise ValueError("The kept reports are not the 4,470 modelling records")
    by_lunation = elig.groupby(GROUP)["kept_03"].agg(["sum", "size"])
    whole = by_lunation["sum"] == 0
    all_monday = elig["reason_03"].eq("date not parsed").groupby(elig[GROUP]).all()
    print(f"Reports: kept {(~lost).sum():,}, lost {lost.sum():,} "
          f"(date not parsed {(elig['reason_03'] == 'date not parsed').sum():,}, "
          f"place not geocoded {(elig['reason_03'] == 'place not geocoded').sum():,}).")
    print(f"Lunations: {len(by_lunation)} with eligible reports; "
          f"{(by_lunation['sum'] == by_lunation['size']).sum()} kept whole, "
          f"{((by_lunation['sum'] > 0) & (by_lunation['sum'] < by_lunation['size'])).sum()} partly lost, "
          f"{whole.sum()} lost whole ({(whole & all_monday).sum()} of them because every report "
          f"was dated on a Monday).")

    # Class: shares among kept and lost, with cluster-bootstrap intervals for the difference.
    label = elig[LABEL].astype(int).to_numpy()
    codes, uniq = pd.factorize(elig[GROUP])
    counts = np.zeros((len(uniq), 2, len(CLASSES)))
    np.add.at(counts, (codes, lost.to_numpy().astype(int), label), 1)
    weights = resample_weights(len(uniq), n_boot)
    boot = np.einsum("bg,gkc->bkc", weights, counts)
    boot_share = boot / boot.sum(axis=2, keepdims=True)
    share = counts.sum(axis=0) / counts.sum(axis=(0, 2))[:, None]
    by_reason = pd.crosstab(elig[LABEL].astype(int), elig["reason_03"])
    print(f"\nClass shares (%); difference lost minus kept with a 95% cluster-bootstrap interval "
          f"({n_boot:,} resamples of {len(uniq)} lunations)")
    print(f"{'Class':10s} {'Kept':>6s} {'Kept %':>7s} {'Lost':>5s} {'Lost %':>7s} "
          f"{'Monday':>7s} {'Geocode':>8s} {'Diff (pp)':>10s} {'95% interval (pp)':>20s}")
    for k, name in CLASSES:
        d = 100 * (boot_share[:, 1, k] - boot_share[:, 0, k])
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{name:10s} {int(counts[:, 0, k].sum()):6,d} {100 * share[0, k]:7.1f} "
              f"{int(counts[:, 1, k].sum()):5,d} {100 * share[1, k]:7.1f} "
              f"{by_reason.loc[k].get('date not parsed', 0):7,d} "
              f"{by_reason.loc[k].get('place not geocoded', 0):8,d} "
              f"{100 * (share[1, k] - share[0, k]):10.1f} {f'[{lo:.1f}, {hi:.1f}]':>20s}")
    chi2, p, dof, _ = chi2_contingency(counts.sum(axis=0))
    print(f"Chi-squared (records independent) = {chi2:.1f}, df = {dof}, p = {p:.2g}")

    year = elig["gregorian_date"].str.extract(r"(\d{4})")[0].astype(int)
    year_bin = year.map(lambda y: next(f"{a}-{b}" for a, b in YEAR_BINS if a <= y <= b))
    compare_lost_kept("By Gregorian year of the report (the hijri_year column of the scraped files is empty)",
                      year_bin, lost, [f"{a}-{b}" for a, b in YEAR_BINS])
    country = elig["country"].map(lambda c: COUNTRIES[c][0])
    region = elig["country"].map(lambda c: COUNTRIES[c][1])
    compare_lost_kept("By region (from the country)", region, lost, REGIONS)
    top = country.value_counts().index[:10]
    compare_lost_kept("Ten countries with most reports", country.where(country.isin(top), "other countries"),
                      lost, list(top) + ["other countries"])
    # Lost reports have no coordinates; Monday reports take those of the same place in the kept rows.
    place = list(zip(elig["city"].fillna(""), elig["province"].fillna(""), elig["country"]))
    site = feats.assign(place=list(zip(feats["city"].fillna(""), feats["province"].fillna(""),
                                       feats["country"].fillna(""))))
    lat_of = site.groupby("place")["latitude"].first()
    lat = pd.Series([lat_of.get(p, np.nan) for p in place], index=elig.index).abs()
    lat_bin = lat.map(lambda v: next((f"{a}-{b} deg" for a, b in LAT_BINS if a <= v < b), np.nan)
                      if pd.notna(v) else np.nan)
    has_lat = lat_bin.notna()
    compare_lost_kept(f"By absolute latitude (lost reports placed by the same site among the kept; "
                      f"{(has_lat & lost).sum()} of {lost.sum()} lost reports placed)",
                      lat_bin[has_lat], lost[has_lat], [f"{a}-{b} deg" for a, b in LAT_BINS])


def part3_negatives(df):
    print("\n" + "=" * 100)
    print("PART 3. Means tried in the negative reports, and sky and weather by class")
    print("=" * 100)
    y3 = df[LABEL].to_numpy()
    neg = y3 == 0
    for name, word in MEANS:
        df[f"status_{name}"] = df["raw_text"].map(lambda t, w=word: means_status(t, w))
    attempted_cols = ["attempted_naked_eye", "attempted_binocular", "attempted_telescope",
                      "attempted_camera"]
    agree = np.all([flag(df[c], {"true"}).to_numpy() == df[f"status_{n}"].isin(["seen", "not seen"]).to_numpy()
                    for (n, _), c in zip(MEANS, attempted_cols)], axis=0)
    print(f"Each report states every means as seen, not seen or not sought. The status read here "
          f"agrees with the attempted_* columns of 01 for {agree.sum():,} of {len(df):,} records.")

    print(f"\nNegative reports (N = {neg.sum():,}): status of each means")
    print(f"{'Means':12s} {'Tried, not seen':>16s} {'Not sought':>11s} {'Not stated':>11s} "
          f"{'Other':>6s} {'% tried':>8s}")
    optical = np.zeros(len(df), bool)
    for name, _ in MEANS:
        s = df.loc[neg, f"status_{name}"]
        tried = (s == "not seen").sum()
        print(f"{name:12s} {tried:16,d} {(s == 'not sought').sum():11,d} {(s == 'not stated').sum():11,d} "
              f"{(~s.isin(['not seen', 'not sought', 'not stated'])).sum():6,d} "
              f"{100 * tried / neg.sum():8.1f}")
        if name != "naked eye":
            optical |= df[f"status_{name}"].isin(["seen", "not seen"]).to_numpy()
    print(f"{'any optical':12s} {(optical & neg).sum():16,d} {'':11s} {'':11s} {'':6s} "
          f"{100 * (optical & neg).sum() / neg.sum():8.1f}")

    eye = df["status_naked eye"].isin(["seen", "not seen"]).to_numpy()
    stated = np.all([df[f"status_{n}"].isin(["seen", "not seen", "not sought"]) for n, _ in MEANS], axis=0)
    pattern = pd.Series(np.select(
        [~stated, eye & ~optical, eye & optical, ~eye & optical],
        ["means not stated", "naked eye only", "naked eye and optical aid", "optical aid only"],
        "nothing tried (all not sought)"), index=df.index)
    order = ["naked eye only", "naked eye and optical aid", "optical aid only",
             "nothing tried (all not sought)", "means not stated"]
    print("\nMeans tried, by class: n (% of class)")
    print(f"{'Means tried':32s} " + " ".join(f"{name:>15s}" for _, name in CLASSES))
    for pat in order:
        print(f"{pat:32s} " + " ".join(
            f"{(pattern[y3 == k] == pat).sum():6,d} ({100 * (pattern[y3 == k] == pat).mean():5.1f})"
            for k, _ in CLASSES))
    print(f"Negatives with an optical aid tried: {(optical & neg).sum():,} of {neg.sum():,} "
          f"({100 * (optical & neg).sum() / neg.sum():.1f}%); naked eye only: "
          f"{((pattern == 'naked eye only') & neg).sum():,} "
          f"({100 * ((pattern == 'naked eye only') & neg).sum() / neg.sum():.1f}%). "
          f"Faid et al. (2024): 3,013 of 3,023 negatives naked-eye only (99.7%).")

    header = df["raw_text"].str.extract(r"وقت الرصد:[^.]*\.\s*(.+?)\s*:\s*ذكر")[0].map(HEADER_EN).fillna("other")
    print("\nHeading of the report (ICOP's own summary line), by class")
    tab = pd.crosstab(header.rename("Heading"), pd.Series(y3, index=df.index).map(dict(CLASSES)))
    tab = tab.reindex(columns=[n for _, n in CLASSES], fill_value=0)
    tab.columns.name = None
    print(tab.to_string())

    sky = df["sky_cloud_text"].map(SKY_EN).fillna("not stated")
    weather = df["weather_text"].map(WEATHER_EN).fillna("not stated")
    for title, cat, order_ in [("Sky (cloud cover)", sky, ["clear", "partly cloudy", "totally cloudy", "not stated"]),
                               ("Weather (transparency)", weather,
                                ["very clear", "clear", "dusty or foggy", "very dusty or heavy fog", "not stated"])]:
        print(f"\n{title}, by class: n (% of class), and % not seen within the category")
        print(f"{'Category':26s} " + " ".join(f"{name:>15s}" for _, name in CLASSES) + f" {'% not seen':>11s}")
        for c in order_:
            in_c = (cat == c).to_numpy()
            if in_c.sum() == 0:
                continue
            print(f"{c:26s} " + " ".join(
                f"{(in_c & (y3 == k)).sum():6,d} ({100 * (in_c & (y3 == k)).sum() / (y3 == k).sum():5.1f})"
                for k, _ in CLASSES) + f" {100 * (in_c & neg).sum() / in_c.sum():11.1f}")

    # Sky groups used by the sensitivity variants of Part 4.
    poor = ((sky == "totally cloudy") | (weather == "very dusty or heavy fog")).to_numpy()
    partly = (~poor) & (sky == "partly cloudy").to_numpy()
    sky_group = np.select([poor, partly], ["poor: totally cloudy or very dusty/heavy fog",
                                           "partly cloudy (otherwise not poor)"],
                          "clear sky, not very dusty/heavy fog")
    groups = ["poor: totally cloudy or very dusty/heavy fog", "partly cloudy (otherwise not poor)",
              "clear sky, not very dusty/heavy fog"]
    print("\nSky groups used in Part 4, by class (n), and the means tried in the negatives")
    print(f"{'Sky group':46s} " + " ".join(f"{name:>10s}" for _, name in CLASSES)
          + " | negatives: " + ", ".join(["eye only", "eye+optical", "optical only", "nothing"]))
    for g in groups:
        in_g = sky_group == g
        print(f"{g:46s} " + " ".join(f"{(in_g & (y3 == k)).sum():10,d}" for k, _ in CLASSES)
              + " | " + ", ".join(f"{((pattern.to_numpy() == p) & in_g & neg).sum():,}" for p in order[:4]))
    return poor, partly, optical, eye | optical


def part4_sensitivity(df, poor, partly, optical, tried, oof_check, report, n_boot):
    print("\n" + "=" * 100)
    print("PART 4. Sensitivity of the model to the negative reports (Stage 1; Stage 2 is unchanged)")
    print("=" * 100)
    X = df[FEATURES].to_numpy(float)
    y3 = df[LABEL].to_numpy()
    groups = df[GROUP].to_numpy()
    neg = y3 == 0
    seen = y3 > 0
    variants = [
        ("Base", "all 4,470 records", np.ones(len(df), bool)),
        ("A1", "no negatives under a totally cloudy sky or very dusty air/heavy fog", ~(neg & poor)),
        ("A2", "as A1, and no negatives under a partly cloudy sky", ~(neg & (poor | partly))),
        ("B0", "only negatives where at least one means was tried", ~neg | tried),
        ("B","only negatives where an optical aid was tried", ~neg | optical),
    ]
    print(f"{'Variant':8s} {'Records':>8s} {'Negatives':>10s} {'Lunations':>10s} {'% seen':>7s}  Definition")
    for name, text, m in variants:
        print(f"{name:8s} {m.sum():8,d} {(m & neg).sum():10,d} {len(set(groups[m])):10d} "
              f"{100 * seen[m].mean():7.1f}  {text}")

    # (a) Each variant with its own grouped folds, as in the main evaluation.
    print("\n(a) Out-of-fold scores, each variant on its own rows with its own grouped folds "
          "(StratifiedGroupKFold 5, shuffle, random_state=42, stratified on the three classes)")
    print(f"{'Variant':8s} {'S1 acc':>7s} {'S1 AUC':>7s} {'S1 majority':>12s} "
          f"{'S2 acc':>7s} {'S2 AUC':>7s} {'S2 majority':>12s}")
    for name, _, m in variants:
        idx = np.flatnonzero(m)
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        folds = list(cv.split(X[idx], y3[idx], groups=groups[idx]))
        p_det, p_eye = out_of_fold(X[idx], y3[idx], folds, np.ones(len(idx), bool))
        s, s2 = seen[idx], y3[idx][seen[idx]] == 2
        print(f"{name:8s} {accuracy_score(s, p_det >= 0.5):7.4f} {roc_auc_score(s, p_det):7.4f} "
              f"{max(s.mean(), 1 - s.mean()):12.4f} "
              f"{accuracy_score(s2, p_eye[s] >= 0.5):7.4f} {roc_auc_score(s2, p_eye[s]):7.4f} "
              f"{max(s2.mean(), 1 - s2.mean()):12.4f}")
        if name == "Base":
            diff = max(np.abs(p_det - oof_check["p_det"]).max(), np.abs(p_eye - oof_check["p_eye"]).max())
            base_check = f"Base reproduces two_stage_oof_predictions.csv (max abs difference {diff:.1e})."
    print(base_check)

    # (b) Like with like: the main evaluation's folds, every model scored on every row set.
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(cv.split(X, y3, groups=groups))
    p_det = {name: out_of_fold(X, y3, folds, m)[0] for name, _, m in variants}
    print("\n(b) Like with like: folds of the main evaluation (on all 4,470 records); each model is "
          "trained on its own rows within the training folds and scored on the same test rows")
    print(f"{'Scored on':10s} {'% seen':>7s} {'Model':>6s} {'S1 acc':>7s} {'S1 AUC':>7s} "
          f"{'Mean P_det':>11s}")
    for rows_name, _, rows in variants:
        for name, _, _ in variants:
            p = p_det[name][rows]
            print(f"{rows_name:10s} {100 * seen[rows].mean():7.1f} {name:>6s} "
                  f"{accuracy_score(seen[rows], p >= 0.5):7.4f} {roc_auc_score(seen[rows], p):7.4f} "
                  f"{p.mean():11.4f}")
    print("Mean P_det close to % seen means the model's base rate matches those rows.")

    # (c) Cluster-bootstrap intervals for the paired differences (variant minus Base).
    codes, uniq = pd.factorize(groups)
    weights = resample_weights(len(uniq), n_boot)
    print(f"\n(c) Variant minus Base on the same rows: difference and 95% cluster-bootstrap interval "
          f"({n_boot:,} resamples of {len(uniq)} lunations)")
    print(f"{'Variant':8s} {'Scored on':12s} {'d acc':>8s} {'95% interval':>18s} "
          f"{'d AUC':>8s} {'95% interval':>18s}")
    for name, _, m in variants[1:]:
        for rows_name, rows in [(f"{name} rows", m), ("all rows", np.ones(len(df), bool))]:
            correct_v = (p_det[name] >= 0.5) == seen
            correct_b = (p_det["Base"] >= 0.5) == seen
            d_acc, d_auc = [], []
            for w_g in weights:
                w = w_g[codes] * rows
                keep = w > 0
                d_acc.append(np.average(correct_v[keep], weights=w[keep])
                             - np.average(correct_b[keep], weights=w[keep]))
                d_auc.append(roc_auc_score(seen[keep], p_det[name][keep], sample_weight=w[keep])
                             - roc_auc_score(seen[keep], p_det["Base"][keep], sample_weight=w[keep]))
            point_acc = correct_v[rows].mean() - correct_b[rows].mean()
            point_auc = (roc_auc_score(seen[rows], p_det[name][rows])
                         - roc_auc_score(seen[rows], p_det["Base"][rows]))
            print(f"{name:8s} {rows_name:12s} {point_acc:8.4f} {ci(d_acc):>18s} "
                  f"{point_auc:8.4f} {ci(d_auc):>18s}")

    # (d) Final models on all rows of each variant.
    print("\n(d) Final models (fitted on all rows of each variant): Stage 1 coefficients")
    print(f"{'Variant':8s} {'':12s} {'Intercept':>10s} " + " ".join(f"{n:>9s}" for n in FEATURE_NAMES))
    finals = {}
    for name, _, m in variants:
        finals[name] = fit_logistic(X[m], seen[m].astype(int))
        std, raw = coefficients(finals[name])
        print(f"{name:8s} {'standardised':12s} {std[0]:10.4f} " + " ".join(f"{v:9.4f}" for v in std[1:]))
        print(f"{'':8s} {'per unit':12s} {raw[0]:10.4f} " + " ".join(f"{v:9.4f}" for v in raw[1:]))
    print("Per unit: per degree of ARCV and ARCL, per minute of lag (the scaler differs between "
          "variants, so these are the comparable values).")
    published = report["stage1"]["score_coefficients"]
    std, _ = coefficients(finals["Base"])
    gap = np.abs(std - np.array([published["intercept"]] + [published[f] for f in FEATURES])).max()
    print(f"Base reproduces the published Stage 1 coefficients (max abs difference {gap:.1e}).")
    stage2 = fit_logistic(X[seen], (y3[seen] == 2).astype(int))
    std2, _ = coefficients(stage2)
    print("Stage 2 (same seen rows in every variant), standardised: "
          + ", ".join(f"{n} {v:.4f}" for n, v in zip(["intercept"] + FEATURE_NAMES, std2)))

    # (e) Reference geometry, with cluster-bootstrap intervals from refits on resampled lunations.
    p_eye_ref = stage2.predict_proba(REFERENCE)[0, 1]
    rows_by_group = [np.flatnonzero(codes == g) for g in range(len(uniq))]
    boot = {name: [] for name, _, _ in variants}
    for w_g in weights:
        idx = np.concatenate([np.repeat(rows_by_group[g], w_g[g]) for g in np.flatnonzero(w_g)])
        for name, _, m in variants:
            r = idx[m[idx]]
            boot[name].append(fit_logistic(X[r], seen[r].astype(int)).predict_proba(REFERENCE)[0, 1])
    boot = {k: np.array(v) for k, v in boot.items()}
    print(f"\n(e) Reference geometry (Sharjah, 15 June 2026: ARCV {REFERENCE[0, 0]}, ARCL {REFERENCE[0, 1]}, "
          f"lag {REFERENCE[0, 2]} min); P_eye = {p_eye_ref:.4f} in every variant")
    print(f"{'Variant':8s} {'P_det':>7s} {'95% interval':>18s} {'minus Base':>11s} "
          f"{'95% interval':>18s} {'Band':>5s}")
    for name, _, _ in variants:
        p = finals[name].predict_proba(REFERENCE)[0, 1]
        d = p - finals["Base"].predict_proba(REFERENCE)[0, 1]
        print(f"{name:8s} {p:7.4f} {ci(boot[name]):>18s} {d:11.4f} "
              f"{ci(boot[name] - boot['Base']) if name != 'Base' else '':>18s} {band(p, p_eye_ref):>5s}")
    print("Bands (Eq. 11): E not detectable, D marginal, C aid-assisted likely, "
          "B borderline naked-eye, A naked-eye likely.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA),
                        help="data/icop (scraped, clean, rejected, features, "
                             "out-of-fold predictions)")
    parser.add_argument("--report", default=str(DEFAULT_REPORT),
                        help="two_stage_crescent_criterion_report.json (check of the coefficients)")
    parser.add_argument("--n-boot", type=int, default=2000, help="Cluster-bootstrap resamples")
    args = parser.parse_args()

    data = Path(args.data_dir)
    raw = pd.read_csv(data / "crescent_observations_raw.csv", low_memory=False)
    clean = pd.read_csv(data / "crescent_observations_clean.csv", low_memory=False)
    rejected = pd.read_csv(data / "crescent_observations_rejected.csv", low_memory=False)
    feats = pd.read_csv(data / "crescent_features.csv", low_memory=False)
    oof = pd.read_csv(data / "two_stage_oof_predictions.csv")
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    unknown = set(clean["country"].dropna()) - set(COUNTRIES)
    if unknown:
        raise ValueError(f"Countries without an English name: {sorted(unknown)}")

    clean = part1_attrition(raw, clean, rejected, feats)
    part2_lost_vs_kept(clean, feats, args.n_boot)

    df = feats[(feats["section_type"] == "first_crescent")
               & (feats["obs_relative_to_sunset"] != "before_sunset")
               & feats[LABEL].notna()].reset_index(drop=True)
    df[LABEL] = df[LABEL].astype(int)
    if len(df) != 4470 or tuple(df[LABEL].value_counts().sort_index()) != (2319, 505, 1646):
        raise ValueError("Source data have changed; verify the modelling subset")
    if df[FEATURES].isna().any().any():
        raise ValueError("Missing predictor values in the modelling subset")
    if not (df[GROUP].equals(oof[GROUP]) and np.array_equal(df[LABEL], oof[LABEL])):
        raise ValueError("Row order differs from two_stage_oof_predictions.csv")
    poor, partly, optical, tried = part3_negatives(df)
    # Thousands of small fits: one BLAS thread is several times faster than many.
    with threadpool_limits(limits=1):
        part4_sensitivity(df, poor, partly, optical, tried, oof, report, args.n_boot)


if __name__ == "__main__":
    main()
