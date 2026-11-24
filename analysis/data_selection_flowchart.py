#!/usr/bin/env python3
"""Data-selection flowchart: from the scraped ICOP reports to the modelling records
(reviewer point 1).

Reviewer: "Provide a data-selection flowchart or summary table ... show the
progression from approximately 7,100 raw ICOP reports to approximately 6,100
usable records and the final 4,470 observations used for modelling, together
with the principal exclusion criteria and the distribution of the three
original observation classes. Where the information is available, a brief
indication of the geographical and temporal coverage and the availability of
explicit observation times would further improve transparency."

What the figure shows:
  - left column: the records kept after each step (scraped, cleaned, feature
    table, modelling subset), with the number of reports, the number of
    lunations (ICOP report pages, one per lunar month), the class counts and a
    bar of the class shares (not seen, seen only with an optical aid, seen by
    naked eye);
  - right column: the records removed at each step, with their count, the
    reason and their classes;
  - bottom panel: coverage of the 4,470 modelling records: dates, lunations,
    sites (distinct coordinates), reported countries, reports per Gregorian
    year, reports per region (regions from the site coordinates, as in
    regional_check.py) and what the archive says about the observation time.

Inputs, from data/icop: the scraped reports
(crescent_observations_raw.csv), the clean and rejected reports
(crescent_observations_clean.csv, crescent_observations_rejected.csv) and the
feature table (crescent_features.csv). The counts at each step are made with
the functions of data_attrition_and_negatives.py, and the regions, sites and
countries with those of regional_check.py (Natural Earth 1:50m countries,
through cartopy). The script stops with an error if any count differs from
Table tab:attrition of the manuscript or from the regional check.

Observation time: the time field of every cleaned report holds only 'after
sunset', 'before sunset', 'before sunrise' or 'after sunrise'; no report gives
a clock time there, so the feature table puts every report at local sunset
(obs_time_source = sunset_fallback). Some observers' free-text comments do
mention clock times (of sunset, moonset or the sighting); these are counted
and printed, but not used.

Run: python3 -W ignore -I -B data_selection_flowchart.py
Output: data_selection_flowchart.png (300 dpi, 6.5 in wide, for \\textwidth)
"""
import argparse
import contextlib
import importlib.util
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # the repository root
DEFAULT_DATA = REPO / "data" / "icop"
DEFAULT_OUTPUT = REPO / "figures" / "data_selection_flowchart.png"
GROUP = "page_url"
LABEL = "vis_class_empirical"

# Table tab:attrition of the manuscript (and the attrition check): reports,
# not seen, aided, naked eye, without class, lunations (None = not checked).
EXPECTED = {
    "scraped": (7154, 3344, 1152, 2628, 30, 205),
    "no_outcome": (30, 0, 0, 0, 30, None),
    "cleaned": (7124, 3344, 1152, 2628, 0, 205),
    "date": (981, 469, 163, 349, 0, 97),
    "place": (51, 20, 9, 22, 0, 44),
    "other_03": (0, 0, 0, 0, 0, 0),
    "features": (6092, 2855, 980, 2257, 0, 205),
    "morning": (1164, 438, 209, 517, 0, 195),
    "before_sunset": (458, 98, 266, 94, 0, 123),
    "unlabelled": (0, 0, 0, 0, 0, 0),
    "model": (4470, 2319, 505, 1646, 0, 202),
}
# Regional check (regions from the site coordinates).
EXPECTED_SITES, EXPECTED_COUNTRIES = 616, 67
EXPECTED_REGIONS = {
    "Arabian Peninsula and Gulf": 674, "Levant, Iraq and Iran": 645, "North Africa": 1095,
    "Sub-Saharan Africa": 464, "Europe": 526, "S and SE Asia, Oceania": 775, "Americas": 291,
}
# The time field of the ICOP reports (Arabic), in English.
TIME_EN = {"بعد غروب الشمس": "after sunset", "قبل غروب الشمس": "before sunset",
           "قبل شروق الشمس": "before sunrise", "بعد شروق الشمس": "after sunrise"}

# Colours of the other figures: ink, muted ink, grid, box edge and surface; class
# colours as in minimum_visibility_check.py (grey, palette slots 2 and 3); the
# single-series bars of the coverage panel take palette slot 1 (blue).
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
EDGE, SURFACE, SIDE_FACE = "#c3c2b7", "#fcfcfb", "#f3f2ee"
C_NOT, C_AID, C_EYE = "#6b6a65", "#eb6834", "#1baf7a"
BLUE = "#2a78d6"
CLASS_COLOURS = [C_NOT, C_AID, C_EYE]


# ---------------------------------------------------------------- helpers
def load_sibling(name):
    """A script in this folder, loaded by path so that 'python3 -I' also works."""
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quietly(func, *args):
    """Run a function of a sibling script without its printed tables."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args)


# ---------------------------------------------------------------- counts
def selection_steps(da, raw, clean, rejected, feats):
    """Reports, classes and lunations kept and removed at each step."""
    # Reason why 03 dropped each clean report (date not parsed / place not geocoded).
    clean = quietly(da.part1_attrition, raw, clean, rejected, feats)
    label = da.class_02(raw)
    no_statement = label.isna() & ~da.flag(raw["seen_any"], {"true", "false"})
    seen_no_means = label.isna() & da.flag(raw["seen_any"], {"true"})
    first = feats["section_type"] == "first_crescent"
    before = feats["obs_relative_to_sunset"] == "before_sunset"
    labelled = feats[LABEL].notna()

    def count(df, lab=None):
        return tuple(da.class_counts(df[LABEL] if lab is None else lab, df[GROUP]))

    def lost_03(reason):
        return count(clean[clean["reason_03"] == reason])

    steps = {
        "scraped": count(raw, label),
        "no_outcome": count(raw[label.isna()], label[label.isna()]),
        "cleaned": count(clean),
        "date": lost_03("date not parsed"),
        "place": lost_03("place not geocoded"),
        "other_03": lost_03("other"),
        "features": count(feats),
        "morning": count(feats[~first]),
        "before_sunset": count(feats[first & before]),
        "unlabelled": count(feats[first & ~before & ~labelled]),
        "model": count(feats[first & ~before & labelled]),
    }
    lost_places = clean[clean["reason_03"] == "place not geocoded"]
    details = {
        "no_statement": int(no_statement.sum()),
        "seen_no_means": int(seen_no_means.sum()),
        "rejected_file": len(rejected),
        "duplicates": len(raw) - len(rejected) - len(clean),
        "places": len(lost_places.groupby(["city", "province", "country"], dropna=False)),
        "place_countries": lost_places["country"].nunique(),
    }
    return steps, details


def check_steps(steps, details):
    bad = []
    for key, expected in EXPECTED.items():
        got = steps[key]
        if any(e is not None and g != e for g, e in zip(got, expected)):
            bad.append(f"{key}: expected {expected}, got {got}")
    if (details["no_statement"], details["seen_no_means"]) != (17, 13):
        bad.append(f"no usable outcome split: {details['no_statement']} / {details['seen_no_means']}")
    if details["rejected_file"] != 30 or details["duplicates"] != 0:
        bad.append(f"rejected file {details['rejected_file']}, duplicates {details['duplicates']}")
    if bad:
        raise ValueError("Counts differ from Table tab:attrition:\n  " + "\n  ".join(bad))


def print_steps(steps, details):
    rows = [
        ("Scraped from ICOP", "scraped"),
        ("  removed: no usable outcome", "no_outcome"),
        ("Cleaned", "cleaned"),
        ("  removed: date not parsed (Monday reports, a parsing defect)", "date"),
        ("  removed: place not geocoded", "place"),
        ("  removed: other reasons in feature computation", "other_03"),
        ("Feature table", "features"),
        ("  removed: last crescent (morning)", "morning"),
        ("  removed: first crescent, observed before sunset", "before_sunset"),
        ("  removed: class label missing", "unlabelled"),
        ("Modelling subset", "model"),
    ]
    print("Records at each step (classes: 0 not seen, 1 aided only, 2 naked eye)")
    print(f"{'Step':62s} {'Reports':>8s} {'Not seen':>9s} {'Aided':>7s} {'Naked eye':>10s} "
          f"{'No class':>9s} {'Lunations':>10s}")
    for name, key in rows:
        n, c0, c1, c2, miss, lun = steps[key]
        print(f"{name:62s} {n:8,d} {c0:9,d} {c1:7,d} {c2:10,d} {miss:9,d} {lun:10,d}")
    print(f"No usable outcome: {details['no_statement']} do not say whether the crescent was seen, "
          f"{details['seen_no_means']} say it was seen but mark no means as successful "
          f"(rejected file: {details['rejected_file']} rows; duplicates removed: "
          f"{details['duplicates']}).")
    print(f"Places not geocoded: {details['places']} distinct (city, province, country) entries "
          f"in {details['place_countries']} countries.")
    print("All counts agree with Table tab:attrition.")


# ---------------------------------------------------------------- coverage
def coverage(da, rc, data_dir, clean):
    """Dates, lunations, sites, countries, regions and observation times."""
    df = rc.load_modelling_rows(data_dir / "crescent_features.csv")
    df = quietly(rc.assign_countries, df)
    # Local date of the report: date part of the local sunset (ISO text with offset).
    day = pd.to_datetime(df["sunset_local"].str[:10])
    parsed = pd.to_datetime(df["gregorian_date"].map(da.date_03))
    if not (day == parsed).all():
        raise ValueError("Local sunset date differs from the Gregorian date of the report")
    sites = len(df[["longitude", "latitude"]].drop_duplicates())
    countries = df["country"].nunique()
    if df["named_country"].nunique() != countries:
        raise ValueError("Reported countries do not map one-to-one onto Natural Earth countries")
    regions = df["region"].value_counts().reindex(rc.REGIONS)
    bad = []
    if sites != EXPECTED_SITES:
        bad.append(f"sites {sites} (expected {EXPECTED_SITES})")
    if countries != EXPECTED_COUNTRIES:
        bad.append(f"countries {countries} (expected {EXPECTED_COUNTRIES})")
    if regions.to_dict() != EXPECTED_REGIONS:
        bad.append(f"regions {regions.to_dict()}")
    if bad:
        raise ValueError("Coverage differs from the regional check: " + "; ".join(bad))

    # Observation time: the time field of every cleaned report, and the time source
    # used in the feature table.
    time_text = clean["obs_time_text"].str.strip()
    unknown = set(time_text.dropna()) - set(TIME_EN)
    if unknown or time_text.isna().any():
        raise ValueError(f"Unexpected observation-time text: {sorted(unknown)}")
    time_counts = time_text.map(TIME_EN).value_counts()
    if time_text.str.contains(r"\d").any():
        raise ValueError("A time field holds a digit (a clock time?)")
    source = df["obs_time_source"].value_counts()
    if set(source.index) != {"sunset_fallback"}:
        raise ValueError(f"Observation instant not always local sunset: {source.to_dict()}")
    # Clock-like times (hh:mm) in the observers' free-text comments; not used.
    clock_comment_clean = int(clean["comment_text"].fillna("").str.contains(r"\d{1,2}:\d{2}").sum())
    clock_comment_model = int(df["comment_text"].fillna("").str.contains(r"\d{1,2}:\d{2}").sum())
    return {
        "n": len(df), "lunations": df[GROUP].nunique(), "first": day.min(), "last": day.max(),
        "years": day.dt.year.value_counts().sort_index(), "sites": sites,
        "countries": countries, "regions": regions,
        "time_counts": time_counts, "time_model": df["obs_time_text"].str.strip().map(TIME_EN)
        .value_counts(), "source": source, "clock_comment_clean": clock_comment_clean,
        "clock_comment_model": clock_comment_model, "n_clean": len(clean),
    }


def print_coverage(cov):
    print(f"\nCoverage of the {cov['n']:,} modelling records")
    print(f"Dates (local date of sunset): {cov['first']:%d %B %Y} to {cov['last']:%d %B %Y}; "
          f"{cov['lunations']} lunations; {cov['sites']} sites (distinct coordinates); "
          f"{cov['countries']} reported countries")
    print("Reports per Gregorian year: " + ", ".join(f"{y} {n}" for y, n in cov["years"].items()))
    print("Reports per region (from the site coordinates): "
          + ", ".join(f"{r} {n:,}" for r, n in cov["regions"].items()))
    print(f"\nObservation time, time field of the {cov['n_clean']:,} cleaned reports: "
          + ", ".join(f"'{t}' {n:,}" for t, n in cov["time_counts"].items())
          + "; no clock time in any of them.")
    print("Modelling records: " + ", ".join(f"'{t}' {n:,}" for t, n in cov["time_model"].items())
          + "; observation instant in the feature table: "
          + ", ".join(f"{s} {n:,}" for s, n in cov["source"].items()) + " (local sunset).")
    print(f"Free-text observer comments with a clock-like time (hh:mm, of sunset, moonset or the "
          f"sighting; not used): {cov['clock_comment_clean']:,} of {cov['n_clean']:,} cleaned "
          f"reports, {cov['clock_comment_model']:,} of {cov['n']:,} modelling records.")


# ---------------------------------------------------------------- figure
def class_bar(ax, x, y, width, height, counts):
    """Bar of class shares (not seen, aided, naked eye), with thin white gaps."""
    total = sum(counts)
    left = x
    for c, colour in zip(counts, CLASS_COLOURS):
        w = width * c / total
        ax.add_patch(Rectangle((left, y), w, height, facecolor=colour, edgecolor="white",
                               linewidth=0.8))
        left += w


def class_text(counts):
    c0, c1, c2 = counts
    return f"{c0:,} not seen  ·  {c1:,} aided only  ·  {c2:,} naked eye"


def box(ax, x, y, w, h, face, edge, lw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor=face, edgecolor=edge, linewidth=lw))


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8,
                                 color=INK2, linewidth=0.9, shrinkA=0, shrinkB=0))


def draw_figure(steps, details, cov, path):
    plt.rcParams.update({"font.size": 8, "text.color": INK, "axes.edgecolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK})
    # Layout in inches: kept steps on the left, removals on the right, coverage below.
    W = 6.5
    MAIN_X, MAIN_W = 0.02, 3.36
    SIDE_X, SIDE_W = 3.74, W - 3.74 - 0.02
    TRUNK = MAIN_X + MAIN_W / 2
    HEAD, MB, SB, PAD = 0.30, 0.74, 0.62, 0.10
    GAPS = [0.42, 0.90, 0.90]
    BRANCH = 0.11  # branch lines this far inside a gap that leads to two removals
    COV_GAP, COV_H = 0.16, 1.86
    H = HEAD + 4 * MB + sum(GAPS) + COV_GAP + COV_H + 0.02

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # Column headings, and the key to the class colours under the right heading.
    y = H - 0.14
    ax.text(MAIN_X, y, "Reports kept at each step", va="center", fontsize=9, fontweight="bold")
    ax.text(SIDE_X, y, "Reports removed", va="center", fontsize=9, fontweight="bold")
    key = [("not seen", C_NOT), ("aided only (seen only with an optical aid)", C_AID),
           ("naked eye", C_EYE)]
    ax.text(SIDE_X, y - 0.21, "Bars: share of each class", va="center", color=INK2)
    for j, (name, colour) in enumerate(key):
        yk = y - 0.36 - 0.14 * j
        ax.add_patch(Rectangle((SIDE_X, yk - 0.04), 0.14, 0.08, facecolor=colour,
                               edgecolor="none"))
        ax.text(SIDE_X + 0.20, yk, name, va="center")

    # Main boxes (kept), from the top.
    main = [
        ("Scraped from the ICOP archive", "scraped",
         f"{steps['scraped'][5]} lunations; {steps['scraped'][4]} reports have no usable outcome"),
        ("Cleaned", "cleaned", f"{steps['cleaned'][5]} lunations; no duplicate reports found"),
        ("Usable records (feature table)", "features",
         f"{steps['features'][5]} lunations; geometry computed at local sunset"),
        ("Modelling subset", "model",
         f"{steps['model'][5]} lunations; first crescent, observed after sunset"),
    ]
    tops = []
    top = H - HEAD
    for i, (title, key_, sub) in enumerate(main):
        tops.append(top)
        n, c0, c1, c2 = steps[key_][:4]
        final = key_ == "model"
        box(ax, MAIN_X, top - MB, MAIN_W, MB, SURFACE, INK if final else INK2,
            1.3 if final else 0.9)
        ax.text(MAIN_X + PAD, top - 0.16, title, va="center", fontsize=9, fontweight="bold")
        ax.text(MAIN_X + MAIN_W - PAD, top - 0.16, f"{n:,} reports", va="center", ha="right",
                fontsize=9, fontweight="bold")
        ax.text(MAIN_X + PAD, top - 0.32, sub, va="center", color=INK2)
        class_bar(ax, MAIN_X + PAD, top - 0.52, MAIN_W - 2 * PAD, 0.10, (c0, c1, c2))
        ax.text(MAIN_X + PAD, top - 0.63, class_text((c0, c1, c2)), va="center")
        if i < 3:
            arrow(ax, (TRUNK, top - MB), (TRUNK, top - MB - GAPS[i]))
            top -= MB + GAPS[i]

    # Side boxes (removed), each on a branch from the trunk inside its gap.
    side = [
        (0, "middle", "No usable outcome", "no_outcome",
         [f"{details['no_statement']} do not say whether the crescent was seen;",
          f"{details['seen_no_means']} say seen, but name no means of sighting"]),
        (1, "upper", "Date not parsed", "date", ["Monday reports (a date-reading defect)"]),
        (1, "lower", "Place not geocoded", "place",
         [f"{details['places']} places that could not be located"]),
        (2, "upper", "Last crescent (morning reports)", "morning", ["excluded by design"]),
        (2, "lower", "First crescent, observed before sunset", "before_sunset",
         ["excluded by design"]),
    ]
    for gap, where, title, key_, notes in side:
        gap_top = tops[gap] - MB
        yb = gap_top - {"middle": GAPS[gap] / 2, "upper": BRANCH,
                        "lower": GAPS[gap] - BRANCH}[where]
        n, c0, c1, c2 = steps[key_][:4]
        top_s = yb + SB / 2
        box(ax, SIDE_X, top_s - SB, SIDE_W, SB, SIDE_FACE, EDGE, 0.8)
        arrow(ax, (TRUNK, yb), (SIDE_X, yb))
        ax.text(SIDE_X + PAD, top_s - 0.14, title, va="center", fontweight="bold")
        ax.text(SIDE_X + SIDE_W - PAD, top_s - 0.14, f"\u2212{n:,}", va="center", ha="right",
                fontweight="bold")
        for j, note in enumerate(notes):
            ax.text(SIDE_X + PAD, top_s - 0.29 - 0.14 * j, note, va="center", color=INK2)
        if c0 + c1 + c2:
            class_bar(ax, SIDE_X + PAD, top_s - 0.45, SIDE_W - 2 * PAD, 0.08, (c0, c1, c2))
            ax.text(SIDE_X + PAD, top_s - 0.53, class_text((c0, c1, c2)), va="center")

    # Coverage panel: facts and reports per year on the left, regions on the right.
    cov_top = tops[-1] - MB - COV_GAP
    cov_bottom = cov_top - COV_H
    box(ax, MAIN_X, cov_bottom, W - 2 * MAIN_X, COV_H, SURFACE, EDGE, 0.8)
    ax.text(MAIN_X + PAD, cov_top - 0.16, f"Coverage of the {cov['n']:,} modelling records",
            va="center", fontsize=9, fontweight="bold")
    facts = [
        f"Dates: {cov['first']:%-d %B %Y} to {cov['last']:%-d %B %Y} ({cov['lunations']} lunations)",
        f"{cov['sites']} sites in {cov['countries']} reported countries",
        "Observation time: the time field says only 'after sunset';",
        f"{cov['clock_comment_model']:,} comments give a clock time (not used)",
    ]
    for j, line in enumerate(facts):
        ax.text(MAIN_X + PAD, cov_top - 0.35 - 0.145 * j, line, va="center")

    def inch_axes(x0, y0, w, h):
        a = fig.add_axes([x0 / W, y0 / H, w / W, h / H])
        a.set_facecolor("none")  # the panel surface shows through
        return a

    # Reports per Gregorian year (columns from zero).
    years = cov["years"]
    yax = inch_axes(MAIN_X + 0.42, cov_bottom + 0.27, 2.80, 0.50)
    yax.bar(years.index, years.to_numpy(), width=0.75, color=BLUE, zorder=2)
    yax.set_ylim(0, 450)
    yax.set_yticks([0, 200, 400])
    yax.set_xticks([2010, 2014, 2018, 2022, 2026])
    yax.set_xlim(years.index.min() - 0.6, years.index.max() + 0.6)
    yax.tick_params(labelsize=8, length=2, pad=1.5)
    yax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    for s in ("top", "right"):
        yax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        yax.spines[s].set_color(EDGE)
    yax.set_title(f"Reports per year ({years.index.min()} and {years.index.max()} in part)",
                  fontsize=8, loc="left", color=INK, pad=4)

    # Reports per region (horizontal bars from zero, counts at the bar ends).
    regions = cov["regions"]
    rax = inch_axes(SIDE_X + 1.30, cov_bottom + 0.10, SIDE_W - 1.30 - 0.40, 1.42)
    ypos = np.arange(len(regions))[::-1]
    rax.barh(ypos, regions.to_numpy(), height=0.66, color=BLUE)
    for yp, n in zip(ypos, regions.to_numpy()):
        rax.text(n + 15, yp, f"{n:,}", va="center", fontsize=8)
    rax.set_yticks(ypos)
    rax.set_yticklabels(regions.index, fontsize=8)
    rax.set_xlim(0, regions.max())
    rax.set_ylim(-0.5, len(regions) - 0.5)
    rax.tick_params(axis="y", length=0, pad=3)
    rax.set_xticks([])
    for s in ("top", "right", "bottom"):
        rax.spines[s].set_visible(False)
    rax.spines["left"].set_color(EDGE)
    rax.set_title("Reports by region (from the site coordinates)", fontsize=8, loc="right",
                  color=INK, pad=4)

    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"\nSaved {path} ({W:.2f} x {H:.2f} in, 300 dpi)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA),
                        help="data/icop (scraped, clean, rejected and feature files)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Figure (PNG)")
    args = parser.parse_args()

    da = load_sibling("data_attrition_and_negatives")
    rc = load_sibling("regional_check")
    data = Path(args.data_dir)
    raw = pd.read_csv(data / "crescent_observations_raw.csv", low_memory=False)
    clean = pd.read_csv(data / "crescent_observations_clean.csv", low_memory=False)
    rejected = pd.read_csv(data / "crescent_observations_rejected.csv", low_memory=False)
    feats = pd.read_csv(data / "crescent_features.csv", low_memory=False)

    steps, details = selection_steps(da, raw, clean, rejected, feats)
    check_steps(steps, details)
    print_steps(steps, details)
    cov = coverage(da, rc, data, clean)
    print_coverage(cov)
    draw_figure(steps, details, cov, args.output)


if __name__ == "__main__":
    main()
