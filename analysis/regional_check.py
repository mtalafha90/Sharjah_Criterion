#!/usr/bin/env python3
"""Where the ICOP reports come from, and a leave-one-region-out test (comment 6).

Reviewer comment 6: "The Faid et al. (2025, MJS) review shows that where the
data come from strongly shapes a criterion. The authors should therefore show
where the observations are located and test leaving one region out."

Input: data/icop/crescent_features.csv,
filtered as in 04_train_two_stage_crescent_criterion.py (4,470 first-crescent
reports: 2,319 not seen, 505 aided, 1,646 naked eye; 202 lunations, one ICOP
monthly page each), and the lunation-grouped out-of-fold probabilities in
data/icop/two_stage_oof_predictions.csv (same rows, same order).

Regions are taken from the report coordinates, which are also the coordinates
used for the crescent geometry, not from the reported country (Arabic text).
Each report is placed in the Natural Earth 1:50m admin-0 country whose polygon
contains its coordinates; a coastal point just outside every polygon takes the
nearest one (cartopy downloads the file on first use). Countries are grouped,
in this order:
  Arabian Peninsula and Gulf  Saudi Arabia, UAE, Oman, Yemen, Bahrain, Kuwait, Qatar
  Levant, Iraq and Iran       the rest of UN Western Asia (Palestine, Jordan, Syria,
                              Lebanon, Iraq, Turkey, Azerbaijan, ...) and Iran
  North Africa                UN Northern Africa (Morocco, Algeria, Tunisia, Libya,
                              Egypt, Sudan) and Mauritania
  Sub-Saharan Africa          every other African country (with Mauritius)
  Europe                      Natural Earth continent Europe
  S and SE Asia, Oceania      the rest of Asia and Oceania (South Asia, about 250
                              reports, is merged with Southeast Asia to keep seven
                              regions; East Asia has only 3 reports)
  Americas                    North and South America, with the Caribbean
Every region keeps at least 150 reports (checked). The listing of reports placed
outside their reported country also exposes a few geocoding errors in the feature
table (e.g. Lawrence, USA, placed in the Gulf of St Lawrence); they are kept as
they are, since the crescent geometry was computed at the same coordinates.

Leave one region out: for each region, both stages are refitted on the other
six regions with the pipeline of the final model (ARCV, ARCL and lag; median
imputation, standardisation, logistic regression, lbfgs, max_iter=4000; Stage 2
on the seen reports only) and applied to the held-out region. Stage accuracy
uses a cutoff of 0.50; the three-class accuracy uses the operational rule
(P_det < 0.50 -> not seen; P_det >= 0.50 and P_eye >= 0.60 -> naked eye;
otherwise aided). Stage 2 is scored on the region's sightings.
A lunation usually has reports from several regions, so this split still has
the same nights in training (seen from elsewhere) and test. The stricter split
also holds out lunations: test = region r and lunation fold k, train = other
regions and other folds, with the five folds of the out-of-fold file
(StratifiedGroupKFold, shuffle=True, random_state=42, stratified on the
three-class label, groups = page_url), rebuilt here and checked against it.
Both are compared with the grouped out-of-fold predictions for the same reports.
Majority = share of the region's own most common class (Stage 1: seen or not;
Stage 2: naked eye or aided; three-class), i.e. the best constant rule for that
region, chosen with hindsight.

Station map (--figure): report sites (unique coordinates), marker area
proportional to the number of reports at the site, colour and marker shape by
region, Robinson projection, 300 dpi, no title.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ARCV = "relative_altitude_deg"
ARCL = "elongation_deg"
LAG = "moon_lag_minutes"
FEATURES = [ARCV, ARCL, LAG]
TARGET = "vis_class_empirical"
LUNATION = "page_url"
PDET, PEYE = 0.50, 0.60
N_FOLDS, FOLD_SEED = 5, 42
HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # the repository root
DEFAULT_INPUT = REPO / "data" / "icop" / "crescent_features.csv"
DEFAULT_OOF = REPO / "data" / "icop" / "two_stage_oof_predictions.csv"
DEFAULT_FIGURE = REPO / "figures" / "station_map.png"

ARABIA = {"SAU", "ARE", "OMN", "YEM", "BHR", "KWT", "QAT"}
# Reported country names whose spelling differs from Natural Earth's NAME_AR.
AR_ALIASES = {
    "المملكة العربية السعودية": "Saudi Arabia",
    "الإمارات": "United Arab Emirates",
    "عُمان": "Oman",
    "بنغلادش": "Bangladesh",
    "البوسنة و الهرسك": "Bosnia and Herz.",
    "كاميرون": "Cameroon",
    "الفليبين": "Philippines",
    "بولينيزيا الفرنسية": "Fr. Polynesia",
}
# Fixed order: tables, colours and marker shapes follow it.
REGIONS = [
    "Arabian Peninsula and Gulf",
    "Levant, Iraq and Iran",
    "North Africa",
    "Sub-Saharan Africa",
    "Europe",
    "S and SE Asia, Oceania",
    "Americas",
]
# Categorical palette in its fixed slot order (one colour per region).
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]


# ---------------------------------------------------------------- data
def load_modelling_rows(path):
    df = pd.read_csv(path, low_memory=False)
    df = df[
        (df["section_type"] == "first_crescent")
        & (df["obs_relative_to_sunset"] != "before_sunset")
        & df[TARGET].notna()
    ].reset_index(drop=True)  # unique index, source order kept (as script 04)
    df[TARGET] = df[TARGET].astype(int)
    if len(df) != 4470 or tuple(np.bincount(df[TARGET])) != (2319, 505, 1646):
        raise ValueError("Source data have changed; expected 4,470 rows, 2319/505/1646")
    if df[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Reports without coordinates")
    return df


def load_oof(path, df):
    oof = pd.read_csv(path, low_memory=False)
    same = (len(oof) == len(df)
            and (oof[LUNATION].to_numpy() == df[LUNATION].to_numpy()).all()
            and (oof[TARGET].to_numpy() == df[TARGET].to_numpy()).all()
            and np.allclose(oof[FEATURES], df[FEATURES], rtol=0, atol=1e-12))
    if not same:
        raise ValueError(f"{path} does not hold the modelling rows in the same order")
    return oof["p_det"].to_numpy(), oof["p_eye"].to_numpy()


# ---------------------------------------------------------------- regions
def region_of(attrs):
    a3, sub, cont = attrs["ADM0_A3"], attrs["SUBREGION"], attrs["CONTINENT"]
    if a3 in ARABIA:
        return REGIONS[0]
    if sub == "Western Asia" or a3 == "IRN":
        return REGIONS[1]
    if sub == "Northern Africa" or a3 == "MRT":
        return REGIONS[2]
    if cont == "Africa" or sub.endswith("Africa"):  # Mauritius is in 'Seven seas'
        return REGIONS[3]
    if cont == "Europe":
        return REGIONS[4]
    if cont in ("Asia", "Oceania") or sub.endswith("Asia"):  # Maldives: 'Seven seas'
        return REGIONS[5]
    if cont in ("North America", "South America"):
        return REGIONS[6]
    raise ValueError(f"No region for {attrs['NAME']} ({cont}, {sub})")


def normalise_ar(text):
    """Arabic spelling variants: no diacritics, tatweel or direction marks; one alef form."""
    text = re.sub("[ً-ْـ‎‏]", "", str(text))
    text = re.sub("[أإآ]", "ا", text).replace("ة", "ه").replace("ى", "ي")
    return re.sub(r"\s+", " ", text).strip()


def assign_countries(df):
    """Natural Earth country and region of each report's coordinates.

    Also matches the reported (Arabic) country to its Natural Earth country, by
    NAME_AR or AR_ALIASES, so that reports placed outside it can be listed.
    """
    import cartopy.io.shapereader as shpreader
    from shapely import STRtree, points

    path = shpreader.natural_earth(resolution="50m", category="cultural",
                                   name="admin_0_countries")
    records = list(shpreader.Reader(path).records())
    geoms = [r.geometry for r in records]
    tree = STRtree(geoms)
    sites = df[["longitude", "latitude"]].drop_duplicates().reset_index(drop=True)
    pts = points(sites["longitude"].to_numpy(), sites["latitude"].to_numpy())
    pt_idx, poly_idx = tree.query(pts, predicate="intersects")
    owner = np.full(len(sites), -1)
    owner[pt_idx] = poly_idx  # a point on a shared border keeps the last match
    outside = np.flatnonzero(owner < 0)
    gap = np.zeros(len(sites))
    if len(outside):
        near = tree.nearest(pts[outside])
        owner[outside] = near
        gap[outside] = [pts[i].distance(geoms[j]) for i, j in zip(outside, near)]
    sites["ne_country"] = [records[j].attributes["NAME"] for j in owner]
    sites["region"] = [region_of(records[j].attributes) for j in owner]
    sites["gap_deg"] = gap
    out = df.merge(sites, on=["longitude", "latitude"], how="left", validate="many_to_one")
    n_out = int((out["gap_deg"] > 0).sum())
    print(f"Natural Earth 1:50m countries: {len(sites)} distinct coordinates; "
          f"{len(outside)} sites ({n_out} reports) outside every polygon take the nearest one "
          f"(largest gap {gap.max():.2f} deg)")

    by_name = {normalise_ar(r.attributes["NAME_AR"]): r.attributes for r in records}
    by_name.update({normalise_ar(k): next(r.attributes for r in records
                                          if r.attributes["NAME"] == v)
                    for k, v in AR_ALIASES.items()})
    named = out["country"].map(lambda c: by_name.get(normalise_ar(c)))
    if named.isna().any():
        raise ValueError(f"Unmatched country names: {sorted(set(out.loc[named.isna(), 'country']))}")
    out["named_country"] = [a["NAME"] for a in named]
    out["named_region"] = [region_of(a) for a in named]
    return out


def class_shares(y):
    n = len(y)
    return [100 * np.mean(y == c) if n else np.nan for c in (0, 1, 2)]


def print_regions(df):
    print("\nReports by region (from coordinates)")
    print(f"{'Region':27s} {'Reports':>7s} {'Share':>6s} {'Sites':>5s} {'Lunations':>9s} "
          f"{'Not seen':>8s} {'Aided':>6s} {'Naked':>6s} {'Eye|seen':>8s}")
    for r in REGIONS + ["All"]:
        d = df if r == "All" else df[df["region"] == r]
        y = d[TARGET].to_numpy()
        s0, s1, s2 = class_shares(y)
        eye = 100 * np.mean(y[y > 0] == 2)
        n_sites = len(d[["longitude", "latitude"]].drop_duplicates())
        print(f"{r:27s} {len(d):7d} {100 * len(d) / len(df):5.1f}% {n_sites:5d} "
              f"{d[LUNATION].nunique():9d} {s0:7.1f}% {s1:5.1f}% {s2:5.1f}% {eye:7.1f}%")
    print("Class shares are of the region's reports; Eye|seen = naked-eye share of its sightings.")


def print_countries(df, top=15):
    # Countries as reported (Arabic text, last column), named in English through
    # Natural Earth; 'Elsewhere' counts reports whose coordinates lie in another country.
    g = []
    for name, d in df.groupby("named_country"):
        g.append((len(d), name, d["named_region"].iloc[0], *class_shares(d[TARGET].to_numpy()),
                  len(d[["longitude", "latitude"]].drop_duplicates()),
                  int((d["ne_country"] != name).sum()), " / ".join(sorted(set(d["country"])))))
    g.sort(key=lambda t: (-t[0], t[1]))
    print(f"\nReports by reported country, top {top} of {len(g)}")
    print(f"{'Country':26s} {'Region':27s} {'Reports':>7s} {'Sites':>5s} "
          f"{'Not seen':>8s} {'Aided':>6s} {'Naked':>6s} {'Elsewhere':>9s}  Reported as")
    for n, name, reg, s0, s1, s2, sites, other, text in g[:top]:
        print(f"{name:26s} {reg:27s} {n:7d} {sites:5d} {s0:7.1f}% {s1:5.1f}% {s2:5.1f}% "
              f"{other:9d}  {text}")
    print(f"Top {top}: {sum(t[0] for t in g[:top]):,} of {len(df):,} reports.")

    off = df[df["ne_country"] != df["named_country"]]
    moved = off["region"] != off["named_region"]
    print(f"\nReports whose coordinates lie outside the reported country: {len(off)} "
          f"({int(moved.sum())} of them in another region)")
    print(f"{'Reported country':26s} {'Coordinates in':16s} {'Lat':>7s} {'Lon':>8s} "
          f"{'Reports':>7s}  Site as reported")
    sites = (off.assign(moved=moved).groupby(["named_country", "ne_country", "latitude", "longitude"])
             .agg(n=(TARGET, "size"), moved=("moved", "first"), city=("city", "first"))
             .reset_index().sort_values(["n", "named_country"], ascending=[False, True]))
    for _, s in sites.iterrows():
        city = re.sub("[‎‏]", "", str(s["city"])).strip()
        print(f"{s['named_country']:26s} {s['ne_country'] + (' *' if s['moved'] else ''):16s} "
              f"{s['latitude']:7.2f} {s['longitude']:8.2f} {s['n']:7d}  {city}")
    print("* = another region. Regions follow the coordinates, as the geometry does.")


# ---------------------------------------------------------------- models
def fit_binary_logistic(X, y):
    # Same pipeline as the saved Stage 1 and Stage 2 models.
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, solver="lbfgs")),
    ])
    return model.fit(X, y)


def fit_predict(train, test):
    """Stage 1 on all training rows, Stage 2 on the seen ones; probabilities for test."""
    seen = train[TARGET] > 0
    stage1 = fit_binary_logistic(train[FEATURES], seen.astype(int))
    stage2 = fit_binary_logistic(train.loc[seen, FEATURES],
                                 (train.loc[seen, TARGET] == 2).astype(int))
    return (stage1.predict_proba(test[FEATURES])[:, 1],
            stage2.predict_proba(test[FEATURES])[:, 1])


def three_class(p_det, p_eye):
    return np.where(p_det < PDET, 0, np.where(p_eye >= PEYE, 2, 1))


def leave_one_region_out(df):
    p_det, p_eye = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for r in REGIONS:
        test = (df["region"] == r).to_numpy()
        p_det[test], p_eye[test] = fit_predict(df[~test], df[test])
    return p_det, p_eye


def lunation_folds(df):
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    fold = np.full(len(df), -1)
    for k, (_, test_idx) in enumerate(cv.split(df, df[TARGET].to_numpy(), groups=df[LUNATION])):
        fold[test_idx] = k
    return fold


def out_of_fold(df, fold):
    p_det, p_eye = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for k in range(N_FOLDS):
        test = fold == k
        p_det[test], p_eye[test] = fit_predict(df[~test], df[test])
    return p_det, p_eye


def region_and_lunation_out(df, fold):
    p_det, p_eye = np.full(len(df), np.nan), np.full(len(df), np.nan)
    region = df["region"].to_numpy()
    for r in REGIONS:
        for k in range(N_FOLDS):
            test = (region == r) & (fold == k)
            train = (region != r) & (fold != k)
            p_det[test], p_eye[test] = fit_predict(df[train], df[test])
    return p_det, p_eye


# ---------------------------------------------------------------- scores
def scores(y, p_det, p_eye):
    seen = y > 0
    s = {"n": len(y), "n_seen": int(seen.sum())}
    s["s1_acc"] = accuracy_score(seen, p_det >= 0.50)
    s["s1_auc"] = roc_auc_score(seen, p_det) if 0 < seen.sum() < len(y) else np.nan
    eye = y[seen] == 2
    s["s2_acc"] = accuracy_score(eye, p_eye[seen] >= 0.50)
    s["s2_auc"] = roc_auc_score(eye, p_eye[seen]) if 0 < eye.sum() < len(eye) else np.nan
    s["c3_acc"] = accuracy_score(y, three_class(p_det, p_eye))
    s["s1_maj"] = max(seen.mean(), 1 - seen.mean())
    s["s2_maj"] = max(eye.mean(), 1 - eye.mean())
    s["c3_maj"] = np.bincount(y, minlength=3).max() / len(y)
    s["obs_seen"], s["mean_pdet"] = seen.mean(), p_det.mean()
    s["obs_eye"], s["mean_peye"] = eye.mean(), p_eye[seen].mean()
    return s


def print_tables(df, preds):
    """preds: {label: (p_det, p_eye)} in column order."""
    y = df[TARGET].to_numpy()
    groups = [(r, (df["region"] == r).to_numpy()) for r in REGIONS]
    groups.append(("Pooled", np.ones(len(df), dtype=bool)))
    res = {g: {k: scores(y[m], p[0][m], p[1][m]) for k, p in preds.items()} for g, m in groups}
    keys = list(preds)
    short = {"LORO": "LORO", "LORO+lun": "LORO+lun", "OOF": "OOF"}

    for stage, n_key, metrics in [
        ("Stage 1 (detectability), all reports", "n", [("s1_acc", "Acc"), ("s1_auc", "AUC")]),
        ("Stage 2 (naked eye | seen), sightings only", "n_seen", [("s2_acc", "Acc"), ("s2_auc", "AUC")]),
    ]:
        maj = metrics[0][0].replace("acc", "maj")
        print(f"\n{stage}; cutoff 0.50")
        print(f"{'Region':27s} {'N':>5s} " + " ".join(
            f"{f'{lab} {short[k]}':>12s}" for m, lab in metrics for k in keys)
            + f" {'Majority':>9s}")
        for g, _ in groups:
            print(f"{g:27s} {res[g][keys[0]][n_key]:5d} " + " ".join(
                f"{res[g][k][m]:12.4f}" for m, _ in metrics for k in keys)
                + f" {res[g][keys[0]][maj]:9.4f}")

    print("\nThree-class accuracy, operational rule (P_det 0.50, P_eye 0.60)")
    print(f"{'Region':27s} {'N':>5s} " + " ".join(f"{short[k]:>9s}" for k in keys)
          + f" {'Majority':>9s} {'LORO-OOF':>9s}")
    for g, _ in groups:
        r = res[g]
        print(f"{g:27s} {r[keys[0]]['n']:5d} " + " ".join(f"{r[k]['c3_acc']:9.4f}" for k in keys)
              + f" {r[keys[0]]['c3_maj']:9.4f} {r['LORO']['c3_acc'] - r['OOF']['c3_acc']:+9.4f}")

    print("\nCalibration in the large: observed share and mean predicted probability")
    print(f"{'Region':27s} {'Seen':>7s} " + " ".join(f"{f'P_det {short[k]}':>14s}" for k in keys)
          + f" {'Eye|seen':>9s} " + " ".join(f"{f'P_eye {short[k]}':>14s}" for k in keys))
    for g, _ in groups:
        r = res[g]
        print(f"{g:27s} {r[keys[0]]['obs_seen']:7.4f} "
              + " ".join(f"{r[k]['mean_pdet']:14.4f}" for k in keys)
              + f" {r[keys[0]]['obs_eye']:9.4f} "
              + " ".join(f"{r[k]['mean_peye']:14.4f}" for k in keys))
    print("P_eye is averaged over the sightings, as Stage 2 is scored there.")
    return res


# ---------------------------------------------------------------- figure
def station_map(df, path):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sites = (df.groupby(["longitude", "latitude", "region"]).size()
             .rename("n").reset_index().sort_values("n", ascending=False))
    area = lambda n: 4.0 * n  # marker area (pt^2) per report

    proj, geo = ccrs.Robinson(), ccrs.PlateCarree()
    fig = plt.figure(figsize=(10, 5.6))
    ax = fig.add_axes([0.0, 0.12, 1.0, 0.88], projection=proj)
    ax.set_global()
    # Keep 55 S to 72 N: no report lies outside 38 S to 61 N.
    ax.set_ylim(proj.transform_point(0, -55, geo)[1], proj.transform_point(0, 72, geo)[1])
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#ebebe8", edgecolor="none")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#fcfcfb", edgecolor="none")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#c3c2b7", linewidth=0.3)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#8f8e88", linewidth=0.35)
    ax.spines["geo"].set_edgecolor("#c3c2b7")
    # Large sites first, so that small ones stay visible on top. Every site is a circle,
    # so that marker area is proportional to the report count in all regions.
    for r, colour in zip(REGIONS, COLOURS):
        s = sites[sites["region"] == r]
        ax.scatter(s["longitude"], s["latitude"], s=area(s["n"]), c=colour, marker="o",
                   alpha=0.75, edgecolors="white", linewidths=0.4, transform=geo, zorder=3)

    counts = df["region"].value_counts()
    region_handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=8, markerfacecolor=c,
               markeredgecolor="white", markeredgewidth=0.4, label=f"{r} ({counts[r]:,})")
        for r, c in zip(REGIONS, COLOURS)]
    size_handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=np.sqrt(area(n)),
               markerfacecolor="none", markeredgecolor="#52514e", markeredgewidth=0.7,
               label=f"{n} report{'s' if n > 1 else ''}")
        for n in (1, 10, 50, 100)]
    leg = fig.legend(handles=region_handles, loc="lower left", bbox_to_anchor=(0.02, 0.0),
                     ncol=2, frameon=False, fontsize=10, title="Region (reports)",
                     title_fontsize=10, alignment="left", handletextpad=0.3, columnspacing=1.2)
    fig.legend(handles=size_handles, loc="lower right", bbox_to_anchor=(0.98, 0.0), ncol=4,
               frameon=False, fontsize=10, title="Reports at a site", title_fontsize=10,
               handletextpad=0.8, columnspacing=1.6, borderpad=0.9)
    fig.add_artist(leg)
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"\nStation map: {len(sites)} sites, largest {sites['n'].max()} reports; saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="Path to data/icop/crescent_features.csv")
    parser.add_argument("--oof", default=str(DEFAULT_OOF),
                        help="Lunation-grouped out-of-fold probabilities (script 04)")
    parser.add_argument("--figure", default=str(DEFAULT_FIGURE),
                        help="Station map PNG (default: figures/station_map.png)")
    args = parser.parse_args()

    df = load_modelling_rows(args.input)
    oof = load_oof(args.oof, df)
    df = assign_countries(df)
    print(f"Modelling rows: {len(df):,} reports, {df[LUNATION].nunique()} lunations, "
          f"classes {np.bincount(df[TARGET]).tolist()} (not seen, aided, naked eye)")
    print_regions(df)
    small = [r for r in REGIONS if (df["region"] == r).sum() < 150]
    if small:
        raise ValueError(f"Regions with fewer than 150 reports: {small}")
    print_countries(df)

    fold = lunation_folds(df)
    rebuilt = out_of_fold(df, fold)
    gap = max(np.max(np.abs(rebuilt[0] - oof[0])), np.max(np.abs(rebuilt[1] - oof[1])))
    if not gap <= 1e-9:
        raise ValueError(f"Rebuilt folds do not reproduce {args.oof} (gap {gap:.3g})")
    print(f"\nLunation folds rebuilt: they reproduce {Path(args.oof).name} "
          f"(largest difference {gap:.1e})")
    preds = {"LORO": leave_one_region_out(df),
             "LORO+lun": region_and_lunation_out(df, fold),
             "OOF": oof}
    print("LORO = region held out; LORO+lun = region and lunation fold held out; "
          "OOF = lunation fold held out (all regions in training).")
    print_tables(df, preds)
    if args.figure:
        station_map(df, args.figure)


if __name__ == "__main__":
    main()
