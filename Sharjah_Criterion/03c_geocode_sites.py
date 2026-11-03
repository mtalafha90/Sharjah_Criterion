#!/usr/bin/env python3
"""Check and complete the coordinates of every observing site (after 02, before 03d).

Script 03 geocoded each place (city, province, country) with OpenStreetMap
Nominatim without restricting the search to the reported country, and it lost
every report dated on a Monday through a date-parsing defect, so the places of
those reports were never geocoded. This script:

1. takes the coordinates that script 03 found for each place, from
   crescent_features_script03.csv (the table of scripts 03 and 03b);
2. checks every place against the Natural Earth 1:50m admin-0 polygon of its
   reported country (a point outside every polygon counts as being in the
   nearest one);
3. geocodes again every place that lies outside its reported country, every
   place that script 03 could not geocode, and every place that occurs only in
   the Monday reports, with these rules:
   - the search is restricted to the reported country (Nominatim
     'countrycodes'), so the country name is not repeated in the query;
   - invisible direction marks are removed from the names, a name written in
     both Arabic and Latin script is searched by its Latin part, and the
     abbreviation 'Obs' is written out as 'Observatory';
   - the queries are 'city, province', then 'city', then 'province'; of the
     first five matches of a query, the first that is a settlement, an
     administrative area, an island, a peak or an observatory is used (not a
     street, building, shop or office that carries the name, and not the
     whole country);
   - every match was then inspected, and two that were other places with a
     similar name were rejected (REJECTED below);
   - a place whose city and province only repeat the country name gives no
     site, and a place without a match stays unresolved. Script 03d leaves out
     the reports of unresolved places.

Output: ICOP_Data/site_geocodes.csv, one row per place, with the coordinates
used from now on and how they were obtained. Re-runs reuse the Nominatim
answers stored in ICOP_Data/geocode_restricted_cache.json.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA = Path(__file__).resolve().parent.parent / "ICOP_Data"
CLEAN = DATA / "crescent_observations_clean.csv"
FEATURES = DATA / "crescent_features_script03.csv"  # the table of scripts 03 and 03b
OUT = DATA / "site_geocodes.csv"
CACHE = DATA / "geocode_restricted_cache.json"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "crescent-criterion-builder/1.0 (research use)"}
PLACE = ["city", "province", "country"]

# Reported country names whose spelling differs from Natural Earth's NAME_AR
# (as in regional_check.py of the manuscript repository).
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
ARABIC = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z]")


def norm_space(s) -> str:
    return "" if pd.isna(s) else re.sub(r"\s+", " ", str(s)).strip()


def normalise_ar(text) -> str:
    """Arabic spelling variants: no diacritics, tatweel or direction marks; one alef form."""
    text = re.sub("[ً-ْـ‎‏]", "", str(text))
    text = re.sub("[أإآ]", "ا", text).replace("ة", "ه").replace("ى", "ي")
    return re.sub(r"\s+", " ", text).strip()


def query_name(name: str) -> str:
    """The form of a place name used in the queries."""
    name = re.sub("[​-‏‪-‮⁦-⁩؜ـ]", "", str(name))
    if ARABIC.search(name) and LATIN.search(name):
        name = re.sub(r"[^A-Za-z0-9 .,'’\-]", " ", name)  # keep the Latin part
    name = re.sub(r"\bObs\b\.?", "Observatory", name)
    return re.sub(r"\s+", " ", name).strip(" ,.-()")


def natural_earth():
    import cartopy.io.shapereader as shpreader
    from shapely import STRtree

    path = shpreader.natural_earth(resolution="50m", category="cultural", name="admin_0_countries")
    records = list(shpreader.Reader(path).records())
    return records, STRtree([r.geometry for r in records])


def iso2(attrs) -> str:
    for key in ("ISO_A2_EH", "ISO_A2", "WB_A2"):
        v = str(attrs.get(key, "")).strip()
        if len(v) == 2 and v.isalpha():
            return v.lower()
    raise ValueError(f"No ISO code for {attrs['NAME']}")


def owner_names(lon, lat, records, tree):
    """Natural Earth country of each point (the nearest polygon if it is in none)."""
    from shapely import points

    pts = points(np.asarray(lon, float), np.asarray(lat, float))
    pt_idx, poly_idx = tree.query(pts, predicate="intersects")
    owner = np.full(len(pts), -1)
    owner[pt_idx] = poly_idx
    out = np.flatnonzero(owner < 0)
    if len(out):
        owner[out] = tree.nearest(pts[out])
    return [records[j].attributes["NAME"] for j in owner]


# Kinds of OpenStreetMap object accepted as a site: settlements and administrative
# areas, and a few named observing places.
ACCEPTED = {("man_made", "observatory"), ("landuse", "observatory"), ("natural", "peak"),
            ("place", "island"), ("place", "islet")}
# Matches found on inspection to be other places with a similar name.
REJECTED = {
    ("جزيرة مصيرة", "جنوب الشرقية", "عُمان"): "the match is the village of Junayz al Janubi, not Masirah Island",
    ("العفولة", "الشمال", "فلسطين"): "the match is the hamlet of Khallet al-Foulah, not Afula",
}


def acceptable(match: dict) -> bool:
    if "," not in match["display_name"]:
        return False  # the whole country
    if match["category"] in ("place", "boundary"):
        return match["type"] != "country" and match["place_rank"] > 4
    return (match["category"], match["type"]) in ACCEPTED


def search(q: str, code: str, cache: dict):
    """First five Nominatim matches for one query, restricted to one country (cached)."""
    key = f"{code}|{q}"
    if key not in cache:
        for attempt in range(6):  # Nominatim sometimes answers 503 or an error page under load
            try:
                r = requests.get(NOMINATIM_URL, headers=HEADERS, timeout=60,
                                 params={"q": q, "format": "jsonv2", "limit": 5, "countrycodes": code,
                                         "accept-language": "en"})
                r.raise_for_status()
                arr = r.json()
                break
            except (requests.RequestException, ValueError):
                if attempt == 5:
                    raise
                time.sleep(5 * 2 ** attempt)
        cache[key] = [{"lat": float(a["lat"]), "lon": float(a["lon"]), "category": a.get("category", ""),
                       "type": a.get("type", ""), "place_rank": int(a.get("place_rank", 0)),
                       "display_name": a.get("display_name", q)} for a in arr]
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        time.sleep(1.1)  # Nominatim usage policy: at most one request per second
    return cache[key]


def geocode(city: str, province: str, country: str, code: str, cache: dict):
    """(match, query, reason): the first acceptable match for 'city, province', 'city', 'province'."""
    names = [n for n in (query_name(city), query_name(province))
             if n and normalise_ar(n) != normalise_ar(country)]
    if not names:
        return None, "", "the place names only the country"
    queries = list(dict.fromkeys([", ".join(names), names[0], names[-1]]))
    for q in queries:
        hit = next((m for m in search(q, code, cache) if acceptable(m)), None)
        if hit:
            return hit, q, ""
    return None, "", "no match within the reported country"


def main():
    clean = pd.read_csv(CLEAN, low_memory=False)
    feats = pd.read_csv(FEATURES, low_memory=False)
    for df in (clean, feats):
        for c in PLACE:
            df[c] = df[c].map(norm_space)
    places = clean[PLACE].drop_duplicates().reset_index(drop=True)
    known = (feats.groupby(PLACE, as_index=False)
             .agg(latitude=("latitude", "first"), longitude=("longitude", "first"),
                  display_name=("geocode_display_name", "first"), n_coords=("latitude", "nunique")))
    # A place can have two nearby coordinates when two spellings normalise to the same
    # name (Makkah: 0.3 km apart). Existing reports keep their own coordinates in
    # script 03d; the first pair is used here for the country check.
    for _, r in known[known["n_coords"] > 1].iterrows():
        print(f"Note: {r['city']} / {r['province']} / {r['country']} has {r['n_coords']} coordinate pairs")
    sites = places.merge(known.drop(columns="n_coords"), on=PLACE, how="left")

    records, tree = natural_earth()
    by_name = {normalise_ar(r.attributes["NAME_AR"]): r.attributes for r in records}
    by_name.update({normalise_ar(k): next(r.attributes for r in records if r.attributes["NAME"] == v)
                    for k, v in AR_ALIASES.items()})
    named = sites["country"].map(lambda c: by_name.get(normalise_ar(c)))
    if named.isna().any():
        raise ValueError(f"Unmatched country names: {sorted(set(sites.loc[named.isna(), 'country']))}")
    sites["reported_country"] = [a["NAME"] for a in named]
    sites["iso2"] = [iso2(a) for a in named]

    has = sites["latitude"].notna()
    sites["located_in"] = None
    sites.loc[has, "located_in"] = owner_names(sites.loc[has, "longitude"], sites.loc[has, "latitude"],
                                               records, tree)
    monday = clean["gregorian_date"].astype(str).str.contains("الإثنين")
    all_monday = clean.assign(monday=monday).groupby(PLACE)["monday"].all()
    monday_only = (~has & np.array([bool(all_monday[tuple(r)]) for r in
                                    sites[PLACE].itertuples(index=False)])).to_numpy()
    outside = has & (sites["located_in"] != sites["reported_country"])
    todo = outside | ~has
    print(f"Places: {len(sites)}; with script-03 coordinates {int(has.sum())}, "
          f"of which outside the reported country {int(outside.sum())}; without coordinates "
          f"{int((~has).sum())} (only in Monday reports {int(monday_only.sum())})")

    sites["source"] = np.where(has, "script 03", "")
    sites["query"] = ""
    sites["match_type"] = ""
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    for i in np.flatnonzero(todo):
        row = sites.iloc[i]
        hit, q, reason = geocode(row["city"], row["province"], row["country"], row["iso2"], cache)
        if hit is not None and (row["city"], row["province"], row["country"]) in REJECTED:
            hit, q, reason = None, "", REJECTED[(row["city"], row["province"], row["country"])]
        why = ("outside the reported country" if outside.iloc[i] else
               "Monday report (date-parsing defect)" if monday_only[i] else "not geocoded by script 03")
        if hit is None:
            sites.loc[i, ["latitude", "longitude", "display_name"]] = [np.nan, np.nan, None]
            sites.loc[i, "source"] = f"unresolved ({reason}): {why}"
        else:
            sites.loc[i, ["latitude", "longitude", "display_name"]] = [hit["lat"], hit["lon"],
                                                                      hit["display_name"]]
            sites.loc[i, ["source", "query", "match_type"]] = [f"restricted geocoding: {why}", q,
                                                               f"{hit['category']}={hit['type']}"]
    ok = sites["latitude"].notna()
    sites["located_in"] = None
    sites.loc[ok, "located_in"] = owner_names(sites.loc[ok, "longitude"], sites.loc[ok, "latitude"],
                                              records, tree)
    # A match from the restricted search lies in the reported country by construction.
    # Natural Earth 1:50m has no polygon for Melilla, so its correct position falls in
    # Morocco's polygon; such points are listed below.
    restricted = sites["source"].str.startswith("restricted")
    sites["inside_reported_country"] = ok & ((sites["located_in"] == sites["reported_country"]) | restricted)
    sites.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(sites["source"].value_counts().to_string())
    off = ok & (sites["located_in"] != sites["reported_country"])
    print(f"Resolved places outside the reported country's 1:50m polygon: {int(off.sum())}")
    if off.any():
        print(sites.loc[off, PLACE + ["latitude", "longitude", "located_in", "display_name"]].to_string())
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
