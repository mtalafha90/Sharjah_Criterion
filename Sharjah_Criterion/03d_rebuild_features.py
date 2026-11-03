#!/usr/bin/env python3
"""Rebuild the feature table from every cleaned report (after 03c, before 04).

This step replaces scripts 03 and 03b, whose table is kept as
ICOP_Data/crescent_features_script03.csv. It corrects three faults of that
table and applies written rules for invalid reports:

1. Monday reports. Script 03 could not read the date of any report made on a
   Monday (it spelled Monday 'الاثنين'; the reports write 'الإثنين') and dropped
   all 981 of them. Here every report date is read (crescent_geometry.report_date).
2. Sites. Every report takes the coordinates of its place from
   ICOP_Data/site_geocodes.csv (script 03c), where each place has been checked
   against its reported country. Reports whose place could not be found within
   the reported country are kept in the table without features and are not used.
3. Geometry. The sunset geometry is computed again for every report with
   crescent_geometry.py, because the module used by script 03 is not in this
   repository. The values agree with the old table to within 0.001 deg, except
   that 423 moonsets differ by one minute (rounding) and 15 Moon ages were 1 h
   too large or too small in the old table (a daylight-saving change between
   conjunction and sunset); the ages here are true elapsed hours.

Validity rules. They apply to the reports that can enter the model (first
crescent, after sunset, with a class), in this order; a report is counted under
the first rule that applies:

  R1  ICOP's heading of the report says the crescent was not seen ('لم تتم رؤية
      الهلال'), but the report records a sighting: the report contradicts
      itself, so it is excluded.
  R2  ICOP's heading says the crescent was seen by others ('تمت الرؤية من قبل
      آخرين'): the reporting observer passed on a sighting by other people,
      often at another place, so the outcome at the reported site is not known.
      Excluded, whatever the coded class.
  R3  A sighting on an evening when the Moon set before the Sun (negative lag):
      the crescent cannot have been seen after sunset. Excluded.
  R4  The free-text comment of every sighting not headed 'seen by others' was
      read (1,368 reports; ICOP_Data/observation_audit.csv lists every report
      flagged, with the words that show the problem, and a second reader
      confirmed each flag). Excluded: sightings made only in daytime or before
      sunset; sightings made at another place or by other people while the
      reporting observer did not see the crescent; reports that are not of the
      evening first crescent (a solar eclipse, another date). Recoded: reports
      whose text states a different outcome from the coded one.

An excluded report keeps its row, with vis_class_empirical empty, so that the
filters of script 04 leave it out; vis_class_reported keeps the class coded
from the report and exclusion_reason gives the rule. A recoded report has the
new class in vis_class_empirical and the reason in recode_note.

Output: ICOP_Data/crescent_features.csv (all 7,124 cleaned reports, in the
order of crescent_observations_clean.csv, with clean_row = its row number).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import crescent_geometry as cg  # noqa: E402

DATA = HERE.parent / "ICOP_Data"
CLEAN = DATA / "crescent_observations_clean.csv"
SITES = DATA / "site_geocodes.csv"
AUDIT = DATA / "observation_audit.csv"
OUTPUT = DATA / "crescent_features.csv"
PLACE = ["city", "province", "country"]

HEADING_NOT_SEEN = "لم تتم رؤية الهلال"
HEADING_SEEN_BY_OTHERS = "تمت الرؤية من قبل آخرين"
AUDIT_EXCLUSIONS = {
    "daytime_only": "R4: seen only in daytime or before sunset",
    "other_place_or_observer": "R4: seen at another place or by other people",
    "not_first_crescent": "R4: not an evening first-crescent sighting",
}
RECODE = {"recode_0": 0, "recode_1": 1, "recode_2": 2}


def norm_space(s) -> str:
    return "" if pd.isna(s) else re.sub(r"\s+", " ", str(s)).strip()


def icop_heading(raw_text) -> str:
    """ICOP's summary line of a report, e.g. 'تمت رؤية الهلال' (the crescent was seen)."""
    m = re.search(r"وقت الرصد:[^.]*\.\s*([^:]{3,60}):", str(raw_text))
    return m.group(1).strip() if m else ""


def metadata_row(rec) -> dict:
    """A row without features (no site, or no sunset): metadata copied as in archive_row."""
    row = {c: None for c in cg.ARCHIVE_COLUMNS}
    row.update({c: rec.get(c) for c in cg.METADATA_COLUMNS})
    for c in ("gregorian_date", "country", "city", "province", "obs_time_text"):
        row[c] = cg.normalize_space(rec.get(c))
    row["vis_class_empirical"] = cg.visibility_label(rec)
    return row


def load_sites() -> dict:
    sites = pd.read_csv(SITES, dtype={c: str for c in PLACE}, keep_default_na=False,
                        na_values={"latitude": [""], "longitude": [""]}, encoding="utf-8-sig")
    return {tuple(norm_space(r[c]) for c in PLACE): r for _, r in sites.iterrows()}


def load_audit(clean: pd.DataFrame) -> pd.DataFrame:
    audit = pd.read_csv(AUDIT, encoding="utf-8-sig")
    if audit["clean_row"].duplicated().any():
        raise ValueError("observation_audit.csv lists a report twice")
    rows = clean.loc[audit["clean_row"]]
    if not (rows["page_url"].to_numpy() == audit["page_url"].to_numpy()).all():
        raise ValueError("observation_audit.csv does not match crescent_observations_clean.csv")
    bad = set(audit["action"]) - set(RECODE) - {"exclude"}
    if bad or not set(audit.loc[audit["action"] == "exclude", "category"]) <= set(AUDIT_EXCLUSIONS):
        raise ValueError(f"Unknown audit action or category: {bad}")
    return audit.set_index("clean_row")


def main():
    clean = pd.read_csv(CLEAN, low_memory=False)
    sites = load_sites()
    audit = load_audit(clean)

    rows, failures = [], 0
    for i, rec in clean.iterrows():
        key = tuple(norm_space(rec[c]) for c in PLACE)
        site = sites[key]
        lat, lon = site["latitude"], site["longitude"]
        extra = {"clean_row": i, "site_source": site["source"], "exclusion_reason": "", "recode_note": ""}
        if pd.isna(lat) or pd.isna(lon):
            row = metadata_row(rec)
            extra["exclusion_reason"] = "site not found in the reported country"
        else:
            try:
                row = cg.archive_row(rec, lat, lon, site["display_name"])
            except Exception as exc:  # e.g. no sunset; none in the current data
                failures += 1
                row = metadata_row(rec)
                extra["exclusion_reason"] = f"geometry failed: {exc}"
        rows.append({**row, **extra})
    df = pd.DataFrame(rows, columns=cg.ARCHIVE_COLUMNS + ["clean_row", "site_source",
                                                         "exclusion_reason", "recode_note"])
    df["vis_class_reported"] = df["vis_class_empirical"]

    # Validity rules, for the reports that can enter the model.
    scope = ((df["section_type"] == "first_crescent") & (df["obs_relative_to_sunset"] == "after_sunset")
             & df["vis_class_reported"].notna())
    heading = df["raw_text"].map(icop_heading)
    seen = df["vis_class_reported"] > 0
    free = scope & (df["exclusion_reason"] == "")

    def apply(mask, reason):
        hit = free & mask & (df["exclusion_reason"] == "")
        df.loc[hit, "exclusion_reason"] = reason
        return int(hit.sum())

    counts = {"site not found in the reported country": int((scope & (df["exclusion_reason"] != "")).sum())}
    counts["R1"] = apply(seen & (heading == HEADING_NOT_SEEN), "R1: heading says not seen")
    counts["R2"] = apply(heading == HEADING_SEEN_BY_OTHERS, "R2: seen by others")
    lag = pd.to_numeric(df["moon_lag_minutes"], errors="coerce")
    counts["R3"] = apply(seen & (lag < 0), "R3: Moon set before the Sun")
    for clean_row, a in audit.iterrows():
        j = df.index[df["clean_row"] == clean_row][0]
        if not free[j] or df.at[j, "exclusion_reason"]:
            continue  # out of scope, or already excluded by an earlier rule
        if a["action"] == "exclude":
            df.at[j, "exclusion_reason"] = AUDIT_EXCLUSIONS[a["category"]]
        else:
            new = RECODE[a["action"]]
            if new != df.at[j, "vis_class_reported"]:
                df.at[j, "vis_class_empirical"] = new
                df.at[j, "recode_note"] = (f"R4: coded {int(df.at[j, 'vis_class_reported'])}, "
                                           f"the text gives {new}: {a['reason']}")
    df.loc[df["exclusion_reason"] != "", "vis_class_empirical"] = np.nan
    for reason in AUDIT_EXCLUSIONS.values():
        counts[reason] = int((scope & (df["exclusion_reason"] == reason)).sum())
    counts["R4 recoded"] = int((df["recode_note"] != "").sum())

    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    kept = df[scope & (df["exclusion_reason"] == "")]
    print(f"Rows: {len(df)} (geometry failures: {failures}); reports that can enter the model: "
          f"{int(scope.sum())}")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print("Recodes:", kept.loc[kept["recode_note"] != "", ["vis_class_reported", "vis_class_empirical"]]
          .value_counts().to_dict())
    print(f"Modelling rows: {len(kept)}, classes "
          f"{kept['vis_class_empirical'].value_counts().sort_index().astype(int).to_dict()}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
