from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

_ICOP_DATA_DIR = Path(__file__).parent.parent / "ICOP_Data"

INPUT_CSV = _ICOP_DATA_DIR / "crescent_observations_raw.csv"
OUTPUT_CSV = _ICOP_DATA_DIR / "crescent_observations_clean.csv"
REJECTED_CSV = _ICOP_DATA_DIR / "crescent_observations_rejected.csv"


BOOL_COLS = [
    "seen_any",
    "seen_naked_eye",
    "seen_binocular",
    "seen_telescope",
    "seen_camera",
    "attempted_naked_eye",
    "attempted_binocular",
    "attempted_telescope",
    "attempted_camera",
]

TEXT_COLS = [
    "page_url",
    "hijri_month_name",
    "section_type",
    "gregorian_date",
    "country",
    "city",
    "province",
    "obs_time_text",
    "obs_relative_to_sunset",
    "sky_cloud_text",
    "weather_text",
    "raw_text",
    "comment_text",
]


def normalize_space(s) -> str:
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def normalize_digits(s: str) -> str:
    table = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    return s.translate(table)


def norm_bool(v):
    s = normalize_space(v).lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def build_target(row):
    # 0 = not seen
    # 1 = seen only with aid/instrument
    # 2 = seen with naked eye
    seen_naked = row.get("seen_naked_eye")
    seen_binoc = row.get("seen_binocular")
    seen_tel = row.get("seen_telescope")
    seen_cam = row.get("seen_camera")
    seen_any = row.get("seen_any")

    if seen_naked is True:
        return 2
    if seen_binoc is True or seen_tel is True or seen_cam is True:
        return 1
    if seen_any is False:
        return 0
    return None


def looks_like_real_observation(row) -> bool:
    raw_text = normalize_space(row.get("raw_text"))
    obs_time_text = normalize_space(row.get("obs_time_text"))

    # basic sanity: most real entries mention observation time
    if "وقت الرصد" in raw_text or obs_time_text:
        return True
    return False


def main():
    df = pd.read_csv(INPUT_CSV)

    # Normalize text
    for c in TEXT_COLS:
        if c in df.columns:
            df[c] = df[c].apply(normalize_space)
            if c in {"gregorian_date", "obs_time_text", "raw_text", "comment_text"}:
                df[c] = df[c].apply(normalize_digits)

    # Normalize bools
    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].apply(norm_bool)

    # Numeric year if present
    if "hijri_year" in df.columns:
        df["hijri_year"] = pd.to_numeric(df["hijri_year"], errors="coerce")

    # Build target
    df["vis_class_empirical"] = df.apply(build_target, axis=1)

    # Flags
    df["has_country"] = df.get("country", "").astype(str).str.len() > 0
    df["has_city_or_province"] = (
        (df.get("city", "").astype(str).str.len() > 0) |
        (df.get("province", "").astype(str).str.len() > 0)
    )
    df["has_date"] = df.get("gregorian_date", "").astype(str).str.len() > 0
    df["has_obs_time_text"] = df.get("obs_time_text", "").astype(str).str.len() > 0
    df["looks_like_real_observation"] = df.apply(looks_like_real_observation, axis=1)

    # Reject rows with no usable visibility target
    reject_mask = df["vis_class_empirical"].isna()

    # Reject rows that do not even look like real observation entries
    reject_mask |= ~df["looks_like_real_observation"]

    rejected = df[reject_mask].copy()
    clean = df[~reject_mask].copy()

    # Drop obvious duplicates
    dedup_cols = [
        "page_url",
        "gregorian_date",
        "country",
        "city",
        "province",
        "obs_time_text",
        "raw_text",
    ]
    dedup_cols = [c for c in dedup_cols if c in clean.columns]
    clean = clean.drop_duplicates(subset=dedup_cols).copy()

    # Sort for readability
    sort_cols = [c for c in ["hijri_year", "hijri_month_name", "gregorian_date", "country", "city"] if c in clean.columns]
    if sort_cols:
        clean = clean.sort_values(sort_cols).reset_index(drop=True)

    clean.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    rejected.to_csv(REJECTED_CSV, index=False, encoding="utf-8-sig")

    print(f"Input rows:     {len(df)}")
    print(f"Clean rows:     {len(clean)}")
    print(f"Rejected rows:  {len(rejected)}")
    print("\nTarget distribution:")
    print(clean["vis_class_empirical"].value_counts(dropna=False).sort_index())
    print(f"\nSaved cleaned file to:   {OUTPUT_CSV}")
    print(f"Saved rejected file to:  {REJECTED_CSV}")


if __name__ == "__main__":
    main()