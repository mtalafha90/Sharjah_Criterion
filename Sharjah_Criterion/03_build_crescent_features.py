from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from timezonefinder import TimezoneFinder

from moon_phases_core import build_daily_report, _sky_coords_at


INPUT_CSV = "crescent_observations_clean.csv"
OUTPUT_CSV = "crescent_features.csv"
GEOCODE_CACHE_JSON = "geocode_cache.json"

TF = TimezoneFinder(in_memory=True)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {
    "User-Agent": "crescent-criterion-builder/1.0 (research use)"
}

AR_MONTHS = {
    "يناير": 1,
    "فبراير": 2,
    "مارس": 3,
    "أبريل": 4,
    "ابريل": 4,
    "مايو": 5,
    "يونيو": 6,
    "يوليو": 7,
    "أغسطس": 8,
    "اغسطس": 8,
    "سبتمبر": 9,
    "أكتوبر": 10,
    "اكتوبر": 10,
    "نوفمبر": 11,
    "ديسمبر": 12,
}

WEEKDAYS_AR = {
    "الاثنين", "الثلاثاء", "الأربعاء", "الاربعاء", "الخميس", "الجمعة", "السبت", "الأحد", "الاحد"
}


def load_geocode_cache() -> dict:
    p = Path(GEOCODE_CACHE_JSON)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def save_geocode_cache(cache: dict) -> None:
    Path(GEOCODE_CACHE_JSON).write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def normalize_space(s: str | None) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def normalize_digits(s: str) -> str:
    if not s:
        return s
    table = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    return s.translate(table)


def parse_arabic_gregorian_date(date_text: str) -> datetime.date | None:
    s = normalize_digits(normalize_space(date_text))
    if not s:
        return None

    parts = s.split()
    parts = [p for p in parts if p not in WEEKDAYS_AR]

    # expected after cleanup: [day, month_name, year]
    if len(parts) < 3:
        m = re.search(r"(\d{1,2})\s+([^\d]+?)\s+(\d{4})", s)
        if not m:
            return None
        day = int(m.group(1))
        month_name = normalize_space(m.group(2))
        year = int(m.group(3))
    else:
        try:
            day = int(parts[0])
            month_name = normalize_space(parts[1])
            year = int(parts[2])
        except Exception:
            return None

    month = AR_MONTHS.get(month_name)
    if month is None:
        return None

    try:
        return datetime(year, month, day).date()
    except Exception:
        return None


def parse_obs_time(obs_time_text: str) -> tuple[int, int] | None:
    s = normalize_digits(normalize_space(obs_time_text))
    if not s:
        return None

    m = re.search(r"(\d{1,2})[:\.](\d{2})", s)
    if not m:
        return None

    hour = int(m.group(1))
    minute = int(m.group(2))

    # Arabic AM/PM handling
    if any(x in s for x in ["مساء", "مساءً", "م", "عصراً", "عصرا"]):
        if hour < 12:
            hour += 12
    elif any(x in s for x in ["صباح", "صباحاً", "صباحا"]):
        if hour == 12:
            hour = 0

    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return hour, minute
    return None


def extract_minutes_offset_from_sunset(obs_time_text: str) -> int | None:
    s = normalize_digits(normalize_space(obs_time_text))
    if not s:
        return None

    # examples like:
    # "بعد غروب الشمس بـ 29 دقيقة"
    # "قبل غروب الشمس بـ 10 دقائق"
    m_after = re.search(r"بعد غروب الشمس(?:\s*بـ)?\s*(\d+)\s*د", s)
    if m_after:
        return int(m_after.group(1))

    m_before = re.search(r"قبل غروب الشمس(?:\s*بـ)?\s*(\d+)\s*د", s)
    if m_before:
        return -int(m_before.group(1))

    return None


def geocode_place(city: str, province: str, country: str, cache: dict) -> dict | None:
    queries = []
    city = normalize_space(city)
    province = normalize_space(province)
    country = normalize_space(country)

    if city and province and country:
        queries.append(f"{city}, {province}, {country}")
    if city and country:
        queries.append(f"{city}, {country}")
    if province and country:
        queries.append(f"{province}, {country}")
    if city:
        queries.append(city)

    for q in queries:
        if q in cache:
            return cache[q]

        try:
            r = requests.get(
                NOMINATIM_URL,
                headers=HEADERS,
                params={
                    "q": q,
                    "format": "jsonv2",
                    "limit": 1
                },
                timeout=30
            )
            r.raise_for_status()
            arr = r.json()
            time.sleep(1.1)  # be polite to Nominatim

            if arr:
                hit = arr[0]
                out = {
                    "query": q,
                    "lat": float(hit["lat"]),
                    "lon": float(hit["lon"]),
                    "display_name": hit.get("display_name", q)
                }
                cache[q] = out
                return out

        except Exception:
            continue

    return None


def timezone_from_latlon(lat: float, lon: float) -> str | None:
    return TF.timezone_at(lat=lat, lng=lon)


def yallop_q(arcl_deg: float, arcv_deg: float) -> float | None:
    if arcl_deg is None or arcv_deg is None:
        return None
    return arcv_deg - (
        11.837
        + 6.322 * arcl_deg
        - 0.7319 * arcl_deg ** 2
        + 0.1018 * arcl_deg ** 3
    )


def yallop_class(arcl_deg: float, q: float | None) -> str:
    if arcl_deg is None or q is None:
        return "NO_DATA"
    if arcl_deg < 5 or arcl_deg > 30:
        return "OUTSIDE_REGIME"
    if q > 0.216:
        return "A"
    if q > 0.0:
        return "B"
    if q > -0.014:
        return "C"
    if q > -0.160:
        return "D"
    return "E"


def visibility_label_from_raw(row) -> int | None:
    # 0 = not seen, 1 = seen only with aid, 2 = naked-eye seen
    def is_true(v):
        return str(v).strip().lower() == "true"

    if is_true(row.get("seen_naked_eye", "")):
        return 2
    if any(is_true(row.get(c, "")) for c in ["seen_binocular", "seen_telescope", "seen_camera"]):
        return 1
    if str(row.get("seen_any", "")).strip().lower() == "false":
        return 0
    return None


def build_observation_datetime_local(
    obs_date,
    tz_name: str,
    obs_time_text: str,
    sunset_local_iso: str | None
) -> tuple[datetime | None, str]:
    tz = ZoneInfo(tz_name)

    sunset_dt = None
    if sunset_local_iso:
        sunset_dt = datetime.fromisoformat(sunset_local_iso)
        if sunset_dt.tzinfo is None:
            sunset_dt = sunset_dt.replace(tzinfo=tz)
        else:
            sunset_dt = sunset_dt.astimezone(tz)

    offset_min = extract_minutes_offset_from_sunset(obs_time_text or "")
    if offset_min is not None and sunset_dt is not None:
        return sunset_dt + timedelta(minutes=offset_min), "sunset_offset"

    hm = parse_obs_time(obs_time_text or "")
    if hm is not None:
        h, m = hm
        return datetime(obs_date.year, obs_date.month, obs_date.day, h, m, tzinfo=tz), "explicit_time"

    if sunset_dt is not None:
        return sunset_dt, "sunset_fallback"

    return None, "missing"


def main():
    df = pd.read_csv(INPUT_CSV)
    cache = load_geocode_cache()

    rows_out = []

    for i, row in df.iterrows():
        country = normalize_space(row.get("country"))
        city = normalize_space(row.get("city"))
        province = normalize_space(row.get("province"))
        gregorian_date = normalize_space(row.get("gregorian_date"))
        obs_time_text = normalize_space(row.get("obs_time_text"))

        obs_date = parse_arabic_gregorian_date(gregorian_date)
        if obs_date is None:
            print(f"[{i}] skipped: could not parse date -> {gregorian_date}")
            continue

        geo = geocode_place(city, province, country, cache)
        save_geocode_cache(cache)

        if geo is None:
            print(f"[{i}] skipped: could not geocode -> city={city}, province={province}, country={country}")
            continue

        lat = geo["lat"]
        lon = geo["lon"]
        tz_name = timezone_from_latlon(lat, lon)
        if not tz_name:
            print(f"[{i}] skipped: could not determine timezone -> lat={lat}, lon={lon}")
            continue

        try:
            daily = build_daily_report(
                day=obs_date,
                lat=lat,
                lon=lon,
                tz_name=tz_name,
                elevation_m=0.0,
            )
        except Exception as e:
            print(f"[{i}] skipped: build_daily_report failed -> {e}")
            continue

        obs_dt_local, obs_time_source = build_observation_datetime_local(
            obs_date=obs_date,
            tz_name=tz_name,
            obs_time_text=obs_time_text,
            sunset_local_iso=daily.get("sunset_local")
        )

        if obs_dt_local is None:
            print(f"[{i}] skipped: no usable observation time")
            continue

        try:
            sky = _sky_coords_at(
                dt_local=obs_dt_local,
                lat=lat,
                lon=lon,
                tz_name=tz_name
            )
        except Exception as e:
            print(f"[{i}] skipped: _sky_coords_at failed -> {e}")
            continue

        conjunction_local = daily.get("conjunction_local")
        moon_age_hours_obs = None
        if conjunction_local:
            try:
                conj_dt = datetime.fromisoformat(conjunction_local)
                if conj_dt.tzinfo is None:
                    conj_dt = conj_dt.replace(tzinfo=ZoneInfo(tz_name))
                else:
                    conj_dt = conj_dt.astimezone(ZoneInfo(tz_name))
                moon_age_hours_obs = (obs_dt_local - conj_dt).total_seconds() / 3600.0
            except Exception:
                moon_age_hours_obs = None

        q = yallop_q(
            arcl_deg=sky.get("elongation_deg"),
            arcv_deg=sky.get("relative_altitude_deg")
        )
        q_class = yallop_class(
            arcl_deg=sky.get("elongation_deg"),
            q=q
        )

        vis_class = visibility_label_from_raw(row)

        rows_out.append({
            # original metadata
            "page_url": row.get("page_url"),
            "hijri_year": row.get("hijri_year"),
            "hijri_month_name": row.get("hijri_month_name"),
            "section_type": row.get("section_type"),
            "gregorian_date": gregorian_date,
            "country": country,
            "city": city,
            "province": province,
            "obs_time_text": obs_time_text,
            "obs_relative_to_sunset": row.get("obs_relative_to_sunset"),
            "seen_any": row.get("seen_any"),
            "seen_naked_eye": row.get("seen_naked_eye"),
            "seen_binocular": row.get("seen_binocular"),
            "seen_telescope": row.get("seen_telescope"),
            "seen_camera": row.get("seen_camera"),
            "attempted_naked_eye": row.get("attempted_naked_eye"),
            "attempted_binocular": row.get("attempted_binocular"),
            "attempted_telescope": row.get("attempted_telescope"),
            "attempted_camera": row.get("attempted_camera"),
            "sky_cloud_text": row.get("sky_cloud_text"),
            "weather_text": row.get("weather_text"),
            "raw_text": row.get("raw_text"),
            "comment_text": row.get("comment_text"),

            # geocoding / timing
            "latitude": lat,
            "longitude": lon,
            "timezone": tz_name,
            "geocode_display_name": geo.get("display_name"),
            "obs_datetime_local": obs_dt_local.isoformat(),
            "obs_time_source": obs_time_source,

            # sunset-based daily report values
            "conjunction_local": daily.get("conjunction_local"),
            "sunset_local": daily.get("sunset_local"),
            "moonset_local": daily.get("moonset_local"),
            "best_time_local": daily.get("best_time_local"),
            "moon_age_hours_at_sunset": daily.get("moon_age_hours"),
            "moon_lag_minutes": daily.get("moon_lag_minutes"),
            "best_time_relative_altitude_deg": daily.get("best_time_relative_altitude_deg"),
            "best_time_crescent_width_arcmin": daily.get("best_time_crescent_width_arcmin"),

            # actual observation-time features
            "julian_date": sky.get("julian_date"),
            "sun_ra_hours": sky.get("sun_ra_hours"),
            "sun_dec_deg": sky.get("sun_dec_deg"),
            "moon_ra_hours": sky.get("moon_ra_hours"),
            "moon_dec_deg": sky.get("moon_dec_deg"),
            "sun_lon_deg": sky.get("sun_lon_deg"),
            "sun_lat_deg": sky.get("sun_lat_deg"),
            "moon_lon_deg": sky.get("moon_lon_deg"),
            "moon_lat_deg": sky.get("moon_lat_deg"),
            "sun_alt_deg": sky.get("sun_alt_deg"),
            "sun_az_deg": sky.get("sun_az_deg"),
            "moon_alt_deg": sky.get("moon_alt_deg"),
            "moon_az_deg": sky.get("moon_az_deg"),
            "relative_altitude_deg": sky.get("relative_altitude_deg"),
            "relative_azimuth_deg": sky.get("relative_azimuth_deg"),
            "elongation_deg": sky.get("elongation_deg"),
            "phase_angle_deg": sky.get("phase_angle_deg"),
            "illumination_percent": sky.get("illumination_percent"),
            "moon_semidiameter_arcmin": sky.get("moon_semidiameter_arcmin"),
            "crescent_width_arcmin": sky.get("crescent_width_arcmin"),
            "horizontal_parallax_deg": sky.get("horizontal_parallax_deg"),
            "distance_km": sky.get("distance_km"),
            "magnitude": sky.get("magnitude"),
            "moon_age_hours_obs": moon_age_hours_obs,

            # current criterion values
            "yallop_q": q,
            "yallop_class": q_class,

            # target label for learning
            "vis_class_empirical": vis_class,
        })

        print(
            f"[{i}] ok | {country} | {city or province} | "
            f"{obs_date.isoformat()} | {obs_time_source} | class={vis_class}"
        )

    out = pd.DataFrame(rows_out)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(out)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()