#!/usr/bin/env python3
"""Sunset geometry of the crescent, used by 03d_rebuild_features.py.

Script 03 imports build_daily_report and _sky_coords_at from moon_phases_core,
which is not in this repository, so it cannot be rerun. This module computes
the same columns with PyEphem, in the conventions of the table written by
scripts 03 and 03b (ICOP_Data/crescent_features_script03.csv). Rebuilt from the
same dates and sites, all 6,092 rows of that table agree to within 0.0008 deg
in ARCV, ARCL, DAZ and the Moon's altitude, 0.0003 arcmin in W and 0.012 min in
the lag, with two exceptions: 423 moonsets differ by one minute, because the
original module rounded them slightly differently (the 1,242 moonsets that 03b
recomputed agree exactly), and 15 Moon ages differ by exactly 1 h, because
script 03 subtracted local clock times across a daylight-saving change, while
this module gives the true elapsed time.

compute_features(date, lat, lon) gives, for the evening of that local (civil)
date at that site, the 37 computed columns of the archive (FEATURE_COLUMNS):

  sunset       Sun's centre at altitude -0.8333 deg (the archive's sun_alt_deg),
               observer at elevation 0 with no refraction (pressure 0); the first
               sunset after local civil noon of the date, refined so that the
               Sun's altitude is -0.8333 deg to about 1e-6 deg;
  angles       airless topocentric, at that sunset: ARCV = Moon altitude - Sun
               altitude, ARCL = Sun-Moon separation, DAZ = |Moon azimuth - Sun
               azimuth| wrapped into [0, 180];
  distance     topocentric Earth-Moon distance (km); the semidiameter is
               s = asin(1737.4 km / distance) and the horizontal parallax
               asin(6378.137 km / distance), both with this distance;
  phase        geocentric phase angle i (Sun-Moon-Earth, from apparent
               geocentric positions), illuminated fraction k = (1 + cos i) / 2,
               illumination_percent = 100 k, magnitude = -12.73 - 2.5 log10(k);
  width        W = 2 s k (arcmin);
  RA/Dec, ecl. apparent topocentric positions on J2000 axes (nutation and
               precession removed), in hours / deg; ecliptic of J2000;
  moon lag     moonset - sunset (min), the moonset from PyEphem defaults (upper
               limb, standard refraction) rounded down to the whole minute, as in
               03b_correct_moon_lag.py: if the Moon is already down at sunset the
               moonset before sunset (negative lag), otherwise the first moonset
               after sunset (searched from 10 min before sunset, as in
               visibility_maps.py, because PyEphem can miss a moonset within
               seconds of the start time);
  best time    sunset + 4/9 * lag (Yallop's best time), with ARCV and W there;
               empty when the Moon sets before the Sun (lag < 0), as in the
               original module;
  moon age     sunset - the new moon before it (h); conjunction_local and the
               ages are empty when the nearest new moon is after sunset (archive
               convention; signed_age=True gives the nearest new moon and a
               negative age instead);
  observation  obs_datetime_local = sunset_local and obs_time_source =
               'sunset_fallback': no obs_time_text of the clean table gives a
               clock time or an offset from sunset, so script 03 falls back to
               sunset for every report;
  Yallop       yallop_q and yallop_class with the formulas of script 03 as they
               are (they use ARCL where Yallop uses W and do not divide by 10;
               no downstream script reads them).

Times are ISO strings in the site's IANA time zone (timezonefinder unless
tz_name is given), as written by datetime.isoformat(). All time differences are
taken in UTC.

archive_row(clean_row, lat, lon, display_name) gives a whole 64-column row of
crescent_features.csv, as script 03 writes it, for one row of
crescent_observations_clean.csv read with pandas defaults; report_date(text)
reads the Arabic report date, Monday included.
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import ephem

AU_KM = 149597870.7
MOON_RADIUS_KM = 1737.4
EARTH_RADIUS_KM = 6378.137
SUN_ALT_AT_SUNSET = -0.8333  # deg; the archive's sun_alt_deg
MOONSET_SEARCH_LEAD = 10 * ephem.minute
BEST_TIME_FRACTION = 4 / 9
ARCSEC = math.pi / (180 * 3600)

# Columns copied from crescent_observations_clean.csv by script 03.
METADATA_COLUMNS = [
    "page_url", "hijri_year", "hijri_month_name", "section_type", "gregorian_date",
    "country", "city", "province", "obs_time_text", "obs_relative_to_sunset",
    "seen_any", "seen_naked_eye", "seen_binocular", "seen_telescope", "seen_camera",
    "attempted_naked_eye", "attempted_binocular", "attempted_telescope",
    "attempted_camera", "sky_cloud_text", "weather_text", "raw_text", "comment_text",
]
# Columns computed by compute_features.
FEATURE_COLUMNS = [
    "timezone", "obs_datetime_local", "obs_time_source",
    "conjunction_local", "sunset_local", "moonset_local", "best_time_local",
    "moon_age_hours_at_sunset", "moon_lag_minutes",
    "best_time_relative_altitude_deg", "best_time_crescent_width_arcmin",
    "julian_date", "sun_ra_hours", "sun_dec_deg", "moon_ra_hours", "moon_dec_deg",
    "sun_lon_deg", "sun_lat_deg", "moon_lon_deg", "moon_lat_deg",
    "sun_alt_deg", "sun_az_deg", "moon_alt_deg", "moon_az_deg",
    "relative_altitude_deg", "relative_azimuth_deg", "elongation_deg",
    "phase_angle_deg", "illumination_percent", "moon_semidiameter_arcmin",
    "crescent_width_arcmin", "horizontal_parallax_deg", "distance_km", "magnitude",
    "moon_age_hours_obs", "yallop_q", "yallop_class",
]
# All columns of crescent_features.csv, in its order.
ARCHIVE_COLUMNS = (METADATA_COLUMNS + ["latitude", "longitude", "timezone", "geocode_display_name"]
                   + FEATURE_COLUMNS[1:] + ["vis_class_empirical"])

_TF = None


# ---------------------------------------------------------------- helpers
def timezone_at(lat: float, lon: float) -> str | None:
    """IANA time zone of the site, found as in script 03 (timezonefinder)."""
    global _TF
    if _TF is None:
        from timezonefinder import TimezoneFinder
        _TF = TimezoneFinder(in_memory=True)
    return _TF.timezone_at(lat=lat, lng=lon)


def _observer(lat: float, lon: float, airless: bool) -> ephem.Observer:
    obs = ephem.Observer()
    obs.lat, obs.lon, obs.elevation = str(lat), str(lon), 0
    if airless:
        obs.pressure = 0  # no refraction: geometric topocentric positions
    return obs


def _utc(d: ephem.Date) -> datetime:
    return ephem.Date(d).datetime().replace(tzinfo=timezone.utc)


def _ephem_date(dt: datetime) -> ephem.Date:
    return ephem.Date(dt.astimezone(timezone.utc).replace(tzinfo=None))


def _nutation(jd: float) -> tuple[float, float, float]:
    """Nutation in longitude and obliquity and the mean obliquity (rad), to about 0.5 arcsec."""
    t = (jd - 2451545.0) / 36525
    sun = math.radians(280.4665 + 36000.7698 * t)
    moon = math.radians(218.3165 + 481267.8813 * t)
    node = math.radians(125.04452 - 1934.136261 * t)
    dpsi = (-17.20 * math.sin(node) - 1.32 * math.sin(2 * sun) - 0.23 * math.sin(2 * moon)
            + 0.21 * math.sin(2 * node)) * ARCSEC
    deps = (9.20 * math.cos(node) + 0.57 * math.cos(2 * sun) + 0.10 * math.cos(2 * moon)
            - 0.09 * math.cos(2 * node)) * ARCSEC
    eps0 = math.radians(23.4392911111 - (46.8150 * t + 0.00059 * t ** 2 - 0.001813 * t ** 3) / 3600)
    return dpsi, deps, eps0


def _j2000_axes(ra: float, dec: float, when: ephem.Date) -> ephem.Equatorial:
    """An apparent position of date (true equator and equinox) on J2000 axes."""
    dpsi, deps, eps0 = _nutation(ephem.julian_date(when))
    eps = eps0 + deps
    x, y, z = math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)
    lon = math.atan2(y * math.cos(eps) + z * math.sin(eps), x) - dpsi  # true -> mean ecliptic
    lat = math.asin(-y * math.sin(eps) + z * math.cos(eps))
    x, y, z = math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)
    mean_ra = math.atan2(y * math.cos(eps0) - z * math.sin(eps0), x) % (2 * math.pi)
    mean_dec = math.asin(y * math.sin(eps0) + z * math.cos(eps0))
    return ephem.Equatorial(ephem.Equatorial(mean_ra, mean_dec, epoch=when), epoch=ephem.J2000)


def _phase_angle(when: ephem.Date) -> float:
    """Geocentric phase angle of the Moon (rad), from apparent geocentric positions."""
    sun, moon = ephem.Sun(when), ephem.Moon(when)

    def vec(body):
        d = body.earth_distance * AU_KM
        return [d * math.cos(body.g_dec) * math.cos(body.g_ra),
                d * math.cos(body.g_dec) * math.sin(body.g_ra), d * math.sin(body.g_dec)]

    s, m = vec(sun), vec(moon)
    to_sun, to_earth = [p - q for p, q in zip(s, m)], [-q for q in m]
    dot = sum(p * q for p, q in zip(to_sun, to_earth))
    norm = math.hypot(*to_sun) * math.hypot(*to_earth)
    return math.acos(max(-1.0, min(1.0, dot / norm)))


def _sky(lat: float, lon: float, when: ephem.Date) -> dict:
    """Positions and crescent quantities at one instant, in the archive's conventions."""
    obs = _observer(lat, lon, airless=True)
    obs.date = when
    sun, moon = ephem.Sun(obs), ephem.Moon(obs)
    out = {}
    for name, body in (("sun", sun), ("moon", moon)):
        eq = _j2000_axes(body.ra, body.dec, when)
        ecl = ephem.Ecliptic(eq, epoch=ephem.J2000)
        out.update({f"{name}_ra_hours": math.degrees(eq.ra) / 15, f"{name}_dec_deg": math.degrees(eq.dec),
                    f"{name}_lon_deg": math.degrees(ecl.lon), f"{name}_lat_deg": math.degrees(ecl.lat),
                    f"{name}_alt_deg": math.degrees(body.alt), f"{name}_az_deg": math.degrees(body.az)})
    daz = abs(math.degrees(moon.az - sun.az)) % 360
    distance = moon.earth_distance * AU_KM  # topocentric, because the Moon is computed for an observer
    phase = _phase_angle(when)
    k = (1 + math.cos(phase)) / 2
    semidiameter = math.degrees(math.asin(MOON_RADIUS_KM / distance)) * 60
    out.update({
        "relative_altitude_deg": math.degrees(moon.alt - sun.alt),
        "relative_azimuth_deg": 360 - daz if daz > 180 else daz,
        "elongation_deg": math.degrees(ephem.separation(sun, moon)),
        "phase_angle_deg": math.degrees(phase),
        "illumination_percent": 100 * k,
        "moon_semidiameter_arcmin": semidiameter,
        "crescent_width_arcmin": 2 * semidiameter * k,
        "horizontal_parallax_deg": math.degrees(math.asin(EARTH_RADIUS_KM / distance)),
        "distance_km": distance,
        "magnitude": -12.73 - 2.5 * math.log10(k) if k > 0 else math.inf,
    })
    return out


# ---------------------------------------------------------------- events
def evening_sunset(day: date, lat: float, lon: float, tz: ZoneInfo) -> ephem.Date:
    """Sunset (Sun's centre at -0.8333 deg, airless) after local civil noon of 'day'."""
    obs = _observer(lat, lon, airless=True)
    obs.horizon = str(SUN_ALT_AT_SUNSET)  # a string is read as degrees
    obs.date = _ephem_date(datetime.combine(day, time(12), tzinfo=tz))
    t = obs.next_setting(ephem.Sun(), use_center=True)
    # PyEphem stops within about 3e-4 deg of the horizon; two secant steps reach 1e-6 deg.
    h0 = math.radians(SUN_ALT_AT_SUNSET)
    for _ in range(2):
        obs.date = t
        f0 = ephem.Sun(obs).alt - h0
        obs.date = t + ephem.second
        f1 = ephem.Sun(obs).alt - h0
        t = t - f0 / (f1 - f0) * ephem.second
    return ephem.Date(t)


def moonset_around_sunset(lat: float, lon: float, sunset: ephem.Date) -> ephem.Date | None:
    """Moonset of 03b_correct_moon_lag.py (PyEphem defaults: upper limb, refraction)."""
    obs = _observer(lat, lon, airless=False)
    obs.date = sunset
    moon = ephem.Moon()
    try:
        set_before, rise_before = obs.previous_setting(moon), obs.previous_rising(moon)
        if set_before > rise_before:  # the Moon last set after it last rose: down at sunset
            return ephem.Date(set_before)
        obs.date = sunset - MOONSET_SEARCH_LEAD
        return ephem.Date(obs.next_setting(moon))
    except (ephem.AlwaysUpError, ephem.NeverUpError):
        return None


def nearest_new_moon(when: ephem.Date) -> ephem.Date:
    """The new moon (geocentric conjunction in ecliptic longitude) closest to 'when'."""
    prev, nxt = ephem.previous_new_moon(when), ephem.next_new_moon(when)
    return ephem.Date(prev if when - prev <= nxt - when else nxt)


# ---------------------------------------------------------------- script 03 formulas
def yallop_q(arcl_deg: float, arcv_deg: float) -> float:
    """yallop_q as script 03 computes it (ARCL in place of W, no division by 10)."""
    return arcv_deg - (11.837 + 6.322 * arcl_deg - 0.7319 * arcl_deg ** 2 + 0.1018 * arcl_deg ** 3)


def yallop_class(arcl_deg: float, q: float) -> str:
    """yallop_class as script 03 assigns it."""
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


# ---------------------------------------------------------------- main entry
def compute_features(date_str: str, lat: float, lon: float, tz_name: str | None = None,
                     signed_age: bool = False, wall_clock_age: bool = False) -> dict:
    """The 37 computed archive columns for the evening of 'date_str' (YYYY-MM-DD) at a site.

    signed_age: use the nearest new moon even when it is after sunset, giving a
        negative age, instead of leaving conjunction and age empty (archive).
    wall_clock_age: reproduce the archive's ages, which subtract the local clock
        times of sunset and conjunction and so are 1 h off when a daylight-saving
        change falls between them (15 archive rows). By default the ages are
        true elapsed hours.
    Raises ephem.AlwaysUpError or ephem.NeverUpError if the Sun does not set.
    """
    lat, lon = float(lat), float(lon)
    day = date.fromisoformat(date_str)
    tz_name = tz_name or timezone_at(lat, lon)
    if not tz_name:
        raise ValueError(f"no time zone found for lat={lat}, lon={lon}")
    tz = ZoneInfo(tz_name)

    sunset = evening_sunset(day, lat, lon, tz)
    sunset_utc = _utc(sunset)
    sunset_local = sunset_utc.astimezone(tz).isoformat()
    out = {"timezone": tz_name, "sunset_local": sunset_local, "obs_datetime_local": sunset_local,
           "obs_time_source": "sunset_fallback", "julian_date": ephem.julian_date(sunset)}
    out.update(_sky(lat, lon, sunset))
    q = yallop_q(out["elongation_deg"], out["relative_altitude_deg"])
    out.update(yallop_q=q, yallop_class=yallop_class(out["elongation_deg"], q))

    # Moon lag, and Yallop's best time when the Moon sets after the Sun.
    out.update(moonset_local=None, moon_lag_minutes=math.nan, best_time_local=None,
               best_time_relative_altitude_deg=math.nan, best_time_crescent_width_arcmin=math.nan)
    moonset = moonset_around_sunset(lat, lon, sunset)
    if moonset is not None:
        moonset_local = _utc(moonset).astimezone(tz).replace(second=0, microsecond=0)
        lag = (moonset_local.astimezone(timezone.utc) - sunset_utc) / timedelta(minutes=1)
        out.update(moonset_local=moonset_local.isoformat(), moon_lag_minutes=lag)
        if lag >= 0:
            best_utc = sunset_utc + timedelta(minutes=BEST_TIME_FRACTION * lag)
            best = _sky(lat, lon, _ephem_date(best_utc))
            out.update(best_time_local=best_utc.astimezone(tz).isoformat(),
                       best_time_relative_altitude_deg=best["relative_altitude_deg"],
                       best_time_crescent_width_arcmin=best["crescent_width_arcmin"])

    # Moon age at sunset (= at the observation time, which is the sunset).
    conj = nearest_new_moon(sunset)
    if conj > sunset and not signed_age:
        out.update(conjunction_local=None, moon_age_hours_at_sunset=math.nan, moon_age_hours_obs=math.nan)
    else:
        conj_utc = _utc(conj)
        age = (sunset_utc - conj_utc) / timedelta(hours=1)
        if wall_clock_age:
            age = (sunset_utc.astimezone(tz).replace(tzinfo=None)
                   - conj_utc.astimezone(tz).replace(tzinfo=None)) / timedelta(hours=1)
        out.update(conjunction_local=conj_utc.astimezone(tz).isoformat(),
                   moon_age_hours_at_sunset=age, moon_age_hours_obs=age)
    return {c: out[c] for c in FEATURE_COLUMNS}


# ---------------------------------------------------------------- whole archive rows
AR_MONTHS = {"يناير": 1, "فبراير": 2, "مارس": 3, "أبريل": 4, "ابريل": 4, "مايو": 5, "يونيو": 6,
             "يوليو": 7, "أغسطس": 8, "اغسطس": 8, "سبتمبر": 9, "أكتوبر": 10, "اكتوبر": 10,
             "نوفمبر": 11, "ديسمبر": 12}


def normalize_space(s) -> str:
    """As in script 03 (a missing value becomes the string 'nan')."""
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def report_date(text) -> str | None:
    """The report date as YYYY-MM-DD, from e.g. 'الإثنين 02 ديسمبر 2013'.

    Script 03 drops a weekday only if it is in its list, which spells Monday
    'الاثنين'; the reports spell it 'الإثنين', so every Monday report failed to
    parse (981 of the 7,124 clean rows). This parser ignores the weekday and
    otherwise gives the same date as script 03.
    """
    s = normalize_space(text).translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    m = re.search(r"(\d{1,2})\s+(\S+)\s+(\d{4})", s)
    if not m or m.group(2) not in AR_MONTHS:
        return None
    try:
        return date(int(m.group(3)), AR_MONTHS[m.group(2)], int(m.group(1))).isoformat()
    except ValueError:
        return None


def visibility_label(row) -> int | None:
    """vis_class_empirical as script 03 assigns it: 0 not seen, 1 aid only, 2 naked eye."""
    def is_true(v):
        return str(v).strip().lower() == "true"

    if is_true(row.get("seen_naked_eye", "")):
        return 2
    if any(is_true(row.get(c, "")) for c in ["seen_binocular", "seen_telescope", "seen_camera"]):
        return 1
    if str(row.get("seen_any", "")).strip().lower() == "false":
        return 0
    return None


def archive_row(clean_row, lat: float, lon: float, geocode_display_name: str, **kwargs) -> dict:
    """A whole crescent_features.csv row (ARCHIVE_COLUMNS) for one clean observation.

    clean_row: a row of crescent_observations_clean.csv read with pandas defaults
    (a Series or dict). lat, lon, geocode_display_name: the site. kwargs go to
    compute_features. Metadata are copied as script 03 copies them.
    """
    day = report_date(clean_row.get("gregorian_date"))
    if day is None:
        raise ValueError(f"cannot read the report date {clean_row.get('gregorian_date')!r}")
    row = {c: clean_row.get(c) for c in METADATA_COLUMNS}
    for c in ("gregorian_date", "country", "city", "province", "obs_time_text"):
        row[c] = normalize_space(clean_row.get(c))
    row.update(latitude=float(lat), longitude=float(lon), geocode_display_name=geocode_display_name)
    row.update(compute_features(day, lat, lon, **kwargs))
    row["vis_class_empirical"] = visibility_label(clean_row)
    return {c: row[c] for c in ARCHIVE_COLUMNS}


if __name__ == "__main__":
    import json
    print(json.dumps(compute_features("2026-05-17", 25.3463, 55.4209), indent=2))
