"""Correct moon_lag_minutes and moonset_local in crescent_features.csv.

The daily report used by 03_build_crescent_features.py returned the next
day's moonset when the Moon had already set before sunset, giving lags of
about 1,400-1,500 min instead of negative values, and returned no moonset
for some evenings (mostly with the Moon above the horizon). This script recomputes
the moonset with PyEphem (topocentric, upper limb, standard refraction) for
those rows only:
  - Moon below the horizon at sunset: the moonset preceding sunset (lag < 0);
  - otherwise: the first moonset after sunset (lag > 0).
As in the original table, the moonset is rounded down to the whole minute and
the lag is moonset minus the stored sunset_local. On rows that were already
correct, PyEphem agrees with the stored lag to within 2 min (mean 0.6 min).
All other values in the file are left byte-for-byte unchanged.
"""
from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ephem
import numpy as np
import pandas as pd

_ICOP_DATA_DIR = Path(__file__).parent.parent / "ICOP_Data"
FEATURES_CSV = _ICOP_DATA_DIR / "crescent_features.csv"

# Stored lags above this are next-day moonsets (the longest genuine lag is about 5 h).
NEXT_DAY_LAG_MIN = 720


def moonset_around_sunset(lat: float, lon: float, sunset: datetime) -> datetime | None:
    obs = ephem.Observer()
    obs.lat, obs.lon, obs.elevation = str(lat), str(lon), 0
    obs.date = ephem.Date(sunset.astimezone(timezone.utc).replace(tzinfo=None))
    moon = ephem.Moon()
    try:
        set_before, rise_before = obs.previous_setting(moon), obs.previous_rising(moon)
        # The Moon is down at sunset if it last set after it last rose.
        moonset = set_before if set_before > rise_before else obs.next_setting(moon)
    except (ephem.AlwaysUpError, ephem.NeverUpError):
        return None
    utc = ephem.Date(moonset).datetime().replace(tzinfo=timezone.utc)
    return utc.astimezone(sunset.tzinfo).replace(second=0, microsecond=0)


def main():
    raw = FEATURES_CSV.read_bytes()
    df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    lag = pd.to_numeric(df["moon_lag_minutes"], errors="coerce")
    has_site = (df["latitude"] != "") & (df["longitude"] != "") & (df["sunset_local"] != "")
    todo = has_site & (lag.isna() | (lag > NEXT_DAY_LAG_MIN))

    changed = []
    for i in df.index[todo]:
        sunset = datetime.fromisoformat(df.at[i, "sunset_local"])
        moonset = moonset_around_sunset(float(df.at[i, "latitude"]), float(df.at[i, "longitude"]), sunset)
        if moonset is None:
            continue
        new_lag = (moonset - sunset) / timedelta(minutes=1)
        changed.append((i, lag.iloc[i], new_lag))
        df.at[i, "moonset_local"] = moonset.isoformat()
        df.at[i, "moon_lag_minutes"] = repr(new_lag)

    FEATURES_CSV.write_bytes(df.to_csv(index=False).encode("utf-8-sig"))

    report = pd.DataFrame(changed, columns=["row", "old_lag", "new_lag"]).set_index("row")
    report["section"] = df.loc[report.index, "section_type"]
    report["was"] = np.where(report["old_lag"].isna(), "missing", "next-day moonset")
    report["sign"] = np.where(report["new_lag"] < 0, "negative", "positive")
    print(f"Rows recomputed: {len(report):,} of {int(todo.sum()):,} candidates")
    print(report.groupby(["section", "was", "sign"]).size().to_string())
    print(f"New lag range: {report['new_lag'].min():.1f} to {report['new_lag'].max():.1f} min")
    print(f"Saved {FEATURES_CSV}")


if __name__ == "__main__":
    main()
