#!/usr/bin/env python3
"""Reproduce the local sunset-geometry benchmark in the revised manuscript.

Input: data/icop/crescent_features.csv, and
the out-of-fold Sharjah probabilities two_stage_oof_predictions.csv written by
script 04 (read, not refitted; by default from the same folder).
These tests are pointwise screens, not complete calendar-authority procedures.

Contradiction rates follow HilalPy (hilalpy 0.0.4 equa.py):
  positive = share of sightings (seen by any means) that the criterion calls not visible;
  negative = share of non-sightings that the criterion calls visible.
The positive rate is also split by observing method (naked-eye and aided-only
sightings), as in HilalPy's layout. ICOP non-sightings are 'not seen by any
means', so the negative rate is not split. 95 % intervals: cluster bootstrap
over lunations (2,000 resamples, default_rng(42)).
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def load_sibling(name):
    """A script in this folder; loaded by path, so it also works under 'python3 -I'."""
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize(df, name, flag):
    y_seen = df["vis_class_empirical"].to_numpy() > 0
    p = flag.to_numpy(dtype=bool)
    return {
        "criterion": name,
        "N": len(df),
        "binary_detection_accuracy": round(float((p == y_seen).mean()), 6),
        "no_detection_above_boundary": int((p & ~y_seen).sum()),
        "no_detection_total": int((~y_seen).sum()),
        "detection_below_boundary": int((~p & y_seen).sum()),
        "detection_total": int(y_seen.sum()),
    }


def contradiction_rates(y, flag, codes, counts, interval):
    """Point estimates and 95 % lunation-bootstrap intervals of the four rates."""
    p = np.asarray(flag, dtype=bool)
    parts = {
        "Positive, all sightings": (~p, y > 0),
        "Positive, naked-eye sightings": (~p, y == 2),
        "Positive, aided-only sightings": (~p, y == 1),
        "Negative, non-sightings": (p, y == 0),
    }
    out = {}
    for label, (hit, den) in parts.items():
        # A resample repeats each lunation as often as it was drawn.
        num_l = np.bincount(codes, weights=(hit & den).astype(float), minlength=counts.shape[1])
        den_l = np.bincount(codes, weights=den.astype(float), minlength=counts.shape[1])
        lo, hi = interval((counts @ num_l) / (counts @ den_l))
        out[label] = (int(num_l.sum()), int(den_l.sum()), num_l.sum() / den_l.sum(), lo, hi)
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="Path to data/icop/crescent_features.csv")
    parser.add_argument("--oof", help="two_stage_oof_predictions.csv from script 04 "
                                      "(default: next to crescent_features.csv)")
    args = parser.parse_args()
    oof_path = Path(args.oof) if args.oof else Path(args.csv).parent / "two_stage_oof_predictions.csv"

    df = pd.read_csv(args.csv, low_memory=False)
    df = df[
        (df["section_type"] == "first_crescent")
        & (df["obs_relative_to_sunset"] != "before_sunset")
        & df["vis_class_empirical"].notna()
    ].reset_index(drop=True)
    print(f"Filtered modelling records: {len(df)}")

    # The out-of-fold file holds the same rows in the same order (script 04).
    oof = pd.read_csv(oof_path, low_memory=False)
    if (len(oof) != len(df) or not (oof["page_url"].to_numpy() == df["page_url"].to_numpy()).all()
            or not (oof["vis_class_empirical"].to_numpy() == df["vis_class_empirical"].to_numpy()).all()):
        raise ValueError(f"{oof_path} does not hold the modelling rows in the same order")
    df["p_det"] = oof["p_det"].to_numpy()

    fields = ["moon_alt_deg", "elongation_deg", "moon_age_hours_at_sunset"]
    df[fields] = df[fields].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=fields).copy()
    print(f"Records with Moon altitude, elongation and age at sunset: {len(df)}")
    if not df["obs_time_source"].eq("sunset_fallback").all():
        raise ValueError("Geometry is not consistently at sunset; do not compare without recalculation")
    altitude = df["moon_alt_deg"]
    elongation = df["elongation_deg"]
    age = df["moon_age_hours_at_sunset"]

    criteria = {
        # MABIMS 1995 (2-3-8): altitude >=2 AND elongation >=3, OR age >=8 h.
        "MABIMS 1995 (local)": ((altitude >= 2) & (elongation >= 3)) | (age >= 8),
        # MABIMS revised criterion: >=3 degrees altitude AND >=6.4 degrees elongation.
        "MABIMS 2021 (local)": (altitude >= 3) & (elongation >= 6.4),
        # Istanbul 2016: *local geometric screening only*, NOT its global calendar rule.
        "Istanbul 2016 (local)": (altitude >= 5) & (elongation >= 8),
        # Sharjah detection decision, out-of-fold (models never fitted on the same lunation).
        "Sharjah (P_det >= 0.50, out-of-fold)": df["p_det"] >= 0.50,
        # The lower band boundary of Eq. 11 (E: P_det < 0.30), for a more lenient screen.
        "Sharjah (P_det >= 0.30, out-of-fold)": df["p_det"] >= 0.30,
    }

    output = pd.DataFrame([summarize(df, name, flag) for name, flag in criteria.items()])
    print(output.to_string(index=False))
    print("These are binary local geometric screens on observation-labelled records.")
    print("Istanbul's global geographic calendar rule is not evaluated.")

    # Lunation resampling as in table1_criteria_comparison.py (2,000 resamples, seed 42).
    t1 = load_sibling("table1_criteria_comparison")
    y = df["vis_class_empirical"].astype(int).to_numpy()
    codes, counts = t1.lunation_resamples(df["page_url"])
    print(f"\nContradiction rates (HilalPy definitions), {len(df):,} records, "
          f"{counts.shape[1]} lunations; 95 % cluster-bootstrap intervals over lunations")
    print(f"Classes 0/1/2 = {np.bincount(y, minlength=3).tolist()}")
    for name, flag in criteria.items():
        print(f"\n{name}")
        for label, (k, n, r, lo, hi) in contradiction_rates(y, flag, codes, counts, t1.interval).items():
            print(f"  {label:32s} {r:.4f} [{lo:.4f}, {hi:.4f}]  ({k:,}/{n:,})")


if __name__ == "__main__":
    main()
