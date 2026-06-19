from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).parent

POSITIVE_CSV = BASE_DIR / "Table_I_positive_observations.csv"
NEGATIVE_CSV = BASE_DIR / "Table_II_negative_observations.csv"

MODEL_META_JSON = BASE_DIR / "two_stage_model_meta.json"
STAGE1_MODEL_JOBLIB = BASE_DIR / "stage1_detectability_model.joblib"
STAGE2_MODEL_JOBLIB = BASE_DIR / "stage2_naked_eye_model.joblib"

OUT_STAGE1_CSV = BASE_DIR / "alrefay_stage1_rowwise_comparison.csv"
OUT_STAGE2_CSV = BASE_DIR / "alrefay_stage2_rowwise_comparison.csv"
OUT_REPORT_JSON = BASE_DIR / "alrefay_rowwise_validation_report.json"

# Operational thresholds used by your saved model
PDET_DETECTABLE = 0.50
PEYE_NAKED_EYE = 0.60


# ============================================================
# HELPERS
# ============================================================
def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make column names easier to match.
    """
    new_cols = {}
    for c in df.columns:
        cc = str(c).strip()
        cc = re.sub(r"\s+", "_", cc)
        new_cols[c] = cc
    return df.rename(columns=new_cols)


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def require_columns(df: pd.DataFrame, required: list[str], label: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing required columns: {missing}")


def load_models():
    meta = json.loads(MODEL_META_JSON.read_text(encoding="utf-8"))

    stage1_path = Path(meta.get("stage1_model", str(STAGE1_MODEL_JOBLIB)))
    stage2_path = Path(meta.get("stage2_model", str(STAGE2_MODEL_JOBLIB)))

    stage1 = joblib.load(stage1_path)
    stage2 = joblib.load(stage2_path)

    # Compatibility patch if needed
    for model in (stage1, stage2):
        clf = model.named_steps.get("clf") if hasattr(model, "named_steps") else None
        if clf is not None and not hasattr(clf, "multi_class"):
            clf.multi_class = "ovr"

    feature_cols = meta.get(
        "feature_cols",
        ["relative_altitude_deg", "elongation_deg", "moon_lag_minutes"]
    )
    return stage1, stage2, feature_cols


def parse_means_to_eye_label(value) -> int | None:
    """
    Interpret the Means column from the positive table.

    Returns:
        1 -> naked eye
        0 -> aided / telescope / instrument
        None -> unclear
    """
    s = str(value).strip().upper()

    # Usually the OCR-extracted field ends with N or T
    if re.search(r"N\s*$", s):
        return 1
    if re.search(r"T\s*$", s):
        return 0

    return None


def prepare_positive_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_colnames(df)

    lag_col = find_column(df, ["Lag_min", "Lag", "lag_min"])
    arcv_col = find_column(df, ["ARCV_deg", "ArcV", "ARCV", "arcv_deg"])
    arcl_col = find_column(df, ["ARCL_deg", "ArcL", "ARCL", "arcl_deg"])
    means_col = find_column(df, ["Means", "means"])

    if lag_col is None or arcv_col is None or arcl_col is None:
        raise ValueError(
            "Positive table must contain columns for Lag_min, ARCV_deg, and ARCL_deg "
            "(or equivalent names such as Lag, ArcV, ArcL)."
        )

    df = df.rename(columns={
        lag_col: "Lag_min",
        arcv_col: "ARCV_deg",
        arcl_col: "ARCL_deg",
    })

    if means_col is not None:
        df = df.rename(columns={means_col: "Means"})
    else:
        df["Means"] = np.nan

    df["Lag_min"] = pd.to_numeric(df["Lag_min"], errors="coerce")
    df["ARCV_deg"] = pd.to_numeric(df["ARCV_deg"], errors="coerce")
    df["ARCL_deg"] = pd.to_numeric(df["ARCL_deg"], errors="coerce")

    df["target_detectability"] = 1
    df["target_naked_eye"] = df["Means"].apply(parse_means_to_eye_label)

    return df


def prepare_negative_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_colnames(df)

    lag_col = find_column(df, ["Lag_min", "Lag", "lag_min"])
    arcv_col = find_column(df, ["ARCV_deg", "ArcV", "ARCV", "arcv_deg"])
    arcl_col = find_column(df, ["ARCL_deg", "ArcL", "ARCL", "arcl_deg"])

    if lag_col is None or arcv_col is None or arcl_col is None:
        raise ValueError(
            "Negative table must contain columns for Lag_min, ARCV_deg, and ARCL_deg "
            "(or equivalent names such as Lag, ArcV, ArcL)."
        )

    df = df.rename(columns={
        lag_col: "Lag_min",
        arcv_col: "ARCV_deg",
        arcl_col: "ARCL_deg",
    })

    df["Lag_min"] = pd.to_numeric(df["Lag_min"], errors="coerce")
    df["ARCV_deg"] = pd.to_numeric(df["ARCV_deg"], errors="coerce")
    df["ARCL_deg"] = pd.to_numeric(df["ARCL_deg"], errors="coerce")

    df["target_detectability"] = 0

    return df


def predict_stage1(row: pd.Series, model, feature_cols: list[str]) -> tuple[int, float]:
    feature_map = {
        "relative_altitude_deg": float(row["ARCV_deg"]),
        "elongation_deg": float(row["ARCL_deg"]),
        "moon_lag_minutes": float(row["Lag_min"]),
    }

    X = pd.DataFrame([{col: feature_map[col] for col in feature_cols}])
    p_det = float(model.predict_proba(X)[0, 1])
    pred = 1 if p_det >= PDET_DETECTABLE else 0
    return pred, p_det


def predict_stage2(row: pd.Series, model, feature_cols: list[str]) -> tuple[int, float]:
    feature_map = {
        "relative_altitude_deg": float(row["ARCV_deg"]),
        "elongation_deg": float(row["ARCL_deg"]),
        "moon_lag_minutes": float(row["Lag_min"]),
    }

    X = pd.DataFrame([{col: feature_map[col] for col in feature_cols}])
    p_eye = float(model.predict_proba(X)[0, 1])
    pred = 1 if p_eye >= PEYE_NAKED_EYE else 0
    return pred, p_eye


def add_match_columns(df: pd.DataFrame, truth_col: str, pred_col: str, prefix: str) -> pd.DataFrame:
    df[f"{prefix}_match"] = (df[truth_col] == df[pred_col]).astype(int)
    df[f"{prefix}_mismatch"] = 1 - df[f"{prefix}_match"]
    return df


def summarize_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "difference_rate": float(1.0 - accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, digits=3, zero_division=0
        ),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    # Load models
    stage1_model, stage2_model, feature_cols = load_models()

    # Read tables
    pos = prepare_positive_table(POSITIVE_CSV)
    neg = prepare_negative_table(NEGATIVE_CSV)

    # =========================
    # Stage 1: detectability
    # =========================
    stage1_df = pd.concat([pos.copy(), neg.copy()], ignore_index=True, sort=False)
    stage1_df["source_table"] = ["positive"] * len(pos) + ["negative"] * len(neg)

    # Keep only rows with usable numeric features
    stage1_df = stage1_df.dropna(subset=["Lag_min", "ARCV_deg", "ARCL_deg"]).copy()

    pred_stage1 = []
    pdet_stage1 = []

    for _, row in stage1_df.iterrows():
        pred, pdet = predict_stage1(row, stage1_model, feature_cols)
        pred_stage1.append(pred)
        pdet_stage1.append(pdet)

    stage1_df["pred_twostep_stage1"] = pred_stage1
    stage1_df["pdet_twostep"] = pdet_stage1

    stage1_df = add_match_columns(
        stage1_df,
        truth_col="target_detectability",
        pred_col="pred_twostep_stage1",
        prefix="stage1"
    )

    y_true_stage1 = stage1_df["target_detectability"].astype(int).to_numpy()
    y_pred_stage1 = stage1_df["pred_twostep_stage1"].astype(int).to_numpy()
    stage1_summary = summarize_binary(y_true_stage1, y_pred_stage1)

    # Save row-by-row stage 1 comparison
    stage1_df.to_csv(OUT_STAGE1_CSV, index=False)

    # =========================
    # Stage 2: aided vs naked-eye
    # =========================
    stage2_df = pos.copy()

    # Keep only rows with usable stage 2 truth labels and numeric features
    stage2_df = stage2_df.dropna(subset=["Lag_min", "ARCV_deg", "ARCL_deg", "target_naked_eye"]).copy()
    stage2_df["target_naked_eye"] = stage2_df["target_naked_eye"].astype(int)

    pred_stage2 = []
    peye_stage2 = []

    for _, row in stage2_df.iterrows():
        pred, peye = predict_stage2(row, stage2_model, feature_cols)
        pred_stage2.append(pred)
        peye_stage2.append(peye)

    stage2_df["pred_twostep_stage2"] = pred_stage2
    stage2_df["peye_twostep"] = peye_stage2

    stage2_df = add_match_columns(
        stage2_df,
        truth_col="target_naked_eye",
        pred_col="pred_twostep_stage2",
        prefix="stage2"
    )

    y_true_stage2 = stage2_df["target_naked_eye"].astype(int).to_numpy()
    y_pred_stage2 = stage2_df["pred_twostep_stage2"].astype(int).to_numpy()
    stage2_summary = summarize_binary(y_true_stage2, y_pred_stage2)

    # Save row-by-row stage 2 comparison
    stage2_df.to_csv(OUT_STAGE2_CSV, index=False)

    # =========================
    # Report
    # =========================
    report = {
        "files": {
            "positive_csv": str(POSITIVE_CSV),
            "negative_csv": str(NEGATIVE_CSV),
            "stage1_output_csv": str(OUT_STAGE1_CSV),
            "stage2_output_csv": str(OUT_STAGE2_CSV),
        },
        "dataset_sizes": {
            "positive_rows_raw": int(len(pos)),
            "negative_rows_raw": int(len(neg)),
            "stage1_rows_used": int(len(stage1_df)),
            "stage2_rows_used": int(len(stage2_df)),
        },
        "stage1_detectability": stage1_summary,
        "stage2_aided_vs_naked_eye": stage2_summary,
    }

    OUT_REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Print summary
    print("Saved:")
    print(f"  {OUT_STAGE1_CSV}")
    print(f"  {OUT_STAGE2_CSV}")
    print(f"  {OUT_REPORT_JSON}")

    print("\n=== Stage 1: Detectability row-by-row comparison ===")
    print(f"Rows used:         {len(stage1_df)}")
    print(f"Agreement rate:    {stage1_summary['accuracy']:.4f}")
    print(f"Difference rate:   {stage1_summary['difference_rate']:.4f}")
    print("Confusion matrix:")
    print(stage1_summary["confusion_matrix"])

    print("\n=== Stage 2: Aided vs naked-eye row-by-row comparison ===")
    print(f"Rows used:         {len(stage2_df)}")
    print(f"Agreement rate:    {stage2_summary['accuracy']:.4f}")
    print(f"Difference rate:   {stage2_summary['difference_rate']:.4f}")
    print("Confusion matrix:")
    print(stage2_summary["confusion_matrix"])


if __name__ == "__main__":
    main()