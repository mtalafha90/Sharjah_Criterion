from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).parent.parent
_ICOP_DATA_DIR = _REPO_ROOT / "ICOP_Data"
_ALREFAY_DIR = _REPO_ROOT / "Alrefay_tables_Comparison"

INPUT_CSV = _ICOP_DATA_DIR / "crescent_features.csv"

REPORT_JSON = _ALREFAY_DIR / "two_stage_crescent_criterion_report.json"
STAGE1_PRED_CSV = _ICOP_DATA_DIR / "two_stage_stage1_predictions.csv"
STAGE2_PRED_CSV = _ICOP_DATA_DIR / "two_stage_stage2_predictions.csv"

STAGE1_MODEL_JOBLIB = _ALREFAY_DIR / "stage1_detectability_model.joblib"
STAGE2_MODEL_JOBLIB = _ALREFAY_DIR / "stage2_naked_eye_model.joblib"
MODEL_META_JSON = _ALREFAY_DIR / "two_stage_model_meta.json"

FEATURE_COLS = [
    "relative_altitude_deg",
    "elongation_deg",
    "moon_lag_minutes",
]


def fit_binary_logistic(X_train, y_train) -> Pipeline:
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=4000,
            solver="lbfgs",
        )),
    ])
    model.fit(X_train, y_train)
    return model


def coeff_dict(model: Pipeline, feature_cols: list[str]) -> dict:
    clf = model.named_steps["clf"]
    coefs = clf.coef_.ravel()
    intercept = float(clf.intercept_.ravel()[0])
    return {
        "intercept": intercept,
        **{feature_cols[i]: float(coefs[i]) for i in range(len(feature_cols))}
    }


def evaluate_binary(model: Pipeline, X_test, y_test, label: str):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report_text = classification_report(y_test, y_pred, digits=3, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    acc = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))

    print(f"\n=== {label} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC:  {auc:.4f}")
    print("\nClassification report:")
    print(report_text)
    print("Confusion matrix:")
    print(cm)

    pred_df = X_test.copy()
    pred_df["y_true"] = y_test.values
    pred_df["y_pred"] = y_pred
    pred_df["p_positive"] = y_prob

    summary = {
        "accuracy": acc,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "classification_report_text": report_text,
    }
    return summary, pred_df


def main():
    df = pd.read_csv(INPUT_CSV)

    df = df[df["section_type"] == "first_crescent"].copy()
    df = df[df["obs_relative_to_sunset"] != "before_sunset"].copy()
    df = df[df["vis_class_empirical"].notna()].copy()
    df["vis_class_empirical"] = df["vis_class_empirical"].astype(int)

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    print("Rows used (initial filtered set):", len(df))
    print("Target distribution (0=not seen,1=aid,2=naked eye):")
    print(df["vis_class_empirical"].value_counts().sort_index())
    print("\nFeatures used:")
    print(feature_cols)

    # Stage 1
    df1 = df.copy()
    df1["target_detectability"] = (df1["vis_class_empirical"] > 0).astype(int)
    X1 = df1[feature_cols].copy()
    y1 = df1["target_detectability"].copy()

    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1, y1, test_size=0.25, random_state=42, stratify=y1
    )

    model1 = fit_binary_logistic(X1_train, y1_train)
    stage1_summary, stage1_pred = evaluate_binary(model1, X1_test, y1_test, "Stage 1: Detectability")
    stage1_pred.to_csv(STAGE1_PRED_CSV, index=False)
    stage1_score = coeff_dict(model1, feature_cols)

    # Stage 2
    df2 = df[df["vis_class_empirical"].isin([1, 2])].copy()
    df2["target_naked_eye"] = (df2["vis_class_empirical"] == 2).astype(int)
    X2 = df2[feature_cols].copy()
    y2 = df2["target_naked_eye"].copy()

    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.25, random_state=42, stratify=y2
    )

    model2 = fit_binary_logistic(X2_train, y2_train)
    stage2_summary, stage2_pred = evaluate_binary(model2, X2_test, y2_test, "Stage 2: Naked-eye vs aided")
    stage2_pred.to_csv(STAGE2_PRED_CSV, index=False)
    stage2_score = coeff_dict(model2, feature_cols)

    # Save fitted pipelines
    joblib.dump(model1, STAGE1_MODEL_JOBLIB)
    joblib.dump(model2, STAGE2_MODEL_JOBLIB)

    meta = {
        "feature_cols": feature_cols,
        "stage1_model": STAGE1_MODEL_JOBLIB,
        "stage2_model": STAGE2_MODEL_JOBLIB,
        "thresholds": {
            "pdet_not_detectable": 0.30,
            "pdet_detectable": 0.50,
            "peye_optical_aid": 0.30,
            "peye_naked_eye": 0.60,
        },
    }
    Path(MODEL_META_JSON).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    report = {
        "rows_used_initial": int(len(df)),
        "feature_cols": feature_cols,
        "stage1": {
            "task": "0 = not seen, 1 = seen by any means",
            "rows_used": int(len(df1)),
            "class_counts": {str(k): int(v) for k, v in y1.value_counts().sort_index().items()},
            "score_coefficients": stage1_score,
            "metrics": stage1_summary,
        },
        "stage2": {
            "task": "0 = optical aid/instrument only, 1 = naked eye",
            "rows_used": int(len(df2)),
            "class_counts": {str(k): int(v) for k, v in y2.value_counts().sort_index().items()},
            "score_coefficients": stage2_score,
            "metrics": stage2_summary,
        },
        "saved_models": meta,
    }
    Path(REPORT_JSON).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nSaved stage 1 predictions to {STAGE1_PRED_CSV}")
    print(f"Saved stage 2 predictions to {STAGE2_PRED_CSV}")
    print(f"Saved stage 1 model to {STAGE1_MODEL_JOBLIB}")
    print(f"Saved stage 2 model to {STAGE2_MODEL_JOBLIB}")
    print(f"Saved model metadata to {MODEL_META_JSON}")
    print(f"Saved combined report to {REPORT_JSON}")


if __name__ == "__main__":
    main()
