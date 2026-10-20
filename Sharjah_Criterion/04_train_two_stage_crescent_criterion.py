from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).parent.parent
_ICOP_DATA_DIR = _REPO_ROOT / "ICOP_Data"
_ALREFAY_DIR = _REPO_ROOT / "Alrefay_tables_Comparison"

INPUT_CSV = _ICOP_DATA_DIR / "crescent_features.csv"

REPORT_JSON = _ALREFAY_DIR / "two_stage_crescent_criterion_report.json"
STAGE1_PRED_CSV = _ICOP_DATA_DIR / "two_stage_stage1_predictions.csv"
STAGE2_PRED_CSV = _ICOP_DATA_DIR / "two_stage_stage2_predictions.csv"
OOF_CSV = _ICOP_DATA_DIR / "two_stage_oof_predictions.csv"

STAGE1_MODEL_JOBLIB = _ALREFAY_DIR / "stage1_detectability_model.joblib"
STAGE2_MODEL_JOBLIB = _ALREFAY_DIR / "stage2_naked_eye_model.joblib"
MODEL_META_JSON = _ALREFAY_DIR / "two_stage_model_meta.json"

N_FOLDS = 5

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


def binary_summary(y_true, y_prob, label: str) -> dict:
    """Metrics at the 0.50 cutoff and ROC AUC, from out-of-fold probabilities."""
    y_pred = (y_prob >= 0.5).astype(int)
    report_text = classification_report(y_true, y_pred, digits=3, zero_division=0)
    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "majority_class_accuracy": float(max(np.mean(y_true), 1 - np.mean(y_true))),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report_text": report_text,
    }
    print(f"\n=== {label} (out-of-fold, grouped by lunation) ===")
    print(f"Accuracy: {summary['accuracy']:.4f}  (majority class: {summary['majority_class_accuracy']:.4f})")
    print(f"ROC AUC:  {summary['roc_auc']:.4f}")
    print(report_text)
    print("Confusion matrix:", summary["confusion_matrix"])
    return summary


def out_of_fold(df, feature_cols, folds):
    """P_det for every row and P_eye for every row, each from models not fitted on its lunation."""
    y = df["vis_class_empirical"].to_numpy()
    p_det = np.full(len(df), np.nan)
    p_eye = np.full(len(df), np.nan)
    for train_idx, test_idx in folds:
        train = df.iloc[train_idx]
        seen = train["vis_class_empirical"] > 0
        stage1 = fit_binary_logistic(train[feature_cols], seen.astype(int))
        stage2 = fit_binary_logistic(train.loc[seen, feature_cols],
                                     (train.loc[seen, "vis_class_empirical"] == 2).astype(int))
        test = df.iloc[test_idx][feature_cols]
        p_det[test_idx] = stage1.predict_proba(test)[:, 1]
        p_eye[test_idx] = stage2.predict_proba(test)[:, 1]
    return p_det, p_eye


def stage_scores(y, p_det, p_eye) -> dict:
    seen = y > 0
    return {
        "stage1_accuracy": accuracy_score(seen, p_det >= 0.5),
        "stage1_auc": roc_auc_score(seen, p_det),
        "stage2_accuracy": accuracy_score(y[seen] == 2, p_eye[seen] >= 0.5),
        "stage2_auc": roc_auc_score(y[seen] == 2, p_eye[seen]),
    }


def repeated_cv(df, feature_cols, grouped: bool, repeats: int = 10) -> dict:
    """Mean and SD of the stage scores over repeated 5-fold CV."""
    y = df["vis_class_empirical"].to_numpy()
    runs = []
    for seed in range(repeats):
        if grouped:
            cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            folds = cv.split(df, y, groups=df["page_url"])
        else:
            cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            folds = cv.split(df, y)
        runs.append(stage_scores(y, *out_of_fold(df, feature_cols, folds)))
    runs = pd.DataFrame(runs)
    return {k: {"mean": float(runs[k].mean()), "sd": float(runs[k].std(ddof=1))} for k in runs}


def main():
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    df = df[df["section_type"] == "first_crescent"].copy()
    df = df[df["obs_relative_to_sunset"] != "before_sunset"].copy()
    df = df[df["vis_class_empirical"].notna()].reset_index(drop=True)
    df["vis_class_empirical"] = df["vis_class_empirical"].astype(int)

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    y = df["vis_class_empirical"].to_numpy()
    seen = y > 0

    print("Rows used (initial filtered set):", len(df))
    print("Lunations (ICOP monthly report pages):", df["page_url"].nunique())
    print("Target distribution (0=not seen,1=aid,2=naked eye):")
    print(df["vis_class_empirical"].value_counts().sort_index())
    print("\nFeatures used:")
    print(feature_cols)

    # Out-of-fold predictions: reports from the same new moon never sit in both
    # the training and the test folds.
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    p_det, p_eye = out_of_fold(df, feature_cols, cv.split(df, y, groups=df["page_url"]))

    stage1_summary = binary_summary(seen.astype(int), p_det, "Stage 1: Detectability")
    stage2_summary = binary_summary((y[seen] == 2).astype(int), p_eye[seen], "Stage 2: Naked-eye vs aided")

    oof = df[["page_url"] + feature_cols].copy()
    oof["vis_class_empirical"] = y
    oof["p_det"], oof["p_eye"] = p_det, p_eye
    oof.to_csv(OOF_CSV, index=False)
    # Inputs for 05_choose_thresholds_and_plot_roc.py.
    pd.DataFrame({"y_true": seen.astype(int), "p_positive": p_det}).to_csv(STAGE1_PRED_CSV, index=False)
    pd.DataFrame({"y_true": (y[seen] == 2).astype(int), "p_positive": p_eye[seen]}).to_csv(
        STAGE2_PRED_CSV, index=False)

    grouped = repeated_cv(df, feature_cols, grouped=True)
    ungrouped = repeated_cv(df, feature_cols, grouped=False)
    print("\nRepeated 5-fold CV (10 repeats), mean (SD): grouped by lunation | ungrouped")
    for k in grouped:
        print(f"  {k:16s} {grouped[k]['mean']:.4f} ({grouped[k]['sd']:.4f}) | "
              f"{ungrouped[k]['mean']:.4f} ({ungrouped[k]['sd']:.4f})")

    # Final criterion: both stages fitted on all records.
    model1 = fit_binary_logistic(df[feature_cols], seen.astype(int))
    model2 = fit_binary_logistic(df.loc[seen, feature_cols], (y[seen] == 2).astype(int))
    joblib.dump(model1, STAGE1_MODEL_JOBLIB)
    joblib.dump(model2, STAGE2_MODEL_JOBLIB)

    meta = {
        "feature_cols": feature_cols,
        # File names, relative to the folder of the metadata file.
        "stage1_model": STAGE1_MODEL_JOBLIB.name,
        "stage2_model": STAGE2_MODEL_JOBLIB.name,
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
        "lunations": int(df["page_url"].nunique()),
        "feature_cols": feature_cols,
        "evaluation": f"{N_FOLDS}-fold cross-validation grouped by lunation (ICOP report page), "
                      "stratified on the three-class label, random_state=42",
        "stage1": {
            "task": "0 = not seen, 1 = seen by any means",
            "rows_used": int(len(df)),
            "class_counts": {str(k): int(v) for k, v in pd.Series(seen.astype(int)).value_counts().sort_index().items()},
            "score_coefficients": coeff_dict(model1, feature_cols),
            "metrics": stage1_summary,
        },
        "stage2": {
            "task": "0 = optical aid/instrument only, 1 = naked eye",
            "rows_used": int(seen.sum()),
            "class_counts": {str(k): int(v) for k, v in pd.Series((y[seen] == 2).astype(int)).value_counts().sort_index().items()},
            "score_coefficients": coeff_dict(model2, feature_cols),
            "metrics": stage2_summary,
        },
        "repeated_cv": {"grouped_by_lunation": grouped, "ungrouped": ungrouped},
        "saved_models": meta,
    }
    Path(REPORT_JSON).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nSaved out-of-fold predictions to {OOF_CSV}")
    print(f"Saved stage 1 / stage 2 predictions to {STAGE1_PRED_CSV} / {STAGE2_PRED_CSV}")
    print(f"Saved stage 1 model to {STAGE1_MODEL_JOBLIB}")
    print(f"Saved stage 2 model to {STAGE2_MODEL_JOBLIB}")
    print(f"Saved model metadata to {MODEL_META_JSON}")
    print(f"Saved combined report to {REPORT_JSON}")


if __name__ == "__main__":
    main()
