from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

STAGE1_CSV = "two_stage_stage1_predictions.csv"
STAGE2_CSV = "two_stage_stage2_predictions.csv"
OUT_JSON = "chosen_thresholds.json"


def evaluate_threshold(y_true, p, thr):
    y_pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
    }


def choose_thresholds(y_true, p, label):
    fpr, tpr, roc_thr = roc_curve(y_true, p)
    auc = roc_auc_score(y_true, p)

    # Youden J
    j = tpr - fpr
    i_best_j = int(np.argmax(j))
    thr_j = roc_thr[i_best_j]

    # Best F1 on a dense grid
    grid = np.linspace(0.05, 0.95, 181)
    rows = [evaluate_threshold(y_true, p, thr) for thr in grid]
    df = pd.DataFrame(rows)

    i_best_f1 = int(df["f1"].idxmax())
    best_f1_row = df.loc[i_best_f1].to_dict()

    # Practical low / medium / high confidence thresholds
    # You can tune these later, but these are useful for reporting
    candidates = {
        "youden_j": evaluate_threshold(y_true, p, thr_j),
        "best_f1": best_f1_row,
        "low_confidence": evaluate_threshold(y_true, p, 0.30),
        "balanced": evaluate_threshold(y_true, p, 0.50),
        "high_confidence": evaluate_threshold(y_true, p, 0.70),
    }

    print(f"\n=== {label} ===")
    print(f"ROC AUC = {auc:.4f}")
    for k, v in candidates.items():
        print(
            f"{k:15s} thr={v['threshold']:.3f} "
            f"acc={v['accuracy']:.3f} "
            f"prec={v['precision']:.3f} "
            f"rec={v['recall']:.3f} "
            f"f1={v['f1']:.3f} "
            f"spec={v['specificity']:.3f}"
        )

    return auc, fpr, tpr, candidates, df


def plot_roc(fpr, tpr, auc, title, outfile):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()


def plot_f1_curve(df, title, outfile):
    plt.figure(figsize=(6, 5))
    plt.plot(df["threshold"], df["f1"], label="F1")
    plt.plot(df["threshold"], df["precision"], label="Precision")
    plt.plot(df["threshold"], df["recall"], label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()


def main():
    s1 = pd.read_csv(STAGE1_CSV)
    s2 = pd.read_csv(STAGE2_CSV)

    y1 = s1["y_true"].astype(int).to_numpy()
    p1 = s1["p_positive"].astype(float).to_numpy()

    y2 = s2["y_true"].astype(int).to_numpy()
    p2 = s2["p_positive"].astype(float).to_numpy()

    auc1, fpr1, tpr1, cand1, df1 = choose_thresholds(y1, p1, "Stage 1: Detectability")
    auc2, fpr2, tpr2, cand2, df2 = choose_thresholds(y2, p2, "Stage 2: Naked-eye")

    plot_roc(fpr1, tpr1, auc1, "ROC — Stage 1 Detectability", "roc_stage1_detectability.png")
    plot_roc(fpr2, tpr2, auc2, "ROC — Stage 2 Naked-eye", "roc_stage2_naked_eye.png")

    plot_f1_curve(df1, "Threshold Curves — Stage 1 Detectability", "thresholds_stage1.png")
    plot_f1_curve(df2, "Threshold Curves — Stage 2 Naked-eye", "thresholds_stage2.png")

    out = {
        "stage1_detectability": cand1,
        "stage2_naked_eye": cand2,
    }
    Path(OUT_JSON).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\nSaved:")
    print("  roc_stage1_detectability.png")
    print("  roc_stage2_naked_eye.png")
    print("  thresholds_stage1.png")
    print("  thresholds_stage2.png")
    print("  chosen_thresholds.json")


if __name__ == "__main__":
    main()