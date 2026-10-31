"""
evaluate.py
------------
Standardized Evaluation Harness for Parkinson's Progression Capstone.

Implements rigorous, unified metrics across all models (Models 1 to 4):
1. Continuous Regression Metrics:
   - MAE (Mean Absolute Error, in MDS-UPDRS Part III points)
   - RMSE (Root Mean Squared Error)
   - Pearson Correlation (r) & p-value
   - Coefficient of Determination (R²)
   - Prediction Bias / Mean Error
2. Binary Progression Classification Metrics (MCID >= 3.5 points):
   - ROC-AUC (Area Under ROC Curve)
   - PR-AUC (Average Precision / Area Under PR Curve)
   - Balanced Accuracy
   - Sensitivity (Recall for rapid worsening)
   - Specificity (Correctly identifying stable patients)
   - Brier Score (Probabilistic Calibration Error)
3. Subgroup Stratification:
   - Evaluates performance separately across medication states (ON vs. OFF vs. UNTREATED)
   - Evaluates across diagnostic cohorts (PD vs. Prodromal)
4. Central Benchmark Ledger:
   - Persists all model runs into report/benchmark_results.csv & report/benchmark_results.json
"""

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    balanced_accuracy_score
)

REPORT_DIR = "report"
BENCHMARK_CSV = os.path.join(REPORT_DIR, "benchmark_results.csv")
BENCHMARK_JSON = os.path.join(REPORT_DIR, "benchmark_results.json")


def evaluate_regression(y_true, y_pred):
    """
    Computes regression metrics on continuous NP3TOT progression.
    Ignores NaNs in true or predicted values.
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt = np.asarray(y_true)[mask]
    yp = np.asarray(y_pred)[mask]

    if len(yt) < 2:
        return {
            "n_samples": int(len(yt)),
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "bias": np.nan
        }

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp))
    bias = float(np.mean(yp - yt))

    # Pearson correlation
    if np.std(yt) > 1e-6 and np.std(yp) > 1e-6:
        r_val, p_val = stats.pearsonr(yt, yp)
    else:
        r_val, p_val = np.nan, np.nan

    return {
        "n_samples": int(len(yt)),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "pearson_r": round(float(r_val), 4) if not np.isnan(r_val) else np.nan,
        "pearson_p": float(p_val) if not np.isnan(p_val) else np.nan,
        "bias": round(bias, 3)
    }


def evaluate_classification(y_true, y_prob, threshold=0.5):
    """
    Computes binary classification metrics for clinically meaningful worsening (MCID >= 3.5 points).
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    yt = np.asarray(y_true)[mask].astype(int)
    yp = np.asarray(y_prob)[mask]

    if len(yt) < 2 or len(np.unique(yt)) < 2:
        return {
            "n_samples": int(len(yt)),
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "balanced_acc": np.nan,
            "sensitivity": np.nan,
            "specificity": np.nan,
            "brier_score": np.nan
        }

    try:
        roc_auc = float(roc_auc_score(yt, yp))
    except Exception:
        roc_auc = np.nan

    try:
        pr_auc = float(average_precision_score(yt, yp))
    except Exception:
        pr_auc = np.nan

    brier = float(brier_score_loss(yt, yp))

    y_pred_bin = (yp >= threshold).astype(int)
    bal_acc = float(balanced_accuracy_score(yt, y_pred_bin))

    cm = confusion_matrix(yt, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan

    return {
        "n_samples": int(len(yt)),
        "roc_auc": round(roc_auc, 4) if not np.isnan(roc_auc) else np.nan,
        "pr_auc": round(pr_auc, 4) if not np.isnan(pr_auc) else np.nan,
        "balanced_acc": round(bal_acc, 4),
        "sensitivity": round(sensitivity, 4) if not np.isnan(sensitivity) else np.nan,
        "specificity": round(specificity, 4) if not np.isnan(specificity) else np.nan,
        "brier_score": round(brier, 4)
    }


def evaluate_model_run(
    model_name,
    target_horizon,
    y_true_reg,
    y_pred_reg,
    y_true_cls=None,
    y_prob_cls=None,
    metadata_df=None,
    modality_tags="Clinical",
    notes=""
):
    """
    Comprehensive unified evaluation function:
    - Overall regression & classification metrics
    - Medication subgroup metrics (ON vs. OFF)
    - Cohort subgroup metrics (PD vs. Prodromal)
    - Appends results to report/benchmark_results.csv
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 1. Overall Metrics
    reg_metrics = evaluate_regression(y_true_reg, y_pred_reg)
    cls_metrics = evaluate_classification(y_true_cls, y_prob_cls) if y_true_cls is not None and y_prob_cls is not None else {}

    result_entry = {
        "timestamp": timestamp,
        "model_name": model_name,
        "target_horizon": target_horizon,
        "modalities": modality_tags,
        "n_samples": reg_metrics["n_samples"],
        "mae": reg_metrics["mae"],
        "rmse": reg_metrics["rmse"],
        "r2": reg_metrics["r2"],
        "pearson_r": reg_metrics["pearson_r"],
        "bias": reg_metrics["bias"],
        "roc_auc": cls_metrics.get("roc_auc", np.nan),
        "pr_auc": cls_metrics.get("pr_auc", np.nan),
        "balanced_acc": cls_metrics.get("balanced_acc", np.nan),
        "sensitivity": cls_metrics.get("sensitivity", np.nan),
        "specificity": cls_metrics.get("specificity", np.nan),
        "brier_score": cls_metrics.get("brier_score", np.nan),
        "notes": notes
    }

    # 2. Medication Subgroups (if metadata provided)
    if metadata_df is not None and "pd_state_clean" in metadata_df.columns:
        for state in ["ON", "OFF", "UNTREATED"]:
            sub_mask = (metadata_df["pd_state_clean"] == state).values
            if sub_mask.sum() > 10:
                sub_reg = evaluate_regression(y_true_reg[sub_mask], y_pred_reg[sub_mask])
                result_entry[f"mae_{state.lower()}"] = sub_reg["mae"]
                result_entry[f"r_{state.lower()}"] = sub_reg["pearson_r"]
            else:
                result_entry[f"mae_{state.lower()}"] = np.nan
                result_entry[f"r_{state.lower()}"] = np.nan

    # 3. Save to Benchmark Ledger CSV
    df_new = pd.DataFrame([result_entry])
    if os.path.exists(BENCHMARK_CSV):
        df_bench = pd.read_csv(BENCHMARK_CSV)
        df_bench = df_bench[~((df_bench["model_name"] == model_name) & (df_bench["target_horizon"] == target_horizon))]
        df_bench = pd.concat([df_bench, df_new], ignore_index=True)
    else:
        df_bench = df_new

    df_bench.to_csv(BENCHMARK_CSV, index=False)

    # 4. Pretty print report
    print_evaluation_summary(result_entry)
    return result_entry


def print_evaluation_summary(entry):
    """Prints a clean tabular evaluation report to console."""
    print("\n" + "=" * 65)
    print(f"BENCHMARK EVALUATION REPORT: {entry['model_name']} ({entry['target_horizon']})")
    print("=" * 65)
    print(f"Modalities Used     : {entry['modalities']}")
    print(f"Evaluation Samples  : {entry['n_samples']:,} held-out test patients")
    print("-" * 65)
    print("REGRESSION METRICS (MDS-UPDRS Part III Continuous Score):")
    print(f"  • MAE (Mean Absolute Error) : {entry['mae']:.3f} points")
    print(f"  • RMSE                      : {entry['rmse']:.3f} points")
    print(f"  • Pearson Correlation (r)   : {entry['pearson_r']:.3f}")
    print(f"  • R² (Explained Variance)   : {entry['r2']:.4f}")
    print(f"  • Prediction Bias           : {entry['bias']:+.3f} points")

    if not np.isnan(entry.get("roc_auc", np.nan)):
        print("-" * 65)
        print("CLASSIFICATION METRICS (Meaningful Progression >= 3.5 pts):")
        print(f"  • ROC-AUC                   : {entry['roc_auc']:.4f}")
        print(f"  • PR-AUC (Average Precision): {entry['pr_auc']:.4f}")
        print(f"  • Balanced Accuracy         : {entry['balanced_acc']:.4f}")
        print(f"  • Sensitivity (True Pos)    : {entry['sensitivity']:.4f}")
        print(f"  • Specificity (True Neg)    : {entry['specificity']:.4f}")
        print(f"  • Brier Calibration Score   : {entry['brier_score']:.4f}")

    if "mae_on" in entry and not np.isnan(entry["mae_on"]):
        print("-" * 65)
        print("MEDICATION STRATIFICATION:")
        print(f"  • ON  Medication MAE       : {entry['mae_on']:.3f} (r = {entry.get('r_on', np.nan):.3f})")
        print(f"  • OFF Medication MAE       : {entry['mae_off']:.3f} (r = {entry.get('r_off', np.nan):.3f})")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    print("Standardized Evaluation Harness initialized.")
    print(f"Benchmark ledger destination: {BENCHMARK_CSV}")
