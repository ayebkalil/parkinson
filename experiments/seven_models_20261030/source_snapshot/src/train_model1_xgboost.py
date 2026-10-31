"""
train_model1_xgboost.py
-----------------------
Phase 3, Step 2: Model 1 — Static Tabular Gradient Boosting Baseline (XGBoost)

Addresses Research Question 1 (RQ1 Baseline):
"How accurately can disease progression (MDS-UPDRS Part III at +12m and +24m)
be predicted from a single static clinic visit snapshot?"

Key Components:
1. Ingests patient-level latest visit records (Train: 2,683 | Val: 574 | Test: 583)
2. Features: Motor, Medication, Cognitive, Autonomic, Sleep, Smell, Biospecimens, Imaging, Demographics
3. Trains:
   - Model 1A: Continuous Regression for +12-Month Motor Progression (target_np3tot_12m)
   - Model 1B: Continuous Regression for +24-Month Motor Progression (target_np3tot_24m)
   - Model 1C: Binary Classification for Meaningful Worsening (prog_mcid_12m, MCID >= 3.5 pts)
4. Evaluates strictly on held-out test patients using src.evaluate.evaluate_model_run
5. Generates SHAP feature importance plot saved to report/model1_xgboost_shap_importance.png
6. Saves model artifacts in models/
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap

from evaluate import evaluate_model_run

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
REPORT_DIR = "report"

# Feature set: multimodal static snapshot features
FEATURE_COLS = [
    # Motor Part III subdomains
    "NP3TOT", "np3_tremor", "np3_rigidity", "np3_bradykinesia", "np3_axial",
    "nhy_stage", "dyskinesia_present",
    # Motor/Non-motor daily living
    "np1_total", "np2_total", "np4_total", "schwab_england_pct",
    # Medication controls
    "ledd", "pd_state_on", "pd_state_off", "hr_post_med",
    # Non-motor scales
    "moca_total", "scopa_aut_total", "gds_total", "ess_total", "rbd_total", "upsit_total",
    # Biospecimens (CSF & SAA)
    "csf_abeta42", "csf_ttau", "csf_ptau", "csf_asyn",
    "csf_ttau_abeta_ratio", "csf_ptau_abeta_ratio", "saa_positive",
    # Neuroimaging (DaTSCAN SBR & FreeSurfer MRI)
    "datscan_caudate_sbr", "datscan_putamen_sbr", "datscan_striatum_sbr",
    "datscan_putamen_asym", "datscan_caudate_putamen_ratio",
    "mri_brain_seg_vol", "mri_brain_stem", "mri_putamen_vol",
    "mri_caudate_vol", "mri_hippocampus_vol", "mri_ventricles_vol",
    # Demographics
    "age_at_visit", "is_male", "education_years"
]


def load_tabular_data():
    print("[1/5] Loading Canonical Tabular Splits (Latest Single Visit Snapshot)...")
    train_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "train_tabular_latest.parquet"))
    val_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "val_tabular_latest.parquet"))
    test_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "test_tabular_latest.parquet"))

    print(f"  Train: {len(train_df)} patients | Val: {len(val_df)} patients | Test: {len(test_df)} patients")
    return train_df, val_df, test_df


def train_12m_regressor(train_df, val_df, test_df):
    print("\n[2/5] Training Model 1A: XGBoost +12-Month Motor Score Regressor...")
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["target_np3tot_12m"].values

    X_val = val_df[FEATURE_COLS]
    y_val = val_df["target_np3tot_12m"].values

    X_test = test_df[FEATURE_COLS]
    y_test = test_df["target_np3tot_12m"].values

    model_12m = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=42,
        early_stopping_rounds=40,
        n_jobs=-1
    )

    model_12m.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds_test = model_12m.predict(X_test)

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "model1_xgboost_12m.json")
    model_12m.save_model(model_path)
    print(f"  Saved trained model to {model_path} (Best Iteration: {model_12m.best_iteration})")

    # Evaluate on held-out test set
    metrics = evaluate_model_run(
        model_name="Model 1: Tabular XGBoost",
        target_horizon="+12 Months (NP3TOT)",
        y_true_reg=y_test,
        y_pred_reg=preds_test,
        metadata_df=test_df,
        modality_tags="Multimodal Tabular Snapshot (Latest Visit)",
        notes=f"Trained on {len(X_train)} pats, evaluated on held-out test set."
    )
    return model_12m, preds_test, metrics


def train_24m_regressor(train_df, val_df, test_df):
    print("\n[3/5] Training Model 1B: XGBoost +24-Month Motor Score Regressor...")
    train_24 = train_df[train_df["target_np3tot_24m"].notna()]
    val_24 = val_df[val_df["target_np3tot_24m"].notna()]
    test_24 = test_df[test_df["target_np3tot_24m"].notna()]

    X_train = train_24[FEATURE_COLS]
    y_train = train_24["target_np3tot_24m"].values

    X_val = val_24[FEATURE_COLS]
    y_val = val_24["target_np3tot_24m"].values

    X_test = test_24[FEATURE_COLS]
    y_test = test_24["target_np3tot_24m"].values

    model_24m = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=42,
        early_stopping_rounds=40,
        n_jobs=-1
    )

    model_24m.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds_test = model_24m.predict(X_test)

    model_path = os.path.join(MODELS_DIR, "model1_xgboost_24m.json")
    model_24m.save_model(model_path)
    print(f"  Saved trained model to {model_path} (Best Iteration: {model_24m.best_iteration})")

    metrics = evaluate_model_run(
        model_name="Model 1: Tabular XGBoost",
        target_horizon="+24 Months (NP3TOT)",
        y_true_reg=y_test,
        y_pred_reg=preds_test,
        metadata_df=test_24,
        modality_tags="Multimodal Tabular Snapshot (Latest Visit)",
        notes=f"Trained on {len(X_train)} pats with valid 24m targets."
    )
    return model_24m, preds_test, metrics


def train_progression_classifier(train_df, val_df, test_df):
    print("\n[4/5] Training Model 1C: XGBoost Meaningful Worsening Classifier (MCID >= 3.5 pts)...")
    train_c = train_df[train_df["prog_mcid_12m"].notna()]
    val_c = val_df[val_df["prog_mcid_12m"].notna()]
    test_c = test_df[test_df["prog_mcid_12m"].notna()]

    X_train = train_c[FEATURE_COLS]
    y_train = train_c["prog_mcid_12m"].values.astype(int)

    X_val = val_c[FEATURE_COLS]
    y_val = val_c["prog_mcid_12m"].values.astype(int)

    X_test = test_c[FEATURE_COLS]
    y_test = test_c["prog_mcid_12m"].values.astype(int)

    scale_pos = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-6)

    model_cls = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        scale_pos_weight=scale_pos,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=35,
        eval_metric="auc",
        n_jobs=-1
    )

    model_cls.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    probs_test = model_cls.predict_proba(X_test)[:, 1]

    metrics = evaluate_model_run(
        model_name="Model 1: Tabular XGBoost Classifier",
        target_horizon="+12m Worsening (MCID >= 3.5)",
        y_true_reg=y_test.astype(float),
        y_pred_reg=probs_test,
        y_true_cls=y_test,
        y_prob_cls=probs_test,
        metadata_df=test_c,
        modality_tags="Multimodal Tabular Snapshot (Latest Visit)",
        notes="Classification of rapid motor worsening (MCID >= 3.5 points)."
    )
    return model_cls, probs_test, metrics


def compute_shap_explanations(model_12m, test_df):
    print("\n[5/5] Computing SHAP Feature Importance on Held-Out Test Set...")
    X_test = test_df[FEATURE_COLS]

    explainer = shap.TreeExplainer(model_12m)
    shap_values = explainer(X_test)

    # Clean feature names for publication chart
    pretty_names = {
        "NP3TOT": "Current Total Motor Score (NP3TOT)",
        "np3_bradykinesia": "Bradykinesia (Slowness)",
        "np3_rigidity": "Rigidity (Stiffness)",
        "np3_tremor": "Tremor Score",
        "np3_axial": "Axial & Gait Score",
        "nhy_stage": "Hoehn & Yahr Stage",
        "dyskinesia_present": "Dyskinesia Flag",
        "np1_total": "Non-Motor Daily Living (Part I)",
        "np2_total": "Motor Daily Living (Part II)",
        "np4_total": "Motor Complications (Part IV)",
        "schwab_england_pct": "Schwab-England Independence %",
        "ledd": "Levodopa Daily Dose (LEDD mg)",
        "pd_state_on": "Tested ON Medication",
        "pd_state_off": "Tested OFF Medication",
        "hr_post_med": "Hours Post-Medication",
        "moca_total": "MoCA Cognitive Score",
        "scopa_aut_total": "SCOPA Autonomic Burden",
        "gds_total": "Geriatric Depression Score",
        "ess_total": "Epworth Sleepiness Score",
        "rbd_total": "REM Sleep Disorder (RBDSQ)",
        "upsit_total": "Smell Identification (UPSIT)",
        "csf_abeta42": "CSF Amyloid-Beta 42",
        "csf_ttau": "CSF Total Tau",
        "csf_ptau": "CSF Phospho-Tau 181",
        "csf_asyn": "CSF Alpha-Synuclein",
        "csf_ttau_abeta_ratio": "CSF tTau / ABeta Ratio",
        "csf_ptau_abeta_ratio": "CSF pTau / ABeta Ratio",
        "saa_positive": "Alpha-Synuclein SAA Positive",
        "datscan_caudate_sbr": "DaTSCAN Caudate SBR",
        "datscan_putamen_sbr": "DaTSCAN Putamen SBR",
        "datscan_striatum_sbr": "DaTSCAN Striatum SBR",
        "datscan_putamen_asym": "DaTSCAN Putamen Asymmetry",
        "datscan_caudate_putamen_ratio": "DaTSCAN Caudate/Putamen Ratio",
        "mri_brain_seg_vol": "MRI Brain Parenchymal Fraction",
        "mri_brain_stem": "MRI Brain Stem Volume / eTIV",
        "mri_putamen_vol": "MRI Putamen Volume / eTIV",
        "mri_caudate_vol": "MRI Caudate Volume / eTIV",
        "mri_hippocampus_vol": "MRI Hippocampus Volume / eTIV",
        "mri_ventricles_vol": "MRI Ventricles Volume / eTIV",
        "age_at_visit": "Age at Visit",
        "is_male": "Sex (Male = 1)",
        "education_years": "Years of Education"
    }

    shap_values.feature_names = [pretty_names.get(col, col) for col in FEATURE_COLS]

    # Create Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_test,
        max_display=15,
        show=False
    )
    plt.title("Model 1 (Tabular XGBoost): Top 15 SHAP Features Driving +12-Month Motor Progression",
              fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("SHAP Value (Impact on +12-Month MDS-UPDRS Part III Score)", fontsize=11)

    out_path = os.path.join(REPORT_DIR, "model1_xgboost_shap_importance.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP Feature Importance figure: {out_path}")


def main():
    print("=" * 70)
    print("PHASE 3, STEP 2: MODEL 1 TABULAR GRADIENT BOOSTING BASELINE")
    print("=" * 70)

    train_df, val_df, test_df = load_tabular_data()

    # 1. Train +12m Regressor
    model_12m, preds_12m, metrics_12m = train_12m_regressor(train_df, val_df, test_df)

    # 2. Train +24m Regressor
    model_24m, preds_24m, metrics_24m = train_24m_regressor(train_df, val_df, test_df)

    # 3. Train Meaningful Worsening Classifier
    model_cls, probs_cls, metrics_cls = train_progression_classifier(train_df, val_df, test_df)

    # 4. Compute SHAP explanations
    compute_shap_explanations(model_12m, test_df)

    print("\n" + "=" * 70)
    print("MODEL 1 BASELINE EXECUTION COMPLETE!")
    print(f"Results recorded in: {os.path.join(REPORT_DIR, 'benchmark_results.csv')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
