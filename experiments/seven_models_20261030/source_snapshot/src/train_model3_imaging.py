"""
train_model3_imaging.py
-----------------------
Phase 3, Step 4: Model 3 — Monomodal Neuroimaging Baseline

Evaluates the standalone prognostic capability of brain imaging:
"How accurately can disease progression (+12m and +24m MDS-UPDRS Part III)
and rapid worsening (MCID >= 3.5) be predicted using neuroimaging features
alone (DaTSCAN SPECT striatal binding ratios + FreeSurfer 3D structural MRI),
without touching any clinical motor or cognitive symptoms?"

Key Components:
1. Loads canonical splits: train_tabular_latest, val_tabular_latest, test_tabular_latest.
2. Ingests 13 imaging and normative features:
   - DaTSCAN SPECT: caudate SBR, putamen SBR, striatum SBR, putamen asymmetry, caudate/putamen ratio
   - FreeSurfer 3D MRI: brain segmentation volume, brainstem, putamen volume, caudate volume,
     hippocampus volume, lateral ventricles volume
   - Normative demographic controls: age at visit, biological sex
3. Trains:
   - Model 3A: XGBoost Imaging Regressor (+12 Months, target_np3tot_12m)
   - Model 3B: XGBoost Imaging Regressor (+24 Months, target_np3tot_24m)
   - Model 3C: XGBoost Imaging Classifier (+12m Worsening, prog_mcid_12m)
   - Model 3D: Deep Feedforward Neural Network (PyTorch MLP)
4. Evaluates on held-out test cohort (583 patients) via src.evaluate.evaluate_model_run
5. Generates SHAP feature importance plot saved to report/model3_imaging_shap_importance.png
6. Saves model artifacts in models/
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from evaluate import evaluate_model_run

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
REPORT_DIR = "report"

IMAGING_FEATURES = [
    # DaTSCAN SPECT Striatal Binding Ratios (SBR)
    "datscan_caudate_sbr",
    "datscan_putamen_sbr",
    "datscan_striatum_sbr",
    "datscan_putamen_asym",
    "datscan_caudate_putamen_ratio",
    # FreeSurfer 3D Structural MRI Volumes
    "mri_brain_seg_vol",
    "mri_brain_stem",
    "mri_putamen_vol",
    "mri_caudate_vol",
    "mri_hippocampus_vol",
    "mri_ventricles_vol",
    # Demographic / Normative Controls
    "age_at_visit",
    "is_male"
]


class ImagingMLP(nn.Module):
    """
    3-Layer Deep Feedforward Neural Network for Monomodal Neuroimaging Prognosis.
    """
    def __init__(self, input_dim=13, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU()
        )
        # Dual regression heads (+12m and +24m) and classification head (worsening)
        self.head_12m = nn.Linear(16, 1)
        self.head_24m = nn.Linear(16, 1)
        self.head_cls = nn.Linear(16, 1)

    def forward(self, x):
        feat = self.net(x)
        p12 = self.head_12m(feat).squeeze(-1)
        p24 = self.head_24m(feat).squeeze(-1)
        logits = self.head_cls(feat).squeeze(-1)
        return p12, p24, logits


def load_imaging_data():
    print("[1/6] Loading Canonical Splits for Neuroimaging Baseline...")
    train_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "train_tabular_latest.parquet"))
    val_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "val_tabular_latest.parquet"))
    test_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "test_tabular_latest.parquet"))

    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Check imaging availability
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dat_cnt = df["datscan_caudate_sbr"].notna().sum()
        mri_cnt = df["mri_brain_seg_vol"].notna().sum()
        any_cnt = (df["datscan_caudate_sbr"].notna() | df["mri_brain_seg_vol"].notna()).sum()
        print(f"  {name} Imaging Coverage: DaTSCAN={dat_cnt}/{len(df)} ({dat_cnt/len(df)*100:.1f}%), MRI={mri_cnt}/{len(df)} ({mri_cnt/len(df)*100:.1f}%), Any={any_cnt}/{len(df)} ({any_cnt/len(df)*100:.1f}%)")

    return train_df, val_df, test_df


def train_xgboost_12m(train_df, val_df, test_df):
    print("\n[2/6] Training Model 3A: XGBoost Imaging Regressor (+12 Months)...")
    X_train = train_df[IMAGING_FEATURES]
    y_train = train_df["target_np3tot_12m"].values

    X_val = val_df[IMAGING_FEATURES]
    y_val = val_df["target_np3tot_12m"].values

    X_test = test_df[IMAGING_FEATURES]
    y_test = test_df["target_np3tot_12m"].values

    model_12m = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        early_stopping_rounds=35,
        n_jobs=-1
    )

    model_12m.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds_test = model_12m.predict(X_test)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "model3_imaging_xgboost_12m.json")
    model_12m.save_model(model_path)
    print(f"  Saved trained model to {model_path} (Best Iteration: {model_12m.best_iteration})")

    metrics = evaluate_model_run(
        model_name="Model 3: Monomodal Neuroimaging (XGBoost)",
        target_horizon="+12 Months (NP3TOT)",
        y_true_reg=y_test,
        y_pred_reg=preds_test,
        metadata_df=test_df,
        modality_tags="Monomodal Neuroimaging Alone (DaTSCAN + MRI)",
        notes="Evaluates standalone prognostic value of brain scans without motor/cognitive exams."
    )
    return model_12m, preds_test, metrics


def train_xgboost_24m(train_df, val_df, test_df):
    print("\n[3/6] Training Model 3B: XGBoost Imaging Regressor (+24 Months)...")
    mask_train = train_df["target_np3tot_24m"].notna()
    mask_val = val_df["target_np3tot_24m"].notna()
    mask_test = test_df["target_np3tot_24m"].notna()

    X_train = train_df.loc[mask_train, IMAGING_FEATURES]
    y_train = train_df.loc[mask_train, "target_np3tot_24m"].values

    X_val = val_df.loc[mask_val, IMAGING_FEATURES]
    y_val = val_df.loc[mask_val, "target_np3tot_24m"].values

    X_test = test_df[IMAGING_FEATURES]
    y_test = test_df["target_np3tot_24m"].values

    model_24m = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        early_stopping_rounds=35,
        n_jobs=-1
    )

    model_24m.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds_test = model_24m.predict(X_test)

    model_path = os.path.join(MODELS_DIR, "model3_imaging_xgboost_24m.json")
    model_24m.save_model(model_path)
    print(f"  Saved trained model to {model_path} (Best Iteration: {model_24m.best_iteration})")

    metrics = evaluate_model_run(
        model_name="Model 3: Monomodal Neuroimaging (XGBoost)",
        target_horizon="+24 Months (NP3TOT)",
        y_true_reg=y_test,
        y_pred_reg=preds_test,
        metadata_df=test_df,
        modality_tags="Monomodal Neuroimaging Alone (DaTSCAN + MRI)",
        notes="Evaluates standalone prognostic value of brain scans on 24-month horizon."
    )
    return model_24m, preds_test, metrics


def train_xgboost_classifier(train_df, val_df, test_df):
    print("\n[4/6] Training Model 3C: XGBoost Imaging Classifier (Rapid Worsening >= 3.5 pts)...")
    mask_train = train_df["prog_mcid_12m"].notna()
    mask_val = val_df["prog_mcid_12m"].notna()

    X_train = train_df.loc[mask_train, IMAGING_FEATURES]
    y_train = train_df.loc[mask_train, "prog_mcid_12m"].values.astype(int)

    X_val = val_df.loc[mask_val, IMAGING_FEATURES]
    y_val = val_df.loc[mask_val, "prog_mcid_12m"].values.astype(int)

    test_c = test_df[test_df["prog_mcid_12m"].notna()].copy()
    X_test = test_c[IMAGING_FEATURES]
    y_test = test_c["prog_mcid_12m"].values.astype(int)

    scale_pos = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    model_cls = xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=3,
        scale_pos_weight=scale_pos,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        early_stopping_rounds=30,
        eval_metric="logloss",
        n_jobs=-1
    )

    model_cls.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    probs_test = model_cls.predict_proba(X_test)[:, 1]

    model_path = os.path.join(MODELS_DIR, "model3_imaging_xgboost_classifier.json")
    model_cls.save_model(model_path)
    print(f"  Saved trained classifier to {model_path} (Best Iteration: {model_cls.best_iteration})")

    metrics = evaluate_model_run(
        model_name="Model 3: Monomodal Neuroimaging Classifier",
        target_horizon="+12m Worsening (MCID >= 3.5)",
        y_true_reg=y_test.astype(float),
        y_pred_reg=probs_test,
        y_true_cls=y_test,
        y_prob_cls=probs_test,
        metadata_df=test_c,
        modality_tags="Monomodal Neuroimaging Alone (DaTSCAN + MRI)",
        notes="Classification of rapid motor worsening from imaging alone."
    )
    return model_cls, probs_test, metrics


def train_deep_mlp(train_df, val_df, test_df):
    print("\n[5/6] Training Model 3D: Deep Feedforward Neural Network (PyTorch MLP)...")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_tr = scaler.fit_transform(imputer.fit_transform(train_df[IMAGING_FEATURES]))
    X_va = scaler.transform(imputer.transform(val_df[IMAGING_FEATURES]))
    X_te = scaler.transform(imputer.transform(test_df[IMAGING_FEATURES]))

    y12_tr = train_df["target_np3tot_12m"].values
    y24_tr = train_df["target_np3tot_24m"].values
    ycl_tr = train_df["prog_mcid_12m"].values

    y12_va = val_df["target_np3tot_12m"].values
    y24_va = val_df["target_np3tot_24m"].values
    ycl_va = val_df["prog_mcid_12m"].values

    y12_te = test_df["target_np3tot_12m"].values
    y24_te = test_df["target_np3tot_24m"].values
    ycl_te = test_df["prog_mcid_12m"].values

    # Convert to tensors
    t_X_tr = torch.tensor(X_tr, dtype=torch.float32)
    t_y12_tr = torch.tensor(y12_tr, dtype=torch.float32)
    t_y24_tr = torch.tensor(y24_tr, dtype=torch.float32)
    t_ycl_tr = torch.tensor(ycl_tr, dtype=torch.float32)

    t_X_va = torch.tensor(X_va, dtype=torch.float32)
    t_y12_va = torch.tensor(y12_va, dtype=torch.float32)
    t_y24_va = torch.tensor(y24_va, dtype=torch.float32)

    t_X_te = torch.tensor(X_te, dtype=torch.float32)

    dataset_tr = TensorDataset(t_X_tr, t_y12_tr, t_y24_tr, t_ycl_tr)
    loader_tr = DataLoader(dataset_tr, batch_size=64, shuffle=True)

    mlp = ImagingMLP(input_dim=len(IMAGING_FEATURES), hidden_dim=64, dropout=0.25)
    optimizer = optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion_reg = nn.SmoothL1Loss()
    criterion_cls = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(1, 41):
        mlp.train()
        for bx, by12, by24, bycl in loader_tr:
            optimizer.zero_grad()
            p12, p24, logits = mlp(bx)

            # Masked regression loss
            loss12 = criterion_reg(p12, by12)
            m24 = ~torch.isnan(by24)
            loss24 = criterion_reg(p24[m24], by24[m24]) if m24.sum() > 0 else 0.0

            mcl = ~torch.isnan(bycl)
            losscl = criterion_cls(logits[mcl], bycl[mcl]) if mcl.sum() > 0 else 0.0

            total_loss = loss12 + loss24 + 0.5 * losscl
            total_loss.backward()
            optimizer.step()

        # Validation
        mlp.eval()
        with torch.no_grad():
            vp12, vp24, _ = mlp(t_X_va)
            vloss12 = criterion_reg(vp12, t_y12_va).item()
            vm24 = ~torch.isnan(t_y24_va)
            vloss24 = criterion_reg(vp24[vm24], t_y24_va[vm24]).item() if vm24.sum() > 0 else 0.0
            val_loss = vloss12 + vloss24

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = mlp.state_dict().copy()

    # Load best weights
    mlp.load_state_dict(best_weights)
    mlp_path = os.path.join(MODELS_DIR, "model3_imaging_mlp.pt")
    torch.save(mlp.state_dict(), mlp_path)
    print(f"  Trained Imaging Deep MLP (Best Val Loss: {best_val_loss:.4f}), saved to {mlp_path}")

    # Evaluate on test set
    mlp.eval()
    with torch.no_grad():
        test_p12, test_p24, test_logits = mlp(t_X_te)
        test_p12 = test_p12.numpy()
        test_p24 = test_p24.numpy()
        test_probs = torch.sigmoid(test_logits).numpy()

    evaluate_model_run(
        model_name="Model 3: Monomodal Neuroimaging (Deep MLP)",
        target_horizon="+12 Months (NP3TOT)",
        y_true_reg=y12_te,
        y_pred_reg=test_p12,
        metadata_df=test_df,
        modality_tags="Monomodal Neuroimaging Alone (DaTSCAN + MRI)",
        notes="Deep Feedforward Network on imaging-derived features."
    )

    evaluate_model_run(
        model_name="Model 3: Monomodal Neuroimaging (Deep MLP)",
        target_horizon="+24 Months (NP3TOT)",
        y_true_reg=y24_te,
        y_pred_reg=test_p24,
        metadata_df=test_df,
        modality_tags="Monomodal Neuroimaging Alone (DaTSCAN + MRI)",
        notes="Deep Feedforward Network on 24-month horizon."
    )


def generate_shap_analysis(model, test_df):
    print("\n[6/6] Computing TreeSHAP Feature Importance for Neuroimaging...")
    X_test = test_df[IMAGING_FEATURES]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Clean feature names for publication
    feature_labels = {
        "datscan_caudate_sbr": "DaTSCAN Caudate SBR",
        "datscan_putamen_sbr": "DaTSCAN Putamen SBR",
        "datscan_striatum_sbr": "DaTSCAN Mean Striatum SBR",
        "datscan_putamen_asym": "Putamen SBR Asymmetry",
        "datscan_caudate_putamen_ratio": "Caudate/Putamen SBR Ratio",
        "mri_brain_seg_vol": "MRI Total Brain Volume",
        "mri_brain_stem": "MRI Brainstem Volume",
        "mri_putamen_vol": "MRI Putamen Volume",
        "mri_caudate_vol": "MRI Caudate Volume",
        "mri_hippocampus_vol": "MRI Hippocampus Volume",
        "mri_ventricles_vol": "MRI Ventricles Volume",
        "age_at_visit": "Age at Scan",
        "is_male": "Sex (Male)"
    }
    plot_df = X_test.rename(columns=feature_labels)

    plt.figure(figsize=(10, 6), dpi=300)
    shap.summary_plot(
        shap_values,
        plot_df,
        show=False,
        max_display=13,
        plot_type="dot"
    )
    plt.title("Model 3: Monomodal Neuroimaging Feature Importance (TreeSHAP on Test Set)", fontsize=12, pad=15, fontweight="bold")
    plt.xlabel("SHAP Value (Impact on +12-Month Motor Progression MDS-UPDRS III)", fontsize=10)
    plt.tight_layout()

    out_fig = os.path.join(REPORT_DIR, "model3_imaging_shap_importance.png")
    plt.savefig(out_fig, bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP plot to {out_fig}")


def main():
    print("=" * 75)
    print("PHASE 3, STEP 4: TRAINING MODEL 3 (MONOMODAL NEUROIMAGING BASELINE)")
    print("=" * 75)

    train_df, val_df, test_df = load_imaging_data()

    # Train XGBoost Models
    model_12m, preds_12m, metrics_12m = train_xgboost_12m(train_df, val_df, test_df)
    model_24m, preds_24m, metrics_24m = train_xgboost_24m(train_df, val_df, test_df)
    model_cls, probs_cls, metrics_cls = train_xgboost_classifier(train_df, val_df, test_df)

    # Train Deep Feedforward Neural Network (MLP)
    train_deep_mlp(train_df, val_df, test_df)

    # SHAP Explainability
    generate_shap_analysis(model_12m, test_df)

    print("\n" + "=" * 75)
    print("MODEL 3: MONOMODAL NEUROIMAGING BASELINE TRAINING & EVALUATION COMPLETE!")
    print("=" * 75)


if __name__ == "__main__":
    main()
