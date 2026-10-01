"""
visualize_baselines_comparison.py
---------------------------------
Generates a comprehensive multi-panel publication-grade comparison figure
for all Phase 3 Baselines (Model 1, Model 2, Model 3):
1. Panel A: MAE by Horizon (+12m vs +24m) across all models.
2. Panel B: Explained Variance (R²) by Horizon.
3. Panel C: Pearson Correlation (r) with scatter trend.
4. Panel D: ROC-AUC for Rapid Motor Worsening (MCID >= 3.5 points).

Saves figure to: report/baseline_models_comparison.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

REPORT_DIR = "report"
BENCHMARK_CSV = os.path.join(REPORT_DIR, "benchmark_results.csv")
OUT_FIG = os.path.join(REPORT_DIR, "baseline_models_comparison.png")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


def generate_comparison_plot():
    print("[1/2] Reading benchmark results ledger...")
    df = pd.read_csv(BENCHMARK_CSV)
    
    # Filter for the main regression models
    reg_df = df[df["target_horizon"].isin(["+12 Months (NP3TOT)", "+24 Months (NP3TOT)"])].copy()
    
    # Standardize model names for display
    name_map = {
        "Model 1: Tabular XGBoost": "Model 1: XGBoost\n(Snapshot Tabular)",
        "Model 2: Longitudinal BiLSTM": "Model 2: BiLSTM + Attn\n(Longitudinal Sequence)",
        "Model 3: Monomodal Neuroimaging (XGBoost)": "Model 3: Neuroimaging XGB\n(DaTSCAN + MRI Alone)",
        "Model 3: Monomodal Neuroimaging (Deep MLP)": "Model 3: Neuroimaging MLP\n(Deep MLP Alone)"
    }
    reg_df["display_name"] = reg_df["model_name"].map(name_map)
    reg_df = reg_df.dropna(subset=["display_name"])

    # Color palette
    colors = {
        "Model 1: XGBoost\n(Snapshot Tabular)": "#2563eb",         # Royal Blue
        "Model 2: BiLSTM + Attn\n(Longitudinal Sequence)": "#059669", # Emerald Green (Winner at 24m)
        "Model 3: Neuroimaging XGB\n(DaTSCAN + MRI Alone)": "#d97706", # Amber Orange
        "Model 3: Neuroimaging MLP\n(Deep MLP Alone)": "#dc2626"       # Crimson Red
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    fig.suptitle("Phase 3 Baselines Benchmark: Model 1 vs. Model 2 vs. Model 3\nEvaluated on Held-Out Test Patients (N=583)", 
                 fontsize=16, fontweight="bold", y=0.98, color="#0f172a")

    models_order = [
        "Model 1: XGBoost\n(Snapshot Tabular)",
        "Model 2: BiLSTM + Attn\n(Longitudinal Sequence)",
        "Model 3: Neuroimaging XGB\n(DaTSCAN + MRI Alone)",
        "Model 3: Neuroimaging MLP\n(Deep MLP Alone)"
    ]

    # --- PANEL A: MAE (Mean Absolute Error) ---
    ax_a = axes[0, 0]
    p12_mae = [reg_df[(reg_df["display_name"] == m) & (reg_df["target_horizon"] == "+12 Months (NP3TOT)")]["mae"].values[0] for m in models_order]
    p24_mae = [reg_df[(reg_df["display_name"] == m) & (reg_df["target_horizon"] == "+24 Months (NP3TOT)")]["mae"].values[0] for m in models_order]

    x = np.arange(len(models_order))
    width = 0.35

    rects1 = ax_a.bar(x - width/2, p12_mae, width, label="+12 Months Horizon", color="#3b82f6", alpha=0.9, edgecolor="#1e3a8a")
    rects2 = ax_a.bar(x + width/2, p24_mae, width, label="+24 Months Horizon", color="#10b981", alpha=0.9, edgecolor="#064e3b")

    ax_a.set_title("A. Mean Absolute Error (MAE in Points) — Lower is Better", fontsize=12, fontweight="bold", pad=10)
    ax_a.set_ylabel("MAE (MDS-UPDRS Part III Points)", fontsize=10, fontweight="bold")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(models_order, fontsize=8.5)
    ax_a.set_ylim(0, 10.5)
    ax_a.legend(loc="upper left", frameon=True)
    ax_a.axhline(3.5, color="#ef4444", linestyle="--", linewidth=1.2, alpha=0.7, label="MCID Threshold (3.5 pts)")

    for rect in rects1:
        h = rect.get_height()
        ax_a.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax_a.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # --- PANEL B: R² (Explained Variance) ---
    ax_b = axes[0, 1]
    p12_r2 = [reg_df[(reg_df["display_name"] == m) & (reg_df["target_horizon"] == "+12 Months (NP3TOT)")]["r2"].values[0] * 100 for m in models_order]
    p24_r2 = [reg_df[(reg_df["display_name"] == m) & (reg_df["target_horizon"] == "+24 Months (NP3TOT)")]["r2"].values[0] * 100 for m in models_order]

    rects1_b = ax_b.bar(x - width/2, p12_r2, width, label="+12 Months Horizon", color="#6366f1", alpha=0.9, edgecolor="#312e81")
    rects2_b = ax_b.bar(x + width/2, p24_r2, width, label="+24 Months Horizon", color="#14b8a6", alpha=0.9, edgecolor="#134e4a")

    ax_b.set_title("B. Explained Variance (R² %) — Higher is Better", fontsize=12, fontweight="bold", pad=10)
    ax_b.set_ylabel("R² Variance Explained (%)", fontsize=10, fontweight="bold")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(models_order, fontsize=8.5)
    ax_b.set_ylim(0, 90)
    ax_b.legend(loc="upper right", frameon=True)

    for rect in rects1_b:
        h = rect.get_height()
        ax_b.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for rect in rects2_b:
        h = rect.get_height()
        ax_b.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # --- PANEL C: Pearson Correlation (r) ---
    ax_c = axes[1, 0]
    p12_r = [reg_df[(reg_df["display_name"] == m) & (reg_df["target_horizon"] == "+12 Months (NP3TOT)")]["pearson_r"].values[0] for m in models_order]
    p24_r = [reg_df[(reg_df["display_name"] == m) & (reg_df["target_horizon"] == "+24 Months (NP3TOT)")]["pearson_r"].values[0] for m in models_order]

    rects1_c = ax_c.bar(x - width/2, p12_r, width, label="+12 Months Horizon", color="#8b5cf6", alpha=0.9, edgecolor="#4c1d95")
    rects2_c = ax_c.bar(x + width/2, p24_r, width, label="+24 Months Horizon", color="#f59e0b", alpha=0.9, edgecolor="#78350f")

    ax_c.set_title("C. Pearson Correlation (r) with Observed Trajectories", fontsize=12, fontweight="bold", pad=10)
    ax_c.set_ylabel("Pearson Correlation Coefficient (r)", fontsize=10, fontweight="bold")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(models_order, fontsize=8.5)
    ax_c.set_ylim(0, 1.05)
    ax_c.legend(loc="lower right", frameon=True)

    for rect in rects1_c:
        h = rect.get_height()
        ax_c.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for rect in rects2_c:
        h = rect.get_height()
        ax_c.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # --- PANEL D: ROC-AUC for Rapid Motor Worsening ---
    ax_d = axes[1, 1]
    cls_df = df[df["target_horizon"] == "+12m Worsening (MCID >= 3.5)"].copy()
    cls_map = {
        "Model 1: Tabular XGBoost Classifier": "Model 1: XGBoost\n(Snapshot Tabular)",
        "Model 2: Longitudinal BiLSTM Classifier": "Model 2: BiLSTM + Attn\n(Longitudinal Sequence)",
        "Model 3: Monomodal Neuroimaging Classifier": "Model 3: Neuroimaging XGB\n(DaTSCAN + MRI Alone)"
    }
    cls_df["display_name"] = cls_df["model_name"].map(cls_map)
    cls_df = cls_df.dropna(subset=["display_name"])

    cls_models = [
        "Model 1: XGBoost\n(Snapshot Tabular)",
        "Model 2: BiLSTM + Attn\n(Longitudinal Sequence)",
        "Model 3: Neuroimaging XGB\n(DaTSCAN + MRI Alone)"
    ]
    auc_vals = [cls_df[cls_df["display_name"] == m]["roc_auc"].values[0] for m in cls_models]
    bar_colors = ["#2563eb", "#059669", "#d97706"]

    bars = ax_d.bar(cls_models, auc_vals, width=0.45, color=bar_colors, alpha=0.9, edgecolor="#0f172a")
    ax_d.set_title("D. ROC-AUC for Rapid Worsening Alert (MCID >= 3.5 pts)", fontsize=12, fontweight="bold", pad=10)
    ax_d.set_ylabel("ROC-AUC Score", fontsize=10, fontweight="bold")
    ax_d.set_ylim(0.4, 0.85)
    ax_d.axhline(0.5, color="#64748b", linestyle=":", linewidth=1.5, label="Random Guess (0.50)")
    ax_d.legend(loc="upper right", frameon=True)

    for bar in bars:
        h = bar.get_height()
        ax_d.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(REPORT_DIR, exist_ok=True)
    plt.savefig(OUT_FIG, bbox_inches="tight")
    plt.close()
    print(f"[2/2] Saved comprehensive baseline comparison plot to: {OUT_FIG}")


if __name__ == "__main__":
    generate_comparison_plot()
