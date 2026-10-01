"""
run_statistical_tests.py
-------------------------
Phase 6: Statistical Significance & Bootstrap Hypothesis Testing

Provides rigorous mathematical validation across all models evaluated on
the exact same 583 held-out test patients:
1. Research Question 1 (RQ1): Model 2 (BiLSTM + Attention) vs. Model 1 (Static XGBoost) on +24 Months.
2. Near-Term Horizon (+12 Months): Model 1 (XGBoost) vs. Model 2 (BiLSTM + Attention).
3. Research Question 2 (RQ2): Strategy B (Late Gated Fusion) vs. Strategy A (Early Fusion) on +24 Months.
4. Strategy B (Late Gated Fusion) vs. Strategy C (Cross-Attention Transformer) on +24 Months.
5. Multimodal Synergy: Late Fusion vs. Model 3 (Neuroimaging Alone) on +12m and +24m.

Statistical Methodology:
- Non-parametric Paired Wilcoxon Signed-Rank Test (handles non-normal medical error distributions)
- Paired Student's t-test
- 1,000-Iteration Paired Bootstrap Resampling with replacement:
  * Computes 95% Confidence Intervals (2.5% and 97.5% percentiles) for each model's MAE
  * Computes 95% Confidence Intervals for pairwise MAE differences: Delta = MAE_A - MAE_B
  * Computes empirical bootstrap p-values: P(Delta <= 0)
- Outputs:
  * report/statistical_significance_results.csv
  * report/statistical_significance_results.json
  * report/statistical_significance_distributions.png (publication figure)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from train_model2_lstm import LongitudinalBiLSTM, collate_sequences
from train_fusion_models import (
    EarlyFusionModel,
    LateFusionModel,
    CrossAttentionFusionModel,
    pad_collate_fn,
    ALL_FEATURE_COLS,
    CLINICAL_COLS,
    IMAGING_COLS,
    DEMO_COLS
)
from train_model1_xgboost import FEATURE_COLS as XGB_COLS

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
REPORT_DIR = "report"
N_BOOTSTRAP = 1000


def load_test_data():
    print("[1/4] Loading canonical patient datasets and test splits...")
    test_tabular = pd.read_parquet(os.path.join(PROCESSED_DIR, "test_tabular_latest.parquet"))
    test_longitudinal = pd.read_parquet(os.path.join(PROCESSED_DIR, "test_longitudinal.parquet"))
    train_longitudinal = pd.read_parquet(os.path.join(PROCESSED_DIR, "train_longitudinal.parquet"))

    print(f"  Test Tabular: {len(test_tabular)} patients | Test Longitudinal: {len(test_longitudinal)} visits")
    return test_tabular, test_longitudinal, train_longitudinal


def get_xgboost_predictions(test_tabular):
    print("[2/4] Generating predictions for Model 1 (XGBoost) and Model 3 (Imaging XGBoost)...")
    preds = {}

    # Model 1
    m1_12 = xgb.XGBRegressor()
    m1_12.load_model(os.path.join(MODELS_DIR, "model1_xgboost_12m.json"))
    m1_24 = xgb.XGBRegressor()
    m1_24.load_model(os.path.join(MODELS_DIR, "model1_xgboost_24m.json"))

    X_m1 = test_tabular[XGB_COLS]
    preds["Model 1: XGBoost"] = {
        "p12": m1_12.predict(X_m1),
        "p24": m1_24.predict(X_m1)
    }

    # Model 3
    m3_12 = xgb.XGBRegressor()
    m3_12.load_model(os.path.join(MODELS_DIR, "model3_imaging_xgboost_12m.json"))
    m3_24 = xgb.XGBRegressor()
    m3_24.load_model(os.path.join(MODELS_DIR, "model3_imaging_xgboost_24m.json"))

    IMG_ONLY_COLS = [
        "datscan_caudate_sbr", "datscan_putamen_sbr", "datscan_striatum_sbr",
        "datscan_putamen_asym", "datscan_caudate_putamen_ratio",
        "mri_brain_seg_vol", "mri_brain_stem", "mri_putamen_vol",
        "mri_caudate_vol", "mri_hippocampus_vol", "mri_ventricles_vol",
        "age_at_visit", "is_male"
    ]
    X_m3 = test_tabular[IMG_ONLY_COLS]
    preds["Model 3: Neuroimaging XGBoost"] = {
        "p12": m3_12.predict(X_m3),
        "p24": m3_24.predict(X_m3)
    }

    return preds


def get_deep_model_predictions(test_tabular, test_longitudinal, train_longitudinal):
    print("  Generating predictions for Model 2 (BiLSTM), Early, Late, and Cross-Attention Fusion...")
    preds = {}

    # 1. Feature scaler & imputation for Fusion & BiLSTM
    train_medians = train_longitudinal[ALL_FEATURE_COLS].median().fillna(0.0)
    scaler = StandardScaler()
    scaler.fit(train_longitudinal[ALL_FEATURE_COLS].fillna(train_medians))

    df = test_longitudinal.copy().sort_values(["PATNO", "visit_date"]).reset_index(drop=True)
    df[ALL_FEATURE_COLS] = df.groupby("PATNO")[ALL_FEATURE_COLS].ffill().fillna(train_medians)
    df[ALL_FEATURE_COLS] = scaler.transform(df[ALL_FEATURE_COLS])

    # Extract sequences for test_tabular patients strictly in order
    pats_order = test_tabular["PATNO"].values
    pat_to_rows = df.groupby("PATNO")

    sequences = []
    cur_scores = []
    for patno in pats_order:
        p_df = pat_to_rows.get_group(patno)
        sequences.append(p_df[ALL_FEATURE_COLS].values)
        cur_scores.append(p_df["NP3TOT"].iloc[-1])

    # Build tensor batch
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    max_len = max(lengths).item()
    batch_size = len(sequences)
    feat_dim = len(ALL_FEATURE_COLS)

    padded_x = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        padded_x[i, :seq_len] = torch.tensor(seq, dtype=torch.float32)
        mask[i, :seq_len] = True

    cur_scores_t = torch.tensor(cur_scores, dtype=torch.float32)

    # Model 2: BiLSTM
    # Note: Model 2 uses 43 features (ALL_FEATURE_COLS except delta_t is placed at the end in Model 2)
    # Let's inspect Model 2's feature columns
    from train_model2_lstm import FEATURE_COLS as LSTM_COLS
    scaler_lstm = StandardScaler()
    train_medians_lstm = train_longitudinal[LSTM_COLS].median().fillna(0.0)
    scaler_lstm.fit(train_longitudinal[LSTM_COLS].fillna(train_medians_lstm))

    df_lstm = test_longitudinal.copy().sort_values(["PATNO", "visit_date"]).reset_index(drop=True)
    df_lstm[LSTM_COLS] = df_lstm.groupby("PATNO")[LSTM_COLS].ffill().fillna(train_medians_lstm)
    df_lstm[LSTM_COLS] = scaler_lstm.transform(df_lstm[LSTM_COLS])

    pat_to_rows_lstm = df_lstm.groupby("PATNO")
    seq_lstm = [pat_to_rows_lstm.get_group(p)[LSTM_COLS].values for p in pats_order]
    padded_x_lstm = torch.zeros(batch_size, max_len, len(LSTM_COLS), dtype=torch.float32)
    for i, seq in enumerate(seq_lstm):
        padded_x_lstm[i, :len(seq)] = torch.tensor(seq, dtype=torch.float32)

    model2 = LongitudinalBiLSTM(input_dim=len(LSTM_COLS), hidden_dim=64, num_layers=2, dropout=0.25)
    ckpt2 = torch.load(os.path.join(MODELS_DIR, "model2_bilstm.pt"), map_location="cpu", weights_only=False)
    if isinstance(ckpt2, dict) and "model_state_dict" in ckpt2:
        model2.load_state_dict(ckpt2["model_state_dict"])
    else:
        model2.load_state_dict(ckpt2)
    model2.eval()
    with torch.no_grad():
        p12_m2, p24_m2, _, _ = model2(padded_x_lstm, mask, lengths, cur_scores_t)
        preds["Model 2: Longitudinal BiLSTM"] = {
            "p12": p12_m2.numpy(),
            "p24": p24_m2.numpy()
        }

    # Strategy A: Early Fusion
    model_early = EarlyFusionModel(input_dim=len(ALL_FEATURE_COLS), hidden_dim=64, dropout=0.25)
    model_early.load_state_dict(torch.load(os.path.join(MODELS_DIR, "early_fusion.pt"), map_location="cpu", weights_only=False))
    model_early.eval()
    with torch.no_grad():
        p12_ef, p24_ef, _, _ = model_early(padded_x, mask, lengths, cur_scores_t)
        preds["Fusion: Early Fusion"] = {
            "p12": p12_ef.numpy(),
            "p24": p24_ef.numpy()
        }

    # Strategy B: Late Fusion
    model_late = LateFusionModel(clin_dim=len(CLINICAL_COLS), img_dim=len(IMAGING_COLS), demo_dim=len(DEMO_COLS), hidden_dim=48, dropout=0.25)
    model_late.load_state_dict(torch.load(os.path.join(MODELS_DIR, "late_fusion.pt"), map_location="cpu", weights_only=False))
    model_late.eval()
    with torch.no_grad():
        p12_lf, p24_lf, _, _ = model_late(padded_x, mask, lengths, cur_scores_t)
        preds["Fusion: Late Gated Fusion"] = {
            "p12": p12_lf.numpy(),
            "p24": p24_lf.numpy()
        }

    # Strategy C: Cross-Attention
    model_cross = CrossAttentionFusionModel(clin_dim=len(CLINICAL_COLS), img_dim=len(IMAGING_COLS), demo_dim=len(DEMO_COLS), hidden_dim=64, num_heads=4, dropout=0.25)
    model_cross.load_state_dict(torch.load(os.path.join(MODELS_DIR, "cross-attention_transformer.pt"), map_location="cpu", weights_only=False))
    model_cross.eval()
    with torch.no_grad():
        p12_ca, p24_ca, _, _ = model_cross(padded_x, mask, lengths, cur_scores_t)
        preds["Fusion: Cross-Attention Transformer"] = {
            "p12": p12_ca.numpy(),
            "p24": p24_ca.numpy()
        }

    return preds


def compute_bootstrap_and_tests(y_true, pred_a, pred_b, name_a, name_b, horizon, n_boot=1000):
    """
    Computes rigorous paired statistical tests and 1,000 bootstrap confidence intervals.
    """
    valid = ~np.isnan(y_true) & ~np.isnan(pred_a) & ~np.isnan(pred_b)
    y = y_true[valid]
    pa = pred_a[valid]
    pb = pred_b[valid]
    n = len(y)

    # Absolute errors per patient
    ae_a = np.abs(y - pa)
    ae_b = np.abs(y - pb)
    diff = ae_a - ae_b  # Positive if model B is better (lower error)

    mae_a = np.mean(ae_a)
    mae_b = np.mean(ae_b)
    delta_mae = mae_a - mae_b

    # 1. Non-parametric Paired Wilcoxon Signed-Rank Test
    wilc_stat, wilc_p = stats.wilcoxon(ae_a, ae_b, alternative="two-sided")

    # 2. Paired t-test
    ttest_stat, ttest_p = stats.ttest_rel(ae_a, ae_b)

    # 3. 1,000 Paired Bootstrap Resampling
    boot_mae_a = []
    boot_mae_b = []
    boot_delta = []

    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        b_a = np.mean(ae_a[idx])
        b_b = np.mean(ae_b[idx])
        boot_mae_a.append(b_a)
        boot_mae_b.append(b_b)
        boot_delta.append(b_a - b_b)

    boot_mae_a = np.array(boot_mae_a)
    boot_mae_b = np.array(boot_mae_b)
    boot_delta = np.array(boot_delta)

    ci_a_low, ci_a_high = np.percentile(boot_mae_a, [2.5, 97.5])
    ci_b_low, ci_b_high = np.percentile(boot_mae_b, [2.5, 97.5])
    ci_delta_low, ci_delta_high = np.percentile(boot_delta, [2.5, 97.5])

    # Empirical one-sided bootstrap p-value
    if delta_mae > 0:
        boot_p = np.mean(boot_delta <= 0)
    else:
        boot_p = np.mean(boot_delta >= 0)

    # Superiority declaration
    if wilc_p < 0.05:
        if delta_mae > 0:
            winner = f"{name_b} (p = {wilc_p:.4e})"
        else:
            winner = f"{name_a} (p = {wilc_p:.4e})"
    else:
        winner = f"Not Significant (p = {wilc_p:.4f})"

    return {
        "comparison": f"{name_a} vs. {name_b}",
        "horizon": horizon,
        "n_samples": n,
        "mae_model_a": mae_a,
        "ci95_a": [ci_a_low, ci_a_high],
        "mae_model_b": mae_b,
        "ci95_b": [ci_b_low, ci_b_high],
        "delta_mae (A - B)": delta_mae,
        "ci95_delta": [ci_delta_low, ci_delta_high],
        "wilcoxon_stat": float(wilc_stat),
        "wilcoxon_p": float(wilc_p),
        "ttest_p": float(ttest_p),
        "bootstrap_p": float(boot_p),
        "winner": winner,
        "boot_delta_dist": boot_delta
    }


def generate_publication_figure(tests_results, all_models_ci, out_fig_path):
    print("\n[Plotting] Generating publication-grade Statistical Significance figure...")
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    # -------------------------------------------------------------
    # Panel A: Forest Plot of All Models MAE with 95% Bootstrap CIs
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    models = list(all_models_ci.keys())
    y_pos = np.arange(len(models))

    # +12m
    mae12 = [all_models_ci[m]["12m"]["mae"] for m in models]
    err12_low = [all_models_ci[m]["12m"]["mae"] - all_models_ci[m]["12m"]["ci"][0] for m in models]
    err12_high = [all_models_ci[m]["12m"]["ci"][1] - all_models_ci[m]["12m"]["mae"] for m in models]

    # +24m
    mae24 = [all_models_ci[m]["24m"]["mae"] for m in models]
    err24_low = [all_models_ci[m]["24m"]["mae"] - all_models_ci[m]["24m"]["ci"][0] for m in models]
    err24_high = [all_models_ci[m]["24m"]["ci"][1] - all_models_ci[m]["24m"]["mae"] for m in models]

    offset = 0.18
    ax1.errorbar(mae12, y_pos - offset, xerr=[err12_low, err12_high], fmt='o', color='#2563eb',
                 ecolor='#2563eb', elinewidth=2, capsize=4, capthick=1.5, label='+12 Months (95% CI)')
    ax1.errorbar(mae24, y_pos + offset, xerr=[err24_low, err24_high], fmt='s', color='#059669',
                 ecolor='#059669', elinewidth=2, capsize=4, capthick=1.5, label='+24 Months (95% CI)')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontsize=9.5, fontweight='bold')
    ax1.set_xlabel("Mean Absolute Error (MDS-UPDRS Part III Points)", fontsize=10, fontweight='bold')
    ax1.set_title("A. Forest Plot: Model Performance with 95% Bootstrap CIs", fontsize=11, fontweight='bold', pad=10)
    ax1.grid(True, linestyle="--", alpha=0.5, axis='x')
    ax1.legend(loc="lower right", frameon=True)

    # -------------------------------------------------------------
    # Panel B: RQ1 Bootstrap Difference: Model 1 XGBoost - Model 2 BiLSTM (+24m)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    res_rq1 = tests_results["RQ1: Model 1 XGBoost vs Model 2 BiLSTM (+24m)"]
    boot_dist_rq1 = res_rq1["boot_delta_dist"]
    ci_low, ci_high = res_rq1["ci95_delta"]
    p_val = res_rq1["wilcoxon_p"]

    sns.histplot(boot_dist_rq1, kde=True, color="#0284c7", ax=ax2, bins=35, stat="density", alpha=0.6)
    ax2.axvline(0.0, color="#ef4444", linestyle="--", linewidth=2, label="Null Hypothesis (Delta = 0)")
    ax2.axvline(res_rq1["delta_mae (A - B)"], color="#0f172a", linestyle="-", linewidth=2.5,
                label=f"Observed Delta = +{res_rq1['delta_mae (A - B)']:.3f} pts")
    ax2.axvline(ci_low, color="#0284c7", linestyle=":", linewidth=1.5)
    ax2.axvline(ci_high, color="#0284c7", linestyle=":", linewidth=1.5, label=f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")

    ax2.set_title(f"B. RQ1 Hypothesis Test (+24 Months)\nModel 1 (XGBoost) vs. Model 2 (BiLSTM) [p = {p_val:.4f}]",
                  fontsize=11, fontweight='bold', pad=10)
    ax2.set_xlabel("Paired Error Difference: XGBoost Error - BiLSTM Error (Points)", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Bootstrap Density", fontsize=10, fontweight='bold')
    ax2.legend(loc="upper right", frameon=True, fontsize=8.5)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel C: RQ2 Bootstrap Difference: Early Fusion - Late Gated Fusion (+24m)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    res_rq2 = tests_results["RQ2: Early Fusion vs Late Fusion (+24m)"]
    boot_dist_rq2 = res_rq2["boot_delta_dist"]
    ci_low_q2, ci_high_q2 = res_rq2["ci95_delta"]
    p_val_q2 = res_rq2["wilcoxon_p"]

    sns.histplot(boot_dist_rq2, kde=True, color="#10b981", ax=ax3, bins=35, stat="density", alpha=0.6)
    ax3.axvline(0.0, color="#ef4444", linestyle="--", linewidth=2, label="Null Hypothesis (Delta = 0)")
    ax3.axvline(res_rq2["delta_mae (A - B)"], color="#0f172a", linestyle="-", linewidth=2.5,
                label=f"Observed Delta = +{res_rq2['delta_mae (A - B)']:.3f} pts")
    ax3.axvline(ci_low_q2, color="#059669", linestyle=":", linewidth=1.5)
    ax3.axvline(ci_high_q2, color="#059669", linestyle=":", linewidth=1.5, label=f"95% CI: [{ci_low_q2:.3f}, {ci_high_q2:.3f}]")

    ax3.set_title(f"C. RQ2 Hypothesis Test (+24 Months)\nEarly Fusion vs. Late Gated Fusion [p = {p_val_q2:.4f}]",
                  fontsize=11, fontweight='bold', pad=10)
    ax3.set_xlabel("Paired Error Difference: Early Fusion - Late Fusion Error (Points)", fontsize=10, fontweight='bold')
    ax3.set_ylabel("Bootstrap Density", fontsize=10, fontweight='bold')
    ax3.legend(loc="upper right", frameon=True, fontsize=8.5)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel D: Rigorous Statistical Significance Summary Table
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    table_data = [
        ["Research Hypothesis", "Horizon", "Obs. Delta", "95% Bootstrap CI", "Wilcoxon p", "Decision"]
    ]
    for key, r in tests_results.items():
        # Shorten hypothesis name
        short_hyp = key.split(" (")[0]
        horiz = "+24m" if "+24m" in key else "+12m"
        delta_str = f"{r['delta_mae (A - B)']:+.3f}"
        ci_str = f"[{r['ci95_delta'][0]:.3f}, {r['ci95_delta'][1]:.3f}]"
        p_str = f"{r['wilcoxon_p']:.4e}" if r['wilcoxon_p'] < 0.001 else f"{r['wilcoxon_p']:.4f}"
        sig_str = "SIG (p < 0.05)" if r['wilcoxon_p'] < 0.05 else "Not Sig"
        table_data.append([short_hyp, horiz, delta_str, ci_str, p_str, sig_str])

    col_widths = [0.33, 0.10, 0.13, 0.21, 0.12, 0.13]
    table = ax4.table(cellText=table_data, colWidths=col_widths, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1.0, 1.7)

    # Header styling
    for col in range(6):
        cell = table[(0, col)]
        cell.set_facecolor("#0f172a")
        cell.set_text_props(color="white", weight="bold")

    # Alternate row colors
    for row in range(1, len(table_data)):
        bg_col = "#f1f5f9" if row % 2 == 1 else "#ffffff"
        for col in range(6):
            cell = table[(row, col)]
            cell.set_facecolor(bg_col)
            if col == 5:
                if "SIG" in table_data[row][col]:
                    cell.set_text_props(color="#15803d", weight="bold")
                else:
                    cell.set_text_props(color="#64748b")

    ax4.set_title("D. Formal Hypothesis Testing Ledger (N=583 Held-Out Patients)",
                  fontsize=11, fontweight='bold', pad=15)

    plt.suptitle("Statistical Significance & Bootstrap Hypothesis Testing Protocol (1,000 Iterations)\nRigorous Validation of Research Questions 1 & 2 on Held-Out PPMI Cohort",
                 fontsize=14, fontweight="bold", y=0.98, color="#0f172a")

    plt.savefig(out_fig_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved Statistical Significance figure to: {out_fig_path}")


def main():
    print("=" * 80)
    print("PHASE 6: STATISTICAL SIGNIFICANCE & BOOTSTRAP HYPOTHESIS TESTING")
    print("=" * 80)

    test_tabular, test_longitudinal, train_longitudinal = load_test_data()

    # Ground truth targets
    y12 = test_tabular["target_np3tot_12m"].values
    y24 = test_tabular["target_np3tot_24m"].values

    # Predictions
    xgb_preds = get_xgboost_predictions(test_tabular)
    deep_preds = get_deep_model_predictions(test_tabular, test_longitudinal, train_longitudinal)

    all_preds = {**xgb_preds, **deep_preds}

    # 1. Compute 95% CIs for individual models
    print("\n[3/4] Computing 1,000-iteration Bootstrap Confidence Intervals for all models...")
    all_models_ci = {}
    for m_name, preds_dict in all_preds.items():
        # +12m
        v12 = ~np.isnan(y12) & ~np.isnan(preds_dict["p12"])
        ae12 = np.abs(y12[v12] - preds_dict["p12"][v12])
        boot12 = [np.mean(np.random.choice(ae12, size=len(ae12), replace=True)) for _ in range(N_BOOTSTRAP)]
        ci12 = np.percentile(boot12, [2.5, 97.5])

        # +24m
        v24 = ~np.isnan(y24) & ~np.isnan(preds_dict["p24"])
        ae24 = np.abs(y24[v24] - preds_dict["p24"][v24])
        boot24 = [np.mean(np.random.choice(ae24, size=len(ae24), replace=True)) for _ in range(N_BOOTSTRAP)]
        ci24 = np.percentile(boot24, [2.5, 97.5])

        all_models_ci[m_name] = {
            "12m": {"mae": float(np.mean(ae12)), "ci": [float(ci12[0]), float(ci12[1])]},
            "24m": {"mae": float(np.mean(ae24)), "ci": [float(ci24[0]), float(ci24[1])]}
        }
        print(f"  {m_name:35s} | 12m MAE: {np.mean(ae12):.3f} [{ci12[0]:.3f}, {ci12[1]:.3f}] | 24m MAE: {np.mean(ae24):.3f} [{ci24[0]:.3f}, {ci24[1]:.3f}]")

    # 2. Key Pairwise Hypothesis Tests
    print("\n[4/4] Conducting Paired Wilcoxon Signed-Rank and Bootstrap Difference Tests...")
    tests_results = {}

    # Test 1: RQ1 (+24m) Model 1 XGBoost vs Model 2 BiLSTM
    tests_results["RQ1: Model 1 XGBoost vs Model 2 BiLSTM (+24m)"] = compute_bootstrap_and_tests(
        y24, all_preds["Model 1: XGBoost"]["p24"], all_preds["Model 2: Longitudinal BiLSTM"]["p24"],
        "Model 1: XGBoost", "Model 2: Longitudinal BiLSTM", "+24 Months", n_boot=N_BOOTSTRAP
    )

    # Test 2: Near-Term (+12m) Model 1 XGBoost vs Model 2 BiLSTM
    tests_results["Near-Term: Model 1 XGBoost vs Model 2 BiLSTM (+12m)"] = compute_bootstrap_and_tests(
        y12, all_preds["Model 1: XGBoost"]["p12"], all_preds["Model 2: Longitudinal BiLSTM"]["p12"],
        "Model 1: XGBoost", "Model 2: Longitudinal BiLSTM", "+12 Months", n_boot=N_BOOTSTRAP
    )

    # Test 3: RQ2 (+24m) Early Fusion vs Late Gated Fusion
    tests_results["RQ2: Early Fusion vs Late Fusion (+24m)"] = compute_bootstrap_and_tests(
        y24, all_preds["Fusion: Early Fusion"]["p24"], all_preds["Fusion: Late Gated Fusion"]["p24"],
        "Fusion: Early Fusion", "Fusion: Late Gated Fusion", "+24 Months", n_boot=N_BOOTSTRAP
    )

    # Test 4: RQ2 (+24m) Cross-Attention vs Late Gated Fusion
    tests_results["RQ2: Cross-Attention vs Late Fusion (+24m)"] = compute_bootstrap_and_tests(
        y24, all_preds["Fusion: Cross-Attention Transformer"]["p24"], all_preds["Fusion: Late Gated Fusion"]["p24"],
        "Fusion: Cross-Attention", "Fusion: Late Gated Fusion", "+24 Months", n_boot=N_BOOTSTRAP
    )

    # Test 5: RQ2 (+12m) Early Fusion vs Late Gated Fusion
    tests_results["RQ2: Early Fusion vs Late Fusion (+12m)"] = compute_bootstrap_and_tests(
        y12, all_preds["Fusion: Early Fusion"]["p12"], all_preds["Fusion: Late Gated Fusion"]["p12"],
        "Fusion: Early Fusion", "Fusion: Late Gated Fusion", "+12 Months", n_boot=N_BOOTSTRAP
    )

    # Test 6: Synergy (+24m) Standalone Neuroimaging vs Late Fusion
    tests_results["Synergy: Imaging Alone vs Late Fusion (+24m)"] = compute_bootstrap_and_tests(
        y24, all_preds["Model 3: Neuroimaging XGBoost"]["p24"], all_preds["Fusion: Late Gated Fusion"]["p24"],
        "Model 3: Neuroimaging", "Fusion: Late Gated Fusion", "+24 Months", n_boot=N_BOOTSTRAP
    )

    # Print summary
    print("\n" + "=" * 95)
    print(f"{'Comparison':<45} | {'Horizon':<8} | {'Delta MAE':<10} | {'95% CI':<18} | {'Wilcoxon p':<10}")
    print("=" * 95)
    csv_rows = []
    for k, v in tests_results.items():
        delta_str = f"{v['delta_mae (A - B)']:+.3f}"
        ci_str = f"[{v['ci95_delta'][0]:.3f}, {v['ci95_delta'][1]:.3f}]"
        p_str = f"{v['wilcoxon_p']:.4e}" if v['wilcoxon_p'] < 0.001 else f"{v['wilcoxon_p']:.4f}"
        print(f"{k:<45} | {v['horizon']:<8} | {delta_str:<10} | {ci_str:<18} | {p_str:<10}")

        csv_rows.append({
            "test_name": k,
            "comparison": v["comparison"],
            "horizon": v["horizon"],
            "n_samples": v["n_samples"],
            "mae_model_a": v["mae_model_a"],
            "ci95_a_low": v["ci95_a"][0],
            "ci95_a_high": v["ci95_a"][1],
            "mae_model_b": v["mae_model_b"],
            "ci95_b_low": v["ci95_b"][0],
            "ci95_b_high": v["ci95_b"][1],
            "delta_mae_a_minus_b": v["delta_mae (A - B)"],
            "ci95_delta_low": v["ci95_delta"][0],
            "ci95_delta_high": v["ci95_delta"][1],
            "wilcoxon_p": v["wilcoxon_p"],
            "ttest_p": v["ttest_p"],
            "bootstrap_p": v["bootstrap_p"],
            "winner": v["winner"]
        })

    # Save CSV & JSON
    out_csv = os.path.join(REPORT_DIR, "statistical_significance_results.csv")
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    print(f"\n  Saved Statistical Significance CSV to: {out_csv}")

    # Clean dict for JSON without numpy arrays
    json_dict = {
        "all_models_ci": all_models_ci,
        "pairwise_tests": {
            k: {kk: vv for kk, vv in v.items() if kk != "boot_delta_dist"}
            for k, v in tests_results.items()
        }
    }
    out_json = os.path.join(REPORT_DIR, "statistical_significance_results.json")
    with open(out_json, "w") as f:
        json.dump(json_dict, f, indent=2)
    print(f"  Saved Statistical Significance JSON to: {out_json}")

    # Generate Publication Figure
    out_fig = os.path.join(REPORT_DIR, "statistical_significance_distributions.png")
    generate_publication_figure(tests_results, all_models_ci, out_fig)

    print("\n" + "=" * 80)
    print("PHASE 6: STATISTICAL SIGNIFICANCE PROTOCOL COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
