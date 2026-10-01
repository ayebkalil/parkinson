"""
visualize_explainability.py
----------------------------
Phase 6, Step 2: Attention & Gating Explainability Visualizations

Addresses:
1. Temporal Attention Mechanism:
   - How does the BiLSTM distribute attention across past visits?
   - Do rapid motor progressors (MCID >= 3.5 pts) trigger sharp attention shifts?
2. Multimodal Gating Dynamics (Late Fusion Mixture-of-Experts):
   - What proportion of decision weight is assigned to Clinical Trajectories (w_clin)
     versus Neuroimaging Biomarkers (w_img)?
   - How does gating adapt across Disease Duration, Levodopa medication state, and DaTSCAN SBR?
3. Individual Patient Case Studies:
   - Visualizing real patient trajectories from the held-out test set with attention heatmaps.

Output Artifact:
- report/explainability_attention_and_gating.png (4-panel publication figure)
- report/explainability_summary_metrics.json
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.preprocessing import StandardScaler

from train_model2_lstm import LongitudinalBiLSTM
from train_fusion_models import (
    LateFusionModel,
    pad_collate_fn,
    ALL_FEATURE_COLS,
    CLINICAL_COLS,
    IMAGING_COLS,
    DEMO_COLS
)
from train_model2_lstm import FEATURE_COLS as LSTM_COLS

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
REPORT_DIR = "report"


def load_data_and_models():
    print("[1/3] Loading test data and trained model checkpoints...")
    test_tabular = pd.read_parquet(os.path.join(PROCESSED_DIR, "test_tabular_latest.parquet"))
    test_longitudinal = pd.read_parquet(os.path.join(PROCESSED_DIR, "test_longitudinal.parquet"))
    train_longitudinal = pd.read_parquet(os.path.join(PROCESSED_DIR, "train_longitudinal.parquet"))

    # Scalers
    train_medians_all = train_longitudinal[ALL_FEATURE_COLS].median().fillna(0.0)
    scaler_all = StandardScaler()
    scaler_all.fit(train_longitudinal[ALL_FEATURE_COLS].fillna(train_medians_all))

    train_medians_lstm = train_longitudinal[LSTM_COLS].median().fillna(0.0)
    scaler_lstm = StandardScaler()
    scaler_lstm.fit(train_longitudinal[LSTM_COLS].fillna(train_medians_lstm))

    # Prep longitudinal data
    df_all = test_longitudinal.copy().sort_values(["PATNO", "visit_date"]).reset_index(drop=True)
    df_all[ALL_FEATURE_COLS] = df_all.groupby("PATNO")[ALL_FEATURE_COLS].ffill().fillna(train_medians_all)
    df_all[ALL_FEATURE_COLS] = scaler_all.transform(df_all[ALL_FEATURE_COLS])

    df_lstm = test_longitudinal.copy().sort_values(["PATNO", "visit_date"]).reset_index(drop=True)
    df_lstm[LSTM_COLS] = df_lstm.groupby("PATNO")[LSTM_COLS].ffill().fillna(train_medians_lstm)
    df_lstm[LSTM_COLS] = scaler_lstm.transform(df_lstm[LSTM_COLS])

    pats_order = test_tabular["PATNO"].values
    pat_to_all = df_all.groupby("PATNO")
    pat_to_lstm = df_lstm.groupby("PATNO")
    pat_to_raw = test_longitudinal.sort_values(["PATNO", "visit_date"]).groupby("PATNO")

    sequences_all = []
    sequences_lstm = []
    raw_histories = []
    cur_scores = []

    for p in pats_order:
        sequences_all.append(pat_to_all.get_group(p)[ALL_FEATURE_COLS].values)
        sequences_lstm.append(pat_to_lstm.get_group(p)[LSTM_COLS].values)
        raw_df = pat_to_raw.get_group(p)
        raw_histories.append({
            "dates": raw_df["visit_date"].values,
            "np3tot": raw_df["NP3TOT"].values,
            "months": raw_df["elapsed_months_bl"].values if "elapsed_months_bl" in raw_df.columns else np.arange(len(raw_df))*6
        })
        cur_scores.append(raw_df["NP3TOT"].iloc[-1])

    lengths = torch.tensor([len(s) for s in sequences_all], dtype=torch.long)
    max_len = max(lengths).item()
    batch_size = len(sequences_all)

    padded_x_all = torch.zeros(batch_size, max_len, len(ALL_FEATURE_COLS), dtype=torch.float32)
    padded_x_lstm = torch.zeros(batch_size, max_len, len(LSTM_COLS), dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i in range(batch_size):
        l = lengths[i].item()
        padded_x_all[i, :l] = torch.tensor(sequences_all[i], dtype=torch.float32)
        padded_x_lstm[i, :l] = torch.tensor(sequences_lstm[i], dtype=torch.float32)
        mask[i, :l] = True

    cur_scores_t = torch.tensor(cur_scores, dtype=torch.float32)

    # Load Late Fusion Model
    late_model = LateFusionModel(clin_dim=len(CLINICAL_COLS), img_dim=len(IMAGING_COLS), demo_dim=len(DEMO_COLS), hidden_dim=48, dropout=0.25)
    late_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "late_fusion.pt"), map_location="cpu", weights_only=False))
    late_model.eval()

    # Load BiLSTM Model
    bilstm_model = LongitudinalBiLSTM(input_dim=len(LSTM_COLS), hidden_dim=64, num_layers=2, dropout=0.25)
    ckpt2 = torch.load(os.path.join(MODELS_DIR, "model2_bilstm.pt"), map_location="cpu", weights_only=False)
    if isinstance(ckpt2, dict) and "model_state_dict" in ckpt2:
        bilstm_model.load_state_dict(ckpt2["model_state_dict"])
    else:
        bilstm_model.load_state_dict(ckpt2)
    bilstm_model.eval()

    return {
        "test_tabular": test_tabular,
        "raw_histories": raw_histories,
        "lengths": lengths,
        "mask": mask,
        "padded_x_all": padded_x_all,
        "padded_x_lstm": padded_x_lstm,
        "cur_scores_t": cur_scores_t,
        "late_model": late_model,
        "bilstm_model": bilstm_model
    }


def extract_weights(data):
    print("[2/3] Extracting attention weights and multimodal gating decisions...")
    late_model = data["late_model"]
    bilstm_model = data["bilstm_model"]
    padded_x_all = data["padded_x_all"]
    padded_x_lstm = data["padded_x_lstm"]
    mask = data["mask"]
    lengths = data["lengths"]
    cur_scores_t = data["cur_scores_t"]
    test_tabular = data["test_tabular"].copy()

    with torch.no_grad():
        # Late Fusion Gating
        p12_lf, p24_lf, logits_lf, gates = late_model(padded_x_all, mask, lengths, cur_scores_t)
        # BiLSTM Attention
        p12_bi, p24_bi, logits_bi, attn_weights_bi = bilstm_model(padded_x_lstm, mask, lengths, cur_scores_t)

    w_clin = gates[:, 0].numpy()
    w_img = gates[:, 1].numpy()
    attn_bi = attn_weights_bi.numpy()

    test_tabular["w_clin"] = w_clin
    test_tabular["w_img"] = w_img
    test_tabular["pred12_lf"] = p12_lf.numpy()
    test_tabular["pred24_lf"] = p24_lf.numpy()

    return test_tabular, attn_bi, lengths.numpy()


def generate_explainability_figure(test_df, attn_bi, lengths, raw_histories, out_path):
    print("[3/3] Generating publication-grade explainability figure...")
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    # -------------------------------------------------------------
    # Panel A: Gating Distribution Across Test Cohort (Mixture-of-Experts)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    w_c = test_df["w_clin"].values * 100
    w_i = test_df["w_img"].values * 100

    sns.kdeplot(w_c, ax=ax1, color="#2563eb", fill=True, alpha=0.4, linewidth=2.5, label="Clinical Trajectory Weight (w_clin)")
    sns.kdeplot(w_i, ax=ax1, color="#10b981", fill=True, alpha=0.4, linewidth=2.5, label="Neuroimaging Scans Weight (w_img)")

    mean_c = np.mean(w_c)
    mean_i = np.mean(w_i)
    ax1.axvline(mean_c, color="#1e40af", linestyle="--", linewidth=1.8, label=f"Mean Clinical: {mean_c:.1f}%")
    ax1.axvline(mean_i, color="#065f46", linestyle="--", linewidth=1.8, label=f"Mean Imaging: {mean_i:.1f}%")

    ax1.set_title("A. Learned Multimodal Gating Distribution (N=583 Patients)\nAdaptive Decision-Level Mixture-of-Experts",
                  fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Learned Modality Contribution (%)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Kernel Density", fontsize=10, fontweight="bold")
    ax1.set_xlim(0, 100)
    ax1.legend(loc="upper center", frameon=True, fontsize=8.5)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel B: Gating Dynamics vs. Striatal Dopamine Depletion (DaTSCAN)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    # Stratify by DaTSCAN Putamen SBR quartiles
    valid_dat = test_df.dropna(subset=["datscan_putamen_sbr"]).copy()
    valid_dat["dat_quartile"] = pd.qcut(valid_dat["datscan_putamen_sbr"], q=4, labels=["Q1: Severe Loss", "Q2: Moderate", "Q3: Mild", "Q4: Preserved"])

    palette = ["#dc2626", "#ea580c", "#3b82f6", "#10b981"]
    sns.boxplot(data=valid_dat, x="dat_quartile", y="w_img", ax=ax2, palette=palette, width=0.45, showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6})

    ax2.set_title("B. Neuroimaging Gating Weight vs. Dopamine Transporter Loss\nDoes the Model Prioritize Scans When Denervation is Severe?",
                  fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("DaTSCAN Putamen Striatal Binding Ratio (SBR)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Neuroimaging Weight (w_img)", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5, axis='y')

    # -------------------------------------------------------------
    # Panel C: Temporal Attention Profiles: Rapid Progressors vs Stable
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    # Patients with >= 4 visits
    multi_visit_idx = np.where(lengths >= 4)[0]
    rapid_idx = [i for i in multi_visit_idx if test_df["prog_mcid_12m"].iloc[i] == 1.0]
    stable_idx = [i for i in multi_visit_idx if test_df["prog_mcid_12m"].iloc[i] == 0.0]

    # Align attention weights to last 4 visits: [t-3, t-2, t-1, t_latest]
    rapid_attn = []
    for i in rapid_idx:
        L = lengths[i]
        rapid_attn.append(attn_bi[i, L-4:L])
    rapid_attn = np.mean(rapid_attn, axis=0) * 100

    stable_attn = []
    for i in stable_idx:
        L = lengths[i]
        stable_attn.append(attn_bi[i, L-4:L])
    stable_attn = np.mean(stable_attn, axis=0) * 100

    visit_labels = ["Visit (t - 3)", "Visit (t - 2)", "Visit (t - 1)", "Latest Visit (t)"]
    x = np.arange(4)
    width = 0.35

    r1 = ax3.bar(x - width/2, rapid_attn, width, label="Rapid Progressors (MCID >= 3.5)", color="#ef4444", edgecolor="#7f1d1d")
    r2 = ax3.bar(x + width/2, stable_attn, width, label="Stable Patients (< 3.5 pts)", color="#3b82f6", edgecolor="#1e3a8a")

    ax3.set_title("C. Temporal Attention Allocation (Last 4 Historical Visits)\nRapid Progressors vs. Clinically Stable Cohort",
                  fontsize=11, fontweight="bold", pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(visit_labels, fontsize=9.5, fontweight="bold")
    ax3.set_ylabel("Attention Weight (%)", fontsize=10, fontweight="bold")
    ax3.legend(loc="upper left", frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.5, axis='y')

    for rect in list(r1) + list(r2):
        h = rect.get_height()
        ax3.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # -------------------------------------------------------------
    # Panel D: Real Patient Trajectory Case Study with Attention
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])

    # Find a patient with clear progression and >= 5 visits
    candidate_pats = [i for i in range(len(test_df)) if lengths[i] >= 5 and test_df["prog_mcid_12m"].iloc[i] == 1.0]
    case_idx = candidate_pats[0] if candidate_pats else 0
    case_patno = test_df["PATNO"].iloc[case_idx]
    hist = raw_histories[case_idx]
    L = lengths[case_idx]
    attn_case = attn_bi[case_idx, :L] * 100

    months = hist["months"]
    scores = hist["np3tot"]
    pred12 = test_df["pred12_lf"].iloc[case_idx]
    pred24 = test_df["pred24_lf"].iloc[case_idx]
    true12 = test_df["target_np3tot_12m"].iloc[case_idx]
    true24 = test_df["target_np3tot_24m"].iloc[case_idx]

    # Plot historical trajectory with scatter colored by attention weight
    line = ax4.plot(months, scores, color="#0f172a", linestyle="-", linewidth=2, label="Observed Clinical Trajectory (MDS-UPDRS III)")
    sc = ax4.scatter(months, scores, c=attn_case, cmap="YlOrRd", s=180, edgecolor="#0f172a", linewidth=1.5, zorder=5)
    cbar = plt.colorbar(sc, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label("Temporal Attention (%)", fontsize=9, fontweight="bold")

    # Plot forecast
    last_m = months[-1]
    last_s = scores[-1]
    ax4.plot([last_m, last_m + 12, last_m + 24], [last_s, pred12, pred24], color="#dc2626", linestyle="--", linewidth=2.5, marker="o", label="AI Multimodal Forecast (Late Fusion)")
    if not np.isnan(true12) and not np.isnan(true24):
        ax4.scatter([last_m + 12, last_m + 24], [true12, true24], color="#16a34a", marker="*", s=220, zorder=6, label="Actual Ground Truth Outcomes")

    ax4.set_title(f"D. Clinical Case Study: Patient #{case_patno}\nDynamic Attention Spotlight & 24-Month Prognosis",
                  fontsize=11, fontweight="bold", pad=10)
    ax4.set_xlabel("Elapsed Months from Baseline Visit", fontsize=10, fontweight="bold")
    ax4.set_ylabel("MDS-UPDRS Part III Score", fontsize=10, fontweight="bold")
    ax4.legend(loc="upper left", frameon=True, fontsize=8)
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Phase 6 Explainable AI & Biological Attribution Analysis\nUnraveling Multimodal Gating Decisions and Temporal Sequence Attention",
                 fontsize=14, fontweight="bold", y=0.98, color="#0f172a")

    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved Explainability figure to: {out_path}")


def main():
    print("=" * 80)
    print("PHASE 6: ATTENTION & MULTIMODAL GATING EXPLAINABILITY ANALYSIS")
    print("=" * 80)

    data = load_data_and_models()
    test_df, attn_bi, lengths = extract_weights(data)

    out_fig = os.path.join(REPORT_DIR, "explainability_attention_and_gating.png")
    generate_explainability_figure(test_df, attn_bi, lengths, data["raw_histories"], out_fig)

    # Compute summary metrics
    summary = {
        "mean_clinical_weight_pct": float(np.mean(test_df["w_clin"]) * 100),
        "mean_imaging_weight_pct": float(np.mean(test_df["w_img"]) * 100),
        "std_clinical_weight_pct": float(np.std(test_df["w_clin"]) * 100),
        "std_imaging_weight_pct": float(np.std(test_df["w_img"]) * 100),
        "min_imaging_weight_pct": float(np.min(test_df["w_img"]) * 100),
        "max_imaging_weight_pct": float(np.max(test_df["w_img"]) * 100),
        "n_patients": len(test_df)
    }

    out_json = os.path.join(REPORT_DIR, "explainability_summary_metrics.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved Explainability summary JSON to: {out_json}")

    print("\n  Summary Metrics:")
    print(f"    Clinical Branch Weight (Mean +/- SD): {summary['mean_clinical_weight_pct']:.1f}% +/- {summary['std_clinical_weight_pct']:.1f}%")
    print(f"    Imaging Branch Weight  (Mean +/- SD): {summary['mean_imaging_weight_pct']:.1f}% +/- {summary['std_imaging_weight_pct']:.1f}%")

    print("\n" + "=" * 80)
    print("PHASE 6: EXPLAINABILITY & BIOLOGICAL ATTRIBUTION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
