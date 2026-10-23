"""
train_model2_lstm.py
--------------------
Phase 3, Step 3: Model 2 — Longitudinal Sequence Deep Learning Model
(Bidirectional LSTM with Temporal Attention and Irregular Spacing Awareness)

Addresses Research Question 1 (RQ1 Core Comparison):
"Can disease progression (MDS-UPDRS Part III at +12m and +24m) be predicted more
accurately from longitudinal visit trajectories than from a single static clinic visit snapshot?"

Key Architectural & Methodological Highlights:
1. Longitudinal Sequence Ingestion: Ingests full variable-length visit history (v_0 -> v_1 -> ... -> v_k)
   preceding the index prediction visit, capturing velocity and rate-of-change across time.
2. Irregular Time-Spacing Awareness: Explicitly ingests delta_t_days (inter-visit interval) and
   elapsed_months_bl (cumulative time since baseline), preventing distorted uniform-interval assumptions.
3. Temporal Attention Pooling: Dynamically attends to critical transition visits while retaining
   the latest visit's direct hidden representation.
4. Multi-Task Residual Learning: Predicts future continuous scores (+12m, +24m) and binary rapid
   worsening (MCID >= 3.5 pts) with residual skip-connections anchored to current motor disability.
5. Strict Anti-Leakage Protocol: Imputation and scaling fit exclusively on Train partition;
   evaluation conducted strictly on the canonical 583 held-out test patients via src.evaluate.
6. Saves model checkpoint to models/model2_bilstm.pt and attention trajectory plots to report/.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from evaluate import evaluate_model_run

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
REPORT_DIR = "report"

# Feature set for longitudinal sequence modeling (43 features)
FEATURE_COLS = [
    # Motor Part III subdomains & disability
    "NP3TOT", "np3_tremor", "np3_rigidity", "np3_bradykinesia", "np3_axial",
    "nhy_stage", "dyskinesia_present",
    # Functional / Daily living
    "np1_total", "np2_total", "np4_total", "schwab_england_pct",
    # Medication controls & timing
    "ledd", "pd_state_on", "pd_state_off", "hr_post_med",
    # Non-motor clinical scales
    "moca_total", "scopa_aut_total", "gds_total", "ess_total", "rbd_total", "upsit_total",
    # Fluid biospecimens (CSF & SAA)
    "csf_abeta42", "csf_ttau", "csf_ptau", "csf_asyn",
    "csf_ttau_abeta_ratio", "csf_ptau_abeta_ratio", "saa_positive",
    # Neuroimaging (DaTSCAN SBR & FreeSurfer MRI subcortical volumes)
    "datscan_caudate_sbr", "datscan_putamen_sbr", "datscan_striatum_sbr",
    "datscan_putamen_asym", "datscan_caudate_putamen_ratio",
    "mri_brain_seg_vol", "mri_brain_stem", "mri_putamen_vol",
    "mri_caudate_vol", "mri_hippocampus_vol", "mri_ventricles_vol",
    # Demographics
    "age_at_visit", "is_male", "education_years",
    # Irregular time features
    "delta_t_days", "elapsed_months_bl"
]


class LongitudinalSequenceDataset(Dataset):
    """
    Dataset representing patient visit trajectories.
    Each item:
      - x: [seq_len, feature_dim]
      - cur_np3tot: float (motor score at index prediction visit)
      - target_12m: float (NP3TOT at +12m, or NaN)
      - target_24m: float (NP3TOT at +24m, or NaN)
      - target_mcid: float (1.0 if worsening >= 3.5, 0.0 otherwise, or NaN)
      - patno: int
    """
    def __init__(self, sequences, cur_scores, targets_12m, targets_24m, targets_mcid, patnos):
        self.sequences = sequences
        self.cur_scores = cur_scores
        self.targets_12m = targets_12m
        self.targets_24m = targets_24m
        self.targets_mcid = targets_mcid
        self.patnos = patnos

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.cur_scores[idx], dtype=torch.float32),
            torch.tensor(self.targets_12m[idx], dtype=torch.float32),
            torch.tensor(self.targets_24m[idx], dtype=torch.float32),
            torch.tensor(self.targets_mcid[idx], dtype=torch.float32),
            self.patnos[idx]
        )


def collate_sequences(batch):
    """
    Dynamic padding collate function for variable-length visit sequences.
    Returns:
      padded_x: [batch_size, max_seq_len, feature_dim]
      mask: [batch_size, max_seq_len] (True for valid visits, False for padded)
      lengths: [batch_size]
      cur_scores: [batch_size]
      targets_12m: [batch_size]
      targets_24m: [batch_size]
      targets_mcid: [batch_size]
      patnos: list of int
    """
    sequences, cur_scores, t12, t24, tmcid, patnos = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    max_len = max(lengths).item()
    batch_size = len(sequences)
    feat_dim = sequences[0].shape[-1]

    padded_x = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        padded_x[i, :seq_len] = seq
        mask[i, :seq_len] = True

    return (
        padded_x,
        mask,
        lengths,
        torch.stack(cur_scores),
        torch.stack(t12),
        torch.stack(t24),
        torch.stack(tmcid),
        list(patnos)
    )


class TemporalAttention(nn.Module):
    """
    Additive Temporal Attention to learn importance weights over historical visits.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h, mask):
        # h: [B, T, hidden_dim]
        # mask: [B, T]
        u = torch.tanh(self.proj(h))               # [B, T, hidden_dim]
        scores = self.v(u).squeeze(-1)             # [B, T]
        scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=-1)        # [B, T]
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # [B, hidden_dim]
        return context, weights


class LongitudinalBiLSTM(nn.Module):
    """
    Bidirectional LSTM with Temporal Attention and Multi-Task Residual Heads.
    """
    def __init__(self, input_dim=43, hidden_dim=64, num_layers=2, dropout=0.25):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input feature projection and normalization
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Bidirectional LSTM backbone
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Temporal attention over bidirectional states
        lstm_out_dim = hidden_dim * 2
        self.attention = TemporalAttention(lstm_out_dim)

        # Representation fusion: Context vector + Latest visit state
        fusion_dim = lstm_out_dim * 2  # 256
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_dim, lstm_out_dim),
            nn.LayerNorm(lstm_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Multi-task heads:
        # Residual delta heads: predict Delta = Target - Current_Score
        self.head_delta_12m = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.head_delta_24m = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # Binary classification head: logit for worsening >= 3.5 points
        self.head_mcid = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, mask, lengths, cur_scores):
        """
        x: [B, T, D]
        mask: [B, T]
        lengths: [B]
        cur_scores: [B]
        """
        B, T, _ = x.shape
        proj = self.input_proj(x)  # [B, T, H]

        packed = nn.utils.rnn.pack_padded_sequence(
            proj, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=T
        )  # [B, T, 2H]

        # 1. Temporal Attention Context
        context, attn_weights = self.attention(lstm_out, mask)  # [B, 2H], [B, T]

        # 2. Latest Visit State (index representation)
        idx_last = (lengths - 1).view(-1, 1, 1).expand(-1, 1, lstm_out.size(2))
        h_last = lstm_out.gather(1, idx_last).squeeze(1)       # [B, 2H]

        # 3. Fuse attention trajectory with current index state
        fused = torch.cat([context, h_last], dim=-1)           # [B, 4H]
        rep = self.fusion_fc(fused)                            # [B, 2H]

        # 4. Multi-task Predictions (Residual anchored to cur_scores)
        delta_12m = self.head_delta_12m(rep).squeeze(-1)       # [B]
        delta_24m = self.head_delta_24m(rep).squeeze(-1)       # [B]
        logits_mcid = self.head_mcid(rep).squeeze(-1)          # [B]

        pred_12m = cur_scores + delta_12m
        pred_24m = cur_scores + delta_24m

        return pred_12m, pred_24m, logits_mcid, attn_weights


def build_trajectories(longitudinal_df, feature_cols, scaler, train_medians):
    """
    Transforms a longitudinal DataFrame into a standardized array matrix per patient.
    Applies within-patient forward-fill and training-median imputation.
    """
    df = longitudinal_df.copy()
    df = df.sort_values(["PATNO", "visit_date"]).reset_index(drop=True)

    # Within-patient forward fill (LOCF)
    df[feature_cols] = df.groupby("PATNO")[feature_cols].ffill()

    # Impute remaining initial missing values strictly with Train medians
    df[feature_cols] = df[feature_cols].fillna(train_medians)

    # Normalize using fitted scaler
    scaled_feats = scaler.transform(df[feature_cols])
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaled_feats

    return df_scaled


def extract_labeled_sequences(df_scaled, feature_cols, mode="all_labeled", latest_pats_df=None):
    """
    Extracts trajectory sequences for training, validation, or testing.
    - If mode == 'all_labeled': extracts all visit prefixes ending at any visit with valid target_np3tot_12m.
    - If mode == 'latest_only': extracts prefix up to the specific latest visit for patients in latest_pats_df.
    """
    sequences = []
    cur_scores = []
    targets_12m = []
    targets_24m = []
    targets_mcid = []
    patnos = []

    grouped = {pat: group.reset_index(drop=True) for pat, group in df_scaled.groupby("PATNO")}

    if mode == "all_labeled":
        for patno, group in grouped.items():
            feat_mat = group[feature_cols].values
            np3_raw = group["NP3TOT"].values
            t12_arr = group["target_np3tot_12m"].values
            t24_arr = group["target_np3tot_24m"].values
            mcid_arr = group["prog_mcid_12m"].values

            for i in range(len(group)):
                if not np.isnan(t12_arr[i]):
                    sequences.append(feat_mat[:i + 1])
                    cur_scores.append(np3_raw[i])
                    targets_12m.append(t12_arr[i])
                    targets_24m.append(t24_arr[i])
                    targets_mcid.append(mcid_arr[i])
                    patnos.append(patno)

    elif mode == "latest_only":
        assert latest_pats_df is not None
        # Map each patient to their target latest visit_date
        target_dates = dict(zip(latest_pats_df["PATNO"], latest_pats_df["visit_date"]))

        for _, row in latest_pats_df.iterrows():
            patno = row["PATNO"]
            vdate = row["visit_date"]
            if patno in grouped:
                group = grouped[patno]
                # Find all visits up to and including vdate
                sub = group[group["visit_date"] <= vdate]
                if len(sub) > 0:
                    idx = len(sub) - 1
                    sequences.append(sub[feature_cols].values)
                    cur_scores.append(sub["NP3TOT"].values[idx])
                    targets_12m.append(row["target_np3tot_12m"])
                    targets_24m.append(row.get("target_np3tot_24m", np.nan))
                    targets_mcid.append(row.get("prog_mcid_12m", np.nan))
                    patnos.append(patno)

    return sequences, cur_scores, targets_12m, targets_24m, targets_mcid, patnos


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_count = 0

    criterion_reg = nn.SmoothL1Loss(reduction="none")
    criterion_cls = nn.BCEWithLogitsLoss(reduction="none")

    for x, mask, lengths, cur_scores, t12, t24, tmcid, _ in loader:
        x = x.to(device)
        mask = mask.to(device)
        lengths = lengths.to(device)
        cur_scores = cur_scores.to(device)
        t12 = t12.to(device)
        t24 = t24.to(device)
        tmcid = tmcid.to(device)

        optimizer.zero_grad()
        pred_12, pred_24, logits_mcid, _ = model(x, mask, lengths, cur_scores)

        # Loss 1: 12-month regression (all samples in train have valid t12)
        loss_12 = criterion_reg(pred_12, t12).mean()

        # Loss 2: 24-month regression (masked for non-null)
        valid_24 = ~torch.isnan(t24)
        if valid_24.sum() > 0:
            loss_24 = criterion_reg(pred_24[valid_24], t24[valid_24]).mean()
        else:
            loss_24 = torch.tensor(0.0, device=device)

        # Loss 3: MCID classification (masked for non-null)
        valid_mcid = ~torch.isnan(tmcid)
        if valid_mcid.sum() > 0:
            loss_mcid = criterion_cls(logits_mcid[valid_mcid], tmcid[valid_mcid]).mean()
        else:
            loss_mcid = torch.tensor(0.0, device=device)

        loss = loss_12 + 0.6 * loss_24 + 0.4 * loss_mcid
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item() * len(x)
        total_count += len(x)

    return total_loss / max(1, total_count)


def evaluate_dataset(model, loader, device):
    model.eval()
    preds_12m, preds_24m, probs_mcid = [], [], []
    trues_12m, trues_24m, trues_mcid = [], [], []
    cur_list = []
    attn_list = []
    patno_list = []

    with torch.no_grad():
        for x, mask, lengths, cur_scores, t12, t24, tmcid, patnos in loader:
            x = x.to(device)
            mask = mask.to(device)
            lengths = lengths.to(device)
            cur_scores = cur_scores.to(device)

            pred_12, pred_24, logits_mcid, attn = model(x, mask, lengths, cur_scores)
            prob_mcid = torch.sigmoid(logits_mcid)

            preds_12m.extend(pred_12.cpu().numpy().tolist())
            preds_24m.extend(pred_24.cpu().numpy().tolist())
            probs_mcid.extend(prob_mcid.cpu().numpy().tolist())

            trues_12m.extend(t12.numpy().tolist())
            trues_24m.extend(t24.numpy().tolist())
            trues_mcid.extend(tmcid.numpy().tolist())
            cur_list.extend(cur_scores.cpu().numpy().tolist())
            patno_list.extend(patnos)

            for i in range(len(lengths)):
                seq_l = lengths[i].item()
                attn_list.append(attn[i, :seq_l].cpu().numpy().tolist())

    return {
        "pred_12m": np.array(preds_12m),
        "pred_24m": np.array(preds_24m),
        "prob_mcid": np.array(probs_mcid),
        "true_12m": np.array(trues_12m),
        "true_24m": np.array(trues_24m),
        "true_mcid": np.array(trues_mcid),
        "cur_np3tot": np.array(cur_list),
        "attn_weights": attn_list,
        "patnos": patno_list
    }


def plot_attention_trajectories(test_results, df_longitudinal):
    """
    Visualizes attention trajectories for representative test patients.
    """
    os.makedirs(REPORT_DIR, exist_ok=True)
    attn_weights = test_results["attn_weights"]
    patnos = test_results["patnos"]
    trues = test_results["true_12m"]
    preds = test_results["pred_12m"]
    curs = test_results["cur_np3tot"]

    # Select representative patients with multiple visits: rapid progressor, moderate, stable
    deltas = trues - curs
    candidates = []
    for i, p in enumerate(patnos):
        if len(attn_weights[i]) >= 5:  # at least 5 visits
            candidates.append((i, p, deltas[i], len(attn_weights[i])))

    if not candidates:
        return

    # Sort by progression delta
    candidates.sort(key=lambda c: c[2])
    # Pick stable (near 0 delta), moderate, and rapid progressor
    pick_indices = [
        candidates[len(candidates) // 6][0],  # stable
        candidates[len(candidates) // 2][0],  # moderate
        candidates[-2][0]                     # rapid progressor
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#ffffff")

    labels = ["Stable / Slow Progression", "Moderate Progression", "Rapid Disease Worsening"]

    for ax_idx, idx in enumerate(pick_indices):
        ax = axes[ax_idx]
        weights = attn_weights[idx]
        p_id = patnos[idx]
        actual_delta = deltas[idx]
        pred_delta = preds[idx] - curs[idx]
        n_visits = len(weights)

        visits_x = [f"V{v + 1}" for v in range(n_visits)]
        bars = ax.bar(visits_x, weights, color="#1f77b4", alpha=0.8, edgecolor="#0d47a1")
        bars[-1].set_color("#d62728")  # Highlight index prediction visit

        ax.set_title(
            f"{labels[ax_idx]} (PATNO {p_id})\n"
            f"True +12m: {trues[idx]:.1f} (Δ={actual_delta:+.1f}) | Pred: {preds[idx]:.1f} (Δ={pred_delta:+.1f})",
            fontsize=11, fontweight="bold"
        )
        ax.set_xlabel("Historical Clinical Visits", fontsize=10)
        ax.set_ylabel("Temporal Attention Weight α_t", fontsize=10)
        ax.set_ylim(0, max(0.4, max(weights) * 1.25))
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.suptitle("Model 2 (BiLSTM): Learned Temporal Attention Weights Across Patient Trajectories",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plot_path = os.path.join(REPORT_DIR, "model2_lstm_attention_trajectories.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved attention trajectory visualization to {plot_path}")


def plot_head_to_head_comparison():
    """
    Plots direct head-to-head comparison between Model 1 (Static XGBoost) and Model 2 (Sequence BiLSTM).
    """
    bench_path = os.path.join(REPORT_DIR, "benchmark_results.csv")
    if not os.path.exists(bench_path):
        return

    df_bench = pd.read_csv(bench_path)
    # Filter for 12m continuous regression
    reg_12m = df_bench[df_bench["target_horizon"].str.contains(r"\+12 Months \(NP3TOT\)", na=False, regex=True)]

    if len(reg_12m) < 2:
        return

    models = reg_12m["model_name"].tolist()
    maes = reg_12m["mae"].tolist()
    rmses = reg_12m["rmse"].tolist()
    r2s = reg_12m["r2"].tolist()
    rs = reg_12m["pearson_r"].tolist()

    x = np.arange(len(models))
    width = 0.2

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#ffffff")

    rects1 = ax1.bar(x - width * 1.5, maes, width, label="MAE (Points)", color="#1976D2")
    rects2 = ax1.bar(x - width * 0.5, rmses, width, label="RMSE (Points)", color="#E53935")
    rects3 = ax1.bar(x + width * 0.5, [v * 10 for v in r2s], width, label="R² (Variance x10)", color="#388E3C")
    rects4 = ax1.bar(x + width * 1.5, [v * 10 for v in rs], width, label="Pearson r (x10)", color="#F57C00")

    ax1.set_ylabel("Error Score (Lower is Better) / Correlation Scale", fontsize=11, fontweight="bold")
    ax1.set_title("Head-to-Head Benchmark on Held-Out Test Cohort (583 Patients)\nModel 1 (Static Snapshot XGBoost) vs Model 2 (Longitudinal Sequence BiLSTM)",
                  fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    def autolabel(rects, is_scaled=False):
        for rect in rects:
            height = rect.get_height()
            val = height / 10.0 if is_scaled else height
            ax1.annotate(f"{val:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3, is_scaled=True)
    autolabel(rects4, is_scaled=True)

    plt.tight_layout()
    plot_path = os.path.join(REPORT_DIR, "model1_vs_model2_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved benchmark comparison plot to {plot_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_only", action="store_true", help="Skip training and evaluate saved checkpoint")
    args = parser.parse_args()

    print("=" * 80)
    print("PHASE 3, STEP 3: MODEL 2 — LONGITUDINAL SEQUENCE BiLSTM TRAINING")
    print("=" * 80)

    device = torch.device("cpu")
    print(f"Using compute device: {device}")

    # 1. Load Datasets
    print("\n[1/6] Loading Longitudinal Datasets & Split Information...")
    train_long = pd.read_parquet(os.path.join(PROCESSED_DIR, "train_longitudinal.parquet"))
    val_long = pd.read_parquet(os.path.join(PROCESSED_DIR, "val_longitudinal.parquet"))
    test_long = pd.read_parquet(os.path.join(PROCESSED_DIR, "test_longitudinal.parquet"))

    val_tab = pd.read_parquet(os.path.join(PROCESSED_DIR, "val_tabular_latest.parquet"))
    test_tab = pd.read_parquet(os.path.join(PROCESSED_DIR, "test_tabular_latest.parquet"))

    print(f"  Train Longitudinal: {len(train_long):,} rows across {train_long['PATNO'].nunique():,} patients")
    print(f"  Val Longitudinal:   {len(val_long):,} rows across {val_long['PATNO'].nunique():,} patients")
    print(f"  Test Longitudinal:  {len(test_long):,} rows across {test_long['PATNO'].nunique():,} patients")
    print(f"  Val Labeled Latest: {len(val_tab)} patients")
    print(f"  Test Held-Out Set:  {len(test_tab)} patients")

    # 2. Strict Preprocessing & Anti-Leakage Feature Scaling
    print("\n[2/6] Fitting Feature Scaler & Imputers Strictly on Training Partition...")
    # Calculate training medians
    train_medians = train_long[FEATURE_COLS].median().fillna(0.0)

    # Temporary dataframe for fitting scaler
    train_fit_df = train_long.sort_values(["PATNO", "visit_date"]).reset_index(drop=True)
    train_fit_df[FEATURE_COLS] = train_fit_df.groupby("PATNO")[FEATURE_COLS].ffill().fillna(train_medians)

    scaler = StandardScaler()
    scaler.fit(train_fit_df[FEATURE_COLS])
    print(f"  Fitted StandardScaler on {len(train_fit_df):,} training visit observations across {len(FEATURE_COLS)} features.")

    # Build standardized trajectories
    val_scaled = build_trajectories(val_long, FEATURE_COLS, scaler, train_medians)
    test_scaled = build_trajectories(test_long, FEATURE_COLS, scaler, train_medians)

    # 3. Extract Trajectory Sequences
    print("\n[3/6] Assembling Sequence Trajectories...")
    batch_size = 64
    if not args.eval_only:
        train_scaled = build_trajectories(train_long, FEATURE_COLS, scaler, train_medians)
        tr_seqs, tr_curs, tr_t12, tr_t24, tr_tmcid, tr_pats = extract_labeled_sequences(
            train_scaled, FEATURE_COLS, mode="all_labeled"
        )
        print(f"  Train: Built {len(tr_seqs):,} sequences (Max length: {max(len(s) for s in tr_seqs)} visits)")
        train_ds = LongitudinalSequenceDataset(tr_seqs, tr_curs, tr_t12, tr_t24, tr_tmcid, tr_pats)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_sequences)
    else:
        train_loader = None

    # Val sequences: up to latest visit
    val_seqs, val_curs, val_t12, val_t24, val_tmcid, val_pats = extract_labeled_sequences(
        val_scaled, FEATURE_COLS, mode="latest_only", latest_pats_df=val_tab
    )
    print(f"  Val:   Built {len(val_seqs):,} sequences (Max length: {max(len(s) for s in val_seqs)} visits)")

    # Test sequences: up to latest visit for the canonical 583 test patients
    te_seqs, te_curs, te_t12, te_t24, te_tmcid, te_pats = extract_labeled_sequences(
        test_scaled, FEATURE_COLS, mode="latest_only", latest_pats_df=test_tab
    )
    print(f"  Test:  Built {len(te_seqs):,} sequences (Max length: {max(len(s) for s in te_seqs)} visits)")

    val_ds = LongitudinalSequenceDataset(val_seqs, val_curs, val_t12, val_t24, val_tmcid, val_pats)
    test_ds = LongitudinalSequenceDataset(te_seqs, te_curs, te_t12, te_t24, te_tmcid, te_pats)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences)

    # 4. Instantiate Model & Training Pipeline
    print("\n[4/6] Initializing Longitudinal BiLSTM with Temporal Attention...")
    model = LongitudinalBiLSTM(
        input_dim=len(FEATURE_COLS),
        hidden_dim=64,
        num_layers=2,
        dropout=0.20
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    epochs = 45
    patience = 12
    best_val_mae = float("inf")
    best_epoch = 0
    patience_counter = 0

    os.makedirs(MODELS_DIR, exist_ok=True)
    best_model_path = os.path.join(MODELS_DIR, "model2_bilstm.pt")

    if not args.eval_only:
        print("\n[5/6] Training Longitudinal BiLSTM (Multi-Task Optimization)...")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            ep_start = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_res = evaluate_dataset(model, val_loader, device)

            val_mae_12 = np.mean(np.abs(val_res["pred_12m"] - val_res["true_12m"]))
            val_rmse_12 = np.sqrt(np.mean((val_res["pred_12m"] - val_res["true_12m"]) ** 2))
            val_r_12 = np.corrcoef(val_res["pred_12m"], val_res["true_12m"])[0, 1]

            scheduler.step(val_mae_12)
            ep_duration = time.time() - ep_start

            if val_mae_12 < best_val_mae:
                best_val_mae = val_mae_12
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mae_12": val_mae_12,
                    "feature_cols": FEATURE_COLS,
                    "train_medians": train_medians.to_dict(),
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist()
                }, best_model_path)
                mark = " [BEST SAVED]"
            else:
                patience_counter += 1
                mark = ""

            if epoch % 5 == 0 or epoch == 1 or mark:
                print(f"  Epoch {epoch:2d}/{epochs:2d} [{ep_duration:4.1f}s] | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val 12m MAE: {val_mae_12:.3f} | RMSE: {val_rmse_12:.3f} | Pearson r: {val_r_12:.3f}{mark}")

            if patience_counter >= patience:
                print(f"\n  Early stopping triggered at Epoch {epoch} (Best Epoch: {best_epoch} with Val MAE {best_val_mae:.3f})")
                break

        print(f"  Training completed in {time.time() - start_time:.1f}s.")
    else:
        print("\n[5/6] Skipping training (--eval_only specified). Loading checkpoint directly.")

    # 5. Evaluate Best Model on Held-Out Test Set
    print("\n[6/6] Rigorous Evaluation on Canonical Held-Out Test Cohort (583 Patients)...")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded best checkpoint from Epoch {checkpoint['epoch']} (Val MAE: {checkpoint['val_mae_12']:.3f})")

    test_res = evaluate_dataset(model, test_loader, device)

    # 5A: Evaluate +12 Month Continuous Progression
    print("\n--- Model 2: Longitudinal BiLSTM (+12-Month Motor Progression) ---")
    metrics_12m = evaluate_model_run(
        model_name="Model 2: Longitudinal BiLSTM",
        target_horizon="+12 Months (NP3TOT)",
        y_true_reg=test_res["true_12m"],
        y_pred_reg=test_res["pred_12m"],
        metadata_df=test_tab,
        modality_tags="Multimodal Sequences (Trajectory History + Delta_t)",
        notes="BiLSTM with Temporal Attention trained on full visit histories."
    )

    # 5B: Evaluate +24 Month Continuous Progression (on subset with valid 24m targets)
    valid_idx_24 = ~np.isnan(test_res["true_24m"])
    print(f"\n--- Model 2: Longitudinal BiLSTM (+24-Month Motor Progression, N={valid_idx_24.sum()}) ---")
    metrics_24m = evaluate_model_run(
        model_name="Model 2: Longitudinal BiLSTM",
        target_horizon="+24 Months (NP3TOT)",
        y_true_reg=test_res["true_24m"][valid_idx_24],
        y_pred_reg=test_res["pred_24m"][valid_idx_24],
        metadata_df=test_tab.iloc[valid_idx_24].reset_index(drop=True),
        modality_tags="Multimodal Sequences (Trajectory History + Delta_t)",
        notes="BiLSTM with Temporal Attention on 24-month horizon."
    )

    # 5C: Evaluate Binary Clinically Meaningful Worsening (MCID >= 3.5 pts)
    valid_idx_mcid = ~np.isnan(test_res["true_mcid"])
    print(f"\n--- Model 2: Longitudinal BiLSTM Classifier (MCID >= 3.5 Worsening, N={valid_idx_mcid.sum()}) ---")
    metrics_mcid = evaluate_model_run(
        model_name="Model 2: Longitudinal BiLSTM Classifier",
        target_horizon="+12m Worsening (MCID >= 3.5)",
        y_true_reg=test_res["true_mcid"][valid_idx_mcid].astype(float),
        y_pred_reg=test_res["prob_mcid"][valid_idx_mcid],
        y_true_cls=test_res["true_mcid"][valid_idx_mcid],
        y_prob_cls=test_res["prob_mcid"][valid_idx_mcid],
        metadata_df=test_tab.iloc[valid_idx_mcid].reset_index(drop=True),
        modality_tags="Multimodal Sequences (Trajectory History + Delta_t)",
        notes="BiLSTM classification of rapid motor worsening."
    )

    # 6. Generate Explainability and Head-to-Head Plots
    print("\nGenerating Attention Trajectory & Benchmark Plots...")
    plot_attention_trajectories(test_res, test_long)
    plot_head_to_head_comparison()

    print("\n" + "=" * 80)
    print("PHASE 3, STEP 3 (MODEL 2) COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
