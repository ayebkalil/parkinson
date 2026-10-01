"""
train_fusion_models.py
----------------------
Phase 4 & 5: Multimodal Deep Fusion Architectures

Directly addresses Research Question 2 (RQ2):
"Among early (feature-level), late (decision-level), and attention-based (learned)
fusion of clinical, biomarker, and imaging-derived modalities, which strategy gives
the best — and most stable across prediction horizons — performance?"

Implements & Benchmarks Head-to-Head on Held-Out Test Patients (N=583):
1. Strategy A: Early Fusion (Joint Feature Vector -> BiLSTM + Temporal Attention)
2. Strategy B: Late Fusion (Modality-Specific Encoders + Adaptive Gating / Mixture of Experts)
3. Strategy C: Cross-Attention Multimodal Transformer (Clinical queries attend to Neuroimaging tokens)

Unified Evaluation:
- All models evaluated strictly on the 583 held-out test patients using src.evaluate.evaluate_model_run.
- Results logged to report/benchmark_results.csv and report/benchmark_results.json.
- Generates publication figure: report/fusion_strategies_comparison.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from evaluate import evaluate_model_run

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
REPORT_DIR = "report"

# Modality Feature Definitions
CLINICAL_COLS = [
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
    # Irregular time features
    "delta_t_days", "elapsed_months_bl"
]

IMAGING_COLS = [
    # DaTSCAN SPECT Striatal Binding Ratios
    "datscan_caudate_sbr", "datscan_putamen_sbr", "datscan_striatum_sbr",
    "datscan_putamen_asym", "datscan_caudate_putamen_ratio",
    # FreeSurfer 3D Structural MRI subcortical volumes
    "mri_brain_seg_vol", "mri_brain_stem", "mri_putamen_vol",
    "mri_caudate_vol", "mri_hippocampus_vol", "mri_ventricles_vol"
]

DEMO_COLS = [
    "age_at_visit", "is_male", "education_years"
]

ALL_FEATURE_COLS = CLINICAL_COLS + IMAGING_COLS + DEMO_COLS


# =========================================================================
# DATA PREPARATION & SEQUENCE EXTRACTION (ZERO-LEAKAGE)
# =========================================================================

def build_trajectories(longitudinal_df, feature_cols, scaler, train_medians):
    """
    Transforms longitudinal DataFrame into normalized trajectory matrix.
    Applies LOCF forward-fill per patient, followed by train_medians imputation.
    """
    df = longitudinal_df.copy()
    df = df.sort_values(["PATNO", "visit_date"]).reset_index(drop=True)

    # Within-patient forward fill
    df[feature_cols] = df.groupby("PATNO")[feature_cols].ffill()

    # Remaining missing filled with train medians
    df[feature_cols] = df[feature_cols].fillna(train_medians)

    # Scale using pre-fitted scaler
    scaled_feats = scaler.transform(df[feature_cols])
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaled_feats
    return df_scaled


def extract_labeled_sequences(df_scaled, feature_cols, mode="all_labeled", latest_pats_df=None):
    """
    Extracts variable-length sequences.
    mode='all_labeled': all prefixes with valid 12m targets (for training)
    mode='latest_only': sequence ending at the exact latest visit for patients in latest_pats_df (for test)
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
        for _, row in latest_pats_df.iterrows():
            patno = row["PATNO"]
            vdate = row["visit_date"]
            if patno in grouped:
                group = grouped[patno]
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


class SequenceDataset(Dataset):
    def __init__(self, sequences, cur_scores, t12, t24, tmcid, patnos):
        self.sequences = sequences
        self.cur_scores = [torch.tensor(s, dtype=torch.float32) for s in cur_scores]
        self.t12 = [torch.tensor(t, dtype=torch.float32) for t in t12]
        self.t24 = [torch.tensor(t, dtype=torch.float32) for t in t24]
        self.tmcid = [torch.tensor(t, dtype=torch.float32) for t in tmcid]
        self.patnos = patnos

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            self.cur_scores[idx],
            self.t12[idx],
            self.t24[idx],
            self.tmcid[idx],
            self.patnos[idx]
        )


def pad_collate_fn(batch):
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


# =========================================================================
# ATTENTION MODULE
# =========================================================================

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h, mask):
        u = torch.tanh(self.proj(h))
        scores = self.v(u).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)
        return context, weights


# =========================================================================
# 1. STRATEGY A: EARLY FUSION MODEL
# =========================================================================

class EarlyFusionModel(nn.Module):
    def __init__(self, input_dim=44, hidden_dim=64, dropout=0.25):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.attention = TemporalAttention(hidden_dim * 2)

        fused_dim = hidden_dim * 2
        self.head_12m = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.head_24m = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.head_cls = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, mask, lengths, cur_scores):
        proj = self.input_proj(x)
        packed = nn.utils.rnn.pack_padded_sequence(proj, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        context, attn = self.attention(lstm_out, mask)

        delta12 = self.head_12m(context).squeeze(-1)
        delta24 = self.head_24m(context).squeeze(-1)
        logits_mcid = self.head_cls(context).squeeze(-1)

        return cur_scores + delta12, cur_scores + delta24, logits_mcid, attn


# =========================================================================
# 2. STRATEGY B: LATE FUSION MODEL (DECISION GATING / MOE)
# =========================================================================

class LateFusionModel(nn.Module):
    def __init__(self, clin_dim=30, img_dim=11, demo_dim=3, hidden_dim=48, dropout=0.25):
        super().__init__()
        # Branch 1: Clinical Trajectory Encoder
        self.clin_proj = nn.Sequential(
            nn.Linear(clin_dim + demo_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.clin_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.clin_attn = TemporalAttention(hidden_dim * 2)

        # Branch 2: Neuroimaging Encoder (Operating on latest scan)
        self.img_encoder = nn.Sequential(
            nn.Linear(img_dim + demo_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Branch Predictors
        self.clin_head_12 = nn.Linear(hidden_dim * 2, 1)
        self.clin_head_24 = nn.Linear(hidden_dim * 2, 1)
        self.img_head_12 = nn.Linear(hidden_dim, 1)
        self.img_head_24 = nn.Linear(hidden_dim, 1)

        # Learned Gating Network (weights in [0, 1])
        self.gate_net = nn.Sequential(
            nn.Linear((hidden_dim * 2) + hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )

        self.head_cls = nn.Linear((hidden_dim * 2) + hidden_dim, 1)

    def forward(self, x, mask, lengths, cur_scores):
        clin_x = torch.cat([x[:, :, :30], x[:, :, 41:]], dim=-1)  # clin (30) + demo (3)
        img_latest = torch.cat([x[:, -1, 30:41], x[:, -1, 41:]], dim=-1)  # img (11) + demo (3)

        c_proj = self.clin_proj(clin_x)
        packed = nn.utils.rnn.pack_padded_sequence(c_proj, lengths.cpu(), batch_first=True, enforce_sorted=False)
        c_out, _ = self.clin_lstm(packed)
        c_out, _ = nn.utils.rnn.pad_packed_sequence(c_out, batch_first=True)
        h_clin, attn = self.clin_attn(c_out, mask)

        h_img = self.img_encoder(img_latest)

        pred12_clin = cur_scores + self.clin_head_12(h_clin).squeeze(-1)
        pred24_clin = cur_scores + self.clin_head_24(h_clin).squeeze(-1)
        pred12_img = cur_scores + self.img_head_12(h_img).squeeze(-1)
        pred24_img = cur_scores + self.img_head_24(h_img).squeeze(-1)

        joint = torch.cat([h_clin, h_img], dim=-1)
        gates = self.gate_net(joint)
        w_clin = gates[:, 0]
        w_img = gates[:, 1]

        pred12 = w_clin * pred12_clin + w_img * pred12_img
        pred24 = w_clin * pred24_clin + w_img * pred24_img

        logits_mcid = self.head_cls(joint).squeeze(-1)

        return pred12, pred24, logits_mcid, gates


# =========================================================================
# 3. STRATEGY C: CROSS-ATTENTION MULTIMODAL TRANSFORMER
# =========================================================================

class CrossAttentionFusionModel(nn.Module):
    def __init__(self, clin_dim=30, img_dim=11, demo_dim=3, hidden_dim=64, num_heads=4, dropout=0.25):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Clinical Sequence Encoder (Query generator)
        self.clin_proj = nn.Sequential(
            nn.Linear(clin_dim + demo_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.clin_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Neuroimaging Tokenizer: Projects imaging measures into discrete biological tokens (Key & Value)
        self.token_datscan = nn.Linear(5, hidden_dim)
        self.token_mri = nn.Linear(6, hidden_dim)
        self.token_demo = nn.Linear(demo_dim, hidden_dim)

        # Multi-Head Cross-Attention Layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.temporal_attn = TemporalAttention(hidden_dim)

        # Multi-Task Heads
        self.head_12m = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.head_24m = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.head_cls = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, mask, lengths, cur_scores):
        clin_input = torch.cat([x[:, :, :30], x[:, :, 41:]], dim=-1)

        c_proj = self.clin_proj(clin_input)
        packed = nn.utils.rnn.pack_padded_sequence(c_proj, lengths.cpu(), batch_first=True, enforce_sorted=False)
        queries, _ = self.clin_lstm(packed)
        queries, _ = nn.utils.rnn.pad_packed_sequence(queries, batch_first=True)

        # Biological Tokens (Key & Value)
        dat_tok = self.token_datscan(x[:, -1, 30:35]).unsqueeze(1)
        mri_tok = self.token_mri(x[:, -1, 35:41]).unsqueeze(1)
        demo_tok = self.token_demo(x[:, -1, 41:]).unsqueeze(1)
        kv_tokens = torch.cat([dat_tok, mri_tok, demo_tok], dim=1)

        # Cross-Attention
        attn_out, cross_weights = self.cross_attn(
            query=queries,
            key=kv_tokens,
            value=kv_tokens
        )
        enhanced_seq = self.norm_cross(queries + attn_out)

        # Temporal Pooling
        context, temp_weights = self.temporal_attn(enhanced_seq, mask)

        delta12 = self.head_12m(context).squeeze(-1)
        delta24 = self.head_24m(context).squeeze(-1)
        logits_mcid = self.head_cls(context).squeeze(-1)

        return cur_scores + delta12, cur_scores + delta24, logits_mcid, cross_weights


# =========================================================================
# UNIFIED TRAINING & EVALUATION LOOP
# =========================================================================

def train_and_eval_fusion_model(model, model_tag, loader_tr, loader_va, loader_te, test_df, num_epochs=35, lr=1e-3):
    print(f"\n---> Training {model_tag}...")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    criterion_reg = nn.SmoothL1Loss(reduction="none")
    criterion_cls = nn.BCEWithLogitsLoss(reduction="none")

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x, mask, lengths, cur_scores, t12, t24, tmcid, _ in loader_tr:
            optimizer.zero_grad()
            p12, p24, logits, _ = model(x, mask, lengths, cur_scores)

            loss_12 = criterion_reg(p12, t12).mean()

            valid_24 = ~torch.isnan(t24)
            if valid_24.sum() > 0:
                loss_24 = criterion_reg(p24[valid_24], t24[valid_24]).mean()
            else:
                loss_24 = torch.tensor(0.0)

            valid_mcid = ~torch.isnan(tmcid)
            if valid_mcid.sum() > 0:
                loss_mcid = criterion_cls(logits[valid_mcid], tmcid[valid_mcid]).mean()
            else:
                loss_mcid = torch.tensor(0.0)

            batch_loss = loss_12 + 1.2 * loss_24 + 0.5 * loss_mcid
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()

            total_loss += batch_loss.item()
            n_batches += 1

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for vx, vmask, vlengths, vcur_scores, vt12, vt24, vtmcid, _ in loader_va:
                vp12, vp24, vlogits, _ = model(vx, vmask, vlengths, vcur_scores)
                vloss_12 = criterion_reg(vp12, vt12).mean()

                v_24 = ~torch.isnan(vt24)
                vloss_24 = criterion_reg(vp24[v_24], vt24[v_24]).mean() if v_24.sum() > 0 else 0.0

                val_loss += (vloss_12 + 1.2 * vloss_24).item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights = model.state_dict().copy()

    # Load best weights
    model.load_state_dict(best_weights)
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, f"{model_tag.lower().replace(' ', '_')}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"  {model_tag} Training Complete (Best Val Loss: {best_val_loss:.4f}) -> Saved {save_path}")

    # Evaluate strictly on test set (583 patients)
    model.eval()
    all_p12, all_p24, all_probs = [], [], []
    all_t12, all_t24, all_tmcid = [], [], []

    with torch.no_grad():
        for tx, tmask, tlengths, tcur_scores, tt12, tt24, ttmcid, _ in loader_te:
            tp12, tp24, tlogits, _ = model(tx, tmask, tlengths, tcur_scores)
            all_p12.extend(tp12.numpy())
            all_p24.extend(tp24.numpy())
            all_probs.extend(torch.sigmoid(tlogits).numpy())
            all_t12.extend(tt12.numpy())
            all_t24.extend(tt24.numpy())
            all_tmcid.extend(ttmcid.numpy())

    all_p12 = np.array(all_p12)
    all_p24 = np.array(all_p24)
    all_probs = np.array(all_probs)
    all_t12 = np.array(all_t12)
    all_t24 = np.array(all_t24)
    all_tmcid = np.array(all_tmcid)

    # 12M Regression
    m12 = evaluate_model_run(
        model_name=f"Fusion: {model_tag}",
        target_horizon="+12 Months (NP3TOT)",
        y_true_reg=all_t12,
        y_pred_reg=all_p12,
        metadata_df=test_df,
        modality_tags="Multimodal Deep Fusion (Clinical + Biofluid + Neuroimaging + Genetics)",
        notes=f"RQ2 benchmark: {model_tag} evaluated on 583 held-out test patients."
    )

    # 24M Regression
    m24 = evaluate_model_run(
        model_name=f"Fusion: {model_tag}",
        target_horizon="+24 Months (NP3TOT)",
        y_true_reg=all_t24,
        y_pred_reg=all_p24,
        metadata_df=test_df,
        modality_tags="Multimodal Deep Fusion (Clinical + Biofluid + Neuroimaging + Genetics)",
        notes=f"RQ2 benchmark: {model_tag} 24-month horizon."
    )

    # Classification
    m_cls = evaluate_model_run(
        model_name=f"Fusion: {model_tag} Classifier",
        target_horizon="+12m Worsening (MCID >= 3.5)",
        y_true_reg=all_tmcid,
        y_pred_reg=all_probs,
        y_true_cls=all_tmcid,
        y_prob_cls=all_probs,
        metadata_df=test_df,
        modality_tags="Multimodal Deep Fusion (Clinical + Biofluid + Neuroimaging + Genetics)",
        notes=f"RQ2 benchmark: {model_tag} rapid worsening classification."
    )

    return {
        "m12": m12,
        "m24": m24,
        "m_cls": m_cls
    }


def generate_fusion_comparison_plot(results_dict):
    print("\n[Plotting] Generating publication-grade RQ2 Fusion Comparison figure...")
    strategies = list(results_dict.keys())

    mae_12 = [results_dict[s]["m12"]["mae"] for s in strategies]
    mae_24 = [results_dict[s]["m24"]["mae"] for s in strategies]
    r2_12 = [results_dict[s]["m12"]["r2"] * 100 for s in strategies]
    r2_24 = [results_dict[s]["m24"]["r2"] * 100 for s in strategies]
    auc_cls = [results_dict[s]["m_cls"]["roc_auc"] for s in strategies]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    fig.suptitle("RQ2 Multimodal Fusion Showdown: Early vs. Late vs. Cross-Attention Transformer\nEvaluated on Held-Out Test Patients (N=583)", 
                 fontsize=14, fontweight="bold", y=1.02, color="#0f172a")

    x = np.arange(len(strategies))
    width = 0.35

    # Panel 1: MAE
    ax1 = axes[0]
    r1 = ax1.bar(x - width/2, mae_12, width, label="+12 Months", color="#3b82f6", edgecolor="#1e3a8a")
    r2 = ax1.bar(x + width/2, mae_24, width, label="+24 Months", color="#10b981", edgecolor="#064e3b")
    ax1.set_title("A. Mean Absolute Error (MAE in pts) — Lower is Better", fontsize=11, fontweight="bold")
    ax1.set_ylabel("MAE (Points)", fontsize=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=9)
    ax1.set_ylim(4.0, 5.5)
    ax1.legend(loc="upper right", frameon=True)
    for rect in list(r1) + list(r2):
        h = rect.get_height()
        ax1.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel 2: R²
    ax2 = axes[1]
    r3 = ax2.bar(x - width/2, r2_12, width, label="+12 Months", color="#6366f1", edgecolor="#312e81")
    r4 = ax2.bar(x + width/2, r2_24, width, label="+24 Months", color="#14b8a6", edgecolor="#134e4a")
    ax2.set_title("B. Explained Variance (R² %) — Higher is Better", fontsize=11, fontweight="bold")
    ax2.set_ylabel("R² (%)", fontsize=10, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, fontsize=9)
    ax2.set_ylim(70, 82)
    ax2.legend(loc="upper right", frameon=True)
    for rect in list(r3) + list(r4):
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel 3: ROC-AUC
    ax3 = axes[2]
    colors_auc = ["#2563eb", "#059669", "#7c3aed"]
    bars = ax3.bar(strategies, auc_cls, width=0.45, color=colors_auc, edgecolor="#0f172a")
    ax3.set_title("C. Rapid Worsening AUC (MCID >= 3.5)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("ROC-AUC Score", fontsize=10, fontweight="bold")
    ax3.set_ylim(0.55, 0.78)
    ax3.axhline(0.7099, color="#ef4444", linestyle="--", linewidth=1.2, label="Model 1 Baseline (0.710)")
    ax3.legend(loc="upper right", frameon=True)
    for bar in bars:
        h = bar.get_height()
        ax3.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_fig = os.path.join(REPORT_DIR, "fusion_strategies_comparison.png")
    plt.savefig(out_fig, bbox_inches="tight")
    plt.close()
    print(f"  Saved RQ2 Fusion Comparison figure to: {out_fig}")


def main():
    print("=" * 80)
    print("PHASE 4 & 5: MULTIMODAL DEEP FUSION ARCHITECTURES (ANSWERING RQ2)")
    print("=" * 80)

    train_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "train_longitudinal.parquet"))
    val_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "val_longitudinal.parquet"))
    test_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "test_longitudinal.parquet"))
    test_tabular = pd.read_parquet(os.path.join(PROCESSED_DIR, "test_tabular_latest.parquet"))
    val_tabular = pd.read_parquet(os.path.join(PROCESSED_DIR, "val_tabular_latest.parquet"))

    # Compute training medians and fit scaler strictly on train
    train_medians = train_df[ALL_FEATURE_COLS].median().fillna(0.0)
    scaler = StandardScaler()
    scaler.fit(train_df[ALL_FEATURE_COLS].fillna(train_medians))

    print("  Transforming trajectories with zero-leakage within-patient LOCF + train medians...")
    train_scaled = build_trajectories(train_df, ALL_FEATURE_COLS, scaler, train_medians)
    val_scaled = build_trajectories(val_df, ALL_FEATURE_COLS, scaler, train_medians)
    test_scaled = build_trajectories(test_df, ALL_FEATURE_COLS, scaler, train_medians)

    print("  Extracting labeled sequences for train (all prefixes) and val/test (latest visits)...")
    seq_tr, cur_tr, t12_tr, t24_tr, mcid_tr, pat_tr = extract_labeled_sequences(
        train_scaled, ALL_FEATURE_COLS, mode="all_labeled"
    )
    seq_va, cur_va, t12_va, t24_va, mcid_va, pat_va = extract_labeled_sequences(
        val_scaled, ALL_FEATURE_COLS, mode="latest_only", latest_pats_df=val_tabular
    )
    seq_te, cur_te, t12_te, t24_te, mcid_te, pat_te = extract_labeled_sequences(
        test_scaled, ALL_FEATURE_COLS, mode="latest_only", latest_pats_df=test_tabular
    )

    print(f"  Extracted Sequences -> Train: {len(seq_tr)} | Val: {len(seq_va)} | Test: {len(seq_te)} patients")

    ds_train = SequenceDataset(seq_tr, cur_tr, t12_tr, t24_tr, mcid_tr, pat_tr)
    ds_val = SequenceDataset(seq_va, cur_va, t12_va, t24_va, mcid_va, pat_va)
    ds_test = SequenceDataset(seq_te, cur_te, t12_te, t24_te, mcid_te, pat_te)

    loader_tr = DataLoader(ds_train, batch_size=32, shuffle=True, collate_fn=pad_collate_fn)
    loader_va = DataLoader(ds_val, batch_size=64, shuffle=False, collate_fn=pad_collate_fn)
    loader_te = DataLoader(ds_test, batch_size=64, shuffle=False, collate_fn=pad_collate_fn)

    results = {}

    # 1. Strategy A: Early Fusion
    model_early = EarlyFusionModel(input_dim=len(ALL_FEATURE_COLS), hidden_dim=64, dropout=0.25)
    results["Early Fusion"] = train_and_eval_fusion_model(
        model_early, "Early Fusion", loader_tr, loader_va, loader_te, test_tabular, num_epochs=35
    )

    # 2. Strategy B: Late Fusion
    model_late = LateFusionModel(clin_dim=len(CLINICAL_COLS), img_dim=len(IMAGING_COLS), demo_dim=len(DEMO_COLS), hidden_dim=48, dropout=0.25)
    results["Late Fusion"] = train_and_eval_fusion_model(
        model_late, "Late Fusion", loader_tr, loader_va, loader_te, test_tabular, num_epochs=35
    )

    # 3. Strategy C: Cross-Attention Multimodal Transformer
    model_cross = CrossAttentionFusionModel(clin_dim=len(CLINICAL_COLS), img_dim=len(IMAGING_COLS), demo_dim=len(DEMO_COLS), hidden_dim=64, num_heads=4, dropout=0.25)
    results["Cross-Attention"] = train_and_eval_fusion_model(
        model_cross, "Cross-Attention Transformer", loader_tr, loader_va, loader_te, test_tabular, num_epochs=35
    )

    # Generate Comparison Figure
    generate_fusion_comparison_plot(results)

    print("\n" + "=" * 80)
    print("PHASE 4 & 5 MULTIMODAL DEEP FUSION BENCHMARK COMPLETE (RQ2 ANSWERED)!")
    print("=" * 80)


if __name__ == "__main__":
    main()
