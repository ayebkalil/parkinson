"""
visualize_data_engineering.py
------------------------------
Generates comprehensive Phase 2 Data Engineering validation figures:
1. Distribution of Current vs 12-Month vs 24-Month Motor Scores (NP3TOT).
2. Disease progression rates (MCID >= 3.5 pts worsening) by cohort.
3. Multimodal completeness matrix & sample sizes.
4. Correlation between multimodal baseline features and 12-month motor delta.

Saves: report/phase2_data_engineering_validation.png
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROCESSED_DIR = "data/processed"
REPORT_DIR = "report"


def main():
    sns.set_theme(style="whitegrid", font="sans-serif")
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "multimodal_longitudinal.parquet"))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.32, wspace=0.25)

    # 1. Target Distributions: Current vs +12m vs +24m
    ax1 = axes[0, 0]
    pd_df = df[df["COHORT_DEFINITION"] == "Parkinson's Disease"]
    sns.kdeplot(pd_df["NP3TOT"].dropna(), ax=ax1, label="Current Visit ($t$)", color="#2b5c8f", fill=True, alpha=0.3, lw=2)
    sns.kdeplot(pd_df["target_np3tot_12m"].dropna(), ax=ax1, label="12-Month Target ($t+12$m)", color="#d95f02", fill=True, alpha=0.25, lw=2)
    sns.kdeplot(pd_df["target_np3tot_24m"].dropna(), ax=ax1, label="24-Month Target ($t+24$m)", color="#7570b3", fill=True, alpha=0.2, lw=2)
    ax1.set_title("A. PD Motor Progression Distributions (MDS-UPDRS Part III)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("MDS-UPDRS Part III Score (NP3TOT, 0–132)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.legend(loc="upper right", frameon=True, fontsize=10)
    ax1.set_xlim(0, 90)

    # 2. Progression Rates (MCID >= 3.5 points) by Cohort
    ax2 = axes[0, 1]
    cohort_order = ["Parkinson's Disease", "Prodromal", "Healthy Control"]
    prog_data = []
    for c in cohort_order:
        sub = df[df["COHORT_DEFINITION"] == c]
        rate_12 = sub["prog_mcid_12m"].dropna().mean() * 100
        rate_24 = sub["prog_mcid_24m"].dropna().mean() * 100
        prog_data.append({"Cohort": c, "Window": "+12 Months", "Progression_Rate": rate_12})
        prog_data.append({"Cohort": c, "Window": "+24 Months", "Progression_Rate": rate_24})
    
    df_prog = pd.DataFrame(prog_data)
    sns.barplot(data=df_prog, x="Cohort", y="Progression_Rate", hue="Window", ax=ax2, palette=["#386cb0", "#e7298a"])
    ax2.set_title("B. Clinically Meaningful Worsening (Delta >= 3.5 pts MCID)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_ylabel("Percentage Worsening (%)", fontsize=11)
    ax2.set_xlabel("Diagnostic Cohort", fontsize=11)
    for p in ax2.patches:
        h = p.get_height()
        if h > 0:
            ax2.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h + 1.2),
                         ha="center", va="bottom", fontsize=10, fontweight="semibold")
    ax2.set_ylim(0, 60)
    ax2.legend(title="Progression Horizon", loc="upper right")

    # 3. Multimodal Coverage Matrix
    ax3 = axes[1, 0]
    modality_counts = {
        "Motor Part III": (df["NP3TOT"].notna().sum(), len(df)),
        "Non-Motor (MoCA)": (df["moca_total"].notna().sum(), len(df)),
        "Autonomic (SCOPA)": (df["scopa_aut_total"].notna().sum(), len(df)),
        "DaTSCAN SPECT (SBR)": (df["datscan_caudate_sbr"].notna().sum(), len(df)),
        "Biospecimens (CSF)": (df["csf_ttau"].notna().sum(), len(df)),
        "Structural MRI (FS7)": (df["mri_putamen_vol"].notna().sum(), len(df)),
        "12m Future Target": (df["target_np3tot_12m"].notna().sum(), len(df)),
        "24m Future Target": (df["target_np3tot_24m"].notna().sum(), len(df))
    }
    labels = list(modality_counts.keys())
    percents = [cnt / tot * 100 for cnt, tot in modality_counts.values()]
    counts = [cnt for cnt, tot in modality_counts.values()]
    
    colors = ["#2b5c8f" if "Target" not in l else "#2ca02c" for l in labels]
    bars = ax3.barh(labels[::-1], percents[::-1], color=colors[::-1], alpha=0.85, edgecolor="none", height=0.6)
    ax3.set_title("C. Multimodal Modality Completeness across 32,743 Visits", fontsize=13, fontweight="bold", pad=10)
    ax3.set_xlabel("Dataset Completeness (%)", fontsize=11)
    ax3.set_xlim(0, 115)
    for bar, pct, cnt in zip(bars, percents[::-1], counts[::-1]):
        ax3.text(pct + 1.5, bar.get_y() + bar.get_height()/2, f"{pct:.1f}% ({cnt:,})", 
                 va="center", fontsize=9, fontweight="semibold")

    # 4. Multimodal Correlation with 12-Month Motor Delta
    ax4 = axes[1, 1]
    feature_cols = [
        ("Current NP3TOT", "NP3TOT"),
        ("Active LEDD", "ledd"),
        ("Bradykinesia", "np3_bradykinesia"),
        ("Rigidity", "np3_rigidity"),
        ("Tremor", "np3_tremor"),
        ("MoCA Score", "moca_total"),
        ("DaTSCAN Putamen SBR", "datscan_putamen_sbr"),
        ("DaTSCAN Asymmetry", "datscan_putamen_asym"),
        ("CSF t-tau", "csf_ttau"),
        ("CSF ABeta42", "csf_abeta42"),
        ("MRI Putamen Vol", "mri_putamen_vol"),
        ("Age at Visit", "age_at_visit")
    ]
    corrs = []
    names = []
    for label, col in feature_cols:
        if col in pd_df.columns:
            valid = pd_df[[col, "delta_np3tot_12m"]].dropna()
            if len(valid) > 50:
                r = np.corrcoef(valid[col], valid["delta_np3tot_12m"])[0, 1]
                corrs.append(r)
                names.append(label)

    order = np.argsort(corrs)
    sorted_corrs = [corrs[i] for i in order]
    sorted_names = [names[i] for i in order]
    bar_cols = ["#d95f02" if r < 0 else "#1f78b4" for r in sorted_corrs]
    ax4.barh(sorted_names, sorted_corrs, color=bar_cols, alpha=0.85, height=0.6)
    ax4.set_title("D. Feature Pearson Correlation with Delta NP3TOT (+12m)", fontsize=13, fontweight="bold", pad=10)
    ax4.set_xlabel("Pearson Correlation Coefficient (r)", fontsize=11)
    ax4.axvline(0, color="gray", linestyle="--", alpha=0.7)
    for i, r in enumerate(sorted_corrs):
        if r < 0:
            ax4.text(r - 0.015, i, f"{r:+.3f}", va="center", ha="right", fontsize=9, fontweight="semibold")
        else:
            ax4.text(r + 0.015, i, f"{r:+.3f}", va="center", ha="left", fontsize=9, fontweight="semibold")
    ax4.set_xlim(-0.46, 0.20)

    plt.suptitle("PPMI Multimodal Cohort: Phase 2 Data Engineering & Progression Target Validation", 
                 fontsize=16, fontweight="bold", y=0.98)
    
    out_path = os.path.join(REPORT_DIR, "phase2_data_engineering_validation.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated validation chart: {out_path}")


if __name__ == "__main__":
    main()
