import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
REPORT_DIR = os.path.join("report")
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 60)
print("PHASE 2 - STEP 1: COHORT FILTERING & TARGET TRAJECTORY EDA")
print("=" * 60)

# 1. Load Participant Status & Split Table
status_file = [f for f in os.listdir(RAW_DIR) if f.startswith("Participant_Status") and f.endswith(".csv")][0]
df_status = pd.read_csv(os.path.join(RAW_DIR, status_file), dtype={"PATNO": str})

split_file = os.path.join(PROCESSED_DIR, "patient_split.csv")
df_split = pd.read_csv(split_file, dtype={"PATNO": str})

print(f"Total Participants in Status File: {len(df_status)}")
print(f"Total Participants in Canonical Split: {len(df_split)}")

# Cohort Mapping
cohort_map = {
    1: "Parkinson's Disease (PD)",
    2: "Healthy Control (HC)",
    3: "SWEDD",
    4: "Prodromal / Other"
}
df_split["COHORT_NAME"] = df_split["COHORT"].map(cohort_map).fillna("Unknown")

# 2. Load MDS-UPDRS Part III
updrs_file = [f for f in os.listdir(RAW_DIR) if f.startswith("MDS-UPDRS_Part_III") and not "Survey" in f and f.endswith(".csv")][0]
updrs_path = os.path.join(RAW_DIR, updrs_file)

cols_to_load = ["REC_ID", "PATNO", "EVENT_ID", "PAG_NAME", "INFODT", "PDSTATE", "NP3TOT"]
df_updrs = pd.read_csv(updrs_path, usecols=lambda c: c in cols_to_load, dtype={"PATNO": str})
print(f"\nLoaded {updrs_file}: {len(df_updrs)} assessment records")

# Parse dates (INFODT format is MM/YYYY)
df_updrs["VISIT_DATE"] = pd.to_datetime(df_updrs["INFODT"], format="%m/%Y", errors="coerce")

# Filter out rows missing NP3TOT
df_updrs_clean = df_updrs.dropna(subset=["NP3TOT"]).copy()
df_updrs_clean["NP3TOT"] = pd.to_numeric(df_updrs_clean["NP3TOT"], errors="coerce")
df_updrs_clean = df_updrs_clean.dropna(subset=["NP3TOT"])
print(f"Assessments with valid NP3TOT scores: {len(df_updrs_clean)}")

# Merge with patient split
df_merged = df_updrs_clean.merge(df_split, on="PATNO", how="inner")
print(f"Assessments linked to patient split: {len(df_merged)}")

# 3. Analyze Core Cohort (PD vs HC)
core_cohorts = ["Parkinson's Disease (PD)", "Healthy Control (HC)"]
df_core = df_merged[df_merged["COHORT_NAME"].isin(core_cohorts)].copy()

print("\n--- Core Cohort Summary (PD vs HC) ---")
print(f"Total assessments in core cohort: {len(df_core)}")
unique_patients_core = df_core["PATNO"].nunique()
print(f"Unique core patients with motor exams: {unique_patients_core}")

pat_summary = df_core.groupby("COHORT_NAME")["PATNO"].nunique()
print("\nUnique Patients per Cohort:")
for cohort, count in pat_summary.items():
    print(f"  - {cohort}: {count} patients")

# 4. Visit Frequency Distribution
visit_counts = df_core.groupby(["COHORT_NAME", "PATNO"]).size().reset_index(name="NUM_VISITS")
print("\nVisit Frequency per Patient:")
print(visit_counts.groupby("COHORT_NAME")["NUM_VISITS"].describe().round(1))

# Check top EVENT_IDs
print("\nTop In-Clinic Visit Types (EVENT_ID):")
top_events = df_core["EVENT_ID"].value_counts().head(10)
print(top_events)

# 5. Target Score (NP3TOT) Statistics across Cohorts
print("\n--- NP3TOT (Motor Score) Distribution ---")
score_stats = df_core.groupby("COHORT_NAME")["NP3TOT"].describe().round(2)
print(score_stats)

# Medication State Analysis
med_state_counts = df_core[df_core["COHORT_NAME"] == "Parkinson's Disease (PD)"]["PDSTATE"].value_counts(dropna=False)
print("\nPD Patients Medication State (PDSTATE) distribution:")
print(med_state_counts)

# 6. Longitudinal Progression Analysis (Tracking Baseline to Future Visits)
# Identify Baseline (BL) and Year 1 (V04) and Year 2 (V06)
bl_scores = df_core[df_core["EVENT_ID"] == "BL"][["PATNO", "COHORT_NAME", "NP3TOT", "PDSTATE"]].rename(columns={"NP3TOT": "NP3TOT_BL", "PDSTATE": "PDSTATE_BL"})
v04_scores = df_core[df_core["EVENT_ID"] == "V04"][["PATNO", "NP3TOT", "PDSTATE"]].rename(columns={"NP3TOT": "NP3TOT_12M", "PDSTATE": "PDSTATE_12M"})
v06_scores = df_core[df_core["EVENT_ID"] == "V06"][["PATNO", "NP3TOT", "PDSTATE"]].rename(columns={"NP3TOT": "NP3TOT_24M", "PDSTATE": "PDSTATE_24M"})

# Merge horizons
progression_df = bl_scores.merge(v04_scores, on="PATNO", how="inner").merge(v06_scores, on="PATNO", how="inner")
print(f"\nPatients with complete 2-Year Trajectory (BL -> 12M -> 24M): {len(progression_df)}")
print(f"  - PD Patients: {(progression_df['COHORT_NAME'] == 'Parkinson\'s Disease (PD)').sum()}")
print(f"  - Healthy Controls: {(progression_df['COHORT_NAME'] == 'Healthy Control (HC)').sum()}")

# Calculate actual score deltas
progression_df["DELTA_12M"] = progression_df["NP3TOT_12M"] - progression_df["NP3TOT_BL"]
progression_df["DELTA_24M"] = progression_df["NP3TOT_24M"] - progression_df["NP3TOT_BL"]

pd_prog = progression_df[progression_df["COHORT_NAME"] == "Parkinson's Disease (PD)"]
hc_prog = progression_df[progression_df["COHORT_NAME"] == "Healthy Control (HC)"]

print("\n--- PD Progression Rate ---")
print(f"Baseline Mean Score: {pd_prog['NP3TOT_BL'].mean():.2f} ± {pd_prog['NP3TOT_BL'].std():.2f}")
print(f"12-Month Mean Score: {pd_prog['NP3TOT_12M'].mean():.2f} ± {pd_prog['NP3TOT_12M'].std():.2f} (Change: +{pd_prog['DELTA_12M'].mean():.2f} points)")
print(f"24-Month Mean Score: {pd_prog['NP3TOT_24M'].mean():.2f} ± {pd_prog['NP3TOT_24M'].std():.2f} (Change: +{pd_prog['DELTA_24M'].mean():.2f} points)")

print("\n--- Healthy Control Progression Rate ---")
print(f"Baseline Mean Score: {hc_prog['NP3TOT_BL'].mean():.2f} ± {hc_prog['NP3TOT_BL'].std():.2f}")
print(f"12-Month Mean Score: {hc_prog['NP3TOT_12M'].mean():.2f} ± {hc_prog['NP3TOT_12M'].std():.2f} (Change: {hc_prog['DELTA_12M'].mean():.2f} points)")
print(f"24-Month Mean Score: {hc_prog['NP3TOT_24M'].mean():.2f} ± {hc_prog['NP3TOT_24M'].std():.2f} (Change: {hc_prog['DELTA_24M'].mean():.2f} points)")

# 7. Generate Visual EDA Charts
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Score Distribution by Cohort
sns.histplot(data=df_core, x="NP3TOT", hue="COHORT_NAME", kde=True, bins=35, palette={"Parkinson's Disease (PD)": "#e11d48", "Healthy Control (HC)": "#0284c7"}, ax=axes[0, 0], alpha=0.6)
axes[0, 0].set_title("A. MDS-UPDRS Part III (Motor Score) Distribution", fontsize=12, fontweight="bold")
axes[0, 0].set_xlabel("NP3TOT Total Motor Score (0 - 132)")
axes[0, 0].set_ylabel("Assessment Count")

# Subplot 2: Longitudinal Trajectory (BL -> 12m -> 24m)
traj_data = pd.DataFrame({
    "Timepoint": ["Baseline", "+12 Months", "+24 Months"] * 2,
    "Cohort": ["Parkinson's Disease (PD)"] * 3 + ["Healthy Control (HC)"] * 3,
    "Mean_Score": [pd_prog['NP3TOT_BL'].mean(), pd_prog['NP3TOT_12M'].mean(), pd_prog['NP3TOT_24M'].mean(),
                   hc_prog['NP3TOT_BL'].mean(), hc_prog['NP3TOT_12M'].mean(), hc_prog['NP3TOT_24M'].mean()],
    "Std_Err": [pd_prog['NP3TOT_BL'].sem(), pd_prog['NP3TOT_12M'].sem(), pd_prog['NP3TOT_24M'].sem(),
                hc_prog['NP3TOT_BL'].sem(), hc_prog['NP3TOT_12M'].sem(), hc_prog['NP3TOT_24M'].sem()]
})

sns.lineplot(data=traj_data, x="Timepoint", y="Mean_Score", hue="Cohort", marker="o", markersize=8, linewidth=2.5, palette={"Parkinson's Disease (PD)": "#e11d48", "Healthy Control (HC)": "#0284c7"}, ax=axes[0, 1])
axes[0, 1].set_title("B. 2-Year Progression Trajectory (Mean ± SEM)", fontsize=12, fontweight="bold")
axes[0, 1].set_ylabel("MDS-UPDRS Part III Score")
axes[0, 1].set_ylim(0, 35)

# Subplot 3: Number of Visits per Patient
sns.countplot(data=visit_counts[visit_counts["NUM_VISITS"] <= 15], x="NUM_VISITS", hue="COHORT_NAME", palette={"Parkinson's Disease (PD)": "#e11d48", "Healthy Control (HC)": "#0284c7"}, ax=axes[1, 0])
axes[1, 0].set_title("C. Longitudinal Visit Frequency per Patient", fontsize=12, fontweight="bold")
axes[1, 0].set_xlabel("Number of In-Clinic Visits")
axes[1, 0].set_ylabel("Number of Patients")

# Subplot 4: Medication State (ON vs OFF) in PD Patients
med_clean = df_core[df_core["COHORT_NAME"] == "Parkinson's Disease (PD)"]["PDSTATE"].fillna("Untreated / Baseline").value_counts()
axes[1, 1].pie(med_clean.values, labels=med_clean.index, autopct="%1.1f%%", colors=["#3b82f6", "#f59e0b", "#10b981", "#94a3b8"], startangle=140, textprops={'fontsize': 10, 'weight': 'bold'})
axes[1, 1].set_title("D. Medication State During Motor Exam (PD)", fontsize=12, fontweight="bold")

plt.tight_layout()
eda_fig_path = os.path.join(REPORT_DIR, "cohort_target_eda.png")
plt.savefig(eda_fig_path, dpi=300)
plt.close()
print(f"\nGenerated EDA Visualization Figure: {eda_fig_path}")

print("=" * 60)
print("PHASE 2 - STEP 1 COMPLETE!")
print("=" * 60)
