import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 1. Load Participant Status
status_file = [f for f in os.listdir(RAW_DIR) if f.startswith("Participant_Status") and f.endswith(".csv")][0]
status_path = os.path.join(RAW_DIR, status_file)
df_status = pd.read_csv(status_path, dtype={"PATNO": str})
print(f"Loaded {status_file}: {len(df_status)} rows")

# 2. Check columns in Participant Status
cols = df_status.columns.tolist()
print("Columns in Participant Status:", cols[:10])

# Determine cohort column
cohort_col = None
for candidate in ["COHORT", "COHORT_DEFINITION", "ENROLL_CAT", "RECRUITMENT_CAT", "STUDY_ARM"]:
    if candidate in df_status.columns:
        cohort_col = candidate
        break

if not cohort_col:
    # Look for any column with cohort in name
    matches = [c for c in df_status.columns if "cohort" in c.lower()]
    if matches:
        cohort_col = matches[0]

print(f"Using cohort column: '{cohort_col}'")

# Deduplicate by PATNO (take most recent status)
df_patients = df_status.dropna(subset=["PATNO"]).drop_duplicates(subset=["PATNO"], keep="last")
print(f"Unique patients found: {len(df_patients)}")

if cohort_col:
    cohort_counts = df_patients[cohort_col].value_counts(dropna=False)
    print("\nCohort distribution:")
    print(cohort_counts)
    strat_col = df_patients[cohort_col].fillna("Unknown")
else:
    strat_col = None

# 3. Cross-reference with MDS-UPDRS Part III
updrs_file = [f for f in os.listdir(RAW_DIR) if f.startswith("MDS-UPDRS_Part_III") and not "Survey" in f and f.endswith(".csv")][0]
updrs_path = os.path.join(RAW_DIR, updrs_file)
df_updrs = pd.read_csv(updrs_path, usecols=["PATNO"], dtype={"PATNO": str})
updrs_patnos = set(df_updrs["PATNO"].dropna().unique())
print(f"\nPatients with MDS-UPDRS Part III assessments: {len(updrs_patnos)}")

df_patients["has_motor_data"] = df_patients["PATNO"].isin(updrs_patnos)
print(f"Patients in status table with motor data: {df_patients['has_motor_data'].sum()}")

# 4. Perform Stratified Train/Val/Test Split (70% Train, 15% Val, 15% Test)
# Seed = 42 for complete reproducibility
np.random.seed(42)

# If rare classes have < 3 samples, group them into 'Other' for stratification
if strat_col is not None:
    counts = strat_col.value_counts()
    valid_strat = strat_col.apply(lambda x: x if counts[x] >= 3 else "Other")
else:
    valid_strat = None

# Step 1: 70% Train vs 30% Temp (Val + Test)
train_df, temp_df = train_test_split(
    df_patients,
    test_size=0.30,
    random_state=42,
    stratify=valid_strat
)

# Step 2: Split 30% Temp equally into 15% Val and 15% Test
temp_strat = valid_strat.loc[temp_df.index] if valid_strat is not None else None
counts_temp = temp_strat.value_counts() if temp_strat is not None else None
if temp_strat is not None:
    temp_strat_clean = temp_strat.apply(lambda x: x if counts_temp[x] >= 2 else "Other")
else:
    temp_strat_clean = None

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_strat_clean
)

# 5. Label splits
train_df = train_df.copy()
train_df["SPLIT"] = "TRAIN"

val_df = val_df.copy()
val_df["SPLIT"] = "VAL"

test_df = test_df.copy()
test_df["SPLIT"] = "TEST"

# Combine into master split table
df_split = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Select key columns
out_cols = ["PATNO", "SPLIT", "has_motor_data"]
if cohort_col and cohort_col in df_split.columns:
    out_cols.insert(1, cohort_col)

df_split_out = df_split[out_cols].sort_values("PATNO")

# Save canonical shared artifact
out_file = os.path.join(PROCESSED_DIR, "patient_split.csv")
df_split_out.to_csv(out_file, index=False)
print(f"\nSaved canonical patient split to: {out_file}")

# Summary Report
print("\n=== Canonical Patient Split Summary ===")
print(df_split_out["SPLIT"].value_counts(normalize=True).round(3) * 100)
print(f"\nTotal Patients in Split: {len(df_split_out)}")
print(f"  - TRAIN: {(df_split_out['SPLIT'] == 'TRAIN').sum()} patients")
print(f"  - VAL:   {(df_split_out['SPLIT'] == 'VAL').sum()} patients")
print(f"  - TEST:  {(df_split_out['SPLIT'] == 'TEST').sum()} patients")
if cohort_col:
    print("\nSplit cross-tabulation with Cohort:")
    print(pd.crosstab(df_split_out[cohort_col], df_split_out["SPLIT"]))
