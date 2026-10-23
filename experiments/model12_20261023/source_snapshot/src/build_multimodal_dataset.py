"""
build_multimodal_dataset.py
----------------------------
PPMI Multimodal Longitudinal Dataset Construction Pipeline (Phase 2, Steps 2-4)

Integrates:
1. Patient Status & Cohort (PD, HC, Prodromal, SWEDD)
2. Demographics (Age, Sex, Education)
3. Motor Assessments (MDS-UPDRS Part III subscores, PDSTATE, HRPOSTMED, NHY, Part I, Part II, Part IV, Schwab-England)
4. Medication Confounder Control (Active LEDD at visit date, PDSTATE)
5. Non-Motor Assessments (MoCA, SCOPA-AUT, GDS-15, ESS, RBDSQ, UPSIT)
6. Biospecimens (CSF ABeta42, t-tau, p-tau, a-synuclein, NfL, SAA status)
7. Tabular Imaging:
   - DaTSCAN SPECT (Caudate & Putamen SBR, Asymmetry Index)
   - FreeSurfer Structural MRI (Subcortical volumes normalized to eTIV)
8. Longitudinal Progression Targets:
   - target_np3tot_12m: actual NP3TOT score at +12 months (window 250-480 days)
   - target_np3tot_24m: actual NP3TOT score at +24 months (window 550-910 days)
   - delta_np3tot_12m: change in motor score from current visit to +12 months
   - delta_np3tot_24m: change in motor score from current visit to +24 months
   - prog_mcid_12m: binary worsening indicator (>= +3.5 points)
   - prog_mcid_24m: binary worsening indicator (>= +3.5 points)
9. Canonical Patient-Level Split (70% Train / 15% Val / 15% Test)

Outputs saved as Parquet:
- data/processed/multimodal_longitudinal.parquet
- data/processed/train_longitudinal.parquet
- data/processed/val_longitudinal.parquet
- data/processed/test_longitudinal.parquet
- data/processed/train_tabular_latest.parquet
- data/processed/val_tabular_latest.parquet
- data/processed/test_tabular_latest.parquet
"""

import os
import re
import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
REPORT_DIR = "report"


def clean_numeric(val):
    """Clean string biomarker values like '<80' or '>1997' to float."""
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    val_str = re.sub(r"[^\d\.\-]", "", val_str)
    try:
        return float(val_str)
    except ValueError:
        return np.nan


def load_cohort_and_split():
    print("[1/8] Loading Cohort Definitions and Canonical Split...")
    # Canonical split
    df_split = pd.read_csv(os.path.join(PROCESSED_DIR, "patient_split.csv"))
    df_split["split"] = df_split["SPLIT"].str.lower()
    df_split = df_split[["PATNO", "split"]]
    
    # Participant Status
    df_status = pd.read_csv(os.path.join(RAW_DIR, "Participant_Status_26Sep2026.csv"))
    df_status = df_status[["PATNO", "COHORT_DEFINITION"]].drop_duplicates("PATNO")
    
    # Demographics (Sex)
    df_demog = pd.read_csv(os.path.join(RAW_DIR, "Demographics_26Sep2026.csv"))
    df_demog = df_demog.sort_values("INFODT").drop_duplicates("PATNO", keep="last")
    df_demog["is_male"] = (df_demog["SEX"] == 1).astype(int)
    df_demog = df_demog[["PATNO", "is_male"]]
    
    # Socioeconomics (Education Years)
    df_soc = pd.read_csv(os.path.join(RAW_DIR, "Socio-Economics_26Sep2026.csv"))
    df_soc["education_years"] = pd.to_numeric(df_soc["EDUCYRS"], errors="coerce")
    df_soc = df_soc.dropna(subset=["education_years"]).drop_duplicates("PATNO", keep="last")[["PATNO", "education_years"]]
    
    # Merge baseline patient info
    patient_info = df_split.merge(df_status, on="PATNO", how="left")
    patient_info = patient_info.merge(df_demog, on="PATNO", how="left")
    patient_info = patient_info.merge(df_soc, on="PATNO", how="left")
    print(f"  Loaded {len(patient_info)} patients with split and demographics.")
    return patient_info


def load_motor_and_dates():
    print("[2/8] Loading MDS-UPDRS Part III (Motor Exams)...")
    df_p3 = pd.read_csv(os.path.join(RAW_DIR, "MDS-UPDRS_Part_III_26Sep2026.csv"), low_memory=False)
    
    # Parse dates
    df_p3["visit_date"] = pd.to_datetime(df_p3["INFODT"], format="%m/%Y", errors="coerce")
    df_p3 = df_p3[df_p3["visit_date"].notna() & df_p3["NP3TOT"].notna()].copy()
    
    # Standardize PDSTATE: ON, OFF, UNTREATED
    df_p3["PDSTATE"] = df_p3["PDSTATE"].str.upper().str.strip()
    df_p3["pd_state_clean"] = df_p3["PDSTATE"].fillna("UNTREATED")
    df_p3.loc[~df_p3["pd_state_clean"].isin(["ON", "OFF"]), "pd_state_clean"] = "UNTREATED"
    df_p3["pd_state_on"] = (df_p3["pd_state_clean"] == "ON").astype(int)
    df_p3["pd_state_off"] = (df_p3["pd_state_clean"] == "OFF").astype(int)
    
    df_p3["hr_post_med"] = pd.to_numeric(df_p3["HRPOSTMED"], errors="coerce").fillna(0.0)
    df_p3["nhy_stage"] = pd.to_numeric(df_p3["NHY"], errors="coerce")
    df_p3["dyskinesia_present"] = pd.to_numeric(df_p3["DYSKPRES"], errors="coerce").fillna(0.0)
    
    # Motor Subdomain Scores
    tremor_cols = ["NP3PTRMR", "NP3PTRML", "NP3KTRMR", "NP3KTRML", "NP3RTARU", "NP3RTALU", 
                   "NP3RTARL", "NP3RTALL", "NP3RTALJ", "NP3RTCON"]
    for c in tremor_cols:
        df_p3[c] = pd.to_numeric(df_p3[c], errors="coerce").fillna(0.0)
    df_p3["np3_tremor"] = df_p3[tremor_cols].sum(axis=1)
    
    rigidity_cols = ["NP3RIGN", "NP3RIGRU", "NP3RIGLU", "NP3RIGRL", "NP3RIGLL"]
    for c in rigidity_cols:
        df_p3[c] = pd.to_numeric(df_p3[c], errors="coerce").fillna(0.0)
    df_p3["np3_rigidity"] = df_p3[rigidity_cols].sum(axis=1)
    
    brady_cols = ["NP3FTAPR", "NP3FTAPL", "NP3HMOVR", "NP3HMOVL", "NP3PRSPR", "NP3PRSPL", 
                  "NP3TTAPR", "NP3TTAPL", "NP3LGAGR", "NP3LGAGL", "NP3BRADY"]
    for c in brady_cols:
        df_p3[c] = pd.to_numeric(df_p3[c], errors="coerce").fillna(0.0)
    df_p3["np3_bradykinesia"] = df_p3[brady_cols].sum(axis=1)
    
    axial_cols = ["NP3SPCH", "NP3FACXP", "NP3RISNG", "NP3GAIT", "NP3FRZGT", "NP3PSTBL", "NP3POSTR"]
    for c in axial_cols:
        df_p3[c] = pd.to_numeric(df_p3[c], errors="coerce").fillna(0.0)
    df_p3["np3_axial"] = df_p3[axial_cols].sum(axis=1)
    
    group_keys = ["PATNO", "EVENT_ID", "visit_date", "pd_state_clean"]
    num_cols = [
        "NP3TOT", "pd_state_on", "pd_state_off", "hr_post_med", "nhy_stage", "dyskinesia_present",
        "np3_tremor", "np3_rigidity", "np3_bradykinesia", "np3_axial"
    ]
    df_motor = df_p3.groupby(group_keys)[num_cols].mean().reset_index()
    print(f"  Processed {len(df_motor)} motor assessment records.")
    return df_motor


def load_age_and_medications():
    print("[3/8] Loading Age at Visit & Computing Active LEDD...")
    # Age at visit
    df_age = pd.read_csv(os.path.join(RAW_DIR, "Age_at_visit_26Sep2026.csv"))
    df_age = df_age[["PATNO", "EVENT_ID", "AGE_AT_VISIT"]].drop_duplicates(["PATNO", "EVENT_ID"])
    df_age["age_at_visit"] = pd.to_numeric(df_age["AGE_AT_VISIT"], errors="coerce")
    df_age = df_age[["PATNO", "EVENT_ID", "age_at_visit"]]
    
    # LEDD log
    df_ledd = pd.read_csv(os.path.join(RAW_DIR, "LEDD_Concomitant_Medication_Log_26Sep2026.csv"))
    df_ledd["start"] = pd.to_datetime(df_ledd["STARTDT"], format="%m/%Y", errors="coerce")
    df_ledd["stop"] = pd.to_datetime(df_ledd["STOPDT"], format="%m/%Y", errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    df_ledd["ledd_val"] = pd.to_numeric(df_ledd["LEDD"], errors="coerce").fillna(0.0)
    df_ledd = df_ledd[df_ledd["start"].notna() & (df_ledd["ledd_val"] > 0)]
    
    return df_age, df_ledd


def calculate_active_ledd(df_motor, df_ledd):
    """Compute active total LEDD for each motor visit date using vectorized numpy intervals."""
    print("  Vector-matching active LEDD doses to visit dates...")
    grouped_ledd = {
        pat: (g["start"].values, g["stop"].values, g["ledd_val"].values)
        for pat, g in df_ledd.groupby("PATNO")
    }
    
    unique_pairs = df_motor[["PATNO", "visit_date"]].drop_duplicates()
    pat_arr = unique_pairs["PATNO"].values
    date_arr = unique_pairs["visit_date"].values
    
    ledd_results = []
    for patno, vdate in zip(pat_arr, date_arr):
        if patno in grouped_ledd:
            starts, stops, vals = grouped_ledd[patno]
            active_mask = (starts <= vdate) & (stops >= vdate)
            ledd_results.append(float(np.sum(vals[active_mask])))
        else:
            ledd_results.append(0.0)
            
    unique_pairs["ledd"] = ledd_results
    df_motor = df_motor.merge(unique_pairs, on=["PATNO", "visit_date"], how="left")
    df_motor["ledd"] = df_motor["ledd"].fillna(0.0)
    print("  Active LEDD computation complete.")
    return df_motor


def load_clinical_scales():
    print("[4/8] Loading Clinical Scale Assessments (Part I, II, IV, Schwab-England, MoCA, SCOPA, GDS, Sleep, UPSIT)...")
    
    # MDS-UPDRS Part I
    df_p1 = pd.read_csv(os.path.join(RAW_DIR, "MDS-UPDRS_Part_I_26Sep2026.csv"))
    df_p1["np1_total"] = pd.to_numeric(df_p1["NP1RTOT"], errors="coerce")
    df_p1 = df_p1[["PATNO", "EVENT_ID", "np1_total"]].drop_duplicates(["PATNO", "EVENT_ID"])
    
    # MDS-UPDRS Part II
    df_p2 = pd.read_csv(os.path.join(RAW_DIR, "MDS_UPDRS_Part_II__Patient_Questionnaire_26Sep2026.csv"))
    df_p2["np2_total"] = pd.to_numeric(df_p2["NP2PTOT"], errors="coerce")
    df_p2 = df_p2[["PATNO", "EVENT_ID", "np2_total"]].drop_duplicates(["PATNO", "EVENT_ID"])
    
    # MDS-UPDRS Part IV
    df_p4 = pd.read_csv(os.path.join(RAW_DIR, "MDS-UPDRS_Part_IV__Motor_Complications_26Sep2026.csv"))
    df_p4["np4_total"] = pd.to_numeric(df_p4["NP4TOT"], errors="coerce")
    df_p4 = df_p4[["PATNO", "EVENT_ID", "np4_total"]].drop_duplicates(["PATNO", "EVENT_ID"])
    
    # Schwab & England ADL
    df_schwab = pd.read_csv(os.path.join(RAW_DIR, "Modified_Schwab___England_Activities_of_Daily_Living_26Sep2026.csv"))
    df_schwab["schwab_england_pct"] = pd.to_numeric(df_schwab["MSEADLG"], errors="coerce")
    df_schwab = df_schwab[["PATNO", "EVENT_ID", "schwab_england_pct"]].drop_duplicates(["PATNO", "EVENT_ID"])
    
    # MoCA Cognitive
    df_moca = pd.read_csv(os.path.join(RAW_DIR, "Montreal_Cognitive_Assessment__MoCA__26Sep2026.csv"))
    df_moca["moca_total"] = pd.to_numeric(df_moca["MCATOT"], errors="coerce")
    df_moca = df_moca[["PATNO", "EVENT_ID", "moca_total"]].drop_duplicates(["PATNO", "EVENT_ID"])
    
    # SCOPA-AUT
    df_scopa = pd.read_csv(os.path.join(RAW_DIR, "SCOPA-AUT_26Sep2026.csv"))
    scau_items = [c for c in df_scopa.columns if c.startswith("SCAU") and not c.endswith("T") and not c.endswith("A")]
    for c in scau_items:
        df_scopa[c] = pd.to_numeric(df_scopa[c], errors="coerce").fillna(0.0)
    df_scopa["scopa_aut_total"] = df_scopa[scau_items].sum(axis=1)
    df_scopa = df_scopa[["PATNO", "EVENT_ID", "scopa_aut_total"]].drop_duplicates(["PATNO", "EVENT_ID"])
    
    # GDS Depression
    df_gds = pd.read_csv(os.path.join(RAW_DIR, "Geriatric_Depression_Scale__Short_Version__26Sep2026.csv"))
    neg_items = ["GDSDROPD", "GDSEMPTY", "GDSBORED", "GDSAFRAD", "GDSHLPLS", "GDSHOME", "GDSMEMRY", "GDSWRTLS", "GDSHOPLS"]
    pos_items = ["GDSSATIS", "GDSGSPIR", "GDSHAPPY", "GDSALIVE", "GDSENRGY", "GDSBETER"]
    gds_score = 0
    for c in neg_items:
        if c in df_gds.columns:
            gds_score = gds_score + (pd.to_numeric(df_gds[c], errors="coerce") == 1).astype(int)
    for c in pos_items:
        if c in df_gds.columns:
            gds_score = gds_score + (pd.to_numeric(df_gds[c], errors="coerce") == 0).astype(int)
    df_gds["gds_total"] = gds_score
    df_gds = df_gds[["PATNO", "EVENT_ID", "gds_total"]].drop_duplicates(["PATNO", "EVENT_ID"])
    
    # ESS Daytime Sleepiness
    df_ess = pd.read_csv(os.path.join(RAW_DIR, "Epworth_Sleepiness_Scale_26Sep2026.csv"))
    ess_items = [f"ESS{i}" for i in range(1, 9) if f"ESS{i}" in df_ess.columns]
    for c in ess_items:
        df_ess[c] = pd.to_numeric(df_ess[c], errors="coerce").fillna(0.0)
    df_ess["ess_total"] = df_ess[ess_items].sum(axis=1)
    df_ess = df_ess[["PATNO", "EVENT_ID", "ess_total"]].drop_duplicates(["PATNO", "EVENT_ID"])
    
    # RBD Sleep Behavior
    df_rbd = pd.read_csv(os.path.join(RAW_DIR, "REM_Sleep_Behavior_Disorder_Screening_Questionnaire_26Sep2026.csv"))
    rbd_items = ["DRMVIVID", "DRMAGRAC", "DRMNOCTB", "SLPLMBMV", "SLPINJUR", "DRMVERBL", 
                 "DRMFIGHT", "DRMUMV", "DRMOBJFL", "MVAWAKEN", "DRMREMEM", "SLPDSTRB"]
    for c in rbd_items:
        if c in df_rbd.columns:
            df_rbd[c] = pd.to_numeric(df_rbd[c], errors="coerce").fillna(0.0)
    df_rbd["rbd_total"] = df_rbd[[c for c in rbd_items if c in df_rbd.columns]].sum(axis=1)
    df_rbd = df_rbd[["PATNO", "EVENT_ID", "rbd_total"]].drop_duplicates(["PATNO", "EVENT_ID"])
    
    # UPSIT Smell Test
    df_upsit = pd.read_csv(os.path.join(RAW_DIR, "University_of_Pennsylvania_Smell_Identification_Test_UPSIT_26Sep2026.csv"))
    df_upsit["upsit_total"] = pd.to_numeric(df_upsit["TOTAL_CORRECT"], errors="coerce")
    df_upsit = df_upsit[["PATNO", "EVENT_ID", "upsit_total"]].drop_duplicates(["PATNO", "EVENT_ID"])
    
    # Merge all non-motor tables onto a single visit scale table
    scales = [df_p1, df_p2, df_p4, df_schwab, df_moca, df_scopa, df_gds, df_ess, df_rbd, df_upsit]
    df_scales = scales[0]
    for s in scales[1:]:
        df_scales = df_scales.merge(s, on=["PATNO", "EVENT_ID"], how="outer")
        
    print(f"  Combined non-motor clinical scales: {len(df_scales)} records.")
    return df_scales


def load_biospecimens():
    print("[5/8] Loading CSF Biomarkers & Alpha-Synuclein SAA...")
    
    # CSF Tests
    df_bio = pd.read_csv(os.path.join(RAW_DIR, "Current_Biospecimen_Analysis_Results_26Sep2026.csv"), low_memory=False)
    csf = df_bio[df_bio["TYPE"].isin(["Cerebrospinal Fluid", "CSF"])].copy()
    csf["num_val"] = csf["TESTVALUE"].apply(clean_numeric)
    
    # Pivot key assays
    test_map = {
        "ABeta42": "csf_abeta42",
        "ABeta 1-42": "csf_abeta42_alt",
        "tTau": "csf_ttau",
        "pTau": "csf_ptau_alt",
        "pTau181": "csf_ptau",
        "CSF Alpha-synuclein": "csf_asyn",
        "a-Synuclein": "csf_asyn_alt",
        "NfL": "csf_nfl",
        "NFL": "csf_nfl_alt"
    }
    csf = csf[csf["TESTNAME"].isin(test_map.keys())].copy()
    csf["feature_name"] = csf["TESTNAME"].map(test_map)
    
    pivoted_csf = csf.pivot_table(
        index=["PATNO", "CLINICAL_EVENT"],
        columns="feature_name",
        values="num_val",
        aggfunc="mean"
    ).reset_index()
    pivoted_csf.rename(columns={"CLINICAL_EVENT": "EVENT_ID"}, inplace=True)
    
    # Fallback consolidation
    if "csf_abeta42" in pivoted_csf.columns and "csf_abeta42_alt" in pivoted_csf.columns:
        pivoted_csf["csf_abeta42"] = pivoted_csf["csf_abeta42"].combine_first(pivoted_csf["csf_abeta42_alt"])
    if "csf_ptau" in pivoted_csf.columns and "csf_ptau_alt" in pivoted_csf.columns:
        pivoted_csf["csf_ptau"] = pivoted_csf["csf_ptau"].combine_first(pivoted_csf["csf_ptau_alt"])
    if "csf_asyn" in pivoted_csf.columns and "csf_asyn_alt" in pivoted_csf.columns:
        pivoted_csf["csf_asyn"] = pivoted_csf["csf_asyn"].combine_first(pivoted_csf["csf_asyn_alt"])
    if "csf_nfl" in pivoted_csf.columns and "csf_nfl_alt" in pivoted_csf.columns:
        pivoted_csf["csf_nfl"] = pivoted_csf["csf_nfl"].combine_first(pivoted_csf["csf_nfl_alt"])
        
    # Biomarker Ratios
    if "csf_ttau" in pivoted_csf.columns and "csf_abeta42" in pivoted_csf.columns:
        pivoted_csf["csf_ttau_abeta_ratio"] = pivoted_csf["csf_ttau"] / pivoted_csf["csf_abeta42"]
    if "csf_ptau" in pivoted_csf.columns and "csf_abeta42" in pivoted_csf.columns:
        pivoted_csf["csf_ptau_abeta_ratio"] = pivoted_csf["csf_ptau"] / pivoted_csf["csf_abeta42"]
        
    csf_cols = ["PATNO", "EVENT_ID"] + [c for c in ["csf_abeta42", "csf_ttau", "csf_ptau", "csf_asyn", 
                                                    "csf_nfl", "csf_ttau_abeta_ratio", "csf_ptau_abeta_ratio"] if c in pivoted_csf.columns]
    df_csf = pivoted_csf[csf_cols]
    
    # SAA Status
    df_saa = pd.read_csv(os.path.join(RAW_DIR, "SAA_Biospecimen_Analysis_Results_26Sep2026.csv"))
    df_saa = df_saa[["PATNO", "CLINICAL_EVENT", "SAA_Status"]].dropna().drop_duplicates(["PATNO", "CLINICAL_EVENT"])
    df_saa.rename(columns={"CLINICAL_EVENT": "EVENT_ID"}, inplace=True)
    df_saa["saa_positive"] = (df_saa["SAA_Status"] == "Positive").astype(int)
    df_saa = df_saa[["PATNO", "EVENT_ID", "saa_positive"]]
    
    # Combine CSF & SAA
    df_bio_all = df_csf.merge(df_saa, on=["PATNO", "EVENT_ID"], how="outer")
    print(f"  Processed {len(df_bio_all)} biospecimen visit entries.")
    return df_bio_all


def load_imaging():
    print("[6/8] Loading Tabular Imaging: DaTSCAN SPECT SBR & FreeSurfer Structural MRI...")
    
    # 1. DaTSCAN SBR
    df_sbr = pd.read_csv(os.path.join(RAW_DIR, "Xing_Core_Lab_-_Quant_SBR_26Sep2026.csv"))
    sbr_cols = ["CAUDATE_R_REF_CWM", "CAUDATE_L_REF_CWM", "PUTAMEN_R_REF_CWM", "PUTAMEN_L_REF_CWM"]
    for c in sbr_cols:
        df_sbr[c] = pd.to_numeric(df_sbr[c], errors="coerce")
        
    df_sbr["datscan_caudate_sbr"] = (df_sbr["CAUDATE_R_REF_CWM"] + df_sbr["CAUDATE_L_REF_CWM"]) / 2.0
    df_sbr["datscan_putamen_sbr"] = (df_sbr["PUTAMEN_R_REF_CWM"] + df_sbr["PUTAMEN_L_REF_CWM"]) / 2.0
    df_sbr["datscan_striatum_sbr"] = (df_sbr["datscan_caudate_sbr"] + df_sbr["datscan_putamen_sbr"]) / 2.0
    
    # Asymmetry index: |R - L| / (mean + 1e-6)
    df_sbr["datscan_putamen_asym"] = np.abs(df_sbr["PUTAMEN_R_REF_CWM"] - df_sbr["PUTAMEN_L_REF_CWM"]) / (df_sbr["datscan_putamen_sbr"] + 1e-6)
    df_sbr["datscan_caudate_putamen_ratio"] = df_sbr["datscan_caudate_sbr"] / (df_sbr["datscan_putamen_sbr"] + 1e-6)
    
    # In PPMI, SC is baseline screening scan: duplicate SC as BL if BL doesn't exist
    sc_scans = df_sbr[df_sbr["EVENT_ID"] == "SC"].copy()
    sc_scans["EVENT_ID"] = "BL"
    df_sbr_combined = pd.concat([df_sbr, sc_scans]).drop_duplicates(["PATNO", "EVENT_ID"], keep="first")
    
    sbr_feat = ["PATNO", "EVENT_ID", "datscan_caudate_sbr", "datscan_putamen_sbr", 
                "datscan_striatum_sbr", "datscan_putamen_asym", "datscan_caudate_putamen_ratio"]
    df_sbr_clean = df_sbr_combined[sbr_feat]
    
    # 2. FreeSurfer 7 MRI Aseg Volumes
    df_fs7 = pd.read_csv(os.path.join(RAW_DIR, "FS7_ASEG_VOL_26Sep2026.csv"))
    etiv = pd.to_numeric(df_fs7["EstimatedTotalIntraCranialVol"], errors="coerce")
    
    df_fs7["mri_etiv"] = etiv
    df_fs7["mri_brain_seg_vol"] = pd.to_numeric(df_fs7["BrainSegVol"], errors="coerce") / etiv
    df_fs7["mri_brain_stem"] = pd.to_numeric(df_fs7["Brain_Stem"], errors="coerce") / etiv
    
    put_vol = pd.to_numeric(df_fs7["Left_Putamen"], errors="coerce") + pd.to_numeric(df_fs7["Right_Putamen"], errors="coerce")
    df_fs7["mri_putamen_vol"] = put_vol / etiv
    
    caud_vol = pd.to_numeric(df_fs7["Left_Caudate"], errors="coerce") + pd.to_numeric(df_fs7["Right_Caudate"], errors="coerce")
    df_fs7["mri_caudate_vol"] = caud_vol / etiv
    
    hippo_vol = pd.to_numeric(df_fs7["Left_Hippocampus"], errors="coerce") + pd.to_numeric(df_fs7["Right_Hippocampus"], errors="coerce")
    df_fs7["mri_hippocampus_vol"] = hippo_vol / etiv
    
    vent_vol = (pd.to_numeric(df_fs7["Left_Lateral_Ventricle"], errors="coerce") + 
                pd.to_numeric(df_fs7["Right_Lateral_Ventricle"], errors="coerce") + 
                pd.to_numeric(df_fs7["3rd_Ventricle"], errors="coerce") + 
                pd.to_numeric(df_fs7["4th_Ventricle"], errors="coerce"))
    df_fs7["mri_ventricles_vol"] = vent_vol / etiv
    
    mri_feat = ["PATNO", "EVENT_ID", "mri_etiv", "mri_brain_seg_vol", "mri_brain_stem", 
                "mri_putamen_vol", "mri_caudate_vol", "mri_hippocampus_vol", "mri_ventricles_vol"]
    df_mri_clean = df_fs7[mri_feat].drop_duplicates(["PATNO", "EVENT_ID"])
    
    df_img = df_sbr_clean.merge(df_mri_clean, on=["PATNO", "EVENT_ID"], how="outer")
    print(f"  Combined imaging features: {len(df_img)} records.")
    return df_img


def build_longitudinal_trajectories_and_targets(df_merged):
    print("[7/8] Sorting Chronological Trajectories & Engineering Longitudinal Targets...")
    
    # Sort chronologically
    df_merged = df_merged.sort_values(["PATNO", "visit_date", "EVENT_ID", "pd_state_clean"]).reset_index(drop=True)
    
    # For target calculation, aggregate at visit date level (mean score per visit date)
    date_level = df_merged.groupby(["PATNO", "visit_date"])["NP3TOT"].mean().reset_index()
    date_level = date_level.sort_values(["PATNO", "visit_date"]).reset_index(drop=True)
    
    # Target lookup mapping: (patno, visit_date) -> {target_12m, target_24m}
    target_map = {}
    grouped = {pat: group.reset_index(drop=True) for pat, group in date_level.groupby("PATNO")}
    
    for patno, group in grouped.items():
        dates = group["visit_date"].values
        scores = group["NP3TOT"].values
        n = len(dates)
        for i in range(n):
            t_i = dates[i]
            cur_score = scores[i]
            days = (dates - t_i) / np.timedelta64(1, "D")
            
            # +12m target window: [250, 480] days, closest to 365
            idx_12 = np.where((days >= 250) & (days <= 480))[0]
            t12 = scores[idx_12[np.argmin(np.abs(days[idx_12] - 365))]] if len(idx_12) > 0 else np.nan
            
            # +24m target window: [550, 910] days, closest to 730
            idx_24 = np.where((days >= 550) & (days <= 910))[0]
            t24 = scores[idx_24[np.argmin(np.abs(days[idx_24] - 730))]] if len(idx_24) > 0 else np.nan
            
            target_map[(patno, t_i)] = (t12, t24)
            
    # Assign targets back to visit records
    t12_list, t24_list = [], []
    for row in df_merged[["PATNO", "visit_date"]].itertuples(index=False):
        t12, t24 = target_map.get((row.PATNO, row.visit_date), (np.nan, np.nan))
        t12_list.append(t12)
        t24_list.append(t24)
        
    df_merged["target_np3tot_12m"] = t12_list
    df_merged["target_np3tot_24m"] = t24_list
    
    # Compute delta changes
    df_merged["delta_np3tot_12m"] = df_merged["target_np3tot_12m"] - df_merged["NP3TOT"]
    df_merged["delta_np3tot_24m"] = df_merged["target_np3tot_24m"] - df_merged["NP3TOT"]
    
    # Binary progression targets: clinically meaningful worsening >= 3.5 points (MCID)
    df_merged["prog_mcid_12m"] = np.where(df_merged["delta_np3tot_12m"].isna(), np.nan, (df_merged["delta_np3tot_12m"] >= 3.5).astype(float))
    df_merged["prog_mcid_24m"] = np.where(df_merged["delta_np3tot_24m"].isna(), np.nan, (df_merged["delta_np3tot_24m"] >= 3.5).astype(float))
    
    # Temporal step features: visit index, elapsed days, and delta_t_days
    df_merged["visit_seq"] = df_merged.groupby("PATNO").cumcount()
    
    # Baseline visit date per patient
    first_dates = df_merged.groupby("PATNO")["visit_date"].transform("min")
    df_merged["elapsed_days_bl"] = (df_merged["visit_date"] - first_dates).dt.days
    df_merged["elapsed_months_bl"] = df_merged["elapsed_days_bl"] / 30.4375
    
    # Delta t between consecutive visits (for GRU-D / RNN)
    prev_date = df_merged.groupby("PATNO")["visit_date"].shift(1)
    df_merged["delta_t_days"] = (df_merged["visit_date"] - prev_date).dt.days.fillna(0.0)
    
    print(f"  Longitudinal trajectories built: {len(df_merged)} total rows.")
    print(f"  Rows with valid 12m target: {df_merged['target_np3tot_12m'].notna().sum()}")
    print(f"  Rows with valid 24m target: {df_merged['target_np3tot_24m'].notna().sum()}")
    return df_merged


def main():
    print("=" * 70)
    print("PPMI MULTIMODAL LONGITUDINAL PIPELINE (PHASE 2, STEPS 2-4)")
    print("=" * 70)
    
    # 1. Cohort & Demographic info
    patient_info = load_cohort_and_split()
    
    # 2. Motor assessments (Part III)
    df_motor = load_motor_and_dates()
    
    # 3. Age & Medication (LEDD)
    df_age, df_ledd = load_age_and_medications()
    df_motor = df_motor.merge(df_age, on=["PATNO", "EVENT_ID"], how="left")
    df_motor = calculate_active_ledd(df_motor, df_ledd)
    
    # 4. Clinical scales (Part I, II, IV, Schwab-England, MoCA, SCOPA, GDS, ESS, RBD, UPSIT)
    df_scales = load_clinical_scales()
    df_merged = df_motor.merge(df_scales, on=["PATNO", "EVENT_ID"], how="left")
    
    # 5. Biospecimens (CSF & SAA)
    df_bio = load_biospecimens()
    df_merged = df_merged.merge(df_bio, on=["PATNO", "EVENT_ID"], how="left")
    
    # 6. Tabular Imaging (DaTSCAN SBR & FreeSurfer MRI)
    df_img = load_imaging()
    df_merged = df_merged.merge(df_img, on=["PATNO", "EVENT_ID"], how="left")
    
    # Merge patient cohort & split info
    df_merged = df_merged.merge(patient_info, on="PATNO", how="left")
    
    # 7. Longitudinal trajectory ordering and future target computation
    df_final = build_longitudinal_trajectories_and_targets(df_merged)
    
    # 8. Export Datasets to Parquet
    print("[8/8] Exporting Datasets to Apache Parquet...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # Save full longitudinal table
    full_path = os.path.join(PROCESSED_DIR, "multimodal_longitudinal.parquet")
    df_final.to_parquet(full_path, index=False)
    print(f"  Saved full dataset: {full_path} ({os.path.getsize(full_path) / (1024*1024):.2f} MB, {len(df_final)} rows)")
    
    # Split datasets
    for split_name in ["train", "val", "test"]:
        sub_df = df_final[df_final["split"] == split_name].copy()
        split_path = os.path.join(PROCESSED_DIR, f"{split_name}_longitudinal.parquet")
        sub_df.to_parquet(split_path, index=False)
        print(f"  Saved {split_name} split: {split_path} ({len(sub_df)} rows, {sub_df['PATNO'].nunique()} patients)")
        
    # Latest valid visit tabular datasets for XGBoost / LightGBM baseline (Model 1)
    # We take each patient's latest visit that has target_np3tot_12m not null
    labeled_df = df_final[df_final["target_np3tot_12m"].notna()].copy()
    latest_tabular = labeled_df.sort_values(["PATNO", "visit_date"]).groupby("PATNO").last().reset_index()
    
    for split_name in ["train", "val", "test"]:
        sub_tab = latest_tabular[latest_tabular["split"] == split_name].copy()
        tab_path = os.path.join(PROCESSED_DIR, f"{split_name}_tabular_latest.parquet")
        sub_tab.to_parquet(tab_path, index=False)
        print(f"  Saved {split_name} tabular latest: {tab_path} ({len(sub_tab)} patients)")
        
    # Quality Assurance & Verification
    print("\n" + "=" * 70)
    print("DATA ENGINEERING PIPELINE SUMMARY & QUALITY ASSURANCE")
    print("=" * 70)
    print(f"Total Cohort Rows: {len(df_final):,}")
    print(f"Unique Patients: {df_final['PATNO'].nunique():,}")
    print(f"Total Labeled 12-Month Target Rows: {df_final['target_np3tot_12m'].notna().sum():,}")
    print(f"Total Labeled 24-Month Target Rows: {df_final['target_np3tot_24m'].notna().sum():,}")
    print(f"Total Complete 12m & 24m Target Rows: {(df_final['target_np3tot_12m'].notna() & df_final['target_np3tot_24m'].notna()).sum():,}")
    
    # Progression rates (MCID >= 3.5 points)
    pd_prog_12 = df_final[df_final["COHORT_DEFINITION"] == "Parkinson's Disease"]["prog_mcid_12m"].dropna()
    print(f"PD 12-Month Progression Rate (>= 3.5 pts worsening): {pd_prog_12.mean() * 100:.1f}% ({int(pd_prog_12.sum())}/{len(pd_prog_12)})")
    
    pd_prog_24 = df_final[df_final["COHORT_DEFINITION"] == "Parkinson's Disease"]["prog_mcid_24m"].dropna()
    print(f"PD 24-Month Progression Rate (>= 3.5 pts worsening): {pd_prog_24.mean() * 100:.1f}% ({int(pd_prog_24.sum())}/{len(pd_prog_24)})")
    
    hc_prog_12 = df_final[df_final["COHORT_DEFINITION"] == "Healthy Control"]["prog_mcid_12m"].dropna()
    print(f"HC 12-Month Progression Rate: {hc_prog_12.mean() * 100:.1f}% ({int(hc_prog_12.sum())}/{len(hc_prog_12)})")
    
    print("\nData Engineering Phase 2 COMPLETE! Ready for Phase 3 Modeling.")
    print("=" * 70)


if __name__ == "__main__":
    main()
