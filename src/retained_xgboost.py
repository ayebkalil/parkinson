"""Load the retained XGBoost regressors with their aligned prediction rows.

The two horizons can have different index visits. Keep the original models in
models/ untouched, and never substitute an ineligible 24-month index row.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import xgboost as xgb

from train_model1_xgboost import FEATURE_COLS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROUND2 = PROJECT_ROOT / "experiments/model12_round2_20261023"
CORRECTED_DATA = PROJECT_ROOT / "experiments/model12_20261023/data"


def load_retained_regressors(seed=42):
    manifest = json.loads((ROUND2 / "retained_checkpoints.json").read_text(encoding="utf-8"))
    selection = json.loads((ROUND2 / "final_selection.json").read_text(encoding="utf-8"))
    frozen = json.loads((ROUND2 / "final_freeze.json").read_text(encoding="utf-8"))["checkpoints"]
    output = {"seed": seed}
    for horizon in (12, 24):
        task = f"{horizon}m"
        entry = next((item for item in manifest if item["model"] == "xgboost"
                      and item["task"] == task and item["seed"] == seed), None)
        if entry is None or entry["config"] != selection["retained"]["xgboost"][task]:
            raise ValueError(f"No retained XGBoost checkpoint for {task}, seed {seed}")
        checkpoint = PROJECT_ROOT / entry["checkpoint"]
        relative = str(checkpoint.relative_to(ROUND2)).replace("\\", "/")
        expected = next((value for key, value in frozen.items()
                         if key.replace("\\", "/") == relative), None)
        if expected is None or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checkpoint integrity check failed: {checkpoint}")
        model = xgb.XGBRegressor()
        model.load_model(checkpoint)
        rows = pd.read_parquet(CORRECTED_DATA / f"test_tabular_{horizon}m.parquet")
        if rows.PATNO.duplicated().any() or not set(FEATURE_COLS).issubset(rows):
            raise ValueError(f"Invalid aligned {task} rows")
        output[task] = (model, rows.set_index("PATNO", drop=False), str(checkpoint))
    return output


def predict_patient(patient_id, artifacts):
    predictions = {}
    for horizon in (12, 24):
        task = f"{horizon}m"
        model, rows, _ = artifacts[task]
        if patient_id not in rows.index:
            predictions[task] = None
            continue
        row = rows.loc[patient_id]
        value = float(model.predict(pd.DataFrame([row[FEATURE_COLS]]))[0])
        predictions[task] = {"score": value, "index_date": row["visit_date"],
                             "index_score": float(row["NP3TOT"]),
                             "index_month": float(row["elapsed_months_bl"])}
    return predictions
