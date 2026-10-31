"""Read-only inference, independent of Streamlit and the training pipelines.

Models are loaded once in a worker so the status API remains responsive while
the React introduction plays. Model weights, splits and preprocessing follow
the existing dashboard; no training or checkpoint changes are performed here.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import logging
import math
import os
from pathlib import Path
import sys
from threading import Lock, Thread
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


class InferenceService:
    def __init__(self) -> None:
        self._lock = Lock()
        self._inference_lock = Lock()
        self._state = {"state": "idle", "progress": 0,
                       "stage": "Prêt à initialiser", "loaded_models": [],
                       "error": None, "started_at": None, "ready_at": None}

    def status(self) -> dict:
        with self._lock:
            return {**self._state, "loaded_models": list(self._state["loaded_models"])}

    def start(self) -> dict:
        with self._lock:
            if self._state["state"] not in ("idle", "error"):
                return {**self._state}
            self._state.update(state="loading", progress=2,
                               stage="Initialisation du moteur d’analyse", error=None,
                               loaded_models=[],
                               started_at=datetime.now(timezone.utc).isoformat())
            Thread(target=self._load, name="neurovia-model-loader", daemon=True).start()
            return {**self._state}

    def _update(self, progress: int, stage: str, model: str | None = None) -> None:
        with self._lock:
            self._state.update(progress=progress, stage=stage)
            if model:
                self._state["loaded_models"].append(model)

    def _load(self) -> None:
        try:
            # Training modules import matplotlib; keep its writable cache local.
            cache = ROOT / ".venv" / "matplotlib-cache"
            cache.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("MPLCONFIGDIR", str(cache))
            os.environ.setdefault("MPLBACKEND", "Agg")
            src = str(ROOT / "src")
            if src not in sys.path:
                sys.path.insert(0, src)
            import numpy as np
            import pandas as pd
            import torch
            from sklearn.preprocessing import StandardScaler
            from train_model2_lstm import LongitudinalBiLSTM, FEATURE_COLS
            from train_fusion_models import (LateFusionModel, ALL_FEATURE_COLS,
                                            CLINICAL_COLS, IMAGING_COLS, DEMO_COLS)
            from retained_xgboost import load_retained_regressors, predict_patient

            self.np, self.pd, self.torch = np, pd, torch
            self.all_cols, self.lstm_cols = ALL_FEATURE_COLS, FEATURE_COLS
            self.xgb_predict = predict_patient
            self._update(18, "Lecture de la cohorte et des historiques de visites")
            data = ROOT / "data" / "processed"
            self.tabular = pd.read_parquet(data / "test_tabular_latest.parquet")
            self.longitudinal = pd.read_parquet(data / "test_longitudinal.parquet")
            train = pd.read_parquet(data / "train_longitudinal.parquet")
            self._update(32, "Préparation des trajectoires et normalisation")
            sequences = []
            for cols in (ALL_FEATURE_COLS, FEATURE_COLS):
                medians = train[cols].median().fillna(0.0)
                scaler = StandardScaler().fit(train[cols].fillna(medians))
                frame = self.longitudinal.sort_values(["PATNO", "visit_date"]).reset_index(drop=True).copy()
                frame[cols] = frame.groupby("PATNO")[cols].ffill().fillna(medians)
                frame[cols] = scaler.transform(frame[cols])
                sequences.append(frame)
            self.all_sequences, self.lstm_sequences = sequences

            self._update(48, "Chargement du modèle de fusion multimodale")
            self.late_model = LateFusionModel(clin_dim=len(CLINICAL_COLS),
                img_dim=len(IMAGING_COLS), demo_dim=len(DEMO_COLS), hidden_dim=48, dropout=0.25)
            self.late_model.load_state_dict(torch.load(ROOT / "models/late_fusion.pt",
                                                       map_location="cpu", weights_only=True))
            self.late_model.eval()
            self._update(65, "Chargement du modèle longitudinal BiLSTM", "late_fusion")
            self.bilstm_model = LongitudinalBiLSTM(input_dim=len(FEATURE_COLS),
                hidden_dim=64, num_layers=2, dropout=0.25)
            checkpoint = torch.load(ROOT / "models/model2_bilstm.pt",
                                    map_location="cpu", weights_only=False)
            self.bilstm_model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
            self.bilstm_model.eval()
            self._update(80, "Vérification des checkpoints XGBoost retenus", "bilstm")
            self.xgb = load_retained_regressors(seed=42)
            self._update(94, "Préparation du tableau de bord", "xgboost")
            self.benchmarks = pd.read_csv(ROOT / "report/benchmark_results.csv")
            self.visits = self.longitudinal.groupby("PATNO").size().to_dict()
            self.patient_index = [self._patient_row(row) for _, row in self.tabular.iterrows()]
            with self._lock:
                self._state.update(state="ready", progress=100, stage="Votre espace est prêt",
                                   ready_at=datetime.now(timezone.utc).isoformat())
        except Exception:
            logger.exception("Unable to initialize inference service")
            with self._lock:
                self._state.update(state="error", stage="Chargement interrompu",
                    error="Les modèles n’ont pas pu être chargés. Vérifiez les fichiers et consultez les journaux du serveur.")

    def _patient_row(self, row) -> dict:
        return {"id": int(row.PATNO), "age": number(row.age_at_visit),
                "sex": "Homme" if row.is_male == 1 else "Femme" if row.is_male == 0 else "Non renseigné",
                "score": number(row.NP3TOT), "stage": number(row.nhy_stage),
                "visits": int(self.visits.get(row.PATNO, 0)),
                "last_visit": row.visit_date.strftime("%Y-%m-%d"),
                "rapid_progression": number(row.prog_mcid_12m) == 1,
                "has_outcome": number(row.prog_mcid_12m) is not None,
                "putamen_sbr": number(row.datscan_putamen_sbr),
                "cohort": str(row.get("COHORT_DEFINITION", "PPMI"))}

    def patients(self, search: str = "", group: str = "all") -> list[dict]:
        rows = self.patient_index
        if search:
            rows = [r for r in rows if search.lower() in str(r["id"]).lower()]
        if group == "rapid":
            rows = [r for r in rows if r["rapid_progression"]]
        elif group == "stable":
            rows = [r for r in rows if r["has_outcome"] and not r["rapid_progression"]]
        elif group == "history":
            rows = [r for r in rows if r["visits"] >= 4]
        elif group == "imaging":
            rows = [r for r in rows if r["putamen_sbr"] is not None and r["putamen_sbr"] < 1]
        return rows

    def overview(self) -> dict:
        frame = self.tabular
        monthly = self.longitudinal.dropna(subset=["NP3TOT", "elapsed_months_bl"]).copy()
        monthly["period"] = (monthly.elapsed_months_bl / 12).round().astype(int) * 12
        trend = monthly.groupby("period").agg(score=("NP3TOT", "mean"), n=("NP3TOT", "count"))
        return {"patients": len(frame), "visits": len(self.longitudinal),
                "models": 3, "mean_score": number(frame.NP3TOT.mean()),
                "mean_age": number(frame.age_at_visit.mean()),
                "rapid_count": int((frame.prog_mcid_12m == 1).sum()),
                "rapid_rate": number((frame.prog_mcid_12m == 1).sum() / frame.prog_mcid_12m.notna().sum() * 100),
                "imaging_count": int(frame.datscan_putamen_sbr.notna().sum()),
                "trend": [{"month": int(i), "score": number(r.score), "n": int(r.n)}
                          for i, r in trend.iterrows() if 0 <= i <= 120],
                "stages": [{"stage": str(k), "count": int(v)} for k, v in frame.nhy_stage.value_counts().sort_index().items()],
                "ready_at": self.status()["ready_at"]}

    @lru_cache(maxsize=128)
    def prediction(self, patient_id: int) -> dict:
        # Serialize access to the shared PyTorch resources and preserve deterministic inference.
        with self._inference_lock:
            return self._predict(patient_id)

    def _predict(self, patient_id: int) -> dict:
        pd, torch = self.pd, self.torch
        rows = self.tabular[self.tabular.PATNO == patient_id]
        if rows.empty:
            raise KeyError(patient_id)
        row = rows.iloc[0]
        cutoff = row.visit_date
        def history(frame):
            return frame[(frame.PATNO == patient_id) & (frame.visit_date <= cutoff)].sort_values("visit_date")
        raw = history(self.longitudinal)
        all_seq, lstm_seq = history(self.all_sequences), history(self.lstm_sequences)
        length = len(all_seq)
        if length == 0:
            raise ValueError("Patient has no eligible trajectory")
        x_all = torch.tensor(all_seq[self.all_cols].values, dtype=torch.float32).unsqueeze(0)
        x_lstm = torch.tensor(lstm_seq[self.lstm_cols].values, dtype=torch.float32).unsqueeze(0)
        mask = torch.ones((1, length), dtype=torch.bool)
        lengths = torch.tensor([length], dtype=torch.long)
        with torch.inference_mode():
            f12, f24, flogit, gates = self.late_model(x_all, mask, lengths,
                torch.tensor([all_seq.NP3TOT.iloc[-1]], dtype=torch.float32))
            b12, b24, blogit, attention = self.bilstm_model(x_lstm, mask, lengths,
                torch.tensor([lstm_seq.NP3TOT.iloc[-1]], dtype=torch.float32))
        current = float(row.NP3TOT)
        index_month = float(row.elapsed_months_bl)
        index_date = cutoff.strftime("%Y-%m-%d")
        def horizon(score, months):
            return {"score": number(score.item()), "delta": number(score.item() - current),
                    "index_score": current, "index_month": index_month, "index_date": index_date,
                    "month": index_month + months}
        predictions = {
            "late_fusion": {"12m": horizon(f12, 12), "24m": horizon(f24, 24),
                            "probability": number(torch.sigmoid(flogit).item())},
            "bilstm": {"12m": horizon(b12, 12), "24m": horizon(b24, 24),
                       "probability": number(torch.sigmoid(blogit).item())},
            "xgboost": {"probability": None}}
        for key, pred in self.xgb_predict(patient_id, self.xgb).items():
            predictions["xgboost"][key] = ({"score": number(pred["score"]),
                "delta": number(pred["score"] - pred["index_score"]),
                "index_score": number(pred["index_score"]), "index_month": number(pred["index_month"]),
                "index_date": pred["index_date"].strftime("%Y-%m-%d"),
                "month": number(pred["index_month"] + int(key[:-1]))} if pred else None)
        weights = attention[0].tolist()
        visit_rows = [{"date": visit.visit_date.strftime("%Y-%m-%d"),
            "month": number(visit.elapsed_months_bl), "score": number(visit.NP3TOT),
            "stage": number(visit.nhy_stage), "ledd": number(visit.ledd),
            "moca": number(visit.moca_total), "putamen_sbr": number(visit.datscan_putamen_sbr),
            "attention": number(weights[i] * 100)} for i, (_, visit) in enumerate(raw.iterrows())]
        return {"patient": self._patient_row(row), "predictions": predictions,
                "history": visit_rows,
                "gating": {"clinical": number(gates[0, 0].item() * 100),
                           "imaging": number(gates[0, 1].item() * 100)},
                "biomarkers": {key: number(row.get(key)) for key in
                    ["np3_tremor", "np3_rigidity", "np3_bradykinesia", "np3_axial", "moca_total",
                     "ledd", "datscan_caudate_sbr", "datscan_putamen_sbr", "datscan_striatum_sbr"]},
                "provenance": "Checkpoints existants ; prétraitement identique au tableau de bord historique. Les horizons XGBoost peuvent avoir des visites index différentes."}

    def benchmark_rows(self) -> list[dict]:
        # Let pandas handle nullable scalar conversion and ISO timestamps before JSON serialization.
        import json
        return json.loads(self.benchmarks.to_json(orient="records"))


service = InferenceService()
