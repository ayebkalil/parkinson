r"""Run with: .venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000"""
from __future__ import annotations

import csv
import io
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .service import ROOT, service
from .experiment_results import audited_benchmarks

app = FastAPI(title="Neurovia · Parkinson Research API", version="1.0.0",
              description="Read-only PPMI cohort exploration and existing model inference.")
app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


def require_ready():
    state = service.status()
    if state["state"] != "ready":
        raise HTTPException(503, detail={"message": "Le moteur est en cours de préparation.",
                                        "state": state["state"]}, headers={"Retry-After": "2"})
    return service


ReadyService = Annotated[object, Depends(require_ready)]


@app.get("/api/health")
def health():
    return {"status": "ok", "models": service.status()["state"]}


@app.get("/api/status")
def status():
    return service.status()


@app.post("/api/initialize", status_code=202)
def initialize():
    return service.start()


@app.get("/api/overview")
def overview(engine: ReadyService):
    return engine.overview()


@app.get("/api/patients")
def patients(engine: ReadyService,
             search: str = Query("", max_length=30),
             group: Literal["all", "rapid", "stable", "history", "imaging"] = "all"):
    items = engine.patients(search.strip(), group)
    return {"items": items, "total": len(items)}


@app.get("/api/patients/{patient_id}")
def patient(patient_id: int, engine: ReadyService):
    try:
        return engine.prediction(patient_id)
    except KeyError:
        raise HTTPException(404, detail="Ce patient ne fait pas partie de la cohorte.") from None
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from exc


@app.get("/api/benchmarks")
def benchmarks(engine: ReadyService, protocol: Literal['audited','historical']='audited'):
    if protocol=='audited':
        result=audited_benchmarks(ROOT)
        if result is not None:return result
    return {"items": engine.benchmark_rows(), "historical": True, "protocol": "historical",
            "note": "Archive historique : les cibles à 24 mois étaient mal rattachées à la visite index pour 444 des 462 patients. Ces chiffres ne constituent pas une référence valide pour mesurer les gains du protocole corrigé. Les expériences auditées apparaîtront après leur vérification finale."}


@app.get("/api/patients/{patient_id}/export")
def export_patient(patient_id: int, engine: ReadyService):
    try:
        result = engine.prediction(patient_id)
    except KeyError:
        raise HTTPException(404, detail="Patient introuvable.") from None
    stream = io.StringIO(newline="")
    fields = ["patient_id", "model", "horizon", "score", "delta", "index_date", "index_score"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for model, horizons in result["predictions"].items():
        for horizon in ("12m", "24m"):
            prediction = horizons[horizon]
            writer.writerow({"patient_id": patient_id, "model": model, "horizon": horizon,
                **{key: prediction.get(key) if prediction else None
                   for key in ["score", "delta", "index_date", "index_score"]}})
    return Response(content="\ufeff" + stream.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="neurovia-patient-{patient_id}.csv"'})


# Production uses the same origin for React and the API; no separate frontend server is required.
dist = ROOT / "frontend" / "dist"
if dist.is_dir():
    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
