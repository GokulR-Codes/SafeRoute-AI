"""
Namma Safe BLR — FastAPI Backend
Run:  uvicorn app:app --reload --port 8000
"""

import os, sys, json, pickle
from datetime import datetime
from typing   import Optional

import pandas as pd
import numpy  as np

# ── ensure backend/ is on the path ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from fastapi            import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses  import FileResponse
from pydantic           import BaseModel, Field

import data_processing as dp
import safety_score    as ss
import risk_model      as rm
import route_engine    as re

# ─── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Namma Safe BLR API",
    description = "AI-powered safety navigation for Bangalore night travel",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Startup: load data & model ───────────────────────────────────────────────
BASE    = os.path.dirname(__file__)
DATA_PATH  = os.path.join(BASE, "../data/bangalore_crime_dataset.csv")
MODEL_PATH = os.path.join(BASE, "../models/risk_model.pkl")

crime_df    = None
density_grid = []
model_bundle = None
incidents_log: list = []    # in-memory crowd-reported incidents

@app.on_event("startup")
async def startup():
    global crime_df, density_grid, model_bundle

    # ── Load & process crime data ─────────────────────────────────────────────
    crime_df     = dp.load_and_clean(DATA_PATH)
    _, _clusters = dp.cluster_hotspots(crime_df)
    density_grid = dp.compute_density_grid(crime_df)
    ss.load_density_grid(density_grid)

    # ── Train / load ML model ─────────────────────────────────────────────────
    if os.path.exists(MODEL_PATH):
        model_bundle = rm.load_model(MODEL_PATH)
        print(f"✅ Loaded model: {model_bundle['model_name']}")
    else:
        print("Training ML model (first run)…")
        model_bundle = rm.train(DATA_PATH, MODEL_PATH)

    print("🚀 Namma Safe BLR API ready!")


# ─── Pydantic schemas ─────────────────────────────────────────────────────────
class RouteRequest(BaseModel):
    src_lat: float = Field(..., example=12.9716)
    src_lon: float = Field(..., example=77.5946)
    dst_lat: float = Field(..., example=12.9352)
    dst_lon: float = Field(..., example=77.6245)
    hour:    int   = Field(22, ge=0, le=23)
    algorithm: str = Field("astar", pattern="^(astar|dijkstra)$")

class RiskRequest(BaseModel):
    latitude:         float = Field(..., example=12.9716)
    longitude:        float = Field(..., example=77.5946)
    hour:             int   = Field(22, ge=0, le=23)
    lighting_score:   Optional[float] = None
    cctv_score:       Optional[float] = None
    crowd_density:    Optional[float] = None
    police_proximity: Optional[float] = None

class IncidentReport(BaseModel):
    latitude:    float
    longitude:   float
    description: str
    severity:    int = Field(5, ge=1, le=10)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Namma Safe BLR", "version": "1.0"}


@app.post("/find-safe-route", tags=["Navigation"])
def find_safe_route(req: RouteRequest):
    """Find the safest route between two points."""
    result = re.find_safe_route(
        req.src_lat, req.src_lon,
        req.dst_lat, req.dst_lon,
        hour      = req.hour,
        algorithm = req.algorithm,
    )
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/predict-risk", tags=["ML"])
def predict_risk(req: RiskRequest):
    """Predict crime risk level for a location using the ML model."""
    from data_processing import time_risk_factor

    time_risk = time_risk_factor(req.hour)
    crime_density = ss._nearest_density(req.latitude, req.longitude)

    features = {
        "severity_norm":     crime_density,
        "time_risk":         time_risk,
        "lighting_score":    req.lighting_score  if req.lighting_score  is not None else 0.5,
        "cctv_score":        req.cctv_score      if req.cctv_score      is not None else 0.4,
        "crowd_density":     req.crowd_density   if req.crowd_density   is not None else 0.5,
        "police_proximity":  req.police_proximity if req.police_proximity is not None else 0.5,
        "is_night":          1 if (req.hour >= 20 or req.hour < 6) else 0,
        "hour":              req.hour,
    }

    prediction = rm.predict_risk(model_bundle, features)
    safety     = ss.compute_safety_score(req.latitude, req.longitude, req.hour,
                                         req.lighting_score, req.cctv_score, req.crowd_density)

    return {**prediction, **safety, "features_used": features}


@app.get("/get-crime-heatmap", tags=["Visualization"])
def get_crime_heatmap(limit: int = 500):
    """Return crime density grid for heatmap rendering."""
    data = density_grid[:limit]
    return {"heatmap": data, "total_cells": len(density_grid)}


@app.get("/get-crime-points", tags=["Visualization"])
def get_crime_points(limit: int = 300):
    """Return raw crime records for marker rendering."""
    if crime_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    sample = crime_df.sample(min(limit, len(crime_df)), random_state=42)
    return {"crimes": sample[["latitude","longitude","crime_type","crime_severity","hour","area"]].to_dict("records")}


@app.post("/report-incident", tags=["Community"])
def report_incident(report: IncidentReport):
    """Crowd-sourced incident reporting."""
    entry = {
        "id":          len(incidents_log) + 1,
        "latitude":    report.latitude,
        "longitude":   report.longitude,
        "description": report.description,
        "severity":    report.severity,
        "timestamp":   datetime.utcnow().isoformat(),
    }
    incidents_log.append(entry)
    return {"status": "reported", "incident_id": entry["id"]}


@app.get("/get-incidents", tags=["Community"])
def get_incidents():
    """Fetch all crowd-reported incidents."""
    return {"incidents": incidents_log, "total": len(incidents_log)}


@app.get("/analytics/summary", tags=["Analytics"])
def analytics_summary():
    """High-level analytics summary."""
    if crime_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    by_area  = crime_df.groupby("area")["crime_type"].count().sort_values(ascending=False).head(10).to_dict()
    by_type  = crime_df["crime_type"].value_counts().to_dict()
    by_hour  = crime_df.groupby("hour")["crime_type"].count().to_dict()
    night_d  = crime_df[crime_df["is_night"]==1]["crime_type"].count()
    day_d    = crime_df[crime_df["is_night"]==0]["crime_type"].count()

    return {
        "total_records":    len(crime_df),
        "by_area":          by_area,
        "by_crime_type":    by_type,
        "by_hour":          {str(k): int(v) for k, v in by_hour.items()},
        "night_crimes":     int(night_d),
        "day_crimes":       int(day_d),
        "model_name":       model_bundle["model_name"] if model_bundle else "Not loaded",
        "model_accuracy":   round(model_bundle["test_accuracy"]*100, 2) if model_bundle else 0,
    }
