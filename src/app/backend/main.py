from __future__ import annotations

import io

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd

from .config import STATIC_DIR
from . import database as db
from . import cache as disk_cache
from .grid import all_grid_cells, grid_id_bbox
from .ml.predictor import load_bundles, models_status

app = FastAPI(title="ARGO SENTINEL API", version="2.0")


@app.on_event("startup")
def warmup():
    # Load disk cache instantly; heavy DuckDB scan runs in background
    db.all_grid_stats("n_obs")
    db.dataset_meta()
    disk_cache.start_background_warmup(db.warmup_full_cache)
    # Load RF bundles from trained_models/ (bundles already built — no auto_build)
    bundles = load_bundles(auto_build=False)
    print(f"[startup] RF bundles loaded: {list(bundles.keys()) or 'NONE — QC will use z-score fallback'}", flush=True)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/meta")
def meta():
    return db.dataset_meta()


@app.get("/api/models")
def models():
    return {"models": models_status()}


@app.get("/api/grids")
def grids(metric: str = Query("n_obs")):
    stats = {r["grid_id"]: r for r in db.all_grid_stats(metric)}
    cells = []
    for cell in all_grid_cells():
        gid = cell["grid_id"]
        s = stats.get(gid)
        cells.append({
            **cell,
            "has_data": s is not None,
            "n_obs": s["n_obs"] if s else 0,
            "mean_temp": s.get("mean_temp") if s else None,
            "mean_psal": s.get("mean_psal") if s else None,
            "mean_depth": s.get("mean_depth") if s else None,
            "metric_value": s.get("metric_value") if s else None,
        })
    return {"metric": metric, "cells": cells}


@app.get("/api/grids/{grid_id}")
def grid_detail(grid_id: int):
    detail = db.grid_detail(grid_id)
    if detail is None:
        raise HTTPException(404, f"No data for grid {grid_id}")
    return {"bbox": grid_id_bbox(grid_id), **detail}


@app.get("/api/grids/{grid_id}/profiles")
def grid_profiles(grid_id: int, limit: int = Query(400, le=2000)):
    return {"grid_id": grid_id, "profiles": db.grid_profiles(grid_id, limit)}


@app.get("/api/grids/status")
def grids_status():
    """Per-grid QC pipeline status (OK / NO_DATA / PARTIAL / FAILED)."""
    from .config import PIPELINE_GRID_SUMMARY
    if not PIPELINE_GRID_SUMMARY.exists():
        return {"available": False, "grids": {}}
    import pandas as pd
    df = pd.read_csv(PIPELINE_GRID_SUMMARY)
    grids = {}
    for _, r in df.iterrows():
        grids[int(r.grid_id)] = {
            "grid_status": str(r.grid_status),
            "n_obs": int(r.n_obs) if pd.notna(r.n_obs) else 0,
            "pct_temp_match": float(r.pct_temp_match) if pd.notna(r.pct_temp_match) else None,
            "pct_sal_match": float(r.pct_sal_match) if pd.notna(r.pct_sal_match) else None,
        }
    return {"available": True, "grids": grids}


@app.get("/api/grids/{grid_id}/analysis")
def grid_analysis(grid_id: int, sample: int = Query(120, le=300)):
    result = db.grid_analysis(grid_id, sample_size=sample)
    if result is None:
        raise HTTPException(404, f"No data for grid {grid_id}")
    return {"bbox": grid_id_bbox(grid_id), **result}


@app.get("/api/grids/{grid_id}/heatmap")
def grid_heatmap(grid_id: int):
    return {"grid_id": grid_id, **db.grid_heatmap(grid_id)}


@app.post("/api/qc/predict")
def qc_predict(body: dict):
    required = ["lat", "lon", "depth", "temperature", "salinity"]
    missing = [k for k in required if k not in body]
    if missing:
        raise HTTPException(400, f"Missing fields: {missing}")
    month = int(body.get("month", 6))
    return db.predict_qc(
        float(body["lat"]),
        float(body["lon"]),
        float(body["depth"]),
        float(body["temperature"]),
        float(body.get("salinity") or body.get("psal", 35.0)),
        month,
    )


@app.post("/api/qc/batch")
async def qc_batch(file: UploadFile = File(...)):
    content = await file.read()
    try:
        if file.filename and file.filename.lower().endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse file: {e}") from e

    colmap = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in colmap:
                return colmap[n]
        return None

    lat_c, lon_c = pick("lat", "latitude"), pick("lon", "longitude")
    if not lat_c or not lon_c:
        raise HTTPException(400, "CSV needs lat/lon columns")

    results = []
    for _, row in df.head(500).iterrows():
        lat, lon = float(row[lat_c]), float(row[lon_c])
        depth = float(row[pick("depth", "pressure")]) if pick("depth", "pressure") else 0.0
        temp = float(row[pick("temperature", "temp")]) if pick("temperature", "temp") else 20.0
        psal = float(row[pick("salinity", "psal", "sal")]) if pick("salinity", "psal", "sal") else 35.0
        month = int(row[pick("month")]) if pick("month") else 6
        pred = db.predict_qc(lat, lon, depth, temp, psal, month)
        results.append(pred)
    return {"count": len(results), "results": results}


@app.get("/")
def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(404, "Frontend not found")
    return FileResponse(index_path)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
