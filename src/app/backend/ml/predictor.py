"""Load chunked RandomForest bundles and run QC inference."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..config import MODEL_DIR, TARGET_COLS, TRAIN_PATH
from .preprocess import (
    classify_columns,
    collect_stats,
    collect_stats_duckdb,
    feature_columns_for_target,
    get_schema_columns,
    prep_chunk,
    QC_FLAG_LABELS,
)

_bundles: dict[str, dict] = {}
_load_attempted = False


def _bundle_path(target: str) -> Path:
    return MODEL_DIR / f"ALL_REGIONS_{target}_bundle.joblib"


def _plain_model_path(target: str) -> Path:
    return MODEL_DIR / f"ALL_REGIONS_{target}_RandomForest.joblib"


def build_bundle_for_target(target: str, train_path: str | Path | None = None) -> dict:
    """Stats pass + attach existing plain RF model → save bundle."""
    train_path = Path(train_path or TRAIN_PATH)
    model_path = _plain_model_path(target)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    pf, train_cols = get_schema_columns(str(train_path))
    feature_cols = feature_columns_for_target(train_cols, target)
    numeric_cols, categorical_cols = classify_columns(pf, feature_cols)

    stats_cache = MODEL_DIR / f"preprocess_stats_{target}.joblib"
    if stats_cache.exists():
        import joblib
        stats = joblib.load(stats_cache)
    else:
        print(f"  Collecting encoding stats via DuckDB for {target}…", flush=True)
        stats = collect_stats_duckdb(
            str(train_path), feature_cols, numeric_cols, categorical_cols, target
        )
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(stats, stats_cache)
    model = joblib.load(model_path)
    bundle = {
        "model": model,
        "target_col": target,
        **stats,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, _bundle_path(target))
    return bundle


def load_bundles(force_rebuild: bool = False, auto_build: bool = False) -> dict[str, dict]:
    global _bundles, _load_attempted
    if _bundles and not force_rebuild:
        return _bundles

    _bundles = {}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for target in TARGET_COLS:
        bp = _bundle_path(target)
        if bp.exists() and not force_rebuild:
            try:
                print(f"Loading RF bundle: {bp.name}...", flush=True)
                bundle = joblib.load(bp)
                required = ("model", "feature_cols", "numeric_cols", "cat_encoders", "numeric_means", "target_encoder")
                missing = [k for k in required if k not in bundle]
                if missing:
                    raise KeyError(f"bundle missing keys: {missing}")
                _bundles[target] = bundle
                print(f"  OK {target}: {len(bundle['feature_cols'])} features, {len(bundle['target_encoder'].classes_)} classes", flush=True)
            except Exception as e:
                print(f"Warning: failed to load bundle {bp}: {e}", flush=True)
            continue
        if auto_build and _plain_model_path(target).exists():
            try:
                print(f"Building bundle for {target} (one-time stats scan)...", flush=True)
                _bundles[target] = build_bundle_for_target(target)
            except Exception as e:
                print(f"Warning: could not build bundle for {target}: {e}", flush=True)
        elif not bp.exists():
            print(f"Warning: missing RF bundle {bp}", flush=True)

    _load_attempted = True
    return _bundles


def models_status() -> list[dict]:
    load_bundles()
    rows = []
    for target in TARGET_COLS:
        bundle = _bundles.get(target)
        rows.append({
            "target": target,
            "bundle_loaded": bundle is not None,
            "bundle_path": str(_bundle_path(target)),
            "plain_model_path": str(_plain_model_path(target)),
            "plain_exists": _plain_model_path(target).exists(),
            "n_features": len(bundle["feature_cols"]) if bundle else None,
            "n_classes": len(bundle["target_encoder"].classes_) if bundle else None,
            "classes": bundle["target_encoder"].classes_.tolist() if bundle else [],
        })
    return rows


def build_inference_row(
    lat: float,
    lon: float,
    depth: float,
    temperature: float,
    salinity: float,
    month: int,
    grid_id: int,
    grid_context: dict | None = None,
) -> pd.DataFrame:
    """Assemble one raw feature row matching train parquet columns."""
    ctx = grid_context or {}
    t_mean = ctx.get("temp_grid_mean") or ctx.get("clim_temp_mean") or temperature
    t_std = ctx.get("temp_grid_std") or ctx.get("clim_temp_std") or 1.0
    s_mean = ctx.get("sal_grid_mean") or ctx.get("clim_psal_mean") or salinity
    s_std = ctx.get("sal_grid_std") or ctx.get("clim_psal_std") or 0.3
    t_std = t_std if t_std and t_std > 1e-6 else 1.0
    s_std = s_std if s_std and s_std > 1e-6 else 0.3

    tz = (temperature - t_mean) / t_std
    sz = (salinity - s_mean) / s_std

    row = {
        "grid_id": grid_id,
        "lat": lat,
        "lon": lon,
        "depth": depth,
        "pressure": depth,
        "temperature": temperature,
        "salinity": salinity,
        "month": month,
        "year": 2020,
        "season": (month - 1) // 3 + 1,
        "day": 15,
        "temp_grid_mean": t_mean,
        "temp_grid_std": t_std,
        "sal_grid_mean": s_mean,
        "sal_grid_std": s_std,
        "n_obs": ctx.get("n_obs", 1000),
        "temp_zscore": tz,
        "sal_zscore": sz,
        "temp_zscore_abs": abs(tz),
        "sal_zscore_abs": abs(sz),
        "temp_z": tz,
        "psal_z": sz,
        "temp_min": ctx.get("min_temp", temperature - 2),
        "temp_max": ctx.get("max_temp", temperature + 2),
        "temp_mean": ctx.get("mean_temp", temperature),
        "psal_min": ctx.get("min_psal", salinity - 0.5),
        "psal_max": ctx.get("max_psal", salinity + 0.5),
        "psal_mean": ctx.get("mean_psal", salinity),
        "pres_mean": depth,
        "depth_bin": "0-200" if depth < 200 else "200+",
        "source_id": "INFERENCE",
        "source_file": "dashboard_inference",
        "season_name": "INFER",
        "temp_z_flag": 1 if abs(tz) < 2 else 2,
        "sal_z_flag": 1 if abs(sz) < 2 else 2,
    }
    return pd.DataFrame([row])


def predict_target(
    target: str,
    lat: float,
    lon: float,
    depth: float,
    temperature: float,
    salinity: float,
    month: int,
    grid_id: int,
    grid_context: dict | None = None,
) -> dict | None:
    bundles = load_bundles()
    bundle = bundles.get(target)
    if bundle is None:
        return None

    raw = build_inference_row(
        lat, lon, depth, temperature, salinity, month, grid_id, grid_context
    )
    X, _ = prep_chunk(
        raw,
        bundle["feature_cols"],
        bundle["numeric_cols"],
        bundle["cat_encoders"],
        bundle["numeric_means"],
    )
    model = bundle["model"]
    pred_idx = model.predict(X.values)[0]
    proba = None
    conf = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X.values)[0]
        conf = float(np.max(proba))

    flag_str = bundle["target_encoder"].inverse_transform([pred_idx])[0]
    label = QC_FLAG_LABELS.get(str(flag_str), str(flag_str))

    return {
        "flag": str(flag_str),
        "label": label,
        "confidence": conf,
        "class_index": int(pred_idx),
    }


def predict_both(
    lat: float,
    lon: float,
    depth: float,
    temperature: float,
    salinity: float,
    month: int,
    grid_id: int,
    grid_context: dict | None = None,
) -> dict:
    out = {}
    for target in TARGET_COLS:
        out[target] = predict_target(
            target, lat, lon, depth, temperature, salinity, month, grid_id, grid_context
        )
    any_loaded = any(out[t] is not None for t in TARGET_COLS)
    return {
        "predictions": out,
        "model_type": "RandomForest (chunked ALL_REGIONS)" if any_loaded else "none",
        "models_loaded": any_loaded,
    }
