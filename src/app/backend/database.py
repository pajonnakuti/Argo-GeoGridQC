"""DuckDB access layer over ALL_REGIONS_UNIFIED.parquet."""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from .config import DATA_PATH, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, ROWS, COLS, SIZE
from .ml.predictor import load_bundles, predict_both
from . import cache as disk_cache

_con: duckdb.DuckDBPyConnection | None = None
_grid_stats_cache: list | None = None
_meta_cache: dict | None = None


def _gaussian_blur(arr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    rad = int(np.ceil(sigma * 3))
    x = np.arange(-rad, rad + 1)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    tmp = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, arr)
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, tmp)

INSTRUMENT_SQL = """
CASE
    WHEN upper(source_id) LIKE 'CTD%' OR lower(source_id) LIKE '%ctd%' THEN 'CTD'
    WHEN upper(source_id) LIKE 'XBT%' OR lower(source_id) LIKE '%xbt%'
         OR lower(source_id) LIKE '%.edf' THEN 'XBT'
    ELSE 'ARGO'
END
"""

QC_LABEL_SQL = """
CASE CAST(temp_qc AS VARCHAR)
    WHEN '1' THEN 'Good'
    WHEN '2' THEN 'Probably good'
    WHEN '3' THEN 'Bad'
    WHEN '4' THEN 'Bad'
    WHEN '9' THEN 'Missing'
    ELSE COALESCE(CAST(temp_qc AS VARCHAR), 'Unknown')
END
"""

PSAL_QC_LABEL_SQL = """
CASE CAST(psal_qc AS VARCHAR)
    WHEN '1' THEN 'Good'
    WHEN '2' THEN 'Probably good'
    WHEN '3' THEN 'Bad'
    WHEN '4' THEN 'Bad'
    WHEN '9' THEN 'Missing'
    ELSE COALESCE(CAST(psal_qc AS VARCHAR), 'Unknown')
END
"""


def _escape(path: str) -> str:
    return path.replace("'", "''")


def get_con() -> duckdb.DuckDBPyConnection:
    global _con
    if _con is None:
        _con = duckdb.connect(database=":memory:")
        p = _escape(str(DATA_PATH))
        _con.execute(f"""
            CREATE OR REPLACE VIEW obs AS
            SELECT
                CAST(grid_id AS INTEGER) AS grid_id,
                CAST(lat AS DOUBLE) AS lat,
                CAST(lon AS DOUBLE) AS lon,
                CAST(depth AS DOUBLE) AS depth,
                CAST(temperature AS DOUBLE) AS temperature,
                CAST(salinity AS DOUBLE) AS salinity,
                CAST(month AS INTEGER) AS month,
                CAST(temp_qc AS VARCHAR) AS temp_qc,
                CAST(psal_qc AS VARCHAR) AS psal_qc,
                CAST(temp_zscore AS DOUBLE) AS temp_zscore,
                CAST(sal_zscore AS DOUBLE) AS sal_zscore,
                CAST(temp_grid_mean AS DOUBLE) AS temp_grid_mean,
                CAST(temp_grid_std AS DOUBLE) AS temp_grid_std,
                CAST(sal_grid_mean AS DOUBLE) AS sal_grid_mean,
                CAST(sal_grid_std AS DOUBLE) AS sal_grid_std,
                source_id,
                source_file,
                {INSTRUMENT_SQL} AS instrument
            FROM read_parquet('{p}')
            WHERE grid_id IS NOT NULL
        """)
    return _con


def dataset_meta() -> dict:
    global _meta_cache
    if _meta_cache is not None:
        return _meta_cache

    cached = disk_cache.load_meta()
    if cached is not None:
        cached["rf_models_loaded"] = len(load_bundles())
        _meta_cache = cached
        return _meta_cache

    # Fast placeholder until background warmup finishes
    _meta_cache = {
        "rows": 0,
        "populated_grids": 0,
        "total_grids": ROWS * COLS,
        "bounds": {
            "lon_min": LON_MIN,
            "lon_max": LON_MAX,
            "lat_min": LAT_MIN,
            "lat_max": LAT_MAX,
        },
        "grid_size_deg": SIZE,
        "cols": COLS,
        "grid_rows": ROWS,
        "data_path": str(DATA_PATH),
        "rf_models_loaded": len(load_bundles()),
        "warming": True,
    }
    return _meta_cache


def _refresh_meta() -> dict:
    global _meta_cache
    con = get_con()
    n_rows, n_grids = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT grid_id) FROM obs"
    ).fetchone()
    meta = {
        "rows": int(n_rows),
        "populated_grids": int(n_grids),
        "total_grids": ROWS * COLS,
        "bounds": {
            "lon_min": LON_MIN,
            "lon_max": LON_MAX,
            "lat_min": LAT_MIN,
            "lat_max": LAT_MAX,
        },
        "grid_size_deg": SIZE,
        "cols": COLS,
        "grid_rows": ROWS,
        "data_path": str(DATA_PATH),
        "rf_models_loaded": len(load_bundles()),
        "warming": False,
    }
    _meta_cache = meta
    disk_cache.save_meta(meta)
    return meta


def all_grid_stats(metric: str = "n_obs") -> list[dict]:
    global _grid_stats_cache
    if _grid_stats_cache is not None and metric == "n_obs":
        return _grid_stats_cache

    if metric == "n_obs":
        cached = disk_cache.load_grid_stats()
        if cached:
            _grid_stats_cache = cached
            return _grid_stats_cache

    con = get_con()
    metric_col = {
        "mean_temp": "AVG(temperature)",
        "mean_psal": "AVG(salinity)",
        "n_obs": "COUNT(*)",
        "mean_depth": "AVG(depth)",
    }.get(metric, "COUNT(*)")
    df = con.execute(
        f"""
        SELECT
            grid_id,
            COUNT(*) AS n_obs,
            AVG(temperature) AS mean_temp,
            AVG(salinity) AS mean_psal,
            AVG(depth) AS mean_depth,
            MAX(depth) AS max_depth,
            {metric_col} AS metric_value
        FROM obs
        GROUP BY grid_id
        ORDER BY grid_id
        """
    ).fetchdf()
    records = df.replace({np.nan: None}).to_dict(orient="records")
    if metric == "n_obs":
        _grid_stats_cache = records
        disk_cache.save_grid_stats(records)
    return records


def warmup_full_cache() -> None:
    """Heavy DuckDB scan — run in background thread after server is up."""
    print("[warmup] Aggregating grid stats from unified parquet…", flush=True)
    all_grid_stats("n_obs")
    _refresh_meta()
    print("[warmup] Cache ready.", flush=True)


def grid_context(grid_id: int) -> dict:
    con = get_con()
    row = con.execute(
        """
        SELECT
            AVG(temp_grid_mean) AS temp_grid_mean,
            AVG(temp_grid_std) AS temp_grid_std,
            AVG(sal_grid_mean) AS sal_grid_mean,
            AVG(sal_grid_std) AS sal_grid_std,
            AVG(temperature) AS mean_temp,
            MIN(temperature) AS min_temp,
            MAX(temperature) AS max_temp,
            AVG(salinity) AS mean_psal,
            MIN(salinity) AS min_psal,
            MAX(salinity) AS max_psal,
            COUNT(*) AS n_obs
        FROM obs WHERE grid_id = ?
        """,
        [grid_id],
    ).fetchdf()
    if row.empty:
        return {}
    return row.iloc[0].replace({np.nan: None}).to_dict()


def grid_detail(grid_id: int) -> dict | None:
    con = get_con()
    row = con.execute(
        """
        SELECT
            grid_id,
            COUNT(*) AS n_obs,
            AVG(temperature) AS mean_temp,
            MIN(temperature) AS min_temp,
            MAX(temperature) AS max_temp,
            AVG(salinity) AS mean_psal,
            MIN(salinity) AS min_psal,
            MAX(salinity) AS max_psal,
            AVG(depth) AS mean_depth,
            MAX(depth) AS max_depth,
            AVG(temp_grid_mean) AS clim_temp_mean,
            AVG(temp_grid_std) AS clim_temp_std,
            AVG(sal_grid_mean) AS clim_psal_mean,
            AVG(sal_grid_std) AS clim_psal_std
        FROM obs WHERE grid_id = ?
        GROUP BY grid_id
        """,
        [grid_id],
    ).fetchdf()
    if row.empty:
        return None

    inst = con.execute(
        "SELECT instrument, COUNT(*) AS n FROM obs WHERE grid_id = ? GROUP BY instrument ORDER BY n DESC",
        [grid_id],
    ).fetchdf()

    qc = con.execute(
        f"""
        SELECT
            {QC_LABEL_SQL} AS label,
            CAST(temp_qc AS VARCHAR) AS flag,
            COUNT(*) AS n
        FROM obs
        WHERE grid_id = ? AND temp_qc IS NOT NULL
        GROUP BY 1, 2 ORDER BY n DESC
        """,
        [grid_id],
    ).fetchdf()

    detail = row.iloc[0].replace({np.nan: None}).to_dict()
    detail["instruments"] = inst.replace({np.nan: None}).to_dict(orient="records")
    detail["qc_temp"] = qc.replace({np.nan: None}).to_dict(orient="records")
    from .ml.predictor import models_status

    ms = {m["target"]: m for m in models_status()}
    detail["model"] = {
        "name": "Random Forest (ALL_REGIONS chunked)",
        "version": "chunked_warm_start",
        "target": "temp_qc / psal_qc",
        "temp_qc_ready": ms.get("temp_qc", {}).get("bundle_loaded", False),
        "psal_qc_ready": ms.get("psal_qc", {}).get("bundle_loaded", False),
    }
    return detail


def grid_analysis(grid_id: int, sample_size: int = 120) -> dict | None:
    """Research-grade grid summary: T/S stats, raw QC, predicted QC sample comparison."""
    base = grid_detail(grid_id)
    if base is None:
        return None

    con = get_con()

    qc_psal = con.execute(
        f"""
        SELECT
            {PSAL_QC_LABEL_SQL} AS label,
            CAST(psal_qc AS VARCHAR) AS flag,
            COUNT(*) AS n
        FROM obs
        WHERE grid_id = ? AND psal_qc IS NOT NULL
        GROUP BY 1, 2 ORDER BY n DESC
        """,
        [grid_id],
    ).fetchdf()

    depth_bins = con.execute(
        """
        SELECT
            CASE
                WHEN depth < 50 THEN '0–50 m'
                WHEN depth < 200 THEN '50–200 m'
                WHEN depth < 1000 THEN '200–1000 m'
                ELSE '>1000 m'
            END AS bin,
            COUNT(*) AS n
        FROM obs WHERE grid_id = ?
        GROUP BY 1 ORDER BY MIN(depth)
        """,
        [grid_id],
    ).fetchdf()

    monthly = con.execute(
        """
        SELECT month, COUNT(*) AS n
        FROM obs WHERE grid_id = ? AND month IS NOT NULL
        GROUP BY month ORDER BY month
        """,
        [grid_id],
    ).fetchdf()

    zstats = con.execute(
        """
        SELECT
            AVG(temp_zscore) AS mean_temp_z,
            STDDEV(temp_zscore) AS std_temp_z,
            AVG(sal_zscore) AS mean_sal_z,
            STDDEV(sal_zscore) AS std_sal_z,
            AVG(ABS(temp_zscore)) AS mean_abs_temp_z,
            AVG(ABS(sal_zscore)) AS mean_abs_sal_z
        FROM obs WHERE grid_id = ?
        """,
        [grid_id],
    ).fetchdf()

    sample = con.execute(
        """
        SELECT lat, lon, depth, temperature, salinity, month,
               CAST(temp_qc AS VARCHAR) AS temp_qc,
               CAST(psal_qc AS VARCHAR) AS psal_qc
        FROM obs WHERE grid_id = ?
        ORDER BY RANDOM() LIMIT ?
        """,
        [grid_id, sample_size],
    ).fetchdf()

    pred_rows = []
    agree_temp = agree_psal = 0
    n_temp = n_psal = 0
    for _, row in sample.iterrows():
        pred = predict_qc(
            float(row["lat"]),
            float(row["lon"]),
            float(row["depth"] or 0),
            float(row["temperature"]),
            float(row["salinity"]),
            int(row["month"]) if row["month"] is not None and not pd.isna(row["month"]) else 6,
        )
        raw_t = str(row["temp_qc"]) if row["temp_qc"] is not None else None
        raw_s = str(row["psal_qc"]) if row["psal_qc"] is not None else None
        pt = pred.get("temp_qc", {})
        ps = pred.get("psal_qc", {})
        if raw_t and pt.get("flag"):
            n_temp += 1
            if str(pt["flag"]) == raw_t:
                agree_temp += 1
        if raw_s and ps.get("flag"):
            n_psal += 1
            if str(ps["flag"]) == raw_s:
                agree_psal += 1
        pred_rows.append({
            "depth": row["depth"],
            "temperature": row["temperature"],
            "salinity": row["salinity"],
            "raw_temp_qc": raw_t,
            "raw_psal_qc": raw_s,
            "pred_temp_qc": pt.get("flag"),
            "pred_psal_qc": ps.get("flag"),
            "pred_temp_label": pt.get("label"),
            "pred_psal_label": ps.get("label"),
        })

    zrow = zstats.iloc[0].replace({np.nan: None}).to_dict() if not zstats.empty else {}

    return {
        **base,
        "qc_psal": qc_psal.replace({np.nan: None}).to_dict(orient="records"),
        "depth_bins": depth_bins.replace({np.nan: None}).to_dict(orient="records"),
        "monthly": monthly.replace({np.nan: None}).to_dict(orient="records"),
        "zscore_stats": zrow,
        "prediction_sample": {
            "n": len(pred_rows),
            "temp_agreement_pct": round(100 * agree_temp / n_temp, 1) if n_temp else None,
            "psal_agreement_pct": round(100 * agree_psal / n_psal, 1) if n_psal else None,
            "rows": pred_rows[:40],
        },
        "thermohaline": {
            "temp": {
                "mean": base.get("mean_temp"),
                "min": base.get("min_temp"),
                "max": base.get("max_temp"),
                "clim_mean": base.get("clim_temp_mean"),
                "clim_std": base.get("clim_temp_std"),
            },
            "salinity": {
                "mean": base.get("mean_psal"),
                "min": base.get("min_psal"),
                "max": base.get("max_psal"),
                "clim_mean": base.get("clim_psal_mean"),
                "clim_std": base.get("clim_psal_std"),
            },
            "depth": {
                "mean": base.get("mean_depth"),
                "max": base.get("max_depth"),
            },
        },
    }


def grid_profiles(grid_id: int, limit: int = 500) -> list[dict]:
    con = get_con()
    df = con.execute(
        """
        SELECT lat, lon, depth, temperature, salinity, instrument, temp_qc, psal_qc, source_id
        FROM obs
        WHERE grid_id = ?
        ORDER BY RANDOM()
        LIMIT ?
        """,
        [grid_id, limit],
    ).fetchdf()
    return df.replace({np.nan: None}).to_dict(orient="records")


def grid_heatmap(grid_id: int, size: int = 64, sample_limit: int = 50_000) -> dict:
    """Return a density heatmap for a grid. Never raises on missing/empty data."""
    empty = {"width": size, "height": size, "max": 0, "values": []}
    try:
        con = get_con()
        # DuckDB SAMPLE only accepts literals (not bound params); LIMIT is reliable + fast.
        df = con.execute(
            """
            SELECT lat, lon
            FROM obs
            WHERE grid_id = ?
              AND lat IS NOT NULL
              AND lon IS NOT NULL
            LIMIT ?
            """,
            [grid_id, int(sample_limit)],
        ).fetchdf()
    except Exception:
        return empty

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return empty

    from .grid import grid_id_bbox

    try:
        bb = grid_id_bbox(grid_id)
    except Exception:
        return empty
    if not bb:
        return empty

    hist = np.zeros((size, size), dtype=np.float32)
    lat_rng = bb["lat_max"] - bb["lat_min"] or 1
    lon_rng = bb["lon_max"] - bb["lon_min"] or 1
    lats = df["lat"].to_numpy(dtype=np.float64, copy=False)
    lons = df["lon"].to_numpy(dtype=np.float64, copy=False)
    for lat, lon in zip(lats, lons):
        if not np.isfinite(lat) or not np.isfinite(lon):
            continue
        iy = int((bb["lat_max"] - lat) / lat_rng * (size - 1))
        ix = int((lon - bb["lon_min"]) / lon_rng * (size - 1))
        iy = min(max(iy, 0), size - 1)
        ix = min(max(ix, 0), size - 1)
        hist[iy, ix] += 1

    if hist.max() <= 0:
        return empty

    smooth = _gaussian_blur(hist, sigma=2.0)
    mx = float(smooth.max()) if smooth.max() > 0 else 1.0
    return {"width": size, "height": size, "max": mx, "values": smooth.flatten().tolist()}


def predict_qc(lat: float, lon: float, depth: float, temp: float, psal: float, month: int) -> dict:
    from .grid import latlon_to_grid_id

    gid = latlon_to_grid_id(lat, lon)
    if gid is None:
        return {"error": "Coordinates outside study domain", "grid_id": None}

    ctx = grid_context(gid)
    rf = predict_both(lat, lon, depth, temp, psal, month, gid, ctx)

    if rf["models_loaded"]:
        preds = rf["predictions"]
        # Guard: if one target failed, keep the other and mark missing as NO_PRED
        temp_pred = preds.get("temp_qc") or {"flag": "NO_PRED", "label": "No prediction", "confidence": None}
        psal_pred = preds.get("psal_qc") or {"flag": "NO_PRED", "label": "No prediction", "confidence": None}
        return {
            "grid_id": gid,
            "temp_qc": temp_pred,
            "psal_qc": psal_pred,
            "model": rf["model_type"],
            "features": {
                "lat": lat,
                "lon": lon,
                "depth": depth,
                "temperature": temp,
                "salinity": psal,
                "month": month,
            },
        }

    # Fallback z-score if models not trained yet
    t_mean = ctx.get("temp_grid_mean") or temp
    t_std = ctx.get("temp_grid_std") or 1.0
    s_mean = ctx.get("sal_grid_mean") or psal
    s_std = ctx.get("sal_grid_std") or 0.3
    tz = (temp - t_mean) / (t_std or 1.0)
    sz = (psal - s_mean) / (s_std or 0.3)

    def flag_from_z(z):
        az = abs(z)
        if az < 2:
            return {"flag": "1", "label": "Good", "confidence": None, "zscore": round(z, 3)}
        if az < 3:
            return {"flag": "2", "label": "Probably good", "confidence": None, "zscore": round(z, 3)}
        if az < 4.5:
            return {"flag": "3", "label": "Bad", "confidence": None, "zscore": round(z, 3)}
        return {"flag": "9", "label": "Missing", "confidence": None, "zscore": round(z, 3)}

    return {
        "grid_id": gid,
        "temp_qc": flag_from_z(tz),
        "psal_qc": flag_from_z(sz),
        "model": "z-score fallback (train models first)",
        "features": {
            "lat": lat,
            "lon": lon,
            "depth": depth,
            "temperature": temp,
            "salinity": psal,
            "month": month,
        },
    }
