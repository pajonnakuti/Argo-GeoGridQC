"""Disk cache for fast startup (avoids scanning 237M rows on every launch)."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT / "data" / "processed" / ".argo_cache"
GRID_STATS_PATH = CACHE_DIR / "grid_stats.parquet"
META_PATH = CACHE_DIR / "meta.json"

_warmup_lock = threading.Lock()
_warmup_started = False


def load_grid_stats() -> list[dict] | None:
    if GRID_STATS_PATH.exists():
        try:
            df = pd.read_parquet(GRID_STATS_PATH)
            return df.replace({float("nan"): None}).to_dict(orient="records")
        except Exception:
            pass

    # Bootstrap from legacy CSV if present
    csv_path = PROJECT_ROOT / "data" / "processed" / "grid_statistics.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            records = []
            for _, row in df.iterrows():
                records.append({
                    "grid_id": int(row["grid_id"]),
                    "n_obs": int(row.get("profile_count", row.get("n_obs", 0))),
                    "mean_temp": row.get("mean_temp"),
                    "mean_psal": row.get("mean_psal"),
                    "mean_depth": row.get("avg_depth", row.get("mean_depth")),
                    "max_depth": row.get("max_depth"),
                    "metric_value": int(row.get("profile_count", row.get("n_obs", 0))),
                })
            save_grid_stats(records)
            return records
        except Exception:
            pass
    return None


def save_grid_stats(records: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(GRID_STATS_PATH, index=False)


def load_meta() -> dict | None:
    if not META_PATH.exists():
        return None
    try:
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_meta(meta: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def start_background_warmup(warmup_fn) -> None:
    """Run heavy DuckDB aggregation once in a background thread."""
    global _warmup_started
    with _warmup_lock:
        if _warmup_started:
            return
        _warmup_started = True

    def _run():
        try:
            warmup_fn()
        except Exception as exc:
            print(f"[cache] background warmup failed: {exc}", flush=True)

    threading.Thread(target=_run, daemon=True, name="argo-warmup").start()
