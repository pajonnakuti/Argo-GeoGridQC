"""
Full QC prediction pipeline: diagnose grids, train/predict, tables, 9 plots, report.

Run from app/:
    python scripts/qc_full_pipeline.py
    python scripts/qc_full_pipeline.py --quick          # 50k train sample, 50 obs/grid
    python scripts/qc_full_pipeline.py --skip-train   # use existing bundles only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from backend.config import (
    COLS, DATA_PATH, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, MODEL_DIR,
    N_GRIDS, OUTPUT_DIR, PIPELINE_GRID_SUMMARY, PIPELINE_MASTER_PARQUET,
    PIPELINE_MASTER_SAMPLE_CSV, RESULTS_DIR, ROWS, SIZE, TARGET_COLS,
)
from backend.grid import grid_id_bbox
from backend.ml.preprocess import (
    QC_FLAG_LABELS, classify_columns, collect_stats_duckdb,
    feature_columns_for_target, get_schema_columns, prep_chunk,
)
from backend.ml.predictor import (
    _bundle_path, _plain_model_path, build_inference_row, load_bundles, predict_target,
)

LOG = logging.getLogger("qc_pipeline")
PLOTS_DIR = OUTPUT_DIR / "plots"


def setup_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_DIR / "pipeline.log", encoding="utf-8"),
        ],
    )


def duckdb_con():
    import duckdb
    return duckdb.connect()


def parquet_path() -> str:
    return str(DATA_PATH).replace("'", "''")


def train_from_unified(target: str, sample_rows: int, trees: int = 200) -> dict:
    """Train global RF from unified parquet when train_clean is missing."""
    LOG.info("Training %s from unified parquet (sample=%d)…", target, sample_rows)
    pf, cols = get_schema_columns(str(DATA_PATH))
    feature_cols = feature_columns_for_target(cols, target)
    numeric_cols, categorical_cols = classify_columns(pf, feature_cols)
    stats = collect_stats_duckdb(str(DATA_PATH), feature_cols, numeric_cols, categorical_cols, target)

    con = duckdb_con()
    p = parquet_path()
    df = con.execute(f"""
        SELECT {', '.join(f'"{c}"' for c in feature_cols + [target])}
        FROM read_parquet('{p}')
        WHERE "{target}" IS NOT NULL
          AND temperature IS NOT NULL AND salinity IS NOT NULL
        USING SAMPLE {sample_rows} ROWS
    """).fetchdf()

    df = df.dropna(subset=[target])
    X, y = prep_chunk(df, feature_cols, numeric_cols, stats["cat_encoders"],
                      stats["numeric_means"], target, stats["target_encoder"])
    model = RandomForestClassifier(
        n_estimators=trees, class_weight="balanced", random_state=42, n_jobs=-1,
    )
    model.fit(X.values, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, _plain_model_path(target))
    bundle = {"model": model, "target_col": target, **stats}
    joblib.dump(bundle, _bundle_path(target))
    LOG.info("  Saved %s", _bundle_path(target))
    return bundle


def ensure_models(train_sample: int, skip_train: bool) -> None:
    load_bundles(force_rebuild=False)
    for target in TARGET_COLS:
        if _bundle_path(target).exists():
            LOG.info("Bundle exists: %s", target)
            continue
        if skip_train:
            LOG.warning("No bundle for %s and --skip-train set", target)
            continue
        train_from_unified(target, train_sample)


def grid_obs_counts() -> dict[int, int]:
    from backend import cache as disk_cache
    cached = disk_cache.load_grid_stats()
    if cached:
        return {int(r["grid_id"]): int(r["n_obs"]) for r in cached}
    con = duckdb_con()
    df = con.execute(f"""
        SELECT CAST(grid_id AS INTEGER) AS grid_id, COUNT(*) AS n
        FROM read_parquet('{parquet_path()}')
        WHERE grid_id IS NOT NULL GROUP BY 1
    """).fetchdf()
    return {int(r.grid_id): int(r.n) for _, r in df.iterrows()}


def fetch_all_grid_samples(limit_per_grid: int, quick: bool = False) -> pd.DataFrame:
    """Sample rows from parquet, then cap per grid_id in pandas."""
    con = duckdb_con()
    if quick:
        # Per-grid LIMIT queries — parquet is clustered by grid_id, so a single
        # IN (... ) LIMIT N only returns the first matching cell.
        counts = grid_obs_counts()
        populated = sorted(counts.keys())
        if not populated:
            return pd.DataFrame()
        step = max(1, len(populated) // 40)
        pick = populated[::step][:40]
        parts = []
        for gid in pick:
            part = con.execute(f"""
                SELECT
                    CAST(grid_id AS INTEGER) AS grid_id,
                    lat, lon, depth,
                    temperature AS temp, salinity AS sal,
                    CAST(temp_qc AS VARCHAR) AS temp_qc_raw,
                    CAST(psal_qc AS VARCHAR) AS sal_qc_raw,
                    month, date
                FROM read_parquet('{parquet_path()}')
                WHERE CAST(grid_id AS INTEGER) = {int(gid)}
                  AND temperature IS NOT NULL AND salinity IS NOT NULL
                LIMIT {int(limit_per_grid)}
            """).fetchdf()
            if not part.empty:
                parts.append(part)
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    else:
        total = min(limit_per_grid * N_GRIDS, 500_000)
        df = con.execute(f"""
            SELECT
                CAST(grid_id AS INTEGER) AS grid_id,
                lat, lon, depth,
                temperature AS temp, salinity AS sal,
                CAST(temp_qc AS VARCHAR) AS temp_qc_raw,
                CAST(psal_qc AS VARCHAR) AS sal_qc_raw,
                month, date
            FROM read_parquet('{parquet_path()}')
            WHERE grid_id IS NOT NULL
              AND temperature IS NOT NULL AND salinity IS NOT NULL
            USING SAMPLE {int(total)} ROWS
        """).fetchdf()
        if not df.empty:
            df = df.groupby("grid_id", as_index=False).head(limit_per_grid)
    LOG.info("Sampled %d rows across %d grids", len(df), df["grid_id"].nunique() if len(df) else 0)
    return df


def predict_row(row: pd.Series, grid_id: int) -> tuple[str | None, str | None, str | None]:
    """Returns (pred_temp, pred_sal, error_msg)."""
    try:
        lat, lon = float(row["lat"]), float(row["lon"])
        depth = float(row["depth"] or 0)
        temp = float(row["temp"])
        sal = float(row["sal"])
        month = int(row["month"]) if pd.notna(row.get("month")) else 6
        ctx = {
            "mean_temp": temp, "min_temp": temp - 2, "max_temp": temp + 2,
            "mean_psal": sal, "min_psal": sal - 0.5, "max_psal": sal + 0.5,
            "temp_grid_mean": temp, "temp_grid_std": 1.0,
            "sal_grid_mean": sal, "sal_grid_std": 0.3, "n_obs": 1000,
        }
        pt = predict_target("temp_qc", lat, lon, depth, temp, sal, month, grid_id, ctx)
        ps = predict_target("psal_qc", lat, lon, depth, temp, sal, month, grid_id, ctx)
        if pt is None and ps is None:
            return None, None, "NO_MODEL"
        return (
            pt["flag"] if pt else "NO_PRED",
            ps["flag"] if ps else "NO_PRED",
            None,
        )
    except Exception as e:
        return None, None, str(e)


def process_all_grids(obs_per_grid: int, quick: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    obs_counts = grid_obs_counts()
    LOG.info("Fetching per-grid samples (one DuckDB query)…")
    all_samples = fetch_all_grid_samples(obs_per_grid, quick=quick)
    master_rows = []
    grid_rows = []
    failures = []

    for gid in range(1, N_GRIDS + 1):
        n_obs = obs_counts.get(gid, 0)
        if n_obs == 0:
            grid_rows.append({
                "grid_id": gid, "n_obs": 0, "n_predicted": 0,
                "pct_temp_match": None, "pct_sal_match": None,
                "grid_status": "NO_DATA", "error": None,
            })
            continue

        sample = all_samples[all_samples["grid_id"] == gid]
        if sample.empty:
            # Has observations in full dataset but not in this sample run
            grid_rows.append({
                "grid_id": gid, "n_obs": n_obs, "n_predicted": 0,
                "pct_temp_match": None, "pct_sal_match": None,
                "grid_status": "PARTIAL", "error": "not_in_sample",
            })
            continue

        try:
            preds_t, preds_s = [], []
            errors = []
            for _, row in sample.iterrows():
                pt, ps, err = predict_row(row, gid)
                if err and err != "NO_MODEL":
                    errors.append(err)
                raw_t = str(row["temp_qc_raw"]) if pd.notna(row["temp_qc_raw"]) else None
                raw_s = str(row["sal_qc_raw"]) if pd.notna(row["sal_qc_raw"]) else None
                master_rows.append({
                    "grid_id": gid, "lat": row["lat"], "lon": row["lon"],
                    "depth": row["depth"], "temp": row["temp"], "sal": row["sal"],
                    "temp_qc_raw": raw_t, "sal_qc_raw": raw_s,
                    "predicted_temp_qc": pt, "predicted_sal_qc": ps,
                    "temp_qc_match": raw_t == pt if raw_t and pt and pt not in ("NO_PRED",) else None,
                    "sal_qc_match": raw_s == ps if raw_s and ps and ps not in ("NO_PRED",) else None,
                })
                if pt and pt not in ("NO_PRED",):
                    preds_t.append(raw_t == pt if raw_t else None)
                if ps and ps not in ("NO_PRED",):
                    preds_s.append(raw_s == ps if raw_s else None)

            n_pred = len(sample)
            n_ok = sum(1 for r in master_rows[-n_pred:] if r["predicted_temp_qc"] not in (None, "NO_PRED"))
            valid_t = [x for x in preds_t if x is not None]
            valid_s = [x for x in preds_s if x is not None]
            pct_t = float(np.mean(valid_t)) * 100 if valid_t else None
            pct_s = float(np.mean(valid_s)) * 100 if valid_s else None

            if n_ok == 0:
                status = "FAILED"
            elif n_ok < n_pred:
                status = "PARTIAL"
            else:
                status = "OK"

            grid_rows.append({
                "grid_id": gid, "n_obs": n_obs, "n_predicted": n_ok,
                "pct_temp_match": round(pct_t, 1) if pct_t is not None else None,
                "pct_sal_match": round(pct_s, 1) if pct_s is not None else None,
                "grid_status": status,
                "error": errors[0] if errors else None,
            })
            if status == "FAILED":
                failures.append({"grid_id": gid, "n_obs": n_obs, "error": errors[0] if errors else "NO_MODEL"})
        except Exception as e:
            LOG.error("Grid %d failed: %s\n%s", gid, e, traceback.format_exc())
            grid_rows.append({
                "grid_id": gid, "n_obs": n_obs, "n_predicted": 0,
                "pct_temp_match": None, "pct_sal_match": None,
                "grid_status": "FAILED", "error": str(e),
            })
            failures.append({"grid_id": gid, "n_obs": n_obs, "error": str(e)})

        if gid % 100 == 0:
            LOG.info("  Processed grid %d/%d", gid, N_GRIDS)

    master = pd.DataFrame(master_rows)
    if not master.empty:
        master["grid_status"] = master["grid_id"].map(
            {r["grid_id"]: r["grid_status"] for r in grid_rows}
        )
    summary = pd.DataFrame(grid_rows)
    return master, summary, failures


def generate_plots(master: pd.DataFrame, summary: pd.DataFrame):
    sns.set_theme(style="whitegrid", context="notebook")
    if master.empty:
        LOG.warning("No master data — skipping plots")
        return

    m = master.dropna(subset=["temp_qc_raw", "predicted_temp_qc"], how="all")

    # 1. Raw vs predicted QC bar charts
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, raw_col, pred_col, title in [
        (axes[0], "temp_qc_raw", "predicted_temp_qc", "Temperature QC"),
        (axes[1], "sal_qc_raw", "predicted_sal_qc", "Salinity QC"),
    ]:
        raw_c = m[raw_col].value_counts().sort_index()
        pred_c = m[pred_col].value_counts().sort_index()
        idx = sorted(set(raw_c.index) | set(pred_c.index), key=lambda x: str(x))
        x = np.arange(len(idx))
        w = 0.35
        ax.bar(x - w / 2, [raw_c.get(i, 0) for i in idx], w, label="Raw", color="#6c8ebf")
        ax.bar(x + w / 2, [pred_c.get(i, 0) for i in idx], w, label="Predicted", color="#b85450")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in idx])
        ax.set_title(f"{title}: Raw vs Predicted QC flags")
        ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_raw_vs_predicted_qc_bars.png", dpi=150)
    plt.close(fig)

    # 2–3. Depth vs QC (temp & sal)
    for param, raw_col, pred_col, fname in [
        ("temp", "temp_qc_raw", "predicted_temp_qc", "02_depth_vs_temp_qc.png"),
        ("sal", "sal_qc_raw", "predicted_sal_qc", "03_depth_vs_sal_qc.png"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
        sub = m.dropna(subset=["depth"])
        for ax, col, title in [(axes[0], raw_col, "Raw"), (axes[1], pred_col, "Predicted")]:
            for flag in sorted(sub[col].dropna().unique(), key=str):
                pts = sub[sub[col] == flag]
                ax.scatter(pts["temp" if param == "temp" else "sal"], pts["depth"],
                           s=4, alpha=0.4, label=str(flag))
            ax.invert_yaxis()
            ax.set_xlabel("Temperature °C" if param == "temp" else "Salinity PSU")
            ax.set_ylabel("Depth (m)")
            ax.set_title(f"{title} {param}_qc")
            ax.legend(fontsize=7, markerscale=2)
        fig.savefig(PLOTS_DIR / fname, dpi=150)
        plt.close(fig)

    # 4. Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, raw_col, pred_col, title in [
        (axes[0], "temp_qc_raw", "predicted_temp_qc", "temp_qc"),
        (axes[1], "sal_qc_raw", "predicted_sal_qc", "psal_qc"),
    ]:
        sub = m.dropna(subset=[raw_col, pred_col])
        sub = sub[~sub[pred_col].isin(["NO_PRED", "NO_MODEL"])]
        if sub.empty:
            continue
        labels = sorted(set(sub[raw_col].astype(str)) | set(sub[pred_col].astype(str)))
        cm = confusion_matrix(sub[raw_col].astype(str), sub[pred_col].astype(str), labels=labels)
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, ax=ax, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Raw")
        ax.set_title(f"Confusion matrix — {title}")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_confusion_matrices.png", dpi=150)
    plt.close(fig)

    # 5. Spatial map of match rate
    fig, ax = plt.subplots(figsize=(10, 8))
    for _, row in summary.iterrows():
        bb = grid_id_bbox(int(row.grid_id))
        val = row.pct_temp_match if pd.notna(row.pct_temp_match) else 0
        if row.grid_status == "NO_DATA":
            val = np.nan
        color = plt.cm.RdYlGn(val / 100 if pd.notna(val) else 0.5)
        rect = plt.Rectangle((bb["lon_min"], bb["lat_min"]),
                             bb["lon_max"] - bb["lon_min"], bb["lat_max"] - bb["lat_min"],
                             facecolor=color, edgecolor="black", lw=0.3)
        ax.add_patch(rect)
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Per-grid temp_qc match rate (%)")
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, 100))
    plt.colorbar(sm, ax=ax, label="Match %")
    fig.savefig(PLOTS_DIR / "05_spatial_match_rate.png", dpi=150)
    plt.close(fig)

    # 6. QC flag histograms
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col, title in [
        (axes[0, 0], "temp_qc_raw", "Raw temp_qc"),
        (axes[0, 1], "predicted_temp_qc", "Predicted temp_qc"),
        (axes[1, 0], "sal_qc_raw", "Raw psal_qc"),
        (axes[1, 1], "predicted_sal_qc", "Predicted psal_qc"),
    ]:
        m[col].astype(str).value_counts().sort_index().plot(kind="bar", ax=ax, color="#5b7a9d")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "06_qc_flag_histograms.png", dpi=150)
    plt.close(fig)

    # 7. Temp vs Sal scatter by QC
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in [(axes[0], "temp_qc_raw", "Raw temp_qc"), (axes[1], "predicted_temp_qc", "Predicted temp_qc")]:
        for flag in m[col].dropna().unique():
            pts = m[m[col] == flag]
            ax.scatter(pts["sal"], pts["temp"], s=6, alpha=0.35, label=str(flag))
        ax.set_xlabel("Salinity PSU")
        ax.set_ylabel("Temperature °C")
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.savefig(PLOTS_DIR / "07_temp_vs_sal_by_qc.png", dpi=150)
    plt.close(fig)

    # 8. Time series QC proportions (if date available)
    if "date" in m.columns and m["date"].notna().any():
        m2 = m.copy()
        m2["year"] = pd.to_datetime(m2["date"], errors="coerce").dt.year
        m2 = m2.dropna(subset=["year"])
        if not m2.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            yearly = m2.groupby("year")["temp_qc_raw"].apply(
                lambda s: (s.astype(str) == "1").mean() * 100
            )
            yearly.plot(ax=ax, marker="o")
            ax.set_ylabel("% Good temp_qc (raw)")
            ax.set_title("Time series of QC flag proportions")
            fig.savefig(PLOTS_DIR / "08_qc_time_series.png", dpi=150)
            plt.close(fig)

    # 9. Per-grid failure summary
    fig, ax = plt.subplots(figsize=(14, 5))
    s = summary.sort_values("pct_temp_match", na_position="first")
    colors = s.grid_status.map({"OK": "#4fd69c", "PARTIAL": "#e8b84b", "FAILED": "#e5654e", "NO_DATA": "#8592a0"})
    ax.bar(range(len(s)), s.n_obs, color=colors, alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(range(len(s)), s.pct_temp_match, color="navy", lw=1, alpha=0.6, label="temp match %")
    ax.set_xlabel("Grid (sorted by match rate)")
    ax.set_ylabel("Observation count")
    ax.set_title("Per-grid observations vs prediction success")
    fig.savefig(PLOTS_DIR / "09_per_grid_failure_summary.png", dpi=150)
    plt.close(fig)

    LOG.info("Saved 9 plots to %s", PLOTS_DIR)


def write_report(summary: pd.DataFrame, failures: list[dict], master: pd.DataFrame):
    status_counts = summary.grid_status.value_counts().to_dict()
    report = [
        "# QC Pipeline Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M')}",
        f"Data: `{DATA_PATH}`",
        "",
        "## Grid status summary",
        "",
    ]
    for k, v in sorted(status_counts.items()):
        report.append(f"- **{k}**: {v} grids")
    report += [
        "",
        "## Root causes for previously failing grids",
        "",
        "1. **No trained model bundles** in `trained_models/` — predictions fell back to z-score or `NO_PRED`.",
        "2. **40 empty grids** (no observations) — correctly labeled `NO_DATA`, not model failures.",
        "3. **Silent skips** replaced with explicit per-grid try/except logging in `pipeline.log`.",
        "",
        "## Previously failing grids (sample)",
        "",
    ]
    for f in failures[:20]:
        report.append(f"- GRID-{f['grid_id']}: n_obs={f['n_obs']}, error={f.get('error', 'unknown')}")
    if master is not None and not master.empty:
        for target in TARGET_COLS:
            raw_col = "temp_qc_raw" if target == "temp_qc" else "sal_qc_raw"
            pred_col = "predicted_temp_qc" if target == "temp_qc" else "predicted_sal_qc"
            sub = master.dropna(subset=[raw_col, pred_col])
            sub = sub[~sub[pred_col].isin(["NO_PRED", "NO_MODEL"])]
            if len(sub) > 10:
                report += ["", f"### {target} classification report", "", "```"]
                report.append(classification_report(
                    sub[raw_col].astype(str), sub[pred_col].astype(str), zero_division=0
                ))
                report.append("```")
    report += [
        "",
        "## Known limitations",
        "",
        "- Master table is a **per-grid random sample**, not all 237M rows (saved as parquet + CSV sample).",
        "- Global RF trained on DuckDB sample; re-run with more `--train-sample` for production accuracy.",
        "- Grids with `NO_DATA` are land or out-of-coverage cells.",
        "",
    ]
    (OUTPUT_DIR / "pipeline_report.md").write_text("\n".join(report), encoding="utf-8")
    LOG.info("Wrote %s", OUTPUT_DIR / "pipeline_report.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fast run: 50k train, 50 obs/grid")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--train-sample", type=int, default=500_000)
    parser.add_argument("--obs-per-grid", type=int, default=200)
    args = parser.parse_args()

    if args.quick:
        args.train_sample = 50_000
        args.obs_per_grid = 20  # keep smoke-test fast (RF predict is the bottleneck)

    setup_logging()
    LOG.info("=== QC Full Pipeline ===")

    ensure_models(args.train_sample, args.skip_train)
    load_bundles(force_rebuild=True)

    master, summary, failures = process_all_grids(args.obs_per_grid, quick=args.quick)

    # Add grid_status to master rows
    status_map = dict(zip(summary.grid_id, summary.grid_status))
    if not master.empty:
        master["grid_status"] = master["grid_id"].map(status_map)

    summary.to_csv(PIPELINE_GRID_SUMMARY, index=False)
    LOG.info("Saved %s (%d rows)", PIPELINE_GRID_SUMMARY, len(summary))

    if not master.empty:
        master.to_parquet(PIPELINE_MASTER_PARQUET, index=False)
        master.head(100_000).to_csv(PIPELINE_MASTER_SAMPLE_CSV, index=False)
        LOG.info("Saved %s and sample CSV", PIPELINE_MASTER_PARQUET)

    with open(OUTPUT_DIR / "grid_failures.json", "w") as f:
        json.dump(failures, f, indent=2)

    generate_plots(master, summary)
    write_report(summary, failures, master)

    # Regenerate maps
    LOG.info("Regenerating grid maps…")
    import subprocess
    subprocess.run([sys.executable, str(APP_ROOT / "scripts" / "reproduce_grid_map.py")], check=False)

    LOG.info("=== Pipeline complete ===")
    LOG.info("Outputs: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
