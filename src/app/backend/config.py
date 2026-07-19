from pathlib import Path

# app/ folder
APP_ROOT = Path(__file__).resolve().parents[1]
# Agro_project/
PROJECT_ROOT = APP_ROOT.parent

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "Final parquet files" / "ALL_REGIONS_UNIFIED.parquet"
TRAIN_PATH = PROJECT_ROOT / "Indian_ocean" / "train_clean" / "ALL_REGIONS_train_clean.parquet"
TEST_PATH = PROJECT_ROOT / "Indian_ocean" / "test_clean" / "ALL_REGIONS_test_clean.parquet"

# ── Models (chunked RF training output) ───────────────────────────────────────
MODEL_DIR = PROJECT_ROOT / "trained_models"
RESULTS_DIR = PROJECT_ROOT / "Indian_ocean" / "model_results_rf_single"

STATIC_DIR = APP_ROOT / "static"

# ── Grid (5° cells, matches 3.0_map_with profiles.py) ───────────────────────
LON_MIN, LON_MAX = 20, 120
LAT_MIN, LAT_MAX = -70, 30
COLS, ROWS = 20, 20
SIZE = 5
N_GRIDS = COLS * ROWS

TARGET_COLS = ["temp_qc", "psal_qc"]

# ── Pipeline outputs ───────────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "qc_pipeline"
MAPS_DIR = PROJECT_ROOT / "outputs" / "maps"
PIPELINE_GRID_SUMMARY = OUTPUT_DIR / "grid_summary.csv"
PIPELINE_MASTER_PARQUET = OUTPUT_DIR / "master_results.parquet"
PIPELINE_MASTER_SAMPLE_CSV = OUTPUT_DIR / "master_results_sample.csv"
