"""
qc_4models_extra.py
====================
Runs 4 NEW models (HistGradientBoosting, LinearSVC, GaussianNB, MLP)
on the same grid data as the original 6-model script.

Windows-compatible: all multiprocessing guarded under __main__.
Speed improvements:
  - Scaler fitted once per target, shared across all 4 models
  - SMOTE skipped for models with native class balancing
  - HistGBM / MLP: early_stopping=True
  - LinearSVC: dual=False (faster when n_samples >> n_features)
  - GaussianNB: near-zero training time
  - Joblib parallelism disabled inside workers (avoid over-subscription)

After all grids finish, auto-merges with existing 6-model CSVs into:
  All_grids_models/all_grids_10model_results.csv
"""

# ============================================================
# TOP-LEVEL IMPORTS  (must be at module level for Windows spawn)
# ============================================================

import os
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# sklearn
from sklearn.ensemble        import HistGradientBoostingClassifier
from sklearn.svm             import LinearSVC
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.naive_bayes     import GaussianNB
from sklearn.neural_network  import MLPClassifier
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score, f1_score, confusion_matrix

# optional SMOTE
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False


# ============================================================
# CONFIGURATION  — edit paths to match your setup
# ============================================================

TRAIN_FOLDER = (
    r"D:\INCOIS\Agro_project\data"
    r"\Arabian_sea_gridwise_split\train"
)
TEST_FOLDER = (
    r"D:\INCOIS\Agro_project\data"
    r"\Arabian_sea_gridwise_split\test"
)
RESULTS_ROOT = r"D:\INCOIS\Agro_project\results"
OUTPUT_DIR   = os.path.join(RESULTS_ROOT, "All_grids_models")

VAL_SIZE     = 0.15
RANDOM_STATE = 42
RESUME       = True          # skip grids already completed

N_NEW_MODELS           = 4
N_TARGETS              = 2
EXPECTED_ROWS_PER_GRID = N_NEW_MODELS * N_TARGETS

# Leave 1 core free for the OS
# Set to 1 if you hit memory issues
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)

# ============================================================
# FEATURES & TARGETS  (keep in sync with 6-model script)
# ============================================================

FEATURE_COLS = [
    "latitude", "longitude",
    "depth", "depth_bin", "pressure",
    "temperature", "salinity",
    "month", "season",
    "temp_grid_mean", "temp_grid_std",
    "sal_grid_mean",  "sal_grid_std",
    "n_obs",
    "temp_zscore", "sal_zscore",
    "temp_zscore_abs", "sal_zscore_abs",
    "temp_z_flag", "sal_z_flag",
]

TARGET_COLS = ["temp_qc", "psal_qc"]


# ============================================================
# HELPERS
# ============================================================

def grid_output_dir(grid_id):
    """Create and return per-grid results directory."""
    d = os.path.join(RESULTS_ROOT, grid_id)
    os.makedirs(os.path.join(d, "confusion_matrices"), exist_ok=True)
    return d


def build_4models():
    """
    Returns dict of 4 model configs.
    Re-instantiates fresh models every call (important for parallel workers).

    scaled           : True  → pass StandardScaler output to this model
    native_balance   : True  → model handles class imbalance internally;
                               skip SMOTE to save time & memory
    """
    return {
        # ── 1. HistGradientBoosting ─────────────────────────────────────
        "HistGradientBoosting": {
            "model": HistGradientBoostingClassifier(
                max_iter=300,
                max_depth=8,
                learning_rate=0.05,
                min_samples_leaf=20,
                l2_regularization=0.1,
                class_weight="balanced",
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=RANDOM_STATE,
                # n_jobs not supported by HistGradientBoostingClassifier
            ),
            "scaled": False,
            "native_balance": True,
        },

        # ── 2. LinearSVC (Calibrated for predict_proba) ─────────────────
        "LinearSVC": {
            "model": CalibratedClassifierCV(
                LinearSVC(
                    C=0.5,
                    max_iter=2000,
                    class_weight="balanced",
                    dual=False,         # faster when n_samples >> n_features
                    random_state=RANDOM_STATE,
                ),
                cv=3,
                method="sigmoid",
                n_jobs=1,
            ),
            "scaled": True,
            "native_balance": True,
        },

        # ── 3. GaussianNB ───────────────────────────────────────────────
        "GaussianNB": {
            "model": GaussianNB(var_smoothing=1e-8),
            "scaled": True,
            "native_balance": False,    # benefits from SMOTE rebalancing
        },

        # ── 4. MLP (Neural Network) ─────────────────────────────────────
        "MLP": {
            "model": MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=256,
                learning_rate="adaptive",
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                random_state=RANDOM_STATE,
            ),
            "scaled": True,
            "native_balance": False,    # benefits from SMOTE rebalancing
        },
    }


def evaluate_quiet(y_true, y_pred):
    acc    = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_wt  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return acc, f1_mac, f1_wt


# ============================================================
# PER-GRID WORKER
# NOTE: this function runs in a *subprocess* on Windows.
#       All globals (FEATURE_COLS, TARGET_COLS, etc.) must be
#       defined at module level so they are pickled correctly.
# ============================================================

def process_grid_4models(grid_id):
    """Train 4 models on one grid. Returns (grid_id, results_list, skipped_list)."""

    train_csv = os.path.join(TRAIN_FOLDER, f"{grid_id}_train.csv")
    test_csv  = os.path.join(TEST_FOLDER,  f"{grid_id}_test.csv")

    grid_results      = []
    skipped_log_local = []

    # ── RESUME CHECK ────────────────────────────────────────────────────
    grid_dir_path        = os.path.join(RESULTS_ROOT, grid_id)
    done_marker          = os.path.join(grid_dir_path, ".done_4models")
    results_path_existing = os.path.join(
        grid_dir_path, f"{grid_id}_4model_results.csv"
    )

    grid_is_done = False
    if RESUME and os.path.exists(done_marker):
        grid_is_done = True
    elif RESUME and os.path.exists(results_path_existing):
        try:
            prev_df_check = pd.read_csv(results_path_existing)
            if len(prev_df_check) >= EXPECTED_ROWS_PER_GRID:
                grid_is_done = True
                with open(done_marker, "w") as f:
                    f.write(f"backfilled at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"rows_found={len(prev_df_check)}\n")
        except Exception:
            pass

    if grid_is_done:
        try:
            prev_df = pd.read_csv(results_path_existing)
            if len(prev_df) > 0:
                print(f"  [{grid_id}] SKIP — 4 models already done", flush=True)
                return grid_id, prev_df.to_dict("records"), []
        except Exception as e:
            print(f"  [{grid_id}] WARNING: Could not reload results ({e}) — re-running", flush=True)

    # ── INPUT VALIDATION ────────────────────────────────────────────────
    if not os.path.exists(train_csv):
        print(f"  [{grid_id}] SKIP — no train file: {train_csv}", flush=True)
        skipped_log_local.append({"grid_id": grid_id, "reason": "no train file"})
        return grid_id, [], skipped_log_local

    if not os.path.exists(test_csv):
        print(f"  [{grid_id}] SKIP — no test file: {test_csv}", flush=True)
        skipped_log_local.append({"grid_id": grid_id, "reason": "no test file"})
        return grid_id, [], skipped_log_local

    grid_dir    = grid_output_dir(grid_id)
    grid_cm_dir = os.path.join(grid_dir, "confusion_matrices")

    # ── LOAD DATA ───────────────────────────────────────────────────────
    try:
        train_full_df = pd.read_csv(train_csv)
        test_df       = pd.read_csv(test_csv)
    except Exception as e:
        print(f"  [{grid_id}] ERROR loading CSV: {e}", flush=True)
        skipped_log_local.append({"grid_id": grid_id, "reason": f"load error: {e}"})
        return grid_id, [], skipped_log_local

    print(f"  [{grid_id}] Train {train_full_df.shape}  Test {test_df.shape}", flush=True)

    # ── FEATURE SELECTION ───────────────────────────────────────────────
    feat_cols = [c for c in FEATURE_COLS if c in train_full_df.columns]
    missing   = set(FEATURE_COLS) - set(feat_cols)
    if missing:
        print(f"  [{grid_id}] WARNING: Missing feature cols: {missing}", flush=True)

    X_all_raw  = train_full_df[feat_cols].copy()
    X_test_raw = test_df[feat_cols].copy()

    # Median imputation (fast, single pass)
    medians    = X_all_raw.median()
    X_all_raw  = X_all_raw.fillna(medians)
    X_test_raw = X_test_raw.fillna(medians)

    grid_t0 = time.time()

    # ── PER TARGET LOOP ─────────────────────────────────────────────────
    for target in TARGET_COLS:

        if target not in train_full_df.columns or target not in test_df.columns:
            print(f"  [{grid_id}] SKIP target '{target}' — column missing", flush=True)
            continue

        y_all_full  = train_full_df[target].copy()
        y_test_full = test_df[target].copy()

        valid_all  = y_all_full.notna()
        valid_test = y_test_full.notna()

        X_all = X_all_raw[valid_all].reset_index(drop=True)
        y_all = y_all_full[valid_all].reset_index(drop=True)
        X_te  = X_test_raw[valid_test].reset_index(drop=True)
        y_te  = y_test_full[valid_test].reset_index(drop=True)

        if y_all.nunique() < 2 or len(X_all) < 20 or len(X_te) < 5:
            print(
                f"  [{grid_id}] SKIP target '{target}' — insufficient data "
                f"(train={len(X_all)}, test={len(X_te)}, classes={y_all.nunique()})",
                flush=True,
            )
            continue

        # ── STRATIFIED TRAIN/VAL SPLIT ──────────────────────────────────
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_all, y_all,
                test_size=VAL_SIZE,
                stratify=y_all,
                random_state=RANDOM_STATE,
            )
        except ValueError:
            # Fall back if stratify fails (very small minority class)
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_all, y_all,
                test_size=VAL_SIZE,
                random_state=RANDOM_STATE,
            )

        X_tr  = X_tr.reset_index(drop=True);  y_tr  = y_tr.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True); y_val = y_val.reset_index(drop=True)

        print(
            f"  [{grid_id}] TARGET: {target}  "
            f"Train={len(X_tr):,}  Val={len(X_val):,}  Test={len(X_te):,}  "
            f"Classes={sorted(y_all.unique().tolist())}",
            flush=True,
        )

        # ── SMOTE (computed once per target, reused by non-native models) ─
        if HAS_SMOTE:
            class_counts   = y_tr.value_counts()
            min_class_size = class_counts.min()
            k_neighbors    = max(1, min(5, min_class_size - 1))

            if min_class_size > 1 and len(class_counts) > 1:
                try:
                    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
                    X_tr_res_arr, y_tr_res_arr = smote.fit_resample(X_tr, y_tr)
                    X_tr_res = pd.DataFrame(X_tr_res_arr, columns=X_tr.columns)
                    y_tr_res = pd.Series(y_tr_res_arr)
                    print(f"  [{grid_id}]   SMOTE: {len(X_tr):,} -> {len(X_tr_res):,}", flush=True)
                    smote_ok = True
                except Exception as e:
                    print(f"  [{grid_id}]   SMOTE failed ({e}) — using original", flush=True)
                    X_tr_res, y_tr_res = X_tr.copy(), y_tr.copy()
                    smote_ok = False
            else:
                X_tr_res, y_tr_res = X_tr.copy(), y_tr.copy()
                smote_ok = False
        else:
            X_tr_res, y_tr_res = X_tr.copy(), y_tr.copy()
            smote_ok = False

        # ── SCALING — fit ONCE, shared across all 4 models ───────────────
        scaler       = StandardScaler()
        X_tr_res_sc  = scaler.fit_transform(X_tr_res)   # SMOTE'd + scaled
        X_val_sc     = scaler.transform(X_val)
        X_te_sc      = scaler.transform(X_te)

        # Unscaled arrays (for HistGBM which doesn't need scaling)
        X_tr_raw_arr = X_tr_res.values
        X_val_raw_arr = X_val.values
        X_te_raw_arr  = X_te.values

        # Unscaled, unSMOTEd (for native_balance=True models)
        X_tr_native_sc  = scaler.transform(X_tr)
        X_tr_native_raw = X_tr.values

        le = LabelEncoder()
        le.fit(y_all)

        # ── MODEL LOOP ──────────────────────────────────────────────────
        models = build_4models()

        for model_name, cfg in models.items():

            model          = cfg["model"]
            use_scaled     = cfg.get("scaled", False)
            native_balance = cfg.get("native_balance", False)

            # Models with native balancing use original (non-SMOTE'd) data
            if native_balance:
                Xtr_fit = X_tr_native_sc  if use_scaled else X_tr_native_raw
                ytr_fit = y_tr.values
            else:
                Xtr_fit = X_tr_res_sc     if use_scaled else X_tr_raw_arr
                ytr_fit = y_tr_res.values

            Xval_fit = X_val_sc    if use_scaled else X_val_raw_arr
            Xte_fit  = X_te_sc     if use_scaled else X_te_raw_arr

            try:
                t0 = time.time()
                model.fit(Xtr_fit, ytr_fit)
                train_time = time.time() - t0

                y_val_pred = model.predict(Xval_fit)
                val_acc, val_f1_mac, val_f1_wt = evaluate_quiet(y_val, y_val_pred)

                y_te_pred = model.predict(Xte_fit)
                te_acc, te_f1_mac, te_f1_wt = evaluate_quiet(y_te, y_te_pred)

                # Confusion matrix
                labels_sorted = sorted(y_all.unique().tolist())
                cm = confusion_matrix(y_te, y_te_pred, labels=labels_sorted)
                cm_df = pd.DataFrame(
                    cm,
                    index=[f"true_{l}" for l in labels_sorted],
                    columns=[f"pred_{l}" for l in labels_sorted],
                )
                cm_path = os.path.join(
                    grid_cm_dir,
                    f"{grid_id}_{target}_{model_name}_cm.csv",
                )
                cm_df.to_csv(cm_path)

                print(
                    f"  [{grid_id}]   {model_name:<22} "
                    f"val_f1m={val_f1_mac:.3f}  test_f1m={te_f1_mac:.3f}  "
                    f"test_acc={te_acc:.3f}  ({train_time:.1f}s)",
                    flush=True,
                )

                grid_results.append({
                    "grid_id"              : grid_id,
                    "target"               : target,
                    "model"                : model_name,
                    "val_accuracy"         : round(val_acc,    4),
                    "val_f1_macro"         : round(val_f1_mac, 4),
                    "val_f1_weighted"      : round(val_f1_wt,  4),
                    "test_accuracy"        : round(te_acc,     4),
                    "test_f1_macro"        : round(te_f1_mac,  4),
                    "test_f1_weighted"     : round(te_f1_wt,   4),
                    "train_time_s"         : round(train_time, 2),
                    "train_rows_resampled" : len(Xtr_fit),
                    "val_rows"             : len(X_val),
                    "test_rows"            : len(X_te),
                })

            except Exception as e:
                print(f"  [{grid_id}]   ERROR {model_name}: {e}", flush=True)
                skipped_log_local.append({
                    "grid_id": grid_id,
                    "target" : target,
                    "model"  : model_name,
                    "reason" : str(e),
                })

    grid_elapsed = time.time() - grid_t0
    print(f"  [{grid_id}] Done in {grid_elapsed/60:.1f} min", flush=True)

    # ── SAVE PER-GRID RESULTS ────────────────────────────────────────────
    if grid_results:
        grid_results_df   = pd.DataFrame(grid_results)
        grid_results_path = os.path.join(
            os.path.join(RESULTS_ROOT, grid_id),
            f"{grid_id}_4model_results.csv",
        )
        grid_results_df.to_csv(grid_results_path, index=False)
        print(f"  [{grid_id}] Saved -> {grid_results_path}", flush=True)

        # Write .done marker
        marker_path = os.path.join(RESULTS_ROOT, grid_id, ".done_4models")
        with open(marker_path, "w") as f:
            f.write(f"completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"rows_saved={len(grid_results_df)}\n")

    return grid_id, grid_results, skipped_log_local


# ============================================================
# MERGE FUNCTION
# ============================================================

def merge_6_and_4_model_results():
    """
    Combines *_6model_results.csv + *_4model_results.csv for every grid
    into one master: All_grids_models/all_grids_10model_results.csv
    Safe to re-run — always rebuilds from per-grid CSVs.
    """
    print("\n" + "=" * 70)
    print("  MERGING 6-MODEL + 4-MODEL RESULTS  ->  10-MODEL MASTER CSV")
    print("=" * 70)

    all_rows = []
    grid_dirs = [
        d for d in Path(RESULTS_ROOT).iterdir()
        if d.is_dir() and d.name != "All_grids_models"
    ]

    for gd in sorted(grid_dirs):
        grid_id = gd.name
        csv_6   = gd / f"{grid_id}_6model_results.csv"
        csv_4   = gd / f"{grid_id}_4model_results.csv"

        parts = []
        for csv_path, label in [(csv_6, "6-model"), (csv_4, "4-model")]:
            if csv_path.exists():
                try:
                    parts.append(pd.read_csv(csv_path))
                except Exception as e:
                    print(f"  WARNING [{grid_id}] Could not read {label} CSV: {e}")
            else:
                print(f"  WARNING [{grid_id}] No {label} CSV found")

        if parts:
            combined = pd.concat(parts, ignore_index=True)
            all_rows.append(combined)
            print(f"  [{grid_id}] merged {len(combined)} rows "
                  f"({'+'.join(str(len(p)) for p in parts)})")

    if not all_rows:
        print("  ERROR: No results found to merge.")
        return

    master_df  = pd.concat(all_rows, ignore_index=True)
    master_csv = os.path.join(OUTPUT_DIR, "all_grids_10model_results.csv")
    master_df.to_csv(master_csv, index=False)
    print(f"\nMaster CSV saved -> {master_csv}  ({len(master_df)} total rows)")

    # Best model per grid per target
    best_rows = (
        master_df
        .loc[master_df.groupby(["grid_id", "target"])["test_f1_macro"].idxmax()]
        .sort_values(["grid_id", "target"])
    )
    best_path = os.path.join(OUTPUT_DIR, "best_model_per_grid_target_10models.csv")
    best_rows.to_csv(best_path, index=False)
    print(f"Best-model summary -> {best_path}")

    print("\n" + "=" * 70)
    print("  MODEL WIN COUNT  (best f1_macro per grid/target)")
    print("=" * 70)
    print(best_rows["model"].value_counts().to_string())

    print("\n" + "=" * 70)
    print("  AVERAGE TEST F1-MACRO PER MODEL  (all grids + targets)")
    print("=" * 70)
    avg_perf = (
        master_df
        .groupby("model")[["test_accuracy", "test_f1_macro", "test_f1_weighted"]]
        .mean()
        .sort_values("test_f1_macro", ascending=False)
    )
    print(avg_perf.round(4).to_string())

    print("\n" + "=" * 70)
    print("  NEW vs ORIGINAL — avg test_f1_macro")
    print("=" * 70)
    new_models = {"HistGradientBoosting", "LinearSVC", "GaussianNB", "MLP"}
    master_df["model_group"] = master_df["model"].apply(
        lambda m: "NEW (4)" if m in new_models else "ORIGINAL (6)"
    )
    print(
        master_df.groupby("model_group")["test_f1_macro"]
        .mean().round(4).to_string()
    )
    print("=" * 70)


# ============================================================
# MAIN  —  Windows REQUIRES the if __name__ == "__main__" guard
#           Without it, each subprocess re-runs the whole script
#           and spawns infinite processes.
# ============================================================

if __name__ == "__main__":

    # Verify folders exist before starting
    for folder, label in [
        (TRAIN_FOLDER, "TRAIN_FOLDER"),
        (TEST_FOLDER,  "TEST_FOLDER"),
        (RESULTS_ROOT, "RESULTS_ROOT"),
    ]:
        if not os.path.exists(folder):
            raise FileNotFoundError(
                f"{label} not found: {folder}\n"
                "Please update the CONFIGURATION section at the top of this script."
            )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_files = sorted(Path(TRAIN_FOLDER).glob("*_train.csv"))
    GRID_IDS    = [f.stem.replace("_train", "") for f in train_files]

    if not GRID_IDS:
        raise RuntimeError(
            f"No *_train.csv files found in: {TRAIN_FOLDER}\n"
            "Check that TRAIN_FOLDER is correct."
        )

    print("=" * 70)
    print("  4 NEW MODELS  —  ALL GRIDS  (PARALLEL)")
    print("  HistGradientBoosting | LinearSVC | GaussianNB | MLP")
    print("=" * 70)
    print(f"Grids found  : {len(GRID_IDS)}")
    print(f"Grid IDs     : {GRID_IDS}")
    print(f"Max workers  : {MAX_WORKERS}  (cpu_count={os.cpu_count()})")
    print(f"SMOTE        : {'enabled' if HAS_SMOTE else 'DISABLED (pip install imbalanced-learn to enable)'}")
    print(f"RESUME       : {RESUME}")
    print()

    all_results = []
    skipped_log = []
    total_start = time.time()

    # ── PARALLEL GRID PROCESSING ─────────────────────────────────────────
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_grid_4models, gid): gid
            for gid in GRID_IDS
        }

        for future in as_completed(futures):
            gid = futures[future]
            try:
                _, grid_results, grid_skipped = future.result()
                all_results.extend(grid_results)
                skipped_log.extend(grid_skipped)

                # Incremental save after each grid completes (crash-safe)
                if all_results:
                    pd.DataFrame(all_results).to_csv(
                        os.path.join(
                            OUTPUT_DIR,
                            "all_grids_4model_results_incremental.csv",
                        ),
                        index=False,
                    )
                print(
                    f"\n  [{gid}] done — incremental CSV updated "
                    f"({len(all_results)} rows so far)\n",
                    flush=True,
                )

            except Exception as e:
                import traceback
                print(f"\n  ERROR in grid {gid}: {e}", flush=True)
                traceback.print_exc()
                skipped_log.append({"grid_id": gid, "reason": f"worker exception: {e}"})

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  4-MODEL TRAINING COMPLETE  —  {total_elapsed/60:.1f} min total")
    print(f"{'='*70}")

    # ── SAVE STANDALONE 4-MODEL MASTER ──────────────────────────────────
    results_df = pd.DataFrame(all_results)
    standalone_path = os.path.join(OUTPUT_DIR, "all_grids_4model_results.csv")
    results_df.to_csv(standalone_path, index=False)
    print(f"4-model results saved -> {standalone_path}")

    # ── SAVE SKIPPED LOG ─────────────────────────────────────────────────
    skipped_df = (
        pd.DataFrame(skipped_log)
        if skipped_log
        else pd.DataFrame(columns=["grid_id", "reason"])
    )
    skipped_path = os.path.join(OUTPUT_DIR, "skipped_log_4models.csv")
    skipped_df.to_csv(skipped_path, index=False)
    if skipped_log:
        print(f"Skipped log  saved -> {skipped_path}  ({len(skipped_log)} entries)")

    # ── AUTO-MERGE WITH 6-MODEL RESULTS ─────────────────────────────────
    merge_6_and_4_model_results()

    print(f"\nALL DONE")
    print(f"  4-model results : {standalone_path}")
    print(f"  10-model master : {os.path.join(OUTPUT_DIR, 'all_grids_10model_results.csv')}")
    print(f"  Per-grid CSVs   : {RESULTS_ROOT}\\<grid_id>\\<grid_id>_4model_results.csv")