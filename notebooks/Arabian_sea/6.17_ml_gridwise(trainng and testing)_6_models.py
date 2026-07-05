import os
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FOLDER = (
    r"D:\INCOIS\Agro_project\data"
    r"\Arabian_sea_gridwise_split\train"
)
TEST_FOLDER = (
    r"D:\INCOIS\Agro_project\data"
    r"\Arabian_sea_gridwise_split\test"
)

RESULTS_ROOT = (
    r"D:\INCOIS\Agro_project\results"
)

OUTPUT_DIR = os.path.join(RESULTS_ROOT, "All_grids_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VAL_SIZE     = 0.15
RANDOM_STATE = 42

RESUME = True
N_MODELS  = 6
N_TARGETS = 2
EXPECTED_ROWS_PER_GRID = N_MODELS * N_TARGETS

# ============================================================
# PARALLELISM SETTING
# Each worker handles one grid at a time.
# n_jobs inside models is set to 1 so workers don't fight
# over CPU cores — the outer pool provides the parallelism.
# Start with MAX_WORKERS=2 and increase if RAM allows.
# ============================================================
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)

# ============================================================
# FEATURES & TARGETS
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
# IMPORTS
# ============================================================

from sklearn.ensemble        import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (
    accuracy_score, f1_score, confusion_matrix
)
import xgboost  as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False


# ============================================================
# HELPERS (must be module-level for pickling by ProcessPoolExecutor)
# ============================================================

def grid_output_dir(grid_id):
    """Per-grid results folder: results/<grid_id>/"""
    d = os.path.join(RESULTS_ROOT, grid_id)
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(d, "confusion_matrices"), exist_ok=True)
    return d


def build_models():
    """
    Fresh, unfitted models every call.
    n_jobs=1 inside each model — parallelism comes from the outer
    ProcessPoolExecutor, not from within each model.
    """
    return {
        "RandomForest": {
            "model": RandomForestClassifier(
                n_estimators=200, max_depth=20, min_samples_leaf=5,
                n_jobs=1, random_state=RANDOM_STATE, class_weight="balanced",
            ),
            "scaled": False,
        },
        "ExtraTrees": {
            "model": ExtraTreesClassifier(
                n_estimators=200, max_depth=20, min_samples_leaf=5,
                n_jobs=1, random_state=RANDOM_STATE, class_weight="balanced",
            ),
            "scaled": False,
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                n_jobs=1, random_state=RANDOM_STATE, verbosity=0,
            ),
            "scaled": False, "encode_labels": True,
        },
        "CatBoost": {
            "model": CatBoostClassifier(
                iterations=300, depth=8, learning_rate=0.05,
                random_seed=RANDOM_STATE, verbose=0,
                auto_class_weights="Balanced",
            ),
            "scaled": False,
        },
        "LightGBM": {
            "model": lgb.LGBMClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                n_jobs=1, random_state=RANDOM_STATE,
                class_weight="balanced", verbose=-1,
            ),
            "scaled": False,
        },
        "LogisticRegression": {
            "model": LogisticRegression(
                max_iter=1000, solver="saga", multi_class="auto",
                class_weight="balanced", n_jobs=1, random_state=RANDOM_STATE,
            ),
            "scaled": True,
        },
    }


def evaluate_quiet(y_true, y_pred):
    acc    = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_wt  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return acc, f1_mac, f1_wt


# ============================================================
# PER-GRID WORKER FUNCTION
# This is the entire original per-grid block, refactored to
# run in a separate process. No logic changes whatsoever.
# Returns (grid_id, grid_results_list, skipped_log_list)
# ============================================================

def process_grid(grid_id):
    """Process a single grid: train/val/test + 6 models x 2 targets."""

    train_csv = os.path.join(TRAIN_FOLDER, f"{grid_id}_train.csv")
    test_csv  = os.path.join(TEST_FOLDER,  f"{grid_id}_test.csv")

    grid_results = []
    skipped_log_local = []

    # --------------------------------------------------------
    # RESUME CHECK
    # --------------------------------------------------------
    grid_dir_check        = os.path.join(RESULTS_ROOT, grid_id)
    done_marker           = os.path.join(grid_dir_check, ".done")
    results_path_existing = os.path.join(
        grid_dir_check, f"{grid_id}_6model_results.csv"
    )

    grid_is_done = False

    if RESUME and os.path.exists(done_marker):
        grid_is_done = True
    elif RESUME and os.path.exists(results_path_existing):
        try:
            prev_df_check = pd.read_csv(results_path_existing)
            if len(prev_df_check) >= EXPECTED_ROWS_PER_GRID:
                grid_is_done = True
                print(f"  [{grid_id}] ℹ No .done marker but CSV has "
                      f"{len(prev_df_check)} rows — backfilling marker.")
                with open(done_marker, "w") as f:
                    f.write(f"backfilled at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"rows_found={len(prev_df_check)}\n")
        except Exception:
            grid_is_done = False

    if grid_is_done:
        try:
            prev_df = pd.read_csv(results_path_existing)
            if len(prev_df) > 0:
                print(f"  [{grid_id}] ⏩ SKIP (already completed)")
                return grid_id, prev_df.to_dict("records"), []
        except Exception as e:
            print(f"  [{grid_id}] ⚠ Could not reload previous results ({e}) — re-running")

    if not os.path.exists(test_csv):
        print(f"  [{grid_id}] ⚠ SKIP — no matching test file: {test_csv}")
        skipped_log_local.append({"grid_id": grid_id, "reason": "no test file"})
        return grid_id, [], skipped_log_local

    grid_dir    = grid_output_dir(grid_id)
    grid_cm_dir = os.path.join(grid_dir, "confusion_matrices")

    try:
        train_full_df = pd.read_csv(train_csv)
        test_df       = pd.read_csv(test_csv)
    except Exception as e:
        print(f"  [{grid_id}] ❌ ERROR loading: {e}")
        skipped_log_local.append({"grid_id": grid_id, "reason": f"load error: {e}"})
        return grid_id, [], skipped_log_local

    print(f"  [{grid_id}] Train {train_full_df.shape}  Test {test_df.shape}")

    feat_cols = [c for c in FEATURE_COLS if c in train_full_df.columns]
    missing   = set(FEATURE_COLS) - set(feat_cols)
    if missing:
        print(f"  [{grid_id}] ⚠ Missing feature cols: {missing}")

    X_all_raw  = train_full_df[feat_cols].copy()
    X_test_raw = test_df[feat_cols].copy()

    for col in feat_cols:
        med = X_all_raw[col].median()
        X_all_raw[col]  = X_all_raw[col].fillna(med)
        X_test_raw[col] = X_test_raw[col].fillna(med)

    grid_t0 = time.time()

    # --------------------------------------------------------
    # PER TARGET
    # --------------------------------------------------------
    for target in TARGET_COLS:

        if target not in train_full_df.columns or target not in test_df.columns:
            print(f"  [{grid_id}] ⚠ SKIP target '{target}' — column missing")
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
            print(f"  [{grid_id}] ⚠ SKIP target '{target}' — insufficient data "
                  f"(train={len(X_all)}, test={len(X_te)}, classes={y_all.nunique()})")
            continue

        # ----------------------------------------------------
        # STRATIFIED TRAIN / VALIDATION SPLIT
        # ----------------------------------------------------
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_all, y_all,
                test_size=VAL_SIZE,
                stratify=y_all,
                random_state=RANDOM_STATE,
            )
        except ValueError:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_all, y_all,
                test_size=VAL_SIZE,
                random_state=RANDOM_STATE,
            )

        X_tr  = X_tr.reset_index(drop=True);  y_tr  = y_tr.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True); y_val = y_val.reset_index(drop=True)

        print(f"  [{grid_id}] TARGET: {target}  Train={len(X_tr):,}  "
              f"Val={len(X_val):,}  Test={len(X_te):,}  "
              f"Classes={sorted(y_all.unique().tolist())}")

        # ----------------------------------------------------
        # SMOTE — training fold only
        # ----------------------------------------------------
        if HAS_SMOTE:
            class_counts   = y_tr.value_counts()
            min_class_size = class_counts.min()
            k_neighbors    = max(1, min(5, min_class_size - 1))

            if min_class_size > 1 and len(class_counts) > 1:
                try:
                    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
                    X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
                    X_tr_res = pd.DataFrame(X_tr_res, columns=X_tr.columns)
                    y_tr_res = pd.Series(y_tr_res)
                    print(f"  [{grid_id}]   ✅ SMOTE: {len(X_tr):,} → {len(X_tr_res):,} "
                          f"(k={k_neighbors})")
                except Exception as e:
                    print(f"  [{grid_id}]   ⚠ SMOTE failed ({e}) — using original data")
                    X_tr_res, y_tr_res = X_tr, y_tr
            else:
                print(f"  [{grid_id}]   ⚠ SMOTE skipped — too few samples in a class")
                X_tr_res, y_tr_res = X_tr, y_tr
        else:
            X_tr_res, y_tr_res = X_tr, y_tr

        # ----------------------------------------------------
        # SCALING — fit on resampled train only
        # ----------------------------------------------------
        scaler      = StandardScaler()
        X_tr_res_sc = scaler.fit_transform(X_tr_res)
        X_val_sc    = scaler.transform(X_val)
        X_te_sc     = scaler.transform(X_te)

        le = LabelEncoder()
        le.fit(y_all)

        # ----------------------------------------------------
        # LOOP OVER 6 MODELS
        # ----------------------------------------------------
        models = build_models()

        for model_name, cfg in models.items():

            model      = cfg["model"]
            use_scaled = cfg.get("scaled", False)
            enc_labels = cfg.get("encode_labels", False)

            Xtr_fit  = X_tr_res_sc if use_scaled else X_tr_res.values
            Xval_fit = X_val_sc    if use_scaled else X_val.values
            Xte_fit  = X_te_sc     if use_scaled else X_te.values
            ytr_fit  = y_tr_res.values

            if enc_labels:
                ytr_fit = le.transform(ytr_fit)

            try:
                t0 = time.time()
                model.fit(Xtr_fit, ytr_fit)
                train_time = time.time() - t0

                y_val_pred = model.predict(Xval_fit)
                if enc_labels:
                    y_val_pred = le.inverse_transform(y_val_pred)
                val_acc, val_f1_mac, val_f1_wt = evaluate_quiet(y_val, y_val_pred)

                y_te_pred = model.predict(Xte_fit)
                if enc_labels:
                    y_te_pred = le.inverse_transform(y_te_pred)
                te_acc, te_f1_mac, te_f1_wt = evaluate_quiet(y_te, y_te_pred)

                # Confusion matrix -> file
                labels_sorted = sorted(y_all.unique().tolist())
                cm = confusion_matrix(y_te, y_te_pred, labels=labels_sorted)
                cm_df = pd.DataFrame(
                    cm,
                    index=[f"true_{l}" for l in labels_sorted],
                    columns=[f"pred_{l}" for l in labels_sorted],
                )
                cm_path = os.path.join(
                    grid_cm_dir, f"{grid_id}_{target}_{model_name}_cm.csv"
                )
                cm_df.to_csv(cm_path)

                print(f"  [{grid_id}]   {model_name:<20} "
                      f"val_f1m={val_f1_mac:.3f}  test_f1m={te_f1_mac:.3f}  "
                      f"test_acc={te_acc:.3f}  ({train_time:.1f}s)")

                result_row = {
                    "grid_id"              : grid_id,
                    "target"               : target,
                    "model"                : model_name,
                    "val_accuracy"         : round(val_acc,    4),
                    "val_f1_macro"         : round(val_f1_mac, 4),
                    "val_f1_weighted"      : round(val_f1_wt,  4),
                    "test_accuracy"        : round(te_acc,    4),
                    "test_f1_macro"        : round(te_f1_mac, 4),
                    "test_f1_weighted"     : round(te_f1_wt,  4),
                    "train_time_s"         : round(train_time, 2),
                    "train_rows_resampled" : len(Xtr_fit),
                    "val_rows"             : len(X_val),
                    "test_rows"            : len(X_te),
                }
                grid_results.append(result_row)

            except Exception as e:
                print(f"  [{grid_id}]   ❌ {model_name} FAILED: {e}")
                skipped_log_local.append({
                    "grid_id": grid_id, "target": target,
                    "model": model_name, "reason": str(e)
                })

    grid_elapsed = time.time() - grid_t0
    print(f"  [{grid_id}] ⏱ Done in {grid_elapsed/60:.1f} min")

    # --------------------------------------------------------
    # SAVE PER-GRID RESULTS
    # --------------------------------------------------------
    grid_results_df   = pd.DataFrame(grid_results)
    grid_results_path = os.path.join(grid_dir, f"{grid_id}_6model_results.csv")
    grid_results_df.to_csv(grid_results_path, index=False)
    print(f"  [{grid_id}] ✅ Saved → {grid_results_path}")

    if len(grid_results_df) > 0:
        grid_best = (
            grid_results_df.loc[
                grid_results_df.groupby("target")["test_f1_macro"].idxmax()
            ]
        )
        grid_best_path = os.path.join(grid_dir, f"{grid_id}_best_model.csv")
        grid_best.to_csv(grid_best_path, index=False)
        print(f"  [{grid_id}] ✅ Saved → {grid_best_path}")

    # Write .done marker
    with open(os.path.join(grid_dir, ".done"), "w") as f:
        f.write(f"completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"rows_saved={len(grid_results_df)}\n")

    return grid_id, grid_results, skipped_log_local


# ============================================================
# MAIN — parallel dispatch
# ============================================================

if __name__ == "__main__":

    # Discover grids
    train_files = sorted(Path(TRAIN_FOLDER).glob("*_train.csv"))
    GRID_IDS    = [f.stem.replace("_train", "") for f in train_files]

    print("=" * 70)
    print("  6-MODEL QC CLASSIFICATION  —  ALL GRIDS  (PARALLEL)")
    print("  (Train / Validation split + SMOTE + Held-out Test)")
    print("=" * 70)
    print(f"\nGrids found  : {len(GRID_IDS)}")
    print(f"Grid IDs     : {GRID_IDS}")
    print(f"Max workers  : {MAX_WORKERS}  (cpu_count={os.cpu_count()})")

    if not HAS_SMOTE:
        print("\n⚠ imbalanced-learn not installed — run: pip install imbalanced-learn")
        print("  Continuing WITHOUT SMOTE (class_weight balancing only)\n")

    all_results = []
    skipped_log = []
    total_start = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_grid, gid): gid for gid in GRID_IDS}

        for future in as_completed(futures):
            gid = futures[future]
            try:
                _, grid_results, grid_skipped = future.result()
                all_results.extend(grid_results)
                skipped_log.extend(grid_skipped)

                # Incremental master save after every completed grid (crash-safe)
                pd.DataFrame(all_results).to_csv(
                    os.path.join(OUTPUT_DIR, "all_grids_6model_results.csv"),
                    index=False,
                )
                print(f"\n  ✅ [{gid}] collected — master CSV updated "
                      f"({len(all_results)} total rows so far)\n")

            except Exception as e:
                print(f"\n  ❌ Grid {gid} raised an exception in worker: {e}\n")
                skipped_log.append({"grid_id": gid, "reason": f"worker exception: {e}"})

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  ALL GRIDS COMPLETE  —  total time: {total_elapsed/60:.1f} min")
    print(f"{'='*70}")

    # ============================================================
    # FINAL SAVE
    # ============================================================

    results_df = pd.DataFrame(all_results)
    out_csv    = os.path.join(OUTPUT_DIR, "all_grids_6model_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\n✅ Final results saved  → {out_csv}")

    skipped_df   = pd.DataFrame(skipped_log) if skipped_log else pd.DataFrame(columns=["grid_id", "reason"])
    skipped_path = os.path.join(OUTPUT_DIR, "skipped_log.csv")
    skipped_df.to_csv(skipped_path, index=False)
    print(f"✅ Skipped log saved    → {skipped_path}")
    print(f"✅ Per-grid results     → {RESULTS_ROOT}\\<grid_id>\\<grid_id>_6model_results.csv")
    print(f"✅ Confusion matrices   → {RESULTS_ROOT}\\<grid_id>\\confusion_matrices\\")

    # ============================================================
    # SUMMARY — BEST MODEL PER GRID PER TARGET
    # ============================================================

    print("\n" + "=" * 70)
    print("  BEST MODEL PER GRID PER TARGET  (by test_f1_macro)")
    print("=" * 70)

    if len(results_df) > 0:
        best_rows = (
            results_df
            .loc[results_df.groupby(["grid_id", "target"])["test_f1_macro"].idxmax()]
            .sort_values(["grid_id", "target"])
        )
        print(best_rows[[
            "grid_id", "target", "model",
            "test_accuracy", "test_f1_macro", "test_f1_weighted"
        ]].to_string(index=False))

        best_path = os.path.join(OUTPUT_DIR, "best_model_per_grid_target.csv")
        best_rows.to_csv(best_path, index=False)
        print(f"\n✅ Best-model summary saved → {best_path}")

        print("\n" + "=" * 70)
        print("  MODEL WIN COUNT  (how often each model was best)")
        print("=" * 70)
        win_counts = best_rows["model"].value_counts()
        print(win_counts.to_string())

        print("\n" + "=" * 70)
        print("  AVERAGE TEST F1-MACRO PER MODEL  (across all grids/targets)")
        print("=" * 70)
        avg_perf = (
            results_df
            .groupby("model")[["test_accuracy", "test_f1_macro", "test_f1_weighted"]]
            .mean()
            .sort_values("test_f1_macro", ascending=False)
        )
        print(avg_perf.round(4).to_string())

    print("\n" + "=" * 70)
    print(f"  ✅ DONE — {len(GRID_IDS)} grids  |  6 models  |  2 targets each")
    print(f"  ✅ Workers used: {MAX_WORKERS}")
    print("=" * 70)