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
    r"\southern_indian_ocean_gridwise_split\train"
)
TEST_FOLDER = (
    r"D:\INCOIS\Agro_project\data"
    r"\southern_indian_ocean_gridwise_split\test"
)

RESULTS_ROOT = (
    r"D:\INCOIS\Agro_project\results_southern_indian_ocean_v2"
)

OUTPUT_DIR = os.path.join(RESULTS_ROOT, "All_grids_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VAL_SIZE     = 0.15
RANDOM_STATE = 42
RESUME       = True

FAST_MODE = True

if FAST_MODE:
    MODEL_N_JOBS = max(1, min(4, (os.cpu_count() or 4) // 2))
    MAX_WORKERS  = max(1, (os.cpu_count() or 4) // MODEL_N_JOBS)
else:
    MODEL_N_JOBS = 1
    MAX_WORKERS  = max(1, (os.cpu_count() or 4) - 1)

MAX_TRAIN_ROWS_PER_GRID = 150_000 if FAST_MODE else None
USE_SMOTE = not FAST_MODE
WRITE_CONFUSION_MATRICES = not FAST_MODE

N_TARGETS = 2

# ============================================================
# MODEL-SET CHANGE TRACKING
# ------------------------------------------------------------
# AdaBoost has been removed from build_models() and replaced with
# LinearSVC. RandomForest is still present, but its hyperparameters
# changed (more trees, unrestricted depth, sqrt features) — so any
# RandomForest rows saved by a PREVIOUS run used the OLD config and
# must be retrained, not skipped.
#
# REMOVED_MODELS      -> old rows for these are dropped and never
#                         retrained (the model no longer exists).
# FORCE_RETRAIN_MODELS -> old rows for these are dropped and ARE
#                         retrained fresh this run (config changed).
# Every other model's existing rows are left alone and skipped,
# exactly like normal RESUME behavior.
# ============================================================
REMOVED_MODELS       = {"AdaBoost"}
FORCE_RETRAIN_MODELS = {"RandomForest"}
STALE_MODELS         = REMOVED_MODELS | FORCE_RETRAIN_MODELS

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

USECOLS = list(dict.fromkeys(FEATURE_COLS + TARGET_COLS))

# Still 10 models total: HistGB, LightGBM, XGBoost, CatBoost,
# RandomForest, ExtraTrees, LinearSVC, MLP, LogisticRegression, GaussianNB
MODEL_COUNT            = 10
EXPECTED_ROWS_PER_GRID = MODEL_COUNT * N_TARGETS

# ============================================================
# IMPORTS
# ============================================================

from sklearn.ensemble        import (
    RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm             import LinearSVC
from sklearn.linear_model    import LogisticRegression
from sklearn.neural_network  import MLPClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
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

GROUP_CANDIDATE_COLS = [
    "profile_id", "cast_id", "float_id", "cycle_number",
    "juld", "file_name", "date", "time",
]

LOAD_COLS = list(dict.fromkeys(USECOLS + GROUP_CANDIDATE_COLS))


# ============================================================
# HELPERS
# ============================================================

def grid_output_dir(grid_id):
    d = os.path.join(RESULTS_ROOT, grid_id)
    os.makedirs(d, exist_ok=True)
    if WRITE_CONFUSION_MATRICES:
        os.makedirs(os.path.join(d, "confusion_matrices"), exist_ok=True)
    return d


def build_models():
    """
    FAST_MODE=True  -> 10 models, each tuned for speed:
      HistGradientBoosting : sklearn's fastest strong boosting impl.
      LightGBM              : fast, multithreaded (MODEL_N_JOBS).
      XGBoost                : fast, multithreaded (MODEL_N_JOBS),
                               histogram tree method.
      CatBoost                : fewer iterations, shallower depth,
                               multithreaded (MODEL_N_JOBS).
      RandomForest (strong)  : more trees, unrestricted depth,
                               sqrt features, balanced_subsample —
                               tuned to be the strongest model here.
      ExtraTrees (lighter)   : fewer trees, multithreaded.
      LinearSVC (lighter)    : linear SVM baseline, scaled inputs.
      MLP (lighter neural net): smaller hidden layers, fewer iters,
                               early stopping.
      LogisticRegression     : cheap linear baseline.
      GaussianNB              : already cheap, unchanged.

    FAST_MODE=False -> original heavier set, RandomForest strengthened
    and AdaBoost swapped for LinearSVC, everything else unchanged.
    """
    if FAST_MODE:
        return {
            "HistGradientBoosting": {
                "model": HistGradientBoostingClassifier(
                    max_iter=120, max_depth=7, learning_rate=0.1,
                    l2_regularization=0.1,
                    early_stopping=True, n_iter_no_change=10,
                    random_state=RANDOM_STATE,
                ),
                "scaled": False,
            },
            "LightGBM": {
                "model": lgb.LGBMClassifier(
                    n_estimators=150, max_depth=7, learning_rate=0.08,
                    subsample=0.8, colsample_bytree=0.8,
                    n_jobs=MODEL_N_JOBS, random_state=RANDOM_STATE,
                    class_weight="balanced", verbose=-1,
                ),
                "scaled": False,
            },
            "XGBoost": {
                "model": xgb.XGBClassifier(
                    n_estimators=150, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    tree_method="hist", eval_metric="mlogloss",
                    n_jobs=MODEL_N_JOBS, random_state=RANDOM_STATE, verbosity=0,
                ),
                "scaled": False, "encode_labels": True,
            },
            "CatBoost": {
                "model": CatBoostClassifier(
                    iterations=150, depth=6, learning_rate=0.1,
                    random_seed=RANDOM_STATE, verbose=0,
                    auto_class_weights="Balanced",
                    thread_count=MODEL_N_JOBS,
                ),
                "scaled": False,
            },
            "RandomForest": {
                "model": RandomForestClassifier(
                    n_estimators=400, max_depth=None, min_samples_leaf=2,
                    max_features="sqrt",
                    n_jobs=MODEL_N_JOBS, random_state=RANDOM_STATE,
                    class_weight="balanced_subsample",
                ),
                "scaled": False,
            },
            "ExtraTrees": {
                "model": ExtraTreesClassifier(
                    n_estimators=100, max_depth=15, min_samples_leaf=5,
                    n_jobs=MODEL_N_JOBS, random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
                "scaled": False,
            },
            "LinearSVC": {
                "model": LinearSVC(
                    C=1.0, loss="squared_hinge", dual=False,
                    class_weight="balanced", max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
                "scaled": True,
            },
            "MLP": {
                "model": MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    activation="relu", solver="adam",
                    alpha=1e-4, batch_size=256,
                    learning_rate_init=2e-3,
                    max_iter=150,
                    early_stopping=True, n_iter_no_change=8,
                    validation_fraction=0.1,
                    random_state=RANDOM_STATE,
                ),
                "scaled": True,
            },
            "LogisticRegression": {
                "model": LogisticRegression(
                    max_iter=300, solver="lbfgs", multi_class="auto",
                    class_weight="balanced", n_jobs=MODEL_N_JOBS,
                    random_state=RANDOM_STATE,
                ),
                "scaled": True,
            },
            "GaussianNB": {
                "model": GaussianNB(),
                "scaled": False,
            },
        }

    # ---------------- ORIGINAL HEAVIER 10-MODEL SET (slow) ----------------
    return {
        "RandomForest": {
            "model": RandomForestClassifier(
                n_estimators=500, max_depth=None, min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=MODEL_N_JOBS, random_state=RANDOM_STATE,
                class_weight="balanced_subsample",
            ),
            "scaled": False,
        },
        "ExtraTrees": {
            "model": ExtraTreesClassifier(
                n_estimators=200, max_depth=20, min_samples_leaf=5,
                n_jobs=MODEL_N_JOBS, random_state=RANDOM_STATE, class_weight="balanced",
            ),
            "scaled": False,
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                n_jobs=MODEL_N_JOBS, random_state=RANDOM_STATE, verbosity=0,
            ),
            "scaled": False, "encode_labels": True,
        },
        "CatBoost": {
            "model": CatBoostClassifier(
                iterations=300, depth=8, learning_rate=0.05,
                random_seed=RANDOM_STATE, verbose=0,
                auto_class_weights="Balanced",
                thread_count=MODEL_N_JOBS,
            ),
            "scaled": False,
        },
        "LightGBM": {
            "model": lgb.LGBMClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                n_jobs=MODEL_N_JOBS, random_state=RANDOM_STATE,
                class_weight="balanced", verbose=-1,
            ),
            "scaled": False,
        },
        "LogisticRegression": {
            "model": LogisticRegression(
                max_iter=1000, solver="saga", multi_class="auto",
                class_weight="balanced", n_jobs=MODEL_N_JOBS, random_state=RANDOM_STATE,
            ),
            "scaled": True,
        },
        "HistGradientBoosting": {
            "model": HistGradientBoostingClassifier(
                max_iter=200, max_depth=8, learning_rate=0.08,
                l2_regularization=0.1,
                early_stopping=True, n_iter_no_change=15,
                random_state=RANDOM_STATE,
            ),
            "scaled": False,
        },
        "LinearSVC": {
            "model": LinearSVC(
                C=1.0, loss="squared_hinge", dual=False,
                class_weight="balanced", max_iter=2000,
                random_state=RANDOM_STATE,
            ),
            "scaled": True,
        },
        "MLP": {
            "model": MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu", solver="adam",
                alpha=1e-4, batch_size=256,
                learning_rate_init=1e-3,
                max_iter=300,
                early_stopping=True, n_iter_no_change=10,
                validation_fraction=0.1,
                random_state=RANDOM_STATE,
            ),
            "scaled": True,
        },
        "GaussianNB": {
            "model": GaussianNB(),
            "scaled": False,
        },
    }


def evaluate_quiet(y_true, y_pred):
    acc    = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_wt  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return acc, f1_mac, f1_wt


def load_grid_csv(path):
    try:
        return pd.read_csv(path, usecols=lambda c: c in LOAD_COLS)
    except ValueError:
        df = pd.read_csv(path)
        keep = [c for c in LOAD_COLS if c in df.columns]
        return df[keep]


def detect_group_col(df):
    for col in GROUP_CANDIDATE_COLS:
        if col in df.columns:
            return col
    return None


# ============================================================
# PER-GRID WORKER FUNCTION
# ============================================================

def process_grid(grid_id):
    """Process a single grid: train/val/test + N models x 2 targets."""

    train_csv = os.path.join(TRAIN_FOLDER, f"{grid_id}_train.csv")
    test_csv  = os.path.join(TEST_FOLDER,  f"{grid_id}_test.csv")

    grid_results = []
    skipped_log_local = []

    # --------------------------------------------------------
    # RESUME CHECK
    # A grid can only be considered "fully done" if it has
    # EXPECTED_ROWS_PER_GRID rows for models OTHER than the stale
    # ones (removed / force-retrain) — those always need fresh work,
    # so a grid with a stale RandomForest is never short-circuited
    # as "done" even if a .done marker exists from a previous run.
    # --------------------------------------------------------
    grid_dir_check        = os.path.join(RESULTS_ROOT, grid_id)
    done_marker           = os.path.join(grid_dir_check, ".done")
    results_path_existing = os.path.join(
        grid_dir_check, f"{grid_id}_10model_results.csv"
    )

    grid_is_done = False

    if RESUME and os.path.exists(results_path_existing):
        try:
            prev_df_check = pd.read_csv(results_path_existing)
            has_stale_rows = prev_df_check["model"].isin(STALE_MODELS).any()
            if (not has_stale_rows) and os.path.exists(done_marker):
                if len(prev_df_check) >= EXPECTED_ROWS_PER_GRID:
                    grid_is_done = True
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

    # --------------------------------------------------------
    # PARTIAL RESUME — keep rows for models that are unchanged and
    # already done; drop rows for REMOVED_MODELS (no longer trained)
    # and FORCE_RETRAIN_MODELS (config changed, must retrain).
    # --------------------------------------------------------
    completed_pairs = set()
    if RESUME and os.path.exists(results_path_existing):
        try:
            prev_partial_df = pd.read_csv(results_path_existing)
            if len(prev_partial_df) > 0:
                dropped = prev_partial_df[prev_partial_df["model"].isin(STALE_MODELS)]
                prev_partial_df = prev_partial_df[~prev_partial_df["model"].isin(STALE_MODELS)]

                if len(dropped) > 0:
                    dropped_removed = dropped[dropped["model"].isin(REMOVED_MODELS)]
                    dropped_retrain = dropped[dropped["model"].isin(FORCE_RETRAIN_MODELS)]
                    if len(dropped_removed) > 0:
                        print(f"  [{grid_id}] 🗑 Dropped {len(dropped_removed)} row(s) "
                              f"for removed model(s): {sorted(dropped_removed['model'].unique())}")
                    if len(dropped_retrain) > 0:
                        print(f"  [{grid_id}] ♻ Dropped {len(dropped_retrain)} old row(s) "
                              f"for force-retrain model(s): {sorted(dropped_retrain['model'].unique())} "
                              f"— will retrain fresh")

                grid_results.extend(prev_partial_df.to_dict("records"))
                completed_pairs = set(
                    zip(prev_partial_df["target"], prev_partial_df["model"])
                )
                print(f"  [{grid_id}] ↻ Resuming — {len(completed_pairs)} "
                      f"(target, model) combo(s) already done, will skip those")
        except Exception as e:
            print(f"  [{grid_id}] ⚠ Could not read partial results ({e}) — "
                  f"starting this grid fresh")
            completed_pairs = set()

    grid_dir    = grid_output_dir(grid_id)
    grid_cm_dir = os.path.join(grid_dir, "confusion_matrices") if WRITE_CONFUSION_MATRICES else None

    try:
        train_full_df = load_grid_csv(train_csv)
        test_df       = load_grid_csv(test_csv)
    except Exception as e:
        print(f"  [{grid_id}] ❌ ERROR loading: {e}")
        skipped_log_local.append({"grid_id": grid_id, "reason": f"load error: {e}"})
        return grid_id, [], skipped_log_local

    if MAX_TRAIN_ROWS_PER_GRID is not None and len(train_full_df) > MAX_TRAIN_ROWS_PER_GRID:
        train_full_df = train_full_df.sample(
            n=MAX_TRAIN_ROWS_PER_GRID, random_state=RANDOM_STATE
        ).reset_index(drop=True)
        print(f"  [{grid_id}] ✂ Subsampled train to {MAX_TRAIN_ROWS_PER_GRID:,} rows")

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

        target_completed_models = {m for (t, m) in completed_pairs if t == target}
        if RESUME and len(target_completed_models) >= MODEL_COUNT:
            print(f"  [{grid_id}] ⏩ SKIP target '{target}' — all "
                  f"{MODEL_COUNT} models already done for this target")
            continue

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
        # TRAIN / VALIDATION SPLIT — leak-safe
        # ----------------------------------------------------
        group_col = detect_group_col(train_full_df)
        split_method = None

        if group_col is not None:
            groups_all = train_full_df[group_col][valid_all].reset_index(drop=True)
            n_groups   = groups_all.nunique()
            if n_groups >= 2:
                try:
                    gss = GroupShuffleSplit(
                        n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE
                    )
                    tr_idx, val_idx = next(
                        gss.split(X_all, y_all, groups=groups_all)
                    )
                    X_tr, X_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]
                    y_tr, y_val = y_all.iloc[tr_idx], y_all.iloc[val_idx]
                    split_method = f"group ({group_col}, {n_groups} groups)"
                except Exception as e:
                    print(f"  [{grid_id}]   ⚠ GroupShuffleSplit failed ({e}) "
                          f"— falling back to random-stratified split")
                    group_col = None
            else:
                group_col = None

        if group_col is None:
            print(f"  [{grid_id}]   ⚠ No group column found "
                  f"(tried {GROUP_CANDIDATE_COLS}) — using random-stratified "
                  f"split. NOTE: this carries a leakage risk if rows from "
                  f"the same cast/profile are near-duplicates.")
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
            split_method = "random-stratified (no group col — leakage risk)"

        X_tr  = X_tr.reset_index(drop=True);  y_tr  = y_tr.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True); y_val = y_val.reset_index(drop=True)

        print(f"  [{grid_id}] TARGET: {target}  Train={len(X_tr):,}  "
              f"Val={len(X_val):,}  Test={len(X_te):,}  "
              f"Classes={sorted(y_all.unique().tolist())}  "
              f"Split={split_method}")

        # ----------------------------------------------------
        # SMOTE — training fold only
        # ----------------------------------------------------
        if USE_SMOTE and HAS_SMOTE:
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
        # LOOP OVER MODELS
        # ----------------------------------------------------
        models = build_models()

        min_class_count = y_tr_res.value_counts().min() if len(y_tr_res) else 0
        if min_class_count < 10:
            models["HistGradientBoosting"]["model"].set_params(
                early_stopping=False
            )
            print(f"  [{grid_id}]   ℹ HGB early_stopping disabled — "
                  f"smallest class in train has only {min_class_count} rows")

        for model_name, cfg in models.items():

            if RESUME and (target, model_name) in completed_pairs:
                print(f"  [{grid_id}]   ⏩ SKIP {model_name} for '{target}' "
                      f"— already done")
                continue

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

                if WRITE_CONFUSION_MATRICES:
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
    grid_results_path = os.path.join(grid_dir, f"{grid_id}_10model_results.csv")
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

    with open(os.path.join(grid_dir, ".done"), "w") as f:
        f.write(f"completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"rows_saved={len(grid_results_df)}\n")

    return grid_id, grid_results, skipped_log_local


# ============================================================
# MAIN — parallel dispatch
# ============================================================

if __name__ == "__main__":

    train_files = sorted(Path(TRAIN_FOLDER).glob("*_train.csv"))
    GRID_IDS    = [f.stem.replace("_train", "") for f in train_files]

    print("=" * 70)
    print("  FAST QC CLASSIFICATION  —  ALL GRIDS  (PARALLEL)")
    print(f"  FAST_MODE={FAST_MODE}  |  Models=10 (all, {'lean/fast-tuned' if FAST_MODE else 'full/heavy'})")
    print(f"  MAX_WORKERS={MAX_WORKERS}  MODEL_N_JOBS={MODEL_N_JOBS}  "
          f"USE_SMOTE={USE_SMOTE}  MAX_TRAIN_ROWS_PER_GRID={MAX_TRAIN_ROWS_PER_GRID}")
    print(f"  RESULTS_ROOT={RESULTS_ROOT}")
    print(f"  REMOVED_MODELS={sorted(REMOVED_MODELS)}  "
          f"FORCE_RETRAIN_MODELS={sorted(FORCE_RETRAIN_MODELS)}")
    print("=" * 70)
    print(f"\nGrids found  : {len(GRID_IDS)}")
    print(f"Grid IDs     : {GRID_IDS}")

    if USE_SMOTE and not HAS_SMOTE:
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

                pd.DataFrame(all_results).to_csv(
                    os.path.join(OUTPUT_DIR, "all_grids_10model_results.csv"),
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
    out_csv    = os.path.join(OUTPUT_DIR, "all_grids_10model_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\n✅ Final results saved  → {out_csv}")

    skipped_df   = pd.DataFrame(skipped_log) if skipped_log else pd.DataFrame(columns=["grid_id", "reason"])
    skipped_path = os.path.join(OUTPUT_DIR, "skipped_log.csv")
    skipped_df.to_csv(skipped_path, index=False)
    print(f"✅ Skipped log saved    → {skipped_path}")
    print(f"✅ Per-grid results     → {RESULTS_ROOT}\\<grid_id>\\<grid_id>_10model_results.csv")
    if WRITE_CONFUSION_MATRICES:
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
    print(f"  ✅ DONE — {len(GRID_IDS)} grids  |  10 models  |  2 targets each")
    print(f"  ✅ Workers used: {MAX_WORKERS}  (model_n_jobs={MODEL_N_JOBS})")
    print("=" * 70)