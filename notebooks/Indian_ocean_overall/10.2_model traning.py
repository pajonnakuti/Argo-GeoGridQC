"""
train_gridwise_models_parallel_resume.py
==========================================
Parallel + resumable version of train_gridwise_models.py.

Trains 10 classifiers PER GRID, PER TARGET on the train_<grid>.csv / test_<grid>.csv
files produced by split_gridwise_chunked.py (run on the CLEANED parquet output of
clean_and_engineer_v2.py).

WHAT'S NEW vs. the plain script
--------------------------------
1) PARALLEL across grids
   - Each grid is independent work (read csvs -> train 10 models -> write
     results), so grids are farmed out to a process pool via
     concurrent.futures.ProcessPoolExecutor.
   - Control pool size with --workers (default: half your CPU cores).
   - To avoid CPU oversubscription (outer process parallelism x inner
     model n_jobs=-1 parallelism fighting each other), every model's
     internal n_jobs is forced down via --inner-jobs (default: 1) when
     running in worker processes. Total core usage stays roughly
     workers * inner_jobs.

2) RESUMABLE
   - Every grid's results are written to their OWN small file the moment
     that grid finishes:
       <output_dir>/grid_results/<grid>.csv    (per-model metrics)
       <output_dir>/predictions/<grid>.csv     (per-model y_true/y_pred,
                                                 needed to rebuild the
                                                 combined/all-grids report)
   - Before scheduling a grid, the script checks whether
     grid_results/<grid>.csv already exists. If so, that grid is skipped
     entirely (not even handed to a worker process).
   - A grid's result file is written ONLY after that grid finishes
     processing successfully. If a worker crashes partway through a grid
     (OOM, killed, uncaught exception), no partial file is left behind,
     so that grid will be retried automatically on the next run.
   - grid_wise_results.csv and combined_results.csv (the final reports)
     are rebuilt at the end of each run by scanning ALL grid_results/ and
     predictions/ files -- including ones from earlier runs -- so a
     resumed run's final reports are always complete, not just "what ran
     this time".

Run:
    python train_gridwise_models_parallel_resume.py
    python train_gridwise_models_parallel_resume.py --workers 6 --inner-jobs 2
    python train_gridwise_models_parallel_resume.py --workers 1   # effectively serial, still resumable

Requirements:
    pip install pandas numpy scikit-learn xgboost imbalanced-learn joblib
    pip install lightgbm catboost   # optional, for the 2 extra boosters
"""

import os
import re
import glob
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib

from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False


# ======================= CONFIG =======================
DATA_DIR   = "/home/incois/PAJO/pplWorks/geogrid/Indian_ocean/grid_splits"
OUTPUT_DIR = "/home/incois/PAJO/pplWorks/geogrid/Indian_ocean/model_results"

TARGET_COLS = ["temp_qc", "psal_qc"]

ALWAYS_DROP_COLS = ["date"]
QC_RELATED_COLS = ["temp_qc", "psal_qc", "z_flag_label"]

IMBALANCE_THRESHOLD = 1.5
SAVE_MODELS = True
RANDOM_STATE = 42

SCALED_MODEL_NAMES = {"LogisticRegression", "SVM"}

MODEL_NAMES_ALL = ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "ExtraTrees",
                    "DecisionTree", "LogisticRegression", "SVM", "GaussianNB", "KNN"]
# ========================================================


def find_grid_pairs(data_dir):
    """Match train_<grid>.csv with its test_<grid>.csv counterpart."""
    train_files = glob.glob(os.path.join(data_dir, "train_*.csv"))
    pairs = []
    for tf in train_files:
        base = os.path.basename(tf)
        grid_name = re.sub(r"^train_", "", base)
        grid_name = re.sub(r"\.csv$", "", grid_name)
        test_path = os.path.join(data_dir, f"test_{grid_name}.csv")
        if os.path.exists(test_path):
            pairs.append((grid_name, tf, test_path))
        else:
            print(f"  No matching test file for grid '{grid_name}', skipping.")
    return sorted(pairs, key=lambda p: p[0])


def normalize_feature_dtypes(df):
    """
    Cast pandas nullable extension dtypes, introduced by
    clean_and_engineer_v2.py, down to plain numpy dtypes that
    sklearn/XGBoost/joblib handle reliably.
    """
    for c in df.columns:
        dtype = df[c].dtype
        dtype_str = str(dtype)
        # Nullable string dtype doesn't always stringify as exactly
        # "string" (can render as "string[python]", "string[pyarrow]", or
        # "StringDtype(na_value=nan)" depending on pandas version/backend).
        # isinstance + startswith fallback catches all variants.
        is_nullable_string = isinstance(dtype, pd.StringDtype) or dtype_str.startswith("string")

        if dtype_str in ("Int64", "Int32", "Int16", "Int8",
                          "UInt64", "UInt32", "UInt16", "UInt8", "Float64", "Float32"):
            df[c] = df[c].astype("float64")
        elif is_nullable_string:
            df[c] = df[c].astype(object)
        elif dtype_str == "boolean":
            df[c] = df[c].astype("float64")
        elif dtype_str == "bool":
            df[c] = df[c].astype("int8")
        elif dtype_str.startswith("datetime64"):
            df[c] = (df[c].astype("int64", errors="ignore") / 86_400_000_000_000).astype("float64")
            df[c] = df[c].where(df[c].abs() < 1e15, np.nan)
    return df


def prep_features(train_df, test_df, target_col, drop_cols):
    """Normalize dtypes, encode categoricals, impute missing values, keep train/test aligned."""
    feature_cols = [c for c in train_df.columns if c != target_col and c not in drop_cols]

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    X_train = normalize_feature_dtypes(X_train)
    X_test = normalize_feature_dtypes(X_test)

    for c in feature_cols:
        if X_train[c].dtype == object or str(X_train[c].dtype).startswith("category"):
            X_train[c] = X_train[c].fillna("MISSING").astype(str)
            X_test[c] = X_test[c].fillna("MISSING").astype(str)
            le = LabelEncoder()
            combined = pd.concat([X_train[c], X_test[c]], axis=0)
            le.fit(combined)
            X_train[c] = le.transform(X_train[c])
            X_test[c] = le.transform(X_test[c])

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        X_train[numeric_cols] = X_train[numeric_cols].replace([np.inf, -np.inf], np.nan)
        X_test[numeric_cols] = X_test[numeric_cols].replace([np.inf, -np.inf], np.nan)

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        all_nan_cols = [c for c in numeric_cols if X_train[c].isna().all()]
        impute_cols = [c for c in numeric_cols if c not in all_nan_cols]

        if impute_cols:
            imputer = SimpleImputer(strategy="median")
            X_train[impute_cols] = imputer.fit_transform(X_train[impute_cols])
            X_test[impute_cols] = imputer.transform(X_test[impute_cols])

        for c in all_nan_cols:
            X_train[c] = 0.0
            X_test[c] = X_test[c].fillna(0.0)

    bad_train_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    bad_test_cols = X_test.select_dtypes(exclude=[np.number]).columns.tolist()
    bad_cols = sorted(set(bad_train_cols) | set(bad_test_cols))
    if bad_cols:
        print(f"   WARNING: non-numeric columns after prep, label-encoding as fallback: {bad_cols} "
              f"(dtypes: {X_train[bad_cols].dtypes.to_dict()})")
        for c in bad_cols:
            X_train[c] = X_train[c].fillna("MISSING").astype(str)
            X_test[c] = X_test[c].fillna("MISSING").astype(str)
            le = LabelEncoder()
            combined = pd.concat([X_train[c], X_test[c]], axis=0)
            le.fit(combined)
            X_train[c] = le.transform(X_train[c])
            X_test[c] = le.transform(X_test[c])

    X_train = X_train.astype("float64")
    X_test = X_test.astype("float64")

    if not np.isfinite(X_train.values).all():
        X_train = X_train.fillna(0.0)
        X_train = X_train.replace([np.inf, -np.inf], 0.0)
    if not np.isfinite(X_test.values).all():
        X_test = X_test.fillna(0.0)
        X_test = X_test.replace([np.inf, -np.inf], 0.0)

    return X_train, X_test


def maybe_smote(X_train, y_train_enc, threshold, random_state):
    counts = pd.Series(y_train_enc).value_counts()
    if len(counts) < 2:
        return X_train, y_train_enc, False, "single_class_no_smote"

    ratio = counts.max() / counts.min()
    if ratio <= threshold:
        return X_train, y_train_enc, False, "balanced_no_smote_needed"

    if not HAS_SMOTE:
        return X_train, y_train_enc, False, "smote_not_installed"

    min_class_count = counts.min()
    if min_class_count < 2:
        return X_train, y_train_enc, False, "minority_too_small_for_smote"

    k_neighbors = min(5, min_class_count - 1)
    try:
        sm = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
        X_res, y_res = sm.fit_resample(X_train, y_train_enc)
        return X_res, y_res, True, f"smote_applied_ratio_{ratio:.1f}"
    except Exception as e:
        return X_train, y_train_enc, False, f"smote_failed_{type(e).__name__}"


def get_models(n_classes, n_jobs=1):
    """
    n_jobs controls the INNER parallelism of each model. When this script
    is farming grids out across multiple worker PROCESSES, n_jobs should
    be small (1 or 2) so the outer (grid-level) and inner (model-level)
    parallelism don't oversubscribe the CPU. When running with a single
    grid-worker (--workers 1), pass n_jobs=-1 for full inner parallelism.
    """
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=150, n_jobs=n_jobs, random_state=RANDOM_STATE),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=150, n_jobs=n_jobs, random_state=RANDOM_STATE),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "GaussianNB": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=n_jobs),
    }

    if HAS_XGB:
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        params = dict(
            n_estimators=150,
            tree_method="hist",
            n_jobs=n_jobs,
            random_state=RANDOM_STATE,
            eval_metric="logloss" if n_classes == 2 else "mlogloss",
            objective=objective,
        )
        if n_classes > 2:
            params["num_class"] = n_classes
        models["XGBoost"] = XGBClassifier(**params)

    if HAS_LGBM:
        objective = "binary" if n_classes == 2 else "multiclass"
        params = dict(
            n_estimators=150,
            n_jobs=n_jobs,
            random_state=RANDOM_STATE,
            objective=objective,
            verbosity=-1,
        )
        if n_classes > 2:
            params["num_class"] = n_classes
        models["LightGBM"] = LGBMClassifier(**params)

    if HAS_CATBOOST:
        loss_function = "Logloss" if n_classes == 2 else "MultiClass"
        models["CatBoost"] = CatBoostClassifier(
            iterations=150,
            random_state=RANDOM_STATE,
            loss_function=loss_function,
            verbose=0,
            thread_count=n_jobs if n_jobs and n_jobs > 0 else -1,
        )

    return models


def compute_metrics(y_true, y_pred, y_proba, n_classes):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    try:
        if n_classes == 2 and y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        elif n_classes > 2 and y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        else:
            metrics["roc_auc"] = np.nan
    except Exception:
        metrics["roc_auc"] = np.nan
    return metrics


# ---------- resume bookkeeping paths ----------

def grid_results_dir(output_dir):
    return os.path.join(output_dir, "grid_results")


def predictions_dir(output_dir):
    return os.path.join(output_dir, "predictions")


def grid_result_path(output_dir, grid_name):
    return os.path.join(grid_results_dir(output_dir), f"{grid_name}.csv")


def grid_pred_path(output_dir, grid_name):
    return os.path.join(predictions_dir(output_dir), f"{grid_name}.csv")


def grid_already_done(output_dir, grid_name):
    return os.path.exists(grid_result_path(output_dir, grid_name))


# ---------- worker: process exactly one grid, for one target ----------

def process_one_grid(task):
    """
    Runs in a worker process. Trains all models for one grid, writes that
    grid's results + predictions to disk, and returns a small status dict
    for progress printing. Only writes the grid_results file (the resume
    marker) on successful completion, so a crashed/killed worker leaves no
    partial marker and that grid gets retried on the next run.
    """
    (grid_name, train_path, test_path, target_col, drop_cols,
     output_dir, save_models, n_jobs) = task

    t_grid_start = time.time()
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        if target_col not in train_df.columns:
            return {"grid": grid_name, "status": "skipped", "reason": "target_col_missing", "rows_written": 0}

        train_df = train_df.dropna(subset=[target_col])
        test_df = test_df.dropna(subset=[target_col])
        if len(train_df) == 0 or len(test_df) == 0:
            return {"grid": grid_name, "status": "skipped", "reason": "no_rows_after_dropna", "rows_written": 0}

        le_target = LabelEncoder()
        le_target.fit(pd.concat([train_df[target_col], test_df[target_col]], axis=0).astype(str))
        y_train_enc = le_target.transform(train_df[target_col].astype(str))
        y_test_enc = le_target.transform(test_df[target_col].astype(str))
        n_classes = len(le_target.classes_)

        if n_classes < 2:
            _write_grid_result_rows(output_dir, grid_name, [{
                "target": target_col, "grid": grid_name, "model": None,
                "status": "skipped_single_class",
            }])
            return {"grid": grid_name, "status": "skipped", "reason": "single_class", "rows_written": 1}

        X_train, X_test = prep_features(train_df, test_df, target_col, drop_cols + [target_col])

        if X_train.shape[1] == 0:
            _write_grid_result_rows(output_dir, grid_name, [{
                "target": target_col, "grid": grid_name, "model": None,
                "status": "skipped_no_features",
            }])
            return {"grid": grid_name, "status": "skipped", "reason": "no_features", "rows_written": 1}

        X_train_final, y_train_final, smote_applied, smote_note = maybe_smote(
            X_train.values, y_train_enc, IMBALANCE_THRESHOLD, RANDOM_STATE
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_final)
        X_test_scaled = scaler.transform(X_test.values)

        models = get_models(n_classes, n_jobs=n_jobs)

        grid_model_dir = None
        if save_models:
            grid_model_dir = os.path.join(output_dir, "models", str(grid_name))
            os.makedirs(grid_model_dir, exist_ok=True)

        result_rows = []
        pred_rows = []  # long format: model, y_true, y_pred

        for model_name, model in models.items():
            t0 = time.time()
            try:
                if model_name in SCALED_MODEL_NAMES:
                    fit_X_train, fit_X_test = X_train_scaled, X_test_scaled
                else:
                    fit_X_train, fit_X_test = X_train_final, X_test.values

                model.fit(fit_X_train, y_train_final)
                y_pred = model.predict(fit_X_test)
                y_proba = model.predict_proba(fit_X_test) if hasattr(model, "predict_proba") else None
                elapsed = time.time() - t0

                metrics = compute_metrics(y_test_enc, y_pred, y_proba, n_classes)
                result_rows.append({
                    "target": target_col, "grid": grid_name, "model": model_name,
                    "n_train": len(X_train_final), "n_test": len(X_test), "n_classes": n_classes,
                    "smote_applied": smote_applied, "smote_note": smote_note,
                    "train_time_sec": round(elapsed, 3), "status": "ok",
                    **metrics,
                })

                y_pred_orig = le_target.inverse_transform(y_pred)
                y_test_orig = le_target.inverse_transform(y_test_enc)
                for yt, yp in zip(y_test_orig.tolist(), y_pred_orig.tolist()):
                    pred_rows.append({"model": model_name, "y_true": yt, "y_pred": yp})

                if save_models and grid_model_dir is not None:
                    joblib.dump(model, os.path.join(grid_model_dir, f"{model_name}.joblib"))

                print(f"  [{grid_name}] {model_name:20s} acc={metrics['accuracy']:.3f}  "
                      f"f1={metrics['f1_weighted']:.3f}  ({elapsed:.1f}s)", flush=True)

            except Exception as e:
                result_rows.append({
                    "target": target_col, "grid": grid_name, "model": model_name,
                    "status": "error", "error": str(e),
                })
                print(f"  [{grid_name}] {model_name:20s} FAILED: {type(e).__name__}: {e}", flush=True)

        _write_grid_result_rows(output_dir, grid_name, result_rows)
        _write_grid_predictions(output_dir, grid_name, pred_rows)

        elapsed_grid = time.time() - t_grid_start
        return {
            "grid": grid_name, "status": "done", "n_models": len(result_rows),
            "elapsed_sec": round(elapsed_grid, 1),
        }

    except Exception as e:
        # Deliberately do NOT write a grid_results file here -- leaving no
        # resume marker means this grid gets retried on the next run.
        return {"grid": grid_name, "status": "error", "reason": f"{type(e).__name__}: {e}"}


def _write_grid_result_rows(output_dir, grid_name, rows):
    os.makedirs(grid_results_dir(output_dir), exist_ok=True)
    pd.DataFrame(rows).to_csv(grid_result_path(output_dir, grid_name), index=False)


def _write_grid_predictions(output_dir, grid_name, pred_rows):
    os.makedirs(predictions_dir(output_dir), exist_ok=True)
    if pred_rows:
        pd.DataFrame(pred_rows).to_csv(grid_pred_path(output_dir, grid_name), index=False)
    else:
        # still write an (empty-but-headered) file so this grid counts as
        # "done" and isn't retried, e.g. a grid where every model errored.
        pd.DataFrame(columns=["model", "y_true", "y_pred"]).to_csv(
            grid_pred_path(output_dir, grid_name), index=False
        )


def rebuild_reports(target_col, output_dir):
    """
    Scans ALL per-grid result/prediction files on disk (from this run and
    any earlier ones) and rebuilds the two summary reports. Safe to call
    any time -- this is what makes a resumed run's final CSVs complete.
    """
    result_files = glob.glob(os.path.join(grid_results_dir(output_dir), "*.csv"))
    if result_files:
        grid_results_df = pd.concat(
            [pd.read_csv(f) for f in result_files], ignore_index=True, sort=False
        )
    else:
        grid_results_df = pd.DataFrame()
    grid_results_path = os.path.join(output_dir, "grid_wise_results.csv")
    grid_results_df.to_csv(grid_results_path, index=False)
    print(f"\nGrid-wise results saved to: {grid_results_path}  ({len(grid_results_df)} rows)")

    pred_files = glob.glob(os.path.join(predictions_dir(output_dir), "*.csv"))
    combined_records = []
    if pred_files:
        all_preds = pd.concat(
            [pd.read_csv(f) for f in pred_files if os.path.getsize(f) > 0],
            ignore_index=True, sort=False
        )
        if len(all_preds):
            for model_name, sub in all_preds.groupby("model"):
                yt = sub["y_true"].astype(str).tolist()
                yp = sub["y_pred"].astype(str).tolist()
                combined_records.append({
                    "target": target_col,
                    "model": model_name,
                    "total_test_rows_across_all_grids": len(yt),
                    "accuracy": accuracy_score(yt, yp),
                    "precision_weighted": precision_score(yt, yp, average="weighted", zero_division=0),
                    "recall_weighted": recall_score(yt, yp, average="weighted", zero_division=0),
                    "f1_weighted": f1_score(yt, yp, average="weighted", zero_division=0),
                })

    combined_df = pd.DataFrame(combined_records)
    if len(combined_df):
        combined_df = combined_df.sort_values("f1_weighted", ascending=False)
    combined_path = os.path.join(output_dir, "combined_results.csv")
    combined_df.to_csv(combined_path, index=False)

    print(f"Combined (all-grids) results saved to: {combined_path}\n")
    print(f"FINAL COMBINED MODEL COMPARISON -- target: {target_col}")
    print("-" * 60)
    if len(combined_df):
        print(combined_df.to_string(index=False))
    else:
        print("(no results yet)")

    return combined_df


def run_for_target(target_col, pairs, output_dir, n_workers, inner_jobs):
    print("\n" + "=" * 70)
    print(f"TARGET: {target_col}   (grid-workers={n_workers}, inner-jobs-per-model={inner_jobs})")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    if SAVE_MODELS:
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)

    drop_cols = list(ALWAYS_DROP_COLS) + [c for c in QC_RELATED_COLS if c != target_col]

    todo = []
    already_done_count = 0
    for grid_name, train_path, test_path in pairs:
        if grid_already_done(output_dir, grid_name):
            already_done_count += 1
            continue
        todo.append((grid_name, train_path, test_path, target_col, drop_cols,
                     output_dir, SAVE_MODELS, inner_jobs))

    print(f"{already_done_count} grid(s) already done (resumed), {len(todo)} grid(s) to process.")

    if todo:
        done_count = 0
        error_count = 0
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(process_one_grid, t): t[0] for t in todo}
            for fut in as_completed(futures):
                grid_name = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    print(f"[{grid_name}] WORKER CRASHED: {type(e).__name__}: {e} -- will retry on next run.")
                    error_count += 1
                    continue

                status = r.get("status")
                if status == "done":
                    done_count += 1
                    print(f"[{grid_name}] GRID DONE ({r.get('n_models')} models, {r.get('elapsed_sec')}s) "
                          f"[{done_count}/{len(todo)}]", flush=True)
                elif status == "skipped":
                    done_count += 1
                    print(f"[{grid_name}] GRID SKIPPED ({r.get('reason')}) [{done_count}/{len(todo)}]", flush=True)
                else:
                    error_count += 1
                    print(f"[{grid_name}] GRID ERROR: {r.get('reason')} -- will retry on next run.", flush=True)

        print(f"\nThis run: {done_count} grid(s) completed/skipped-cleanly, {error_count} grid(s) errored "
              f"(errored grids left un-marked, will retry next run).")

    return rebuild_reports(target_col, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Parallel, resumable per-grid model training")
    default_workers = max(1, (os.cpu_count() or 4) // 2)
    parser.add_argument("--workers", type=int, default=default_workers,
                         help=f"Number of grids to process concurrently (default: {default_workers})")
    parser.add_argument("--inner-jobs", type=int, default=1,
                         help="n_jobs given to each model (RandomForest/XGBoost/etc). "
                              "Keep low (1-2) when --workers > 1 to avoid CPU oversubscription. "
                              "Use -1 only when --workers 1.")
    args = parser.parse_args()

    if not HAS_LGBM:
        print("Note: lightgbm not installed -- LightGBM will be skipped. pip install lightgbm")
    if not HAS_CATBOOST:
        print("Note: catboost not installed -- CatBoost will be skipped. pip install catboost")
    if not HAS_XGB:
        print("Note: xgboost not installed -- XGBoost will be skipped. pip install xgboost")

    pairs = find_grid_pairs(DATA_DIR)
    print(f"Found {len(pairs)} grid(s) with matching train/test files.")

    if not pairs:
        print("No grid train/test pairs found -- nothing to do.")
        return

    all_summaries = {}
    for target_col in TARGET_COLS:
        target_output_dir = os.path.join(OUTPUT_DIR, target_col)
        summary = run_for_target(target_col, pairs, target_output_dir, args.workers, args.inner_jobs)
        all_summaries[target_col] = summary

    print("\n" + "=" * 70)
    print("ALL TARGETS COMPLETE")
    print("=" * 70)
    for target_col, summary in all_summaries.items():
        print(f"\n{target_col} -> {os.path.join(OUTPUT_DIR, target_col, 'combined_results.csv')}")
        if summary is not None and len(summary):
            print(summary.to_string(index=False))


if __name__ == "__main__":
    main()