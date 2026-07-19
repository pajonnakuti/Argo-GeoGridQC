"""
Chunked RandomForest training on ALL_REGIONS train/test parquet pair.
Saves both plain .joblib and inference bundle (model + encoders + feature cols).

Run from app folder:
    python scripts/train_single_parquet_rf_chunked.py
    python scripts/train_single_parquet_rf_chunked.py --chunk-size 100000 --trees-per-chunk 15
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score

warnings.filterwarnings("ignore")

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from backend.config import MODEL_DIR, RESULTS_DIR, TARGET_COLS, TEST_PATH, TRAIN_PATH
from backend.ml.preprocess import (
    classify_columns,
    collect_stats,
    feature_columns_for_target,
    get_schema_columns,
    iter_pandas_batches,
    normalize_numeric_series,
    prep_chunk,
)

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

RANDOM_STATE = 42
IMBALANCE_THRESHOLD = 1.5
DEFAULT_CHUNK_SIZE = 200_000
DEFAULT_TREES_PER_CHUNK = 20
DEFAULT_MAX_TOTAL_TREES = 400
DEFAULT_ROC_SAMPLE_SIZE = 200_000


def maybe_smote_chunk(X, y, threshold, random_state):
    counts = pd.Series(y).value_counts()
    if len(counts) < 2:
        return X, y
    ratio = counts.max() / counts.min()
    if ratio <= threshold or not HAS_SMOTE:
        return X, y
    min_count = counts.min()
    if min_count < 2:
        return X, y
    k_neighbors = min(5, min_count - 1)
    try:
        sm = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
        return sm.fit_resample(X, y)
    except Exception:
        return X, y


def collect_stats_with_anchors(train_path, test_path, feature_cols, numeric_cols,
                                categorical_cols, target_col, chunk_size):
    numeric_sum = {c: 0.0 for c in numeric_cols}
    numeric_count = {c: 0 for c in numeric_cols}
    cat_values = {c: set() for c in categorical_cols}
    target_values = set()
    target_counts = {}
    anchor_rows = {}

    needed_cols = feature_cols + [target_col]

    def scan(path, is_train):
        for df in iter_pandas_batches(path, needed_cols, chunk_size):
            for c in numeric_cols:
                if c in df.columns:
                    vals = normalize_numeric_series(df[c])
                    vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
                    numeric_sum[c] += float(vals.sum())
                    numeric_count[c] += int(vals.shape[0])
            for c in categorical_cols:
                if c in df.columns:
                    cat_values[c].update(df[c].fillna("MISSING").astype(str).unique().tolist())
            if target_col in df.columns:
                y_str = df[target_col].dropna().astype(str)
                target_values.update(y_str.unique().tolist())
                if is_train:
                    for lbl, cnt in y_str.value_counts().items():
                        target_counts[lbl] = target_counts.get(lbl, 0) + int(cnt)
                    df_valid = df.dropna(subset=[target_col])
                    for lbl, sub in df_valid.groupby(df_valid[target_col].astype(str)):
                        if lbl not in anchor_rows:
                            anchor_rows[lbl] = sub.iloc[0].to_dict()

    print("Pass 1/3: scanning TRAIN…", flush=True)
    scan(train_path, is_train=True)
    print("Pass 1/3: scanning TEST…", flush=True)
    scan(test_path, is_train=False)

    numeric_means = {
        c: (numeric_sum[c] / numeric_count[c] if numeric_count[c] > 0 else 0.0)
        for c in numeric_cols
    }
    from sklearn.preprocessing import LabelEncoder

    cat_encoders = {}
    for c in categorical_cols:
        classes = sorted(cat_values[c]) if cat_values[c] else ["MISSING"]
        if "MISSING" not in classes:
            classes.append("MISSING")
        le = LabelEncoder()
        le.fit(classes)
        cat_encoders[c] = le

    target_encoder = LabelEncoder()
    target_encoder.fit(sorted(target_values))
    anchor_df = pd.DataFrame(list(anchor_rows.values())) if anchor_rows else pd.DataFrame()

    return {
        "numeric_means": numeric_means,
        "cat_encoders": cat_encoders,
        "target_encoder": target_encoder,
        "target_counts": target_counts,
        "anchor_df": anchor_df,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


def train_chunked(train_path, feature_cols, numeric_cols, target_col, stats,
                  chunk_size, trees_per_chunk, max_total_trees, inner_jobs):
    cat_encoders = stats["cat_encoders"]
    numeric_means = stats["numeric_means"]
    target_encoder = stats["target_encoder"]
    anchor_df = stats["anchor_df"]

    anchor_X, anchor_y = pd.DataFrame(), None
    if len(anchor_df):
        anchor_X, anchor_y = prep_chunk(
            anchor_df, feature_cols, numeric_cols, cat_encoders, numeric_means,
            target_col, target_encoder,
        )

    model = None
    total_trees = 0
    n_train_rows = 0
    chunk_idx = 0
    t_start = time.time()
    needed_cols = feature_cols + [target_col]

    for df in iter_pandas_batches(train_path, needed_cols, chunk_size):
        chunk_idx += 1
        df = df.dropna(subset=[target_col])
        if len(df) == 0:
            continue
        X, y = prep_chunk(df, feature_cols, numeric_cols, cat_encoders, numeric_means,
                          target_col, target_encoder)
        if len(anchor_X):
            X = pd.concat([X, anchor_X], ignore_index=True)
            y = np.concatenate([y, anchor_y])
        X, y = maybe_smote_chunk(X.values, y, IMBALANCE_THRESHOLD, RANDOM_STATE)
        trees_this = min(trees_per_chunk, max(1, max_total_trees - total_trees))
        if model is None:
            model = RandomForestClassifier(
                n_estimators=trees_this, warm_start=True,
                n_jobs=inner_jobs, random_state=RANDOM_STATE,
            )
        else:
            model.n_estimators = total_trees + trees_this
        model.fit(X, y)
        total_trees = model.n_estimators
        n_train_rows += len(df)
        print(f"  [train] chunk {chunk_idx}: total_trees={total_trees}, rows={n_train_rows}", flush=True)
        if total_trees >= max_total_trees:
            break

    print(f"Pass 2/3 done: {total_trees} trees, {n_train_rows} rows, {time.time()-t_start:.1f}s", flush=True)
    return model, n_train_rows, total_trees


def evaluate_chunked(test_path, feature_cols, numeric_cols, target_col, stats, model,
                     chunk_size, output_dir, roc_sample_size):
    cat_encoders = stats["cat_encoders"]
    numeric_means = stats["numeric_means"]
    target_encoder = stats["target_encoder"]
    n_classes = len(target_encoder.classes_)
    cm_total = np.zeros((n_classes, n_classes), dtype=np.int64)
    n_test_rows = 0
    roc_true, roc_proba, roc_seen = [], [], 0
    pred_path = os.path.join(output_dir, f"predictions_{target_col}.csv")
    os.makedirs(output_dir, exist_ok=True)
    first_write = True
    needed_cols = feature_cols + [target_col]

    for df in iter_pandas_batches(test_path, needed_cols, chunk_size):
        if target_col in df.columns:
            df = df.dropna(subset=[target_col])
        if len(df) == 0:
            continue
        X, y = prep_chunk(df, feature_cols, numeric_cols, cat_encoders, numeric_means,
                          target_col, target_encoder)
        y_pred = model.predict(X.values)
        y_proba = model.predict_proba(X.values)
        if y is not None:
            cm_total += confusion_matrix(y, y_pred, labels=range(n_classes))
            if roc_seen < roc_sample_size:
                take = min(roc_sample_size - roc_seen, len(y))
                roc_true.append(y[:take])
                roc_proba.append(y_proba[:take])
                roc_seen += take
        n_test_rows += len(df)
        out = pd.DataFrame({"model": "RandomForest", "y_pred": target_encoder.inverse_transform(y_pred)})
        if y is not None:
            out.insert(1, "y_true", target_encoder.inverse_transform(y))
        out.to_csv(pred_path, mode="w" if first_write else "a", header=first_write, index=False)
        first_write = False

    support = cm_total.sum(axis=1)
    tp = np.diag(cm_total)
    pred_sum = cm_total.sum(axis=0)
    total = cm_total.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_sum > 0, tp / np.maximum(pred_sum, 1), 0.0)
        recall = np.where(support > 0, tp / np.maximum(support, 1), 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / np.maximum(denom, 1e-12), 0.0)
    weights = support / total if total > 0 else np.zeros_like(support, dtype=float)
    metrics = {
        "accuracy": float(tp.sum() / total) if total > 0 else np.nan,
        "precision_weighted": float(np.sum(precision * weights)),
        "recall_weighted": float(np.sum(recall * weights)),
        "f1_weighted": float(np.sum(f1 * weights)),
    }
    if roc_true:
        try:
            yt = np.concatenate(roc_true)
            yp = np.concatenate(roc_proba)
            metrics["roc_auc_approx"] = (
                roc_auc_score(yt, yp[:, 1]) if n_classes == 2
                else roc_auc_score(yt, yp, multi_class="ovr", average="weighted")
            )
        except Exception:
            metrics["roc_auc_approx"] = np.nan
    else:
        metrics["roc_auc_approx"] = np.nan
    print(f"Pass 3/3 done: {n_test_rows} test rows", flush=True)
    return metrics, n_test_rows, pred_path


def run_for_target(target_col, train_path, test_path, model_dir, output_dir,
                   chunk_size, trees_per_chunk, max_total_trees, inner_jobs, roc_sample_size):
    print("\n" + "=" * 70)
    print(f"TARGET: {target_col}")
    print("=" * 70)

    pf, train_cols = get_schema_columns(train_path)
    feature_cols = feature_columns_for_target(train_cols, target_col)
    numeric_cols, categorical_cols = classify_columns(pf, feature_cols)

    stats = collect_stats_with_anchors(
        train_path, test_path, feature_cols, numeric_cols, categorical_cols,
        target_col, chunk_size,
    )
    if len(stats["target_encoder"].classes_) < 2:
        print(f"  Skipping '{target_col}': fewer than 2 classes.")
        return None

    model, n_train, total_trees = train_chunked(
        train_path, feature_cols, numeric_cols, target_col, stats,
        chunk_size, trees_per_chunk, max_total_trees, inner_jobs,
    )

    os.makedirs(model_dir, exist_ok=True)
    plain_path = os.path.join(model_dir, f"ALL_REGIONS_{target_col}_RandomForest.joblib")
    bundle_path = os.path.join(model_dir, f"ALL_REGIONS_{target_col}_bundle.joblib")
    joblib.dump(model, plain_path)
    bundle = {"model": model, "target_col": target_col, **stats}
    joblib.dump(bundle, bundle_path)
    print(f"  Saved: {plain_path}")
    print(f"  Saved: {bundle_path}")

    metrics, n_test, pred_path = evaluate_chunked(
        test_path, feature_cols, numeric_cols, target_col, stats, model,
        chunk_size, output_dir, roc_sample_size,
    )
    result = {
        "target": target_col, "model": "RandomForest",
        "n_train_rows": n_train, "n_test_rows": n_test,
        "total_trees": total_trees, **metrics,
    }
    pd.DataFrame([result]).to_csv(os.path.join(output_dir, f"results_{target_col}.csv"), index=False)
    print(pd.DataFrame([result]).to_string(index=False))
    return result


def main():
    parser = argparse.ArgumentParser(description="Chunked RF training on ALL_REGIONS parquet pair")
    parser.add_argument("--train-path", default=str(TRAIN_PATH))
    parser.add_argument("--test-path", default=str(TEST_PATH))
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--targets", default=",".join(TARGET_COLS))
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--trees-per-chunk", type=int, default=DEFAULT_TREES_PER_CHUNK)
    parser.add_argument("--max-total-trees", type=int, default=DEFAULT_MAX_TOTAL_TREES)
    parser.add_argument("--inner-jobs", type=int, default=-1)
    parser.add_argument("--roc-sample-size", type=int, default=DEFAULT_ROC_SAMPLE_SIZE)
    args = parser.parse_args()

    if not os.path.exists(args.train_path):
        raise FileNotFoundError(args.train_path)
    if not os.path.exists(args.test_path):
        raise FileNotFoundError(args.test_path)

    results = []
    for target in [t.strip() for t in args.targets.split(",") if t.strip()]:
        r = run_for_target(
            target, args.train_path, args.test_path, args.model_dir, args.output_dir,
            args.chunk_size, args.trees_per_chunk, args.max_total_trees,
            args.inner_jobs, args.roc_sample_size,
        )
        if r:
            results.append(r)
    if results:
        print("\nALL TARGETS COMPLETE\n", pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
