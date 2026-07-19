"""
Shared preprocessing for chunked RF training and live inference.
Mirrors train_single_parquet_rf_chunked.py so predictions use the same encoding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import LabelEncoder

ALWAYS_DROP_COLS = ["date"]
QC_RELATED_COLS = ["temp_qc", "psal_qc", "z_flag_label"]

QC_FLAG_LABELS = {
    "1": "Good",
    "2": "Probably good",
    "3": "Bad",
    "4": "Bad",
    "9": "Missing",
}


def is_categorical_arrow_type(t):
    return pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_binary(t)


def normalize_numeric_series(s):
    dtype_str = str(s.dtype)
    if dtype_str in ("bool", "boolean"):
        return s.astype("float64")
    if dtype_str.startswith("datetime64"):
        s = s.astype("int64", errors="ignore") / 86_400_000_000_000
        return s.astype("float64")
    return pd.to_numeric(s, errors="coerce").astype("float64")


def get_schema_columns(path):
    pf = pq.ParquetFile(path)
    return pf, list(pf.schema_arrow.names)


def classify_columns(pf, feature_cols):
    numeric_cols, categorical_cols = [], []
    for c in feature_cols:
        field = pf.schema_arrow.field(c)
        if is_categorical_arrow_type(field.type):
            categorical_cols.append(c)
        else:
            numeric_cols.append(c)
    return numeric_cols, categorical_cols


def feature_columns_for_target(train_cols: list[str], target_col: str) -> list[str]:
    drop_cols = list(ALWAYS_DROP_COLS) + [c for c in QC_RELATED_COLS if c != target_col]
    return [c for c in train_cols if c != target_col and c not in drop_cols]


def iter_pandas_batches(path, columns, batch_size):
    pf = pq.ParquetFile(path)
    existing_cols = [c for c in columns if c in pf.schema_arrow.names]
    for batch in pf.iter_batches(batch_size=batch_size, columns=existing_cols):
        yield batch.to_pandas()


def collect_stats_duckdb(train_path, feature_cols, numeric_cols, categorical_cols, target_col):
    """Fast stats via DuckDB — minutes instead of full Python scan."""
    import duckdb

    p = str(train_path).replace("'", "''")
    con = duckdb.connect()
    numeric_means = {}
    for c in numeric_cols:
        try:
            val = con.execute(
                f'SELECT AVG(CAST("{c}" AS DOUBLE)) FROM read_parquet(\'{p}\') '
                f'WHERE "{c}" IS NOT NULL'
            ).fetchone()[0]
            numeric_means[c] = float(val) if val is not None else 0.0
        except Exception:
            numeric_means[c] = 0.0

    cat_encoders = {}
    for c in categorical_cols:
        try:
            vals = con.execute(
                f'SELECT DISTINCT CAST("{c}" AS VARCHAR) AS v FROM read_parquet(\'{p}\') '
                f'WHERE "{c}" IS NOT NULL LIMIT 5000'
            ).fetchdf()["v"].astype(str).tolist()
        except Exception:
            vals = []
        classes = sorted(set(vals)) if vals else ["MISSING"]
        if "MISSING" not in classes:
            classes.append("MISSING")
        le = LabelEncoder()
        le.fit(classes)
        cat_encoders[c] = le

    target_vals = con.execute(
        f'SELECT DISTINCT CAST("{target_col}" AS VARCHAR) FROM read_parquet(\'{p}\') '
        f'WHERE "{target_col}" IS NOT NULL'
    ).fetchdf().iloc[:, 0].astype(str).tolist()
    target_encoder = LabelEncoder()
    target_encoder.fit(sorted(target_vals))

    return {
        "numeric_means": numeric_means,
        "cat_encoders": cat_encoders,
        "target_encoder": target_encoder,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


def collect_stats(train_path, test_path, feature_cols, numeric_cols, categorical_cols,
                  target_col, chunk_size=200_000):
    numeric_sum = {c: 0.0 for c in numeric_cols}
    numeric_count = {c: 0 for c in numeric_cols}
    cat_values = {c: set() for c in categorical_cols}
    target_values = set()

    needed_cols = feature_cols + [target_col]

    def scan(path):
        for df in iter_pandas_batches(path, needed_cols, chunk_size):
            for c in numeric_cols:
                if c in df.columns:
                    vals = normalize_numeric_series(df[c])
                    vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
                    numeric_sum[c] += float(vals.sum())
                    numeric_count[c] += int(vals.shape[0])
            for c in categorical_cols:
                if c in df.columns:
                    vals = df[c].fillna("MISSING").astype(str)
                    cat_values[c].update(vals.unique().tolist())
            if target_col in df.columns:
                y_str = df[target_col].dropna().astype(str)
                target_values.update(y_str.unique().tolist())

    scan(train_path)
    scan(test_path)

    numeric_means = {
        c: (numeric_sum[c] / numeric_count[c] if numeric_count[c] > 0 else 0.0)
        for c in numeric_cols
    }

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

    return {
        "numeric_means": numeric_means,
        "cat_encoders": cat_encoders,
        "target_encoder": target_encoder,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


def prep_chunk(df, feature_cols, numeric_cols, cat_encoders, numeric_means,
               target_col=None, target_encoder=None):
    X = pd.DataFrame(index=df.index)

    for c in numeric_cols:
        if c in df.columns:
            vals = normalize_numeric_series(df[c]).replace([np.inf, -np.inf], np.nan)
            X[c] = vals.fillna(numeric_means.get(c, 0.0))
        else:
            X[c] = numeric_means.get(c, 0.0)

    for c, le in cat_encoders.items():
        if c in df.columns:
            vals = df[c].fillna("MISSING").astype(str)
        else:
            vals = pd.Series(["MISSING"] * len(df), index=df.index)
        classes_set = set(le.classes_.tolist())
        vals = vals.where(vals.isin(classes_set), other=le.classes_[0])
        X[c] = le.transform(vals)

    X = X[feature_cols].astype("float64")

    y = None
    if target_col and target_encoder and target_col in df.columns:
        y_raw = df[target_col].astype(str)
        classes_set = set(target_encoder.classes_.tolist())
        y_raw = y_raw.where(y_raw.isin(classes_set), other=target_encoder.classes_[0])
        y = target_encoder.transform(y_raw)

    return X, y


def qc_label(flag) -> str:
    return QC_FLAG_LABELS.get(str(flag), str(flag))
