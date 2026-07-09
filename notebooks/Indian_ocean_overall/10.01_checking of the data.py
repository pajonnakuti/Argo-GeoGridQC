"""
qc_train_test.py
=================
QC / sanity-check script for the grid-wise train/test split produced by
split_train_test_grid_wise.py.

Checks it performs (all memory-safe / streamed, no full-file load):

  1. Row counts            -> train vs test row counts, and total.
  2. Grid counts            -> how many unique grids ended up in each set,
                                and confirms there is NO grid overlap
                                between train and test (leakage check).
  3. Missing values         -> null count + null % for every column,
                                pulled straight from parquet row-group
                                metadata (fast, no data read needed).
  4. QC flags (temp/sal)    -> auto-detects temperature/salinity value
                                columns AND their QC/flag columns, then
                                prints the value_counts / distribution of
                                each QC flag in train vs test.
  5. Class balance          -> auto-detects categorical / low-cardinality
                                columns (region, source file, season,
                                basin, quality flags, etc.) and compares
                                the % distribution of each class between
                                train and test, flagging any category
                                whose share differs by more than
                                IMBALANCE_THRESHOLD between the two sets.

Output:
  - Full report printed to console.
  - Same report written to a text file:
        <BASE_DIR>/qc_report/qc_report.txt
  - Per-column missing-value table and per-class balance table also saved
    as CSVs in the same folder for easy viewing in Excel.

Run:
    python qc_train_test.py

If your column names differ from the guesses below, just edit the
CONFIG section at the top -- everything else auto-adapts.
"""

import os
import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

# =========================================================================
# CONFIG -- edit these if auto-detection picks the wrong columns
# =========================================================================

BASE_DIR = r"D:\INCOIS\Agro_project\Indian_ocean"
TRAIN_PARQUET = os.path.join(BASE_DIR, "train", "ALL_REGIONS_train.parquet")
TEST_PARQUET  = os.path.join(BASE_DIR, "test",  "ALL_REGIONS_test.parquet")

REPORT_DIR = os.path.join(BASE_DIR, "qc_report")
os.makedirs(REPORT_DIR, exist_ok=True)

GRID_COLUMN = "grid_id"

# Keywords used to auto-detect relevant columns (case-insensitive substring match).
TEMP_KEYWORDS = ["temp", "sst", "temperature"]
SAL_KEYWORDS  = ["sal", "psal", "sss", "salinity"]
QC_KEYWORDS   = ["qc", "flag", "quality"]

# Columns to force-treat as "class" columns for balance checking, in addition
# to whatever gets auto-detected (low-cardinality columns). Leave empty list
# to rely purely on auto-detection.
FORCE_CLASS_COLUMNS = []

# A column is auto-treated as "categorical" for the class-balance check if
# its number of unique values (sampled) is <= this number.
MAX_CLASS_CARDINALITY = 30

# Flag a class as "imbalanced" between train/test if the % share differs
# by more than this many percentage points.
IMBALANCE_THRESHOLD = 3.0

BATCH_SIZE = 300_000

# =========================================================================


def check_files_exist():
    for label, path in [("TRAIN", TRAIN_PARQUET), ("TEST", TEST_PARQUET)]:
        if not os.path.exists(path):
            print(f"ERROR: {label} file not found at: {path}")
            print("Edit TRAIN_PARQUET / TEST_PARQUET at the top of this script "
                  "if your files are elsewhere.")
            sys.exit(1)


def get_schema_and_rowcount(path):
    pf = pq.ParquetFile(path)
    return pf, pf.schema_arrow, pf.metadata.num_rows


def get_null_counts(pf, schema):
    """
    Pull null counts per column directly from parquet row-group statistics.
    No data is actually read -- this is metadata only, so it's fast and
    safe even for huge files.
    """
    n_cols = len(schema.names)
    null_counts = [0] * n_cols
    total_rows = pf.metadata.num_rows

    for rg_idx in range(pf.metadata.num_row_groups):
        rg = pf.metadata.row_group(rg_idx)
        for col_idx in range(rg.num_columns):
            col_meta = rg.column(col_idx)
            stats = col_meta.statistics
            if stats is not None and stats.has_null_count:
                null_counts[col_idx] += stats.null_count
            else:
                null_counts[col_idx] = None  # unknown, stats not available

    result = {}
    for name, cnt in zip(schema.names, null_counts):
        if cnt is None:
            result[name] = None
        else:
            pct = (cnt / total_rows * 100) if total_rows else 0
            result[name] = (cnt, pct)
    return result


def detect_columns(all_columns, keywords):
    return [c for c in all_columns if any(k in c.lower() for k in keywords)]


def detect_qc_pairs(all_columns):
    """
    Pair up value columns (temp/sal) with their QC/flag columns based on
    keyword overlap, e.g. 'temp' <-> 'temp_qc', 'sal_value' <-> 'sal_flag'.
    """
    temp_cols = detect_columns(all_columns, TEMP_KEYWORDS)
    sal_cols  = detect_columns(all_columns, SAL_KEYWORDS)
    qc_cols   = detect_columns(all_columns, QC_KEYWORDS)

    temp_qc = [c for c in qc_cols if any(k in c.lower() for k in TEMP_KEYWORDS)]
    sal_qc  = [c for c in qc_cols if any(k in c.lower() for k in SAL_KEYWORDS)]

    # value columns = temp/sal columns that are NOT themselves QC columns
    temp_value_cols = [c for c in temp_cols if c not in qc_cols]
    sal_value_cols  = [c for c in sal_cols if c not in qc_cols]

    return {
        "temp_value_cols": temp_value_cols,
        "sal_value_cols": sal_value_cols,
        "temp_qc_cols": temp_qc,
        "sal_qc_cols": sal_qc,
    }


def sample_uniques(path, column, sample_rows=500_000):
    """Read just one column, up to sample_rows, to estimate cardinality / get value_counts."""
    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(columns=[column])
    series = table.column(column).to_pandas()
    return series


def full_value_counts(path, column):
    """Stream just one column across the whole file and accumulate value_counts."""
    pf = pq.ParquetFile(path)
    counts = pd.Series(dtype="int64")
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=[column]):
        s = batch.to_pandas()[column]
        vc = s.value_counts(dropna=False)
        counts = counts.add(vc, fill_value=0)
    return counts.astype("int64").sort_values(ascending=False)


def get_grid_set(path):
    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(columns=[GRID_COLUMN])
    return set(table.column(GRID_COLUMN).to_pandas().dropna().unique().tolist())


def main():
    lines = []  # collect everything for the text report

    def log(msg=""):
        print(msg)
        lines.append(msg)

    check_files_exist()

    log("=" * 78)
    log("QC REPORT: TRAIN / TEST SPLIT")
    log("=" * 78)
    log(f"Train file: {TRAIN_PARQUET}")
    log(f"Test file : {TEST_PARQUET}")
    log()

    # ---------------------------------------------------------------
    # 1. Row counts + schema
    # ---------------------------------------------------------------
    pf_train, schema_train, n_train = get_schema_and_rowcount(TRAIN_PARQUET)
    pf_test,  schema_test,  n_test  = get_schema_and_rowcount(TEST_PARQUET)

    log("-" * 78)
    log("1. ROW COUNTS")
    log("-" * 78)
    total = n_train + n_test
    log(f"  Train rows : {n_train:,}  ({n_train/total*100:.2f}%)" if total else f"  Train rows : {n_train:,}")
    log(f"  Test rows  : {n_test:,}  ({n_test/total*100:.2f}%)" if total else f"  Test rows  : {n_test:,}")
    log(f"  Total rows : {total:,}")
    log()

    if schema_train.names != schema_test.names:
        log("  WARNING: train and test have DIFFERENT columns!")
        log(f"    Train only: {set(schema_train.names) - set(schema_test.names)}")
        log(f"    Test only : {set(schema_test.names) - set(schema_train.names)}")
    all_columns = schema_train.names
    log(f"  Columns ({len(all_columns)}): {all_columns}")
    log()

    # ---------------------------------------------------------------
    # 2. Grid overlap / leakage check
    # ---------------------------------------------------------------
    log("-" * 78)
    log("2. GRID LEAKAGE CHECK")
    log("-" * 78)
    if GRID_COLUMN in all_columns:
        train_grids = get_grid_set(TRAIN_PARQUET)
        test_grids  = get_grid_set(TEST_PARQUET)
        overlap = train_grids & test_grids
        log(f"  Unique grids in train: {len(train_grids):,}")
        log(f"  Unique grids in test : {len(test_grids):,}")
        if overlap:
            log(f"  *** LEAKAGE DETECTED: {len(overlap):,} grids appear in BOTH train and test! ***")
            log(f"      Example overlapping grids: {list(overlap)[:10]}")
        else:
            log("  OK: no grid overlap between train and test.")
    else:
        log(f"  Column '{GRID_COLUMN}' not found -- skipping leakage check.")
    log()

    # ---------------------------------------------------------------
    # 3. Missing values
    # ---------------------------------------------------------------
    log("-" * 78)
    log("3. MISSING VALUES (per column)")
    log("-" * 78)

    null_train = get_null_counts(pf_train, schema_train)
    null_test  = get_null_counts(pf_test, schema_test)

    missing_rows = []
    log(f"  {'column':30s} {'train_null':>12s} {'train_%':>8s}   {'test_null':>12s} {'test_%':>8s}")
    for col in all_columns:
        tr = null_train.get(col)
        te = null_test.get(col)
        tr_cnt, tr_pct = tr if tr else ("n/a", "n/a")
        te_cnt, te_pct = te if te else ("n/a", "n/a")
        tr_pct_str = f"{tr_pct:.2f}" if isinstance(tr_pct, float) else tr_pct
        te_pct_str = f"{te_pct:.2f}" if isinstance(te_pct, float) else te_pct
        log(f"  {col:30s} {str(tr_cnt):>12s} {tr_pct_str:>8s}   {str(te_cnt):>12s} {te_pct_str:>8s}")
        missing_rows.append({
            "column": col,
            "train_null_count": tr_cnt, "train_null_pct": tr_pct,
            "test_null_count": te_cnt, "test_null_pct": te_pct,
        })
    log()
    pd.DataFrame(missing_rows).to_csv(os.path.join(REPORT_DIR, "missing_values.csv"), index=False)

    # ---------------------------------------------------------------
    # 4. Temp / Salinity QC flags
    # ---------------------------------------------------------------
    log("-" * 78)
    log("4. TEMPERATURE / SALINITY QC FLAG DISTRIBUTION")
    log("-" * 78)

    detected = detect_qc_pairs(all_columns)
    log(f"  Detected temp value column(s): {detected['temp_value_cols']}")
    log(f"  Detected temp QC column(s)   : {detected['temp_qc_cols']}")
    log(f"  Detected sal value column(s) : {detected['sal_value_cols']}")
    log(f"  Detected sal QC column(s)    : {detected['sal_qc_cols']}")
    log()

    if not (detected["temp_qc_cols"] or detected["sal_qc_cols"]):
        log("  No QC/flag columns auto-detected for temp or salinity.")
        log("  If your QC columns are named differently, add them to QC_KEYWORDS")
        log("  or FORCE_CLASS_COLUMNS at the top of this script.")
        log()

    for qc_col in detected["temp_qc_cols"] + detected["sal_qc_cols"]:
        log(f"  --- QC column: {qc_col} ---")
        vc_train = full_value_counts(TRAIN_PARQUET, qc_col)
        vc_test  = full_value_counts(TEST_PARQUET, qc_col)

        combined = pd.DataFrame({"train_count": vc_train, "test_count": vc_test}).fillna(0)
        combined["train_pct"] = combined["train_count"] / combined["train_count"].sum() * 100
        combined["test_pct"]  = combined["test_count"] / combined["test_count"].sum() * 100
        combined = combined.sort_index()

        log(f"  {'flag_value':>12s} {'train_count':>12s} {'train_%':>8s}   {'test_count':>12s} {'test_%':>8s}")
        for idx, row in combined.iterrows():
            log(f"  {str(idx):>12s} {int(row['train_count']):>12,d} {row['train_pct']:>7.2f}%   "
                f"{int(row['test_count']):>12,d} {row['test_pct']:>7.2f}%")
        combined.to_csv(os.path.join(REPORT_DIR, f"qc_flag_{qc_col}.csv"))
        log()

    # Also report basic stats (min/max/mean) for the raw temp/sal value columns via streaming,
    # since these often reveal bad sentinel values (e.g. -999, 99.99).
    value_cols = detected["temp_value_cols"] + detected["sal_value_cols"]
    if value_cols:
        log("  --- Value range check (helps spot bad sentinel values, e.g. -999) ---")
        for path, label in [(TRAIN_PARQUET, "train"), (TEST_PARQUET, "test")]:
            pf = pq.ParquetFile(path)
            mins = {c: None for c in value_cols}
            maxs = {c: None for c in value_cols}
            sums = {c: 0.0 for c in value_cols}
            counts = {c: 0 for c in value_cols}
            for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=value_cols):
                df = batch.to_pandas()
                for c in value_cols:
                    s = df[c].dropna()
                    if len(s) == 0:
                        continue
                    mn, mx = s.min(), s.max()
                    mins[c] = mn if mins[c] is None else min(mins[c], mn)
                    maxs[c] = mx if maxs[c] is None else max(maxs[c], mx)
                    sums[c] += s.sum()
                    counts[c] += len(s)
            for c in value_cols:
                mean_v = sums[c] / counts[c] if counts[c] else float("nan")
                log(f"    [{label}] {c:20s} min={mins[c]}  max={maxs[c]}  mean={mean_v:.3f}  n={counts[c]:,}")
        log()

    # ---------------------------------------------------------------
    # 5. Class balance (train vs test)
    # ---------------------------------------------------------------
    log("-" * 78)
    log("5. CLASS BALANCE (train vs test)")
    log("-" * 78)

    # Auto-detect categorical columns by sampling cardinality from train.
    auto_class_cols = []
    for col in all_columns:
        if col == GRID_COLUMN:
            continue
        try:
            sample = sample_uniques(TRAIN_PARQUET, col, sample_rows=200_000)
        except Exception:
            continue
        n_unique = sample.nunique(dropna=True)
        if 1 < n_unique <= MAX_CLASS_CARDINALITY:
            auto_class_cols.append(col)

    class_cols = sorted(set(auto_class_cols) | set(FORCE_CLASS_COLUMNS))
    # Don't re-report QC columns here if already fully covered in section 4
    class_cols = [c for c in class_cols]

    log(f"  Auto-detected class-like columns (<= {MAX_CLASS_CARDINALITY} unique values): {class_cols}")
    log()

    imbalance_summary = []

    for col in class_cols:
        vc_train = full_value_counts(TRAIN_PARQUET, col)
        vc_test  = full_value_counts(TEST_PARQUET, col)

        combined = pd.DataFrame({"train_count": vc_train, "test_count": vc_test}).fillna(0)
        combined["train_pct"] = combined["train_count"] / combined["train_count"].sum() * 100
        combined["test_pct"]  = combined["test_count"] / combined["test_count"].sum() * 100
        combined["pct_diff"]  = (combined["train_pct"] - combined["test_pct"]).abs()
        combined = combined.sort_values("train_count", ascending=False)

        max_diff = combined["pct_diff"].max() if len(combined) else 0
        flag = "  <-- IMBALANCED" if max_diff > IMBALANCE_THRESHOLD else ""

        log(f"  --- Class column: {col} (max train/test % diff = {max_diff:.2f}pp){flag} ---")
        log(f"  {'class':>15s} {'train_count':>12s} {'train_%':>8s}   {'test_count':>12s} {'test_%':>8s} {'diff_pp':>8s}")
        for idx, row in combined.iterrows():
            log(f"  {str(idx):>15s} {int(row['train_count']):>12,d} {row['train_pct']:>7.2f}%   "
                f"{int(row['test_count']):>12,d} {row['test_pct']:>7.2f}% {row['pct_diff']:>7.2f}")
        combined.to_csv(os.path.join(REPORT_DIR, f"class_balance_{col}.csv"))

        imbalance_summary.append({"column": col, "max_pct_diff": max_diff, "imbalanced": max_diff > IMBALANCE_THRESHOLD})
        log()

    if imbalance_summary:
        log("-" * 78)
        log("  IMBALANCE SUMMARY (columns with max % diff > threshold = {:.1f}pp)".format(IMBALANCE_THRESHOLD))
        log("-" * 78)
        any_flagged = False
        for row in imbalance_summary:
            if row["imbalanced"]:
                any_flagged = True
                log(f"  ** {row['column']}: max diff {row['max_pct_diff']:.2f}pp -- consider stratifying the split by this column **")
        if not any_flagged:
            log("  All detected class columns are reasonably balanced between train and test.")
        pd.DataFrame(imbalance_summary).to_csv(os.path.join(REPORT_DIR, "imbalance_summary.csv"), index=False)
    else:
        log("  No categorical columns detected to check for class balance.")
        log("  (Increase MAX_CLASS_CARDINALITY at the top of the script if needed.)")

    log()
    log("=" * 78)
    log(f"Full report saved to: {os.path.join(REPORT_DIR, 'qc_report.txt')}")
    log(f"CSV breakdowns saved to: {REPORT_DIR}")
    log("=" * 78)

    with open(os.path.join(REPORT_DIR, "qc_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()