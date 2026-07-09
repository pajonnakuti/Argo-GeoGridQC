"""
audit_and_split_grid_data.py
=============================
Run this LOCALLY where your parquet file lives.

WHY THIS VERSION:
Your earlier issue ("some columns / no data get skipped, not matching in
train/test") almost always comes from one of these root causes, which this
script now explicitly detects BEFORE any split happens:

  1. A column is 100% NULL for a particular grid (but has data in other grids).
     -> When you later concat per-grid CSVs, that grid's column looks "empty"
        or gets dropped/ignored by some ML pipelines.
  2. A column's dtype differs across grids (e.g. numeric in most grids,
     but stored as text/object in one grid because of a stray string like
     "NA" or "-" instead of a real null).
  3. A column has a small but real amount of data in a grid, but overwhelmingly
     null (>50-90%) -> unstable if that grid ends up mostly in test or train
     after a random split.
  4. Grids with very few rows -> after an 80/20 split, test set may end up
     with 0-1 rows for that grid, effectively "skipping" it in evaluation.

WORKFLOW:
  STEP 1: python audit_and_split_grid_data.py audit --input <parquet> --outdir <dir>
          -> review the reports, fix/clean source data if needed
  STEP 2: python audit_and_split_grid_data.py split --input <parquet> --outdir <dir>
          -> only run this once you're satisfied with the audit

Requirements:
    pip install pandas pyarrow numpy
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def guess_grid_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if c.lower() in
                  ("grid", "grid_id", "gridid", "grid_no", "grid_code", "cell_id", "id")]
    if candidates:
        return candidates[0]
    fallback = [c for c in df.columns if "grid" in c.lower()]
    if fallback:
        return fallback[0]
    raise ValueError(
        "Could not auto-detect a grid column. Please pass --grid-col explicitly.\n"
        f"Available columns: {list(df.columns)}"
    )


def load_parquet(path: str) -> pd.DataFrame:
    print(f"Loading parquet file: {path}")
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"ERROR: could not read parquet file.\n{e}")
        sys.exit(1)
    print(f"Loaded shape: {df.shape}\n")
    return df


# ----------------------------------------------------------------------
# STEP 1: Deep audit
# ----------------------------------------------------------------------

def audit(df: pd.DataFrame, grid_col: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    overall_dtypes = df.dtypes.astype(str).to_dict()
    grids = sorted(df[grid_col].dropna().unique().tolist())
    cols = [c for c in df.columns if c != grid_col]

    # ---- A) Row counts per grid (catches tiny grids that break splits) ----
    row_counts = df.groupby(grid_col).size().reset_index(name="row_count")
    row_counts["risk_too_few_rows_for_split"] = row_counts["row_count"] < 10
    row_counts.to_csv(os.path.join(outdir, "1_grid_row_counts.csv"), index=False)

    # ---- B) Column x Grid matrix: null %, dtype, unique count ----
    detail_records = []
    for g in grids:
        gdf = df[df[grid_col] == g]
        n = len(gdf)
        for c in cols:
            series = gdf[c]
            n_non_null = series.notna().sum()
            pct_null = round(1 - (n_non_null / n), 4) if n > 0 else np.nan
            dtype_here = str(series.dropna().infer_objects().dtype) if n_non_null > 0 else "ALL_NULL"
            detail_records.append({
                "grid": g,
                "column": c,
                "row_count": n,
                "non_null_count": n_non_null,
                "pct_null": pct_null,
                "dtype_in_grid": dtype_here,
                "overall_dtype": overall_dtypes[c],
                "dtype_mismatch": (dtype_here != "ALL_NULL") and (dtype_here != overall_dtypes[c]),
                "fully_null_in_this_grid": n_non_null == 0,
            })
    detail = pd.DataFrame(detail_records)
    detail.to_csv(os.path.join(outdir, "2_column_by_grid_detail.csv"), index=False)

    # ---- C) Summary: which columns are problematic, and in how many grids ----
    problem_summary = (
        detail.groupby("column")
        .agg(
            grids_fully_null=("fully_null_in_this_grid", "sum"),
            grids_dtype_mismatch=("dtype_mismatch", "sum"),
            grids_high_null_gt50pct=("pct_null", lambda x: (x > 0.5).sum()),
            total_grids=("grid", "nunique"),
        )
        .reset_index()
    )
    problem_summary["fully_null_in_ALL_grids"] = problem_summary["grids_fully_null"] == problem_summary["total_grids"]
    problem_summary = problem_summary.sort_values(
        ["fully_null_in_ALL_grids", "grids_fully_null", "grids_dtype_mismatch"], ascending=False
    )
    problem_summary.to_csv(os.path.join(outdir, "3_column_problem_summary.csv"), index=False)

    # ---- D) Grids that are missing entire columns' worth of data ----
    grid_issue_summary = (
        detail.groupby("grid")
        .agg(
            columns_fully_null=("fully_null_in_this_grid", "sum"),
            columns_dtype_mismatch=("dtype_mismatch", "sum"),
            columns_high_null_gt50pct=("pct_null", lambda x: (x > 0.5).sum()),
        )
        .reset_index()
        .sort_values("columns_fully_null", ascending=False)
    )
    grid_issue_summary = grid_issue_summary.merge(row_counts, on="grid", how="left")
    grid_issue_summary.to_csv(os.path.join(outdir, "4_grid_problem_summary.csv"), index=False)

    # ---- E) Columns that are useless dataset-wide (always null) ----
    always_null_cols = problem_summary.loc[problem_summary["fully_null_in_ALL_grids"], "column"].tolist()

    # ---- Console summary ----
    print("=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    print(f"Total grids: {len(grids)}")
    print(f"Total columns (excl. grid col): {len(cols)}")
    print(f"Grids with < 10 rows (risky for splitting): "
          f"{int(row_counts['risk_too_few_rows_for_split'].sum())}")
    print(f"Columns that are 100% null in ALL grids (candidates to drop): "
          f"{len(always_null_cols)}")
    if always_null_cols:
        print(f"  -> {always_null_cols}")

    cols_with_partial_null_issue = problem_summary[
        (problem_summary["grids_fully_null"] > 0) & (~problem_summary["fully_null_in_ALL_grids"])
    ]
    print(f"\nColumns that are fully null in SOME grids but not others "
          f"(these are the main cause of 'skipped columns' after split): "
          f"{len(cols_with_partial_null_issue)}")
    if len(cols_with_partial_null_issue) > 0:
        print(cols_with_partial_null_issue[["column", "grids_fully_null", "total_grids"]].to_string(index=False))

    cols_with_dtype_issue = problem_summary[problem_summary["grids_dtype_mismatch"] > 0]
    print(f"\nColumns with dtype mismatches across grids: {len(cols_with_dtype_issue)}")
    if len(cols_with_dtype_issue) > 0:
        print(cols_with_dtype_issue[["column", "grids_dtype_mismatch"]].to_string(index=False))

    print("\nFull reports written to:")
    print(f"  {os.path.join(outdir, '1_grid_row_counts.csv')}")
    print(f"  {os.path.join(outdir, '2_column_by_grid_detail.csv')}  (most detailed - column x grid)")
    print(f"  {os.path.join(outdir, '3_column_problem_summary.csv')}  (which columns are problematic)")
    print(f"  {os.path.join(outdir, '4_grid_problem_summary.csv')}  (which grids are problematic)")
    print("\nReview these before running the 'split' step.")


# ----------------------------------------------------------------------
# STEP 2: Safe split (guarantees identical columns in every train/test file)
# ----------------------------------------------------------------------

def safe_split(df: pd.DataFrame, grid_col: str, outdir: str, test_size: float, seed: int,
               drop_always_null: bool = True, min_rows_warn: int = 10):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)

    all_cols = list(df.columns)

    if drop_always_null:
        always_null_cols = [c for c in all_cols if df[c].isna().all()]
        if always_null_cols:
            print(f"Dropping {len(always_null_cols)} column(s) that are 100% null across the WHOLE dataset: "
                  f"{always_null_cols}")
            df = df.drop(columns=always_null_cols)
            all_cols = list(df.columns)

    summary = []
    warnings = []

    for grid_val, gdf in df.groupby(grid_col):
        n = len(gdf)
        if n < min_rows_warn:
            warnings.append(f"Grid '{grid_val}' has only {n} rows — split may be unreliable.")

        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = max(1, int(round(n * test_size))) if n > 1 else 0
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        gdf_reset = gdf.reset_index(drop=True)
        # Force both frames to have the FULL, IDENTICAL column set/order every time
        train_df = gdf_reset.iloc[train_idx].reindex(columns=all_cols)
        test_df = gdf_reset.iloc[test_idx].reindex(columns=all_cols)

        safe_grid = str(grid_val).replace("/", "_").replace("\\", "_").replace(" ", "_")
        train_path = os.path.join(outdir, f"train_{safe_grid}.csv")
        test_path = os.path.join(outdir, f"test_{safe_grid}.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        # per-grid check: any column fully empty in train or test after split?
        empty_in_train = [c for c in all_cols if train_df[c].isna().all() and not gdf_reset[c].isna().all()]
        empty_in_test = [c for c in all_cols if len(test_df) > 0 and test_df[c].isna().all() and not gdf_reset[c].isna().all()]

        summary.append({
            "grid": grid_val,
            "total_rows": n,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "n_columns_written": len(all_cols),
            "columns_emptied_by_split_in_train": ", ".join(empty_in_train),
            "columns_emptied_by_split_in_test": ", ".join(empty_in_test),
            "train_file": train_path,
            "test_file": test_path,
        })

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(outdir, "split_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\nAll {len(summary_df)} grids split. Every train/test CSV has the SAME {len(all_cols)} columns "
          f"in the SAME order.")
    if warnings:
        print(f"\n⚠ {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")

    risky = summary_df[
        (summary_df["columns_emptied_by_split_in_train"] != "") |
        (summary_df["columns_emptied_by_split_in_test"] != "")
    ]
    if len(risky) > 0:
        print(f"\n⚠ {len(risky)} grid(s) had a column go fully empty in train OR test purely due to random split "
              f"(the column had very sparse data to begin with). See split_summary.csv, columns "
              f"'columns_emptied_by_split_in_train/test'.")
    print(f"\nSplit summary saved to: {summary_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Audit grid data for consistency, then safely split per grid.")
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--input", required=True, help="Path to ALL_REGIONS_UNIFIED.parquet")
    common.add_argument("--outdir", required=True, help="Folder to write reports/CSVs")
    common.add_argument("--grid-col", default=None, help="Grid identifier column (auto-detected if omitted)")

    p_audit = sub.add_parser("audit", parents=[common], help="Run the deep consistency audit only")

    p_split = sub.add_parser("split", parents=[common], help="Run the safe per-grid train/test split")
    p_split.add_argument("--test-size", type=float, default=0.2)
    p_split.add_argument("--seed", type=int, default=42)
    p_split.add_argument("--keep-always-null-cols", action="store_true",
                          help="By default, columns that are 100%% null across the WHOLE dataset are dropped. Pass this flag to keep them.")

    args = parser.parse_args()
    df = load_parquet(args.input)
    grid_col = args.grid_col or guess_grid_column(df)
    print(f"Using grid column: '{grid_col}' ({df[grid_col].nunique()} unique grids)\n")

    if args.mode == "audit":
        audit(df, grid_col, args.outdir)
    elif args.mode == "split":
        safe_split(df, grid_col, args.outdir, args.test_size, args.seed,
                   drop_always_null=not args.keep_always_null_cols)


if __name__ == "__main__":
    main()