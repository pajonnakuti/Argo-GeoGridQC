"""
check_empty_columns.py
========================
Checks EVERY column across the WHOLE parquet file (not per-grid) to find:
  - Columns that are 100% empty (completely useless -> safe to drop)
  - Columns that are mostly empty (e.g. >90% null -> probably drop)
  - Columns that are partially empty (real gaps -> handle case by case)

Memory-safe: streams the file in batches (same approach as
split_gridwise_chunked.py), never loads the whole file into memory.

Output: prints a report AND saves it to column_null_report.csv

Run:
    python check_empty_columns.py
"""

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ======================= CONFIG =======================
INPUT_PATH = r"D:\INCOIS\Agro_project\data\processed\Final parquet files\ALL_REGIONS_UNIFIED.parquet"
OUTPUT_DIR = r"D:\INCOIS\Agro_project\Indian_ocean"
BATCH_SIZE = 50_000
# Thresholds used to label columns in the report
FULLY_EMPTY_THRESHOLD = 100.0   # % null -> "FULLY EMPTY - drop"
MOSTLY_EMPTY_THRESHOLD = 90.0   # % null -> "MOSTLY EMPTY - investigate/likely drop"
# ========================================================


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Opening parquet file (metadata only): {INPUT_PATH}")
    pf = pq.ParquetFile(INPUT_PATH)
    all_cols = [f.name for f in pf.schema_arrow]
    total_rows = pf.metadata.num_rows
    print(f"Total rows: {total_rows:,}")
    print(f"Total columns: {len(all_cols)}\n")

    non_null_counts = {c: 0 for c in all_cols}
    rows_seen = 0

    print(f"Streaming in batches of {BATCH_SIZE:,} rows...\n")
    batch_num = 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE):
        batch_num += 1
        df_chunk = batch.to_pandas()
        rows_seen += len(df_chunk)

        for c in all_cols:
            non_null_counts[c] += df_chunk[c].notna().sum()

        del df_chunk, batch
        print(f"  Batch {batch_num} done — {rows_seen:,} rows checked so far", end="\r")

    print(f"\n\nDone. Checked {rows_seen:,} rows across {batch_num} batches.\n")

    # ---- Build report ----
    records = []
    for c in all_cols:
        non_null = non_null_counts[c]
        null_count = total_rows - non_null
        pct_null = 100 * null_count / total_rows if total_rows else 0

        if pct_null >= FULLY_EMPTY_THRESHOLD:
            verdict = "FULLY EMPTY -> safe to drop"
        elif pct_null >= MOSTLY_EMPTY_THRESHOLD:
            verdict = "MOSTLY EMPTY -> investigate, likely drop"
        elif pct_null > 0:
            verdict = "PARTIALLY EMPTY -> handle gaps (impute/flag), keep column"
        else:
            verdict = "COMPLETE -> no action needed"

        records.append({
            "column": c,
            "non_null_rows": non_null,
            "null_rows": null_count,
            "pct_null": round(pct_null, 2),
            "verdict": verdict,
        })

    report_df = pd.DataFrame(records).sort_values("pct_null", ascending=False)
    report_path = os.path.join(OUTPUT_DIR, "column_null_report.csv")
    report_df.to_csv(report_path, index=False)

    # ---- Print summary ----
    print(f"{'Column':<25} {'% Null':<10} {'Verdict'}")
    print("-" * 80)
    for _, r in report_df.iterrows():
        print(f"{r['column']:<25} {r['pct_null']:<10} {r['verdict']}")

    fully_empty = report_df[report_df["verdict"].str.startswith("FULLY")]
    mostly_empty = report_df[report_df["verdict"].str.startswith("MOSTLY")]

    print(f"\nReport saved to: {report_path}")
    print(f"\nSummary:")
    print(f"  Fully empty columns  : {len(fully_empty)}  {list(fully_empty['column']) if len(fully_empty) else ''}")
    print(f"  Mostly empty columns : {len(mostly_empty)}  {list(mostly_empty['column']) if len(mostly_empty) else ''}")

    if len(fully_empty) > 0:
        print(f"\nSuggested pandas snippet to drop fully-empty columns when you load data later:")
        print(f"  df = df.drop(columns={list(fully_empty['column'])})")


if __name__ == "__main__":
    main()