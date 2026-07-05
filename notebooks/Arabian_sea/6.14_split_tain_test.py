import os
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================================
# CONFIGURATION
# ==========================================================

INPUT_FOLDER = (
    r"D:\INCOIS\Agro_project\data"
    r"\Arabian_sea_gridwise_csv"
)

OUTPUT_FOLDER = (
    r"D:\INCOIS\Agro_project\data"
    r"\Arabian_sea_gridwise_split"
)

TRAIN_FOLDER = os.path.join(OUTPUT_FOLDER, "train")
TEST_FOLDER  = os.path.join(OUTPUT_FOLDER, "test")

# ----------------------------------------------------------
# SPLIT SETTINGS
# ----------------------------------------------------------

TRAIN_END_YEAR  = 2021
TEST_START_YEAR = 2022

DATE_COLUMN = "date"

MIN_TRAIN_ROWS = 10
MIN_TEST_ROWS  = 5

# ==========================================================
# CREATE OUTPUT DIRECTORIES
# ==========================================================

os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER,  exist_ok=True)

print("=" * 60)
print("GRID-WISE TEMPORAL TRAIN / TEST SPLIT")
print("=" * 60)
print(f"\nInput  Folder : {INPUT_FOLDER}")
print(f"Output Folder : {OUTPUT_FOLDER}")
print(f"\nTrain Period  : up to and including {TRAIN_END_YEAR}")
print(f"Test  Period  : from {TEST_START_YEAR} onwards")
print(f"\nMin Train Rows: {MIN_TRAIN_ROWS}")
print(f"Min Test  Rows: {MIN_TEST_ROWS}")


# ==========================================================
# SMART DATE PARSER
# Handles nanosecond timestamps like:
#   2002-04-19 02:35:38.000265472  ← 9 decimal places (ns)
#   2023-04-02 14:57:30            ← normal datetime
# Strategy: truncate sub-microsecond digits, then parse
# NO ROWS ARE DROPPED — bad dates get forward/backward filled
# ==========================================================

def parse_dates_safe(series: pd.Series) -> pd.Series:
    """
    Robustly parse a date series that may contain nanosecond
    precision timestamps (9 decimal places) which overflow
    standard pandas datetime parsing.

    Steps:
      1. Strip nanosecond overflow: truncate fractional seconds
         to max 6 digits (microsecond precision)
      2. Parse with pd.to_datetime
      3. Any remaining NaT → forward fill, then backward fill
         so ZERO rows are lost
    """

    def truncate_ns(val):
        """Truncate fractional seconds to 6 digits max."""
        if not isinstance(val, str):
            return val
        # Match pattern: digits after decimal point in time part
        # e.g. '02:35:38.000265472' → '02:35:38.000265'
        import re
        return re.sub(
            r'(\d{2}:\d{2}:\d{2})\.(\d{6})\d*',
            r'\1.\2',
            val
        )

    # Step 1 — truncate nanoseconds
    cleaned = series.astype(str).apply(truncate_ns)

    # Step 2 — parse
    parsed = pd.to_datetime(cleaned, errors="coerce")

    # Step 3 — report and fill any remaining NaT
    n_nat = parsed.isna().sum()
    if n_nat > 0:
        print(f"      ⚠  {n_nat:,} dates still unparseable"
              f" after truncation → forward/backward filled")
        parsed = parsed.ffill().bfill()

    return parsed


# ==========================================================
# GET ALL GRID CSV FILES
# ==========================================================

all_csv_files = sorted(Path(INPUT_FOLDER).glob("*.csv"))
total_grids   = len(all_csv_files)

print(f"\nTotal Grid CSV Files Found : {total_grids:,}")

if total_grids == 0:
    print("❌ No CSV files found. Check INPUT_FOLDER path.")
    exit()

# ==========================================================
# TRACKING VARIABLES
# ==========================================================

summary_records  = []
skipped_grids    = []
total_train_rows = 0
total_test_rows  = 0
grids_saved      = 0
grids_skipped    = 0

# ==========================================================
# PROCESS EACH GRID
# ==========================================================

print("\n" + "-" * 60)
print("Processing Grids...")
print("-" * 60 + "\n")

for i, csv_file in enumerate(all_csv_files, start=1):

    grid_id = csv_file.stem

    # --------------------------------------------------
    # LOAD
    # --------------------------------------------------
    try:
        grid_df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"  [{i}/{total_grids}] ❌ ERROR loading"
              f" {grid_id} : {e}")
        skipped_grids.append({"grid_id": grid_id,
                               "reason": f"load error: {e}"})
        grids_skipped += 1
        continue

    total_rows = len(grid_df)

    if total_rows == 0:
        print(f"  [{i}/{total_grids}] ⚠ SKIP {grid_id}"
              f" — empty file")
        skipped_grids.append({"grid_id": grid_id,
                               "reason": "empty file"})
        grids_skipped += 1
        continue

    # --------------------------------------------------
    # CHECK DATE COLUMN EXISTS
    # --------------------------------------------------
    if DATE_COLUMN not in grid_df.columns:
        print(f"  [{i}/{total_grids}] ⚠ SKIP {grid_id}"
              f" — no '{DATE_COLUMN}' column")
        skipped_grids.append({"grid_id": grid_id,
                               "reason": f"missing: {DATE_COLUMN}"})
        grids_skipped += 1
        continue

    # --------------------------------------------------
    # PARSE DATES — ZERO ROWS DROPPED
    # --------------------------------------------------
    grid_df[DATE_COLUMN] = parse_dates_safe(grid_df[DATE_COLUMN])

    # Confirm no NaT remain
    remaining_nat = grid_df[DATE_COLUMN].isna().sum()
    assert remaining_nat == 0, \
        f"Still {remaining_nat} NaT after fill — check data!"

    # --------------------------------------------------
    # EXTRACT YEAR
    # --------------------------------------------------
    grid_df["year"] = grid_df[DATE_COLUMN].dt.year

    # --------------------------------------------------
    # TEMPORAL SPLIT
    # --------------------------------------------------
    train_df = grid_df[grid_df["year"] <= TRAIN_END_YEAR].copy()
    test_df  = grid_df[grid_df["year"] >= TEST_START_YEAR].copy()

    n_train = len(train_df)
    n_test  = len(test_df)

    # --------------------------------------------------
    # MINIMUM ROW CHECK
    # --------------------------------------------------
    if n_train < MIN_TRAIN_ROWS:
        reason = f"train rows={n_train} < min={MIN_TRAIN_ROWS}"
        print(f"  [{i}/{total_grids}] ⚠ SKIP {grid_id} — {reason}")
        skipped_grids.append({"grid_id": grid_id, "reason": reason})
        grids_skipped += 1
        continue

    if n_test < MIN_TEST_ROWS:
        reason = f"test rows={n_test} < min={MIN_TEST_ROWS}"
        print(f"  [{i}/{total_grids}] ⚠ SKIP {grid_id} — {reason}")
        skipped_grids.append({"grid_id": grid_id, "reason": reason})
        grids_skipped += 1
        continue

    # --------------------------------------------------
    # RESET INDEX
    # --------------------------------------------------
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    # --------------------------------------------------
    # SAVE TRAIN CSV
    # --------------------------------------------------
    train_file = os.path.join(TRAIN_FOLDER, f"{grid_id}_train.csv")
    train_df.to_csv(train_file, index=False)

    # --------------------------------------------------
    # SAVE TEST CSV
    # --------------------------------------------------
    test_file = os.path.join(TEST_FOLDER, f"{grid_id}_test.csv")
    test_df.to_csv(test_file, index=False)

    # --------------------------------------------------
    # STATS
    # --------------------------------------------------
    total_train_rows += n_train
    total_test_rows  += n_test
    grids_saved      += 1

    train_pct = round(100 * n_train / total_rows, 2)
    test_pct  = round(100 * n_test  / total_rows, 2)

    train_yr_min = int(train_df["year"].min())
    train_yr_max = int(train_df["year"].max())
    test_yr_min  = int(test_df["year"].min())
    test_yr_max  = int(test_df["year"].max())

    depth_min = round(grid_df["depth"].min(), 2) \
        if "depth" in grid_df.columns else np.nan
    depth_max = round(grid_df["depth"].max(), 2) \
        if "depth" in grid_df.columns else np.nan

    temp_mean = round(grid_df["temperature"].mean(), 4) \
        if "temperature" in grid_df.columns else np.nan
    temp_std  = round(grid_df["temperature"].std(),  4) \
        if "temperature" in grid_df.columns else np.nan

    sal_mean = round(grid_df["salinity"].mean(), 4) \
        if "salinity" in grid_df.columns else np.nan
    sal_std  = round(grid_df["salinity"].std(),  4) \
        if "salinity" in grid_df.columns else np.nan

    summary_records.append({
        "grid_id"      : grid_id,
        "total_rows"   : total_rows,
        "train_rows"   : n_train,
        "test_rows"    : n_test,
        "train_pct"    : train_pct,
        "test_pct"     : test_pct,
        "train_yr_min" : train_yr_min,
        "train_yr_max" : train_yr_max,
        "test_yr_min"  : test_yr_min,
        "test_yr_max"  : test_yr_max,
        "depth_min_m"  : depth_min,
        "depth_max_m"  : depth_max,
        "temp_mean"    : temp_mean,
        "temp_std"     : temp_std,
        "sal_mean"     : sal_mean,
        "sal_std"      : sal_std,
        "status"       : "saved"
    })

    print(
        f"  [{i:>4}/{total_grids}]  {grid_id:<20}"
        f"  total={total_rows:>9,}"
        f"  |  train={n_train:>7,} ({train_pct:>5.1f}%)"
        f"  |  test={n_test:>7,} ({test_pct:>5.1f}%)"
        f"  |  years {train_yr_min}-{train_yr_max}"
        f" → {test_yr_min}-{test_yr_max}"
    )

# ==========================================================
# SAVE SUMMARY FILES
# ==========================================================

summary_df = pd.DataFrame(summary_records)
skipped_df = pd.DataFrame(skipped_grids) \
    if skipped_grids \
    else pd.DataFrame(columns=["grid_id", "reason"])

print("\n" + "-" * 60)
print("Saving Summary Files...")
print("-" * 60)

summary_path = os.path.join(OUTPUT_FOLDER, "split_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\n✅ Saved : split_summary.csv → {summary_path}")

skipped_path = os.path.join(OUTPUT_FOLDER, "skipped_grids.csv")
skipped_df.to_csv(skipped_path, index=False)
print(f"✅ Saved : skipped_grids.csv → {skipped_path}")

# ==========================================================
# VERIFY OUTPUT
# ==========================================================

train_files_saved = len(list(Path(TRAIN_FOLDER).glob("*.csv")))
test_files_saved  = len(list(Path(TEST_FOLDER).glob("*.csv")))

print("\n" + "=" * 60)
print("FINAL SPLIT REPORT")
print("=" * 60)
print(f"\n  Total Grids Found    : {total_grids:,}")
print(f"  Grids Saved          : {grids_saved:,}")
print(f"  Grids Skipped        : {grids_skipped:,}")
print(f"\n  Total Train Rows     : {total_train_rows:,}")
print(f"  Total Test  Rows     : {total_test_rows:,}")

combined = total_train_rows + total_test_rows
if combined > 0:
    print(
        f"\n  Overall Split        : "
        f"{100*total_train_rows/combined:.1f}% train / "
        f"{100*total_test_rows/combined:.1f}% test"
    )

if len(summary_df) > 0:
    print(f"\n  Per-Grid Row Statistics:")
    print(f"    Train — min={summary_df['train_rows'].min():,}  "
          f"max={summary_df['train_rows'].max():,}  "
          f"mean={summary_df['train_rows'].mean():.0f}")
    print(f"    Test  — min={summary_df['test_rows'].min():,}  "
          f"max={summary_df['test_rows'].max():,}  "
          f"mean={summary_df['test_rows'].mean():.0f}")

    print(f"\n  Top 5 Grids by Train Rows:")
    top5 = summary_df.nlargest(5, "train_rows")[
        ["grid_id","train_rows","test_rows","train_pct",
         "train_yr_min","train_yr_max"]
    ]
    print(top5.to_string(index=False))

if len(skipped_df) > 0:
    print(f"\n  Skipped Grids:")
    print(skipped_df.to_string(index=False))

print("\n" + "=" * 60)
print("✅ TRAIN / TEST SPLIT COMPLETE ")
print("=" * 60)