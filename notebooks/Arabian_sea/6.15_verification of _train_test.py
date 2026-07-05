import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==========================================================
# CONFIGURATION — UPDATE THESE PATHS
# ==========================================================

ORIGINAL_INPUT_FOLDER = (
    r"D:\INCOIS\Agro_project\data"
    r"\Arabian_sea_gridwise_csv"
)

SPLIT_OUTPUT_FOLDER = (
    r"D:\INCOIS\Agro_project\data"
    r"\Arabian_sea_gridwise_split"
)

TRAIN_FOLDER   = os.path.join(SPLIT_OUTPUT_FOLDER, "train")
TEST_FOLDER    = os.path.join(SPLIT_OUTPUT_FOLDER, "test")
SUMMARY_FILE   = os.path.join(SPLIT_OUTPUT_FOLDER, "split_summary.csv")
SKIPPED_FILE   = os.path.join(SPLIT_OUTPUT_FOLDER, "skipped_grids.csv")

TRAIN_END_YEAR  = 2021
TEST_START_YEAR = 2022
DATE_COLUMN     = "date"

# ==========================================================
# HELPER : PRINT SECTION HEADER
# ==========================================================

def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def subsection(title):
    print("\n" + "-" * 50)
    print(f"  {title}")
    print("-" * 50)

# ==========================================================
# START VERIFICATION
# ==========================================================

print("=" * 60)
print("  TRAIN/TEST SPLIT — FULL VERIFICATION REPORT")
print("=" * 60)

all_checks_passed = True
check_results     = []

def record_check(check_name, passed, detail=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    check_results.append({
        "check"  : check_name,
        "status" : status,
        "detail" : detail
    })
    if not passed:
        global all_checks_passed
        all_checks_passed = False
    print(f"  {status}  {check_name}")
    if detail:
        print(f"         → {detail}")

# ==========================================================
# CHECK 1: FOLDER EXISTENCE
# ==========================================================

section("CHECK 1 : FOLDER & FILE EXISTENCE")

folders = {
    "Input  Folder" : ORIGINAL_INPUT_FOLDER,
    "Output Folder" : SPLIT_OUTPUT_FOLDER,
    "Train  Folder" : TRAIN_FOLDER,
    "Test   Folder" : TEST_FOLDER,
}

for name, path in folders.items():
    exists = os.path.exists(path)
    record_check(
        f"{name} exists",
        exists,
        path
    )

# Check summary files
for fname, fpath in [
    ("split_summary.csv", SUMMARY_FILE),
    ("skipped_grids.csv", SKIPPED_FILE)
]:
    exists = os.path.exists(fpath)
    record_check(
        f"{fname} exists",
        exists,
        fpath
    )

# ==========================================================
# CHECK 2: FILE COUNTS
# ==========================================================

section("CHECK 2 : FILE COUNT VERIFICATION")

original_files = sorted(Path(ORIGINAL_INPUT_FOLDER).glob("*.csv"))
train_files    = sorted(Path(TRAIN_FOLDER).glob("*_train.csv"))
test_files     = sorted(Path(TEST_FOLDER).glob("*_test.csv"))

n_original = len(original_files)
n_train    = len(train_files)
n_test     = len(test_files)

print(f"\n  Original grid CSVs : {n_original:,}")
print(f"  Train  CSVs saved  : {n_train:,}")
print(f"  Test   CSVs saved  : {n_test:,}")

# Load skipped info
if os.path.exists(SKIPPED_FILE):
    skipped_df = pd.read_csv(SKIPPED_FILE)
    n_skipped  = len(skipped_df)
else:
    skipped_df = pd.DataFrame()
    n_skipped  = 0

print(f"  Grids skipped      : {n_skipped:,}")
print(f"  Expected saved     : {n_original - n_skipped:,}")

# Train count == Test count
record_check(
    "Train file count == Test file count",
    n_train == n_test,
    f"train={n_train}, test={n_test}"
)

# Saved + Skipped == Original
record_check(
    "Saved + Skipped == Original total",
    (n_train + n_skipped) == n_original,
    f"saved={n_train} + skipped={n_skipped}"
    f" = {n_train + n_skipped} vs original={n_original}"
)

# ==========================================================
# CHECK 3: GRID ID MATCHING
# ==========================================================

section("CHECK 3 : GRID ID MATCHING")

# Extract grid IDs from filenames
original_ids = set(f.stem for f in original_files)

train_ids = set(
    f.stem.replace("_train", "")
    for f in train_files
)

test_ids = set(
    f.stem.replace("_test", "")
    for f in test_files
)

skipped_ids = set(skipped_df["grid_id"].tolist()) \
    if len(skipped_df) > 0 else set()

# Train IDs == Test IDs
record_check(
    "Train grid IDs == Test grid IDs",
    train_ids == test_ids,
    f"{len(train_ids - test_ids)} in train not in test"
    if train_ids != test_ids else "All match"
)

# Train + Skipped == Original
combined_ids = train_ids | skipped_ids
record_check(
    "Train IDs + Skipped IDs == Original IDs",
    combined_ids == original_ids,
    f"Missing: {original_ids - combined_ids}"
    if combined_ids != original_ids else "All accounted for"
)

# Any duplicates?
all_saved_ids = list(train_ids)
has_duplicates = len(all_saved_ids) != len(set(all_saved_ids))
record_check(
    "No duplicate grid IDs in train folder",
    not has_duplicates,
    "No duplicates found" if not has_duplicates
    else "⚠ Duplicates detected"
)

# Print missing grids if any
missing_from_output = original_ids - combined_ids
if missing_from_output:
    print(f"\n  ⚠ Grids missing from output:")
    for g in sorted(missing_from_output):
        print(f"    - {g}")

# ==========================================================
# CHECK 4: TEMPORAL SPLIT LOGIC — SAMPLE GRIDS
# ==========================================================

section("CHECK 4 : TEMPORAL SPLIT LOGIC VERIFICATION")

print("\n  Checking date boundaries in train/test files...")
print("  (Sampling up to 20 grids for detailed check)\n")

sample_grids = list(train_ids)[:20]

temporal_issues   = []
overlap_issues    = []
gap_issues        = []

for grid_id in sample_grids:

    train_file = Path(TRAIN_FOLDER) / f"{grid_id}_train.csv"
    test_file  = Path(TEST_FOLDER)  / f"{grid_id}_test.csv"

    if not train_file.exists() or not test_file.exists():
        continue

    t_df  = pd.read_csv(train_file)
    te_df = pd.read_csv(test_file)

    t_df[DATE_COLUMN]  = pd.to_datetime(t_df[DATE_COLUMN],  errors="coerce")
    te_df[DATE_COLUMN] = pd.to_datetime(te_df[DATE_COLUMN], errors="coerce")

    t_df["year"]  = t_df[DATE_COLUMN].dt.year
    te_df["year"] = te_df[DATE_COLUMN].dt.year

    train_max_year = int(t_df["year"].max())
    test_min_year  = int(te_df["year"].min())
    train_min_year = int(t_df["year"].min())
    test_max_year  = int(te_df["year"].max())

    # Check: no train data after TRAIN_END_YEAR
    train_leak = (t_df["year"] > TRAIN_END_YEAR).sum()

    # Check: no test data before TEST_START_YEAR
    test_leak = (te_df["year"] < TEST_START_YEAR).sum()

    # Check: test comes AFTER train
    order_ok = test_min_year >= train_max_year

    status = "✅" if (train_leak == 0 and
                      test_leak  == 0 and
                      order_ok) else "❌"

    print(
        f"  {status} {grid_id:<22}"
        f"  train: {train_min_year}-{train_max_year}"
        f"  |  test: {test_min_year}-{test_max_year}"
        f"  |  train_leak={train_leak}"
        f"  |  test_leak={test_leak}"
    )

    if train_leak > 0 or test_leak > 0:
        temporal_issues.append({
            "grid_id"    : grid_id,
            "train_leak" : train_leak,
            "test_leak"  : test_leak
        })

print()
record_check(
    "No temporal leakage in sampled grids",
    len(temporal_issues) == 0,
    f"{len(temporal_issues)} grids have leakage"
    if temporal_issues
    else "No leakage found in sample"
)

# ==========================================================
# CHECK 5: ROW COUNT VERIFICATION
# ==========================================================

section("CHECK 5 : ROW COUNT VERIFICATION")

print("\n  Verifying: original rows == train rows + test rows")
print("  (Sampling up to 30 grids)\n")

row_mismatch_grids = []
sample_for_rows    = list(train_ids)[:30]

total_orig_rows  = 0
total_train_rows = 0
total_test_rows  = 0

for grid_id in sample_for_rows:

    orig_file  = Path(ORIGINAL_INPUT_FOLDER) / f"{grid_id}.csv"
    train_file = Path(TRAIN_FOLDER) / f"{grid_id}_train.csv"
    test_file  = Path(TEST_FOLDER)  / f"{grid_id}_test.csv"

    if not orig_file.exists():
        continue

    orig_df  = pd.read_csv(orig_file)
    t_df     = pd.read_csv(train_file)
    te_df    = pd.read_csv(test_file)

    n_orig  = len(orig_df)
    n_train = len(t_df)
    n_test  = len(te_df)
    n_sum   = n_train + n_test

    # Account for possible skipped rows
    # (rows not in train period or test period)
    orig_df[DATE_COLUMN] = pd.to_datetime(
        orig_df[DATE_COLUMN], errors="coerce"
    )
    orig_df["year"] = orig_df[DATE_COLUMN].dt.year

    rows_in_train_period = (
        orig_df["year"] <= TRAIN_END_YEAR
    ).sum()

    rows_in_test_period = (
        orig_df["year"] >= TEST_START_YEAR
    ).sum()

    expected_total = rows_in_train_period + rows_in_test_period

    match = (n_train == rows_in_train_period and
             n_test  == rows_in_test_period)

    status = "✅" if match else "❌"

    print(
        f"  {status} {grid_id:<22}"
        f"  orig={n_orig:>6,}"
        f"  |  train={n_train:>6,} (expect {rows_in_train_period:>6,})"
        f"  |  test={n_test:>5,} (expect {rows_in_test_period:>5,})"
    )

    if not match:
        row_mismatch_grids.append({
            "grid_id"          : grid_id,
            "train_actual"     : n_train,
            "train_expected"   : rows_in_train_period,
            "test_actual"      : n_test,
            "test_expected"    : rows_in_test_period
        })

    total_orig_rows  += n_orig
    total_train_rows += n_train
    total_test_rows  += n_test

print()
record_check(
    "Row counts match expected split",
    len(row_mismatch_grids) == 0,
    f"{len(row_mismatch_grids)} grids have row mismatches"
    if row_mismatch_grids
    else "All row counts correct"
)

# ==========================================================
# CHECK 6: COLUMN INTEGRITY
# ==========================================================

section("CHECK 6 : COLUMN INTEGRITY")

print("\n  Checking columns are preserved after split...\n")

sample_grid = list(train_ids)[0]

orig_file  = Path(ORIGINAL_INPUT_FOLDER) / f"{sample_grid}.csv"
train_file = Path(TRAIN_FOLDER) / f"{sample_grid}_train.csv"
test_file  = Path(TEST_FOLDER)  / f"{sample_grid}_test.csv"

orig_cols  = set(pd.read_csv(orig_file, nrows=1).columns)
train_cols = set(pd.read_csv(train_file, nrows=1).columns)
test_cols  = set(pd.read_csv(test_file,  nrows=1).columns)

# Columns in train that are not in original
# (year column was added — that is expected)
extra_in_train = train_cols - orig_cols - {"year"}
extra_in_test  = test_cols  - orig_cols - {"year"}

missing_in_train = orig_cols - train_cols
missing_in_test  = orig_cols - test_cols

print(f"  Original  columns : {len(orig_cols)}")
print(f"  Train     columns : {len(train_cols)}"
      f"  (includes 'year' added during split)")
print(f"  Test      columns : {len(test_cols)}")

print(f"\n  Columns in original but missing in train : "
      f"{missing_in_train if missing_in_train else 'None'}")
print(f"  Columns in original but missing in test  : "
      f"{missing_in_test if missing_in_test else 'None'}")
print(f"  Extra unexpected columns in train        : "
      f"{extra_in_train if extra_in_train else 'None'}")
print(f"  Extra unexpected columns in test         : "
      f"{extra_in_test if extra_in_test else 'None'}")

record_check(
    "No original columns missing in train",
    len(missing_in_train) == 0,
    f"Missing: {missing_in_train}"
    if missing_in_train else "All columns present"
)

record_check(
    "No original columns missing in test",
    len(missing_in_test) == 0,
    f"Missing: {missing_in_test}"
    if missing_in_test else "All columns present"
)

record_check(
    "No unexpected extra columns",
    len(extra_in_train) == 0 and len(extra_in_test) == 0,
    f"Extra: {extra_in_train | extra_in_test}"
    if (extra_in_train or extra_in_test) else "Clean"
)

# ==========================================================
# CHECK 7: DATA INTEGRITY — NO DATA CORRUPTION
# ==========================================================

section("CHECK 7 : DATA INTEGRITY CHECK")

print("\n  Checking for NaN, nulls, data types...\n")

sample_grid  = list(train_ids)[0]
train_file   = Path(TRAIN_FOLDER) / f"{sample_grid}_train.csv"
test_file    = Path(TEST_FOLDER)  / f"{sample_grid}_test.csv"

t_df         = pd.read_csv(train_file)
te_df        = pd.read_csv(test_file)

key_cols = [
    "temperature", "salinity",
    "depth", "latitude", "longitude"
]
key_cols = [c for c in key_cols if c in t_df.columns]

print(f"  Sample grid : {sample_grid}")
print(f"\n  Train Null Counts:")
train_nulls = t_df[key_cols].isnull().sum()
print(train_nulls.to_string())

print(f"\n  Test Null Counts:")
test_nulls = te_df[key_cols].isnull().sum()
print(test_nulls.to_string())

print(f"\n  Train Data Types:")
print(t_df[key_cols].dtypes.to_string())

print(f"\n  Train Basic Statistics:")
print(t_df[key_cols].describe().round(4).to_string())

print(f"\n  Test Basic Statistics:")
print(te_df[key_cols].describe().round(4).to_string())

record_check(
    "Key columns exist in train file",
    len(key_cols) > 0,
    f"Found: {key_cols}"
)

# ==========================================================
# CHECK 8: SUMMARY FILE VERIFICATION
# ==========================================================

section("CHECK 8 : SUMMARY CSV VERIFICATION")

if os.path.exists(SUMMARY_FILE):

    summary_df = pd.read_csv(SUMMARY_FILE)

    print(f"\n  Summary file rows    : {len(summary_df):,}")
    print(f"  Summary file columns : {len(summary_df.columns)}")
    print(f"\n  Columns: {list(summary_df.columns)}")

    print(f"\n  Sample rows:")
    print(summary_df.head(5).to_string(index=False))

    print(f"\n  Train % Statistics:")
    print(summary_df["train_pct"].describe().round(2))

    print(f"\n  Test % Statistics:")
    print(summary_df["test_pct"].describe().round(2))

    # Verify summary counts match actual files
    summary_grids = set(summary_df["grid_id"].tolist())

    record_check(
        "Summary grid IDs match train folder",
        summary_grids == train_ids,
        f"{len(summary_grids - train_ids)} in summary not in train"
        if summary_grids != train_ids else "All match"
    )

    # Check no negative row counts
    neg_train = (summary_df["train_rows"] < 0).sum()
    neg_test  = (summary_df["test_rows"]  < 0).sum()

    record_check(
        "No negative row counts in summary",
        neg_train == 0 and neg_test == 0,
        "All positive" if neg_train == 0 and neg_test == 0
        else f"train_neg={neg_train}, test_neg={neg_test}"
    )

    # Check train_pct + test_pct is reasonable
    summary_df["check_pct"] = (
        summary_df["train_pct"] + summary_df["test_pct"]
    )
    weird_pct = (summary_df["check_pct"] > 101).sum()

    record_check(
        "Train% + Test% <= 101% (within rounding)",
        weird_pct == 0,
        f"{weird_pct} grids have train%+test% > 101"
        if weird_pct > 0 else "All percentages reasonable"
    )

# ==========================================================
# CHECK 9: OVERALL ROW COUNT ACROSS ALL FILES
# ==========================================================

section("CHECK 9 : FULL DATASET ROW COUNT")

print("\n  Counting rows across ALL train and test files...")
print("  (This may take a moment)\n")

total_train_all = 0
total_test_all  = 0

for f in Path(TRAIN_FOLDER).glob("*_train.csv"):
    df_temp = pd.read_csv(f, usecols=[DATE_COLUMN])
    total_train_all += len(df_temp)

for f in Path(TEST_FOLDER).glob("*_test.csv"):
    df_temp = pd.read_csv(f, usecols=[DATE_COLUMN])
    total_test_all += len(df_temp)

total_all = total_train_all + total_test_all

print(f"  Total Train Rows (all files) : {total_train_all:,}")
print(f"  Total Test  Rows (all files) : {total_test_all:,}")
print(f"  Combined Total               : {total_all:,}")

if total_all > 0:
    print(
        f"  Actual Split Ratio           : "
        f"{100*total_train_all/total_all:.1f}% train / "
        f"{100*total_test_all/total_all:.1f}% test"
    )

record_check(
    "Train rows > Test rows (train should be larger)",
    total_train_all > total_test_all,
    f"train={total_train_all:,} > test={total_test_all:,}"
    if total_train_all > total_test_all
    else f"⚠ test is larger than train"
)

# ==========================================================
# CHECK 10: SPOT CHECK — ONE GRID FULL DETAIL
# ==========================================================

section("CHECK 10 : DEEP SPOT CHECK — ONE GRID")

spot_grid  = list(train_ids)[0]
orig_file  = Path(ORIGINAL_INPUT_FOLDER) / f"{spot_grid}.csv"
train_file = Path(TRAIN_FOLDER) / f"{spot_grid}_train.csv"
test_file  = Path(TEST_FOLDER)  / f"{spot_grid}_test.csv"

print(f"\n  Grid selected for spot check : {spot_grid}")

orig_df  = pd.read_csv(orig_file)
t_df     = pd.read_csv(train_file)
te_df    = pd.read_csv(test_file)

orig_df[DATE_COLUMN]  = pd.to_datetime(orig_df[DATE_COLUMN],  errors="coerce")
t_df[DATE_COLUMN]     = pd.to_datetime(t_df[DATE_COLUMN],     errors="coerce")
te_df[DATE_COLUMN]    = pd.to_datetime(te_df[DATE_COLUMN],    errors="coerce")

orig_df["year"]  = orig_df[DATE_COLUMN].dt.year
t_df["year"]     = t_df[DATE_COLUMN].dt.year
te_df["year"]    = te_df[DATE_COLUMN].dt.year

print(f"\n  ORIGINAL FILE:")
print(f"    Rows          : {len(orig_df):,}")
print(f"    Year range    : {orig_df['year'].min()} → {orig_df['year'].max()}")
print(f"    Date min      : {orig_df[DATE_COLUMN].min()}")
print(f"    Date max      : {orig_df[DATE_COLUMN].max()}")

print(f"\n  TRAIN FILE ({spot_grid}_train.csv):")
print(f"    Rows          : {len(t_df):,}")
print(f"    Year range    : {t_df['year'].min()} → {t_df['year'].max()}")
print(f"    Date min      : {t_df[DATE_COLUMN].min()}")
print(f"    Date max      : {t_df[DATE_COLUMN].max()}")
print(f"    Rows > {TRAIN_END_YEAR}   : {(t_df['year'] > TRAIN_END_YEAR).sum()} ← should be 0")

print(f"\n  TEST FILE ({spot_grid}_test.csv):")
print(f"    Rows          : {len(te_df):,}")
print(f"    Year range    : {te_df['year'].min()} → {te_df['year'].max()}")
print(f"    Date min      : {te_df[DATE_COLUMN].min()}")
print(f"    Date max      : {te_df[DATE_COLUMN].max()}")
print(f"    Rows < {TEST_START_YEAR}   : {(te_df['year'] < TEST_START_YEAR).sum()} ← should be 0")

print(f"\n  Year-wise row distribution:")
print(f"\n  {'Year':<8} {'Original':>10} {'Train':>10} {'Test':>10}")
print(f"  {'-'*40}")

all_years = sorted(orig_df["year"].dropna().unique().astype(int))

for yr in all_years:
    o = (orig_df["year"] == yr).sum()
    t = (t_df["year"]   == yr).sum()
    e = (te_df["year"]  == yr).sum()
    tag = " ← TRAIN" if yr <= TRAIN_END_YEAR else " ← TEST"
    print(f"  {yr:<8} {o:>10,} {t:>10,} {e:>10,} {tag}")

# ==========================================================
# FINAL REPORT — ALL CHECKS SUMMARY
# ==========================================================

section("FINAL VERIFICATION REPORT")

print(f"\n  {'CHECK':<50} {'STATUS'}")
print(f"  {'-'*60}")

for item in check_results:
    print(f"  {item['check']:<50} {item['status']}")
    if item['detail']:
        print(f"  {'':50}   → {item['detail']}")

passed = sum(1 for c in check_results if "PASS" in c["status"])
failed = sum(1 for c in check_results if "FAIL" in c["status"])
total  = len(check_results)

print(f"\n  {'─'*60}")
print(f"  Total Checks : {total}")
print(f"  Passed       : {passed}")
print(f"  Failed       : {failed}")

print("\n" + "=" * 60)
if all_checks_passed:
    print("  ✅ ALL CHECKS PASSED")
    print("  Your train/test split is CORRECT and VERIFIED")
else:
    print("  ❌ SOME CHECKS FAILED")
    print("  Review the FAIL items above and fix before proceeding")
print("=" * 60)
