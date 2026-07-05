import os
import glob
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ====================================
# FILE PATHS
# ====================================

# Reads directly from the per-grid parquet files produced by 13.1.
# No combined "southern_indian_ocean_profile_depth_level.parquet" is needed.
GRID_PARTS_DIR = r"D:\INCOIS\Agro_project\data\processed\grid_parts"

OUTPUT_DIR  = r"D:\INCOIS\Agro_project\data\processed\southern_indian_ocean_zscore"
SPLIT_DIR   = r"D:\INCOIS\Agro_project\data\southern_indian_ocean_gridwise_split"
TRAIN_FOLDER = os.path.join(SPLIT_DIR, "train")
TEST_FOLDER  = os.path.join(SPLIT_DIR, "test")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)

TRAIN_END_YEAR  = 2021
TEST_START_YEAR = 2022

MIN_TRAIN_ROWS = 10
MIN_TEST_ROWS  = 5

DEPTH_BINS = [
    0, 10, 20, 30, 50, 75,
    100, 125, 150, 200, 300,
    400, 500, 700, 1000,
    1500, 2000, 3000, 4000, 6000, 99999
]
DEPTH_LABELS = DEPTH_BINS[:-1]

OUTPUT_COLS = [
    "grid_id", "file_name", "date", "latitude", "longitude",
    "depth", "depth_bin", "pressure", "temperature", "salinity",
    "month", "year", "season",
    "temp_grid_mean", "temp_grid_std",
    "sal_grid_mean",  "sal_grid_std",
    "n_obs",
    "temp_zscore", "sal_zscore",
    "temp_zscore_abs", "sal_zscore_abs",
    "temp_z_flag", "sal_z_flag",
    "z_final_flag", "z_flag_label",
    "temp_qc", "psal_qc",
]


# ====================================
# DIAGNOSTIC CHECK
# ====================================

def run_diagnostic_checks():
    print("=" * 60)
    print("DIAGNOSTIC CHECKS")
    print("=" * 60)

    all_ok = True
    parquet_files = []

    print(f"\n[CHECK] Grid parts folder:")
    print(f"  Path : {GRID_PARTS_DIR}")

    if os.path.isdir(GRID_PARTS_DIR):
        parquet_files = sorted(glob.glob(os.path.join(GRID_PARTS_DIR, "*.parquet")))
        print(f"  ✅ Exists — {len(parquet_files)} grid parquet files found")
        if not parquet_files:
            print("  ❌ No .parquet files inside! Run 13.1__nc_to_parquet.py first.")
            all_ok = False
    else:
        print("  ❌ Does not exist — run 13.1__nc_to_parquet.py first")
        all_ok = False

    print("=" * 60)
    if all_ok:
        print("✅ ALL CHECKS PASSED — proceeding with pipeline")
    else:
        print("❌ SOME CHECKS FAILED — see above for what to fix")
    print("=" * 60)

    return all_ok, parquet_files


# ====================================
# PER-GRID FEATURE / STAT HELPERS
# ====================================

def add_date_features(df):
    df["date"]  = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["year"]  = df["date"].dt.year

    def get_season(month):
        if   month in [12, 1, 2]: return 1   # DJF - SH Summer
        elif month in [3, 4, 5]:  return 2   # MAM - SH Autumn
        elif month in [6, 7, 8]:  return 3   # JJA - SH Winter
        else:                     return 4   # SON - SH Spring

    df["season"] = df["month"].apply(get_season)
    return df


def add_depth_bins(df):
    df["depth_bin"] = pd.cut(
        df["depth"], bins=DEPTH_BINS, labels=DEPTH_LABELS, right=True
    ).astype(float)
    return df


def calculate_grid_statistics(df):
    """Stats grouped by depth_bin + season, WITHIN a single grid's data."""
    stats = df.groupby(["depth_bin", "season"], observed=True).agg(
        temp_grid_mean   = ("temperature", "mean"),
        temp_grid_std    = ("temperature", "std"),
        temp_grid_median = ("temperature", "median"),
        temp_grid_q1     = ("temperature", lambda x: x.quantile(0.25)),
        temp_grid_q3     = ("temperature", lambda x: x.quantile(0.75)),
        temp_grid_min    = ("temperature", "min"),
        temp_grid_max    = ("temperature", "max"),

        sal_grid_mean    = ("salinity", "mean"),
        sal_grid_std     = ("salinity", "std"),
        sal_grid_median  = ("salinity", "median"),
        sal_grid_q1      = ("salinity", lambda x: x.quantile(0.25)),
        sal_grid_q3      = ("salinity", lambda x: x.quantile(0.75)),
        sal_grid_min     = ("salinity", "min"),
        sal_grid_max     = ("salinity", "max"),

        n_obs            = ("temperature", "count"),
    ).reset_index()
    return stats


def calculate_zscores(df):
    temp_std_safe = df["temp_grid_std"].replace(0, np.nan)
    sal_std_safe  = df["sal_grid_std"].replace(0, np.nan)

    df["temp_zscore"] = (df["temperature"] - df["temp_grid_mean"]) / temp_std_safe
    df["sal_zscore"]  = (df["salinity"]    - df["sal_grid_mean"])  / sal_std_safe

    df["temp_zscore_abs"] = df["temp_zscore"].abs()
    df["sal_zscore_abs"]  = df["sal_zscore"].abs()
    return df


def apply_zscore_flags(df):
    def zscore_to_flag(z):
        if pd.isna(z):     return 9
        abs_z = abs(z)
        if   abs_z <= 2.0: return 1
        elif abs_z <= 3.0: return 2
        elif abs_z <= 4.0: return 3
        else:              return 4

    df["temp_z_flag"] = df["temp_zscore"].apply(zscore_to_flag)
    df["sal_z_flag"]  = df["sal_zscore"].apply(zscore_to_flag)

    temp_bad = (df["temperature"] > 40) | (df["temperature"] < -2.5)
    sal_bad  = (df["salinity"]    > 42) | (df["salinity"]    <  2.0)

    df.loc[temp_bad, "temp_z_flag"] = 4
    df.loc[sal_bad,  "sal_z_flag"]  = 4

    df["z_final_flag"] = df[["temp_z_flag", "sal_z_flag"]].max(axis=1)

    flag_map = {1: "GOOD", 2: "PROBABLY_GOOD", 3: "PROBABLY_BAD", 4: "BAD", 9: "MISSING"}
    df["z_flag_label"] = df["z_final_flag"].map(flag_map)
    return df


def process_one_grid(parquet_path):
    """Load one grid parquet, add features, compute z-scores + flags."""
    grid_df = pd.read_parquet(parquet_path)

    if grid_df.empty:
        return None

    if "grid_id" in grid_df.columns:
        grid_id = grid_df["grid_id"].iloc[0]
    else:
        grid_id = os.path.splitext(os.path.basename(parquet_path))[0]

    grid_df = add_date_features(grid_df)
    grid_df = add_depth_bins(grid_df)

    stats = calculate_grid_statistics(grid_df)
    grid_df = grid_df.merge(stats, on=["depth_bin", "season"], how="left")

    grid_df = calculate_zscores(grid_df)
    grid_df = apply_zscore_flags(grid_df)

    return grid_id, grid_df, stats


def split_train_test(grid_id, grid_df):
    train_mask = grid_df["year"] <= TRAIN_END_YEAR
    test_mask  = grid_df["year"] >= TEST_START_YEAR

    n_train = int(train_mask.sum())
    n_test  = int(test_mask.sum())

    if n_train < MIN_TRAIN_ROWS:
        return None, None, {"grid_id": grid_id, "reason": f"train rows={n_train} < min={MIN_TRAIN_ROWS}"}
    if n_test < MIN_TEST_ROWS:
        return None, None, {"grid_id": grid_id, "reason": f"test rows={n_test} < min={MIN_TEST_ROWS}"}

    train_df = grid_df.loc[train_mask].reset_index(drop=True)
    test_df  = grid_df.loc[test_mask].reset_index(drop=True)

    return train_df, test_df, None


# ====================================
# MAIN
# ====================================

def main():
    print("=" * 60)
    print("ARGO QC — GRID-WISE Z-SCORE + TRAIN/TEST SPLIT")
    print("(no combined parquet needed - reads grid_parts directly)")
    print("=" * 60)

    all_ok, parquet_files = run_diagnostic_checks()
    if not all_ok:
        print("\n❌ Fix the issues above, then run this script again.")
        return

    summary_records = []
    skipped_records = []
    all_grid_stats  = []
    worst_temp_rows = []
    worst_sal_rows  = []
    bad_obs_rows    = []

    total_train_rows = 0
    total_test_rows  = 0
    grids_saved   = 0
    grids_skipped = 0
    grids_failed  = 0

    total_grids = len(parquet_files)
    print(f"\nProcessing {total_grids} grid parquet files...\n")
    print("-" * 60)

    for i, path in enumerate(parquet_files, start=1):
        try:
            result = process_one_grid(path)
        except Exception as e:
            print(f"  ❌ ERROR processing {os.path.basename(path)}: {e}")
            grids_failed += 1
            continue

        if result is None:
            continue

        grid_id, grid_df, stats = result

        stats = stats.copy()
        stats.insert(0, "grid_id", grid_id)
        all_grid_stats.append(stats)

        cols_present = [c for c in OUTPUT_COLS if c in grid_df.columns]
        out_df = grid_df[cols_present]

        train_df, test_df, skip_reason = split_train_test(grid_id, out_df)

        if skip_reason is not None:
            skipped_records.append(skip_reason)
            grids_skipped += 1
            continue

        train_df.to_csv(os.path.join(TRAIN_FOLDER, f"{grid_id}_train.csv"), index=False)
        test_df.to_csv(os.path.join(TEST_FOLDER,  f"{grid_id}_test.csv"),  index=False)

        n_train, n_test = len(train_df), len(test_df)
        total_train_rows += n_train
        total_test_rows  += n_test
        grids_saved += 1

        summary_records.append({
            "grid_id":      grid_id,
            "total_rows":   len(out_df),
            "train_rows":   n_train,
            "test_rows":    n_test,
            "train_pct":    round(100 * n_train / len(out_df), 2),
            "test_pct":     round(100 * n_test  / len(out_df), 2),
            "train_yr_min": int(train_df["year"].min()),
            "train_yr_max": int(train_df["year"].max()),
            "test_yr_min":  int(test_df["year"].min()),
            "test_yr_max":  int(test_df["year"].max()),
            "temp_mean":    round(out_df["temperature"].mean(), 4) if "temperature" in out_df.columns else np.nan,
            "sal_mean":     round(out_df["salinity"].mean(), 4) if "salinity" in out_df.columns else np.nan,
            "bad_count":    int((out_df["z_final_flag"] >= 3).sum()),
            "bad_pct":      round(100 * (out_df["z_final_flag"] >= 3).sum() / len(out_df), 2),
        })

        worst_temp_rows.append(out_df.nlargest(20, "temp_zscore_abs"))
        worst_sal_rows.append(out_df.nlargest(20, "sal_zscore_abs"))
        bad_obs_rows.append(out_df[out_df["z_final_flag"] >= 3])

        if i % 25 == 0 or i == total_grids:
            print(f"  [{i}/{total_grids}] saved={grids_saved} skipped={grids_skipped} failed={grids_failed}")

    # ====================================
    # AGGREGATE SUMMARY OUTPUTS
    # ====================================

    print("\n--- WRITING SUMMARY FILES ---")

    summary_df = pd.DataFrame(summary_records)
    skipped_df = pd.DataFrame(skipped_records) if skipped_records else pd.DataFrame(columns=["grid_id", "reason"])

    summary_df.to_csv(os.path.join(SPLIT_DIR, "split_summary.csv"), index=False)
    skipped_df.to_csv(os.path.join(SPLIT_DIR, "skipped_grids.csv"), index=False)
    print("✅ Saved: split_summary.csv, skipped_grids.csv")

    if all_grid_stats:
        grid_stats_df = pd.concat(all_grid_stats, ignore_index=True)
        grid_stats_df.to_parquet(os.path.join(OUTPUT_DIR, "grid_statistics.parquet"), index=False)
        print("✅ Saved: grid_statistics.parquet")

    if worst_temp_rows:
        pd.concat(worst_temp_rows, ignore_index=True) \
            .sort_values("temp_zscore_abs", ascending=False).head(100) \
            .to_csv(os.path.join(OUTPUT_DIR, "worst_temperature_anomalies.csv"), index=False)
        print("✅ Saved: worst_temperature_anomalies.csv")

    if worst_sal_rows:
        pd.concat(worst_sal_rows, ignore_index=True) \
            .sort_values("sal_zscore_abs", ascending=False).head(100) \
            .to_csv(os.path.join(OUTPUT_DIR, "worst_salinity_anomalies.csv"), index=False)
        print("✅ Saved: worst_salinity_anomalies.csv")

    if bad_obs_rows:
        bad_df = pd.concat(bad_obs_rows, ignore_index=True)
        bad_df.to_csv(os.path.join(OUTPUT_DIR, "bad_observations_zscore.csv"), index=False)
        print(f"✅ Saved: bad_observations_zscore.csv ({len(bad_df):,} rows)")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Grids processed total   : {total_grids:,}")
    print(f"Grids saved (train+test): {grids_saved:,}")
    print(f"Grids skipped           : {grids_skipped:,}")
    print(f"Grids failed            : {grids_failed:,}")
    print(f"Total train rows        : {total_train_rows:,}")
    print(f"Total test rows         : {total_test_rows:,}")
    print(f"\nTrain folder : {TRAIN_FOLDER}")
    print(f"Test folder  : {TEST_FOLDER}")
    print(f"Zscore extras: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()