import os
import re
import glob
import numpy as np
import pandas as pd

# ====================================
# CONFIG
# ====================================

# Folder containing the raw grid-wise CSVs (Grid_29.csv, Grid_132.csv, etc.)
INPUT_FOLDER = r"D:\INCOIS\Agro_project\data\raw\CTD_XBT_DATA\drive-download-20260703T143557Z-3-001"

# Where the single combined parquet file will be written
OUTPUT_PARQUET = r"D:\INCOIS\Agro_project\data\processed\all_grids_ctd_xbt_final.parquet"

# Optional: where to log any files/grids that had problems (never crashes the run)
ERROR_LOG = r"D:\INCOIS\Agro_project\data\processed\ctd_xbt_build_errors.log"

# Standard oceanographic depth-bin edges (lower-bound labels), matching Grid_29_train.csv
DEPTH_BIN_EDGES = [0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 400, 500, 700, 1000, 1500, 2000, np.inf]
DEPTH_BIN_LABELS = DEPTH_BIN_EDGES[:-1]  # label = lower edge of each bin

FINAL_COLUMNS = [
    "grid_id", "file_name", "date", "latitude", "longitude",
    "depth", "depth_bin", "pressure", "temperature", "salinity",
    "month", "year", "season",
    "temp_grid_mean", "temp_grid_std", "sal_grid_mean", "sal_grid_std", "n_obs",
    "temp_zscore", "sal_zscore", "temp_zscore_abs", "sal_zscore_abs",
    "temp_z_flag", "sal_z_flag",
    "temp_qc", "psal_qc",
]

MONTH_TO_SEASON = {
    12: 1, 1: 1, 2: 1,   # DJF -> winter
    3: 2, 4: 2, 5: 2,    # MAM -> spring
    6: 3, 7: 3, 8: 3,    # JJA -> summer
    9: 4, 10: 4, 11: 4,  # SON -> autumn
}


def log_error(msg: str):
    print(msg)
    try:
        with open(ERROR_LOG, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def zscore_flag(z: pd.Series) -> pd.Series:
    """
    Bucketed QC-style flag derived from |z-score|, matching Grid_29_train.csv:
      NaN        -> 9   (undefined, e.g. std == 0 or missing data)
      [0, 2)     -> 1   (good)
      [2, 3)     -> 2
      [3, 4)     -> 3
      >= 4       -> 4   (extreme outlier)
    """
    flag = pd.Series(1, index=z.index, dtype="int64")
    flag[(z >= 2) & (z < 3)] = 2
    flag[(z >= 3) & (z < 4)] = 3
    flag[z >= 4] = 4
    flag[z.isna()] = 9
    return flag


def process_one_grid_csv(csv_path: str) -> pd.DataFrame | None:
    file_label = os.path.basename(csv_path)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        log_error(f"SKIP {file_label}: could not read CSV -- {e}")
        return None

    # --- normalize column names we expect from the raw export ---
    rename_map = {}
    if "datetime" in df.columns and "date" not in df.columns:
        rename_map["datetime"] = "date"
    df = df.rename(columns=rename_map)

    required = ["grid_id", "file_name", "date", "latitude", "longitude",
                "depth", "pressure", "temperature", "salinity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log_error(f"SKIP {file_label}: missing required columns {missing}")
        return None

    # --- grid_id: strip "Grid_" prefix, keep the number only ---
    df["grid_id"] = (
        df["grid_id"].astype(str)
        .str.extract(r"(\d+)", expand=False)
    )
    if df["grid_id"].isna().any():
        bad = df["grid_id"].isna().sum()
        log_error(f"  WARNING {file_label}: {bad} rows had no numeric grid_id, dropping them")
        df = df.dropna(subset=["grid_id"])
    df["grid_id"] = df["grid_id"].astype(int)

    # --- qc columns: keep if present, else fill with NaN so schema still matches ---
    if "temp_qc" not in df.columns:
        df["temp_qc"] = np.nan
    if "psal_qc" not in df.columns:
        df["psal_qc"] = np.nan

    # --- parse date, derive month/year/season ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["season"] = df["month"].map(MONTH_TO_SEASON)

    # --- depth_bin: standard bins, negative/invalid depth -> NaN ---
    depth = pd.to_numeric(df["depth"], errors="coerce")
    df["depth_bin"] = pd.cut(
        depth.where(depth >= 0, np.nan),
        bins=DEPTH_BIN_EDGES,
        labels=DEPTH_BIN_LABELS,
        right=False,
        include_lowest=True,
    ).astype(float)

    # --- per-grid, per-depth_bin, per-season stats ---
    group_cols = ["grid_id", "depth_bin", "season"]
    stats = df.groupby(group_cols, dropna=False).agg(
        temp_grid_mean=("temperature", "mean"),
        temp_grid_std=("temperature", "std"),
        sal_grid_mean=("salinity", "mean"),
        sal_grid_std=("salinity", "std"),
        n_obs=("temperature", "count"),
    ).reset_index()

    df = df.merge(stats, on=group_cols, how="left")

    # --- z-scores + flags ---
    df["temp_zscore"] = (df["temperature"] - df["temp_grid_mean"]) / df["temp_grid_std"]
    df["sal_zscore"] = (df["salinity"] - df["sal_grid_mean"]) / df["sal_grid_std"]
    df["temp_zscore_abs"] = df["temp_zscore"].abs()
    df["sal_zscore_abs"] = df["sal_zscore"].abs()
    df["temp_z_flag"] = zscore_flag(df["temp_zscore_abs"])
    df["sal_z_flag"] = zscore_flag(df["sal_zscore_abs"])

    # --- final column order (missing ones filled as NaN so every grid matches schema) ---
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[FINAL_COLUMNS]

    return df


def main():
    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "Grid_*.csv")))
    print(f"Found {len(csv_files)} grid CSV files in:\n{INPUT_FOLDER}\n")

    if not csv_files:
        print("No Grid_*.csv files found -- nothing to do.")
        return

    all_dfs = []
    ok_count = 0
    fail_count = 0

    for csv_path in csv_files:
        try:
            result = process_one_grid_csv(csv_path)
            if result is not None and len(result) > 0:
                all_dfs.append(result)
                ok_count += 1
                print(f"  OK: {os.path.basename(csv_path)} -> {len(result)} rows")
            else:
                fail_count += 1
        except Exception as e:
            # never let one bad grid file kill the whole run
            fail_count += 1
            log_error(f"SKIP {os.path.basename(csv_path)}: unexpected error -- {e}")
            continue

    if not all_dfs:
        print("\nNo grids were successfully processed. Nothing written.")
        return

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_parquet(OUTPUT_PARQUET, engine="pyarrow", index=False)

    print("\n============================================")
    print("DONE")
    print("============================================")
    print(f"Grids processed OK : {ok_count}")
    print(f"Grids skipped      : {fail_count}")
    print(f"Total rows written : {len(final_df)}")
    print(f"Output parquet     : {OUTPUT_PARQUET}")
    if fail_count:
        print(f"See error details in: {ERROR_LOG}")


if __name__ == "__main__":
    main()