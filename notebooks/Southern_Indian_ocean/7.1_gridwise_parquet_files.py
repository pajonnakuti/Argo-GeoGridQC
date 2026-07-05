import os
import json
import numpy as np
import pandas as pd
import xarray as xr
import gsw
from multiprocessing import Pool, cpu_count

# ====================================
# FILE PATHS
# ====================================

ROOT_FOLDER = r"D:\INCOIS\Agro_project\data\raw\southern_indian_ocean_nc_files"

PROCESSED_DIR = r"D:\INCOIS\Agro_project\data\processed"
OUTPUT_FILE   = os.path.join(PROCESSED_DIR, "southern_indian_ocean_profile_depth_level.parquet")

# per-grid intermediate parquet files (enables resume + avoids huge in-memory concat)
GRID_PARQUET_DIR = os.path.join(PROCESSED_DIR, "grid_parts")

# checkpoint file tracking which .nc files have already been turned into rows
CHECKPOINT_FILE = os.path.join(PROCESSED_DIR, "processed_files_checkpoint.json")

N_WORKERS = max(1, cpu_count() - 1)

# ====================================
# SOUTHERN INDIAN OCEAN GRIDS
# 
# ====================================

southern_indian_ocean_grids = [
    151,
    165, 166, 167, 168, 169, 170, 171,
    184, 185, 187, 188, 189, 190, 191,
    204, 205, 207, 208, 209, 210, 211,
    223, 224, 225, 226, 227, 228, 229, 230, 231,
    243, 244, 245, 246, 247, 248, 249, 250, 251,
    261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271,
    281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291,
    301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311,
    321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331,
    341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351,
    361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371,
    381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391,
]


# ====================================
# CHECKPOINT HELPERS
# ====================================

def load_checkpoint():
    """
    Load checkpoint safely.
    If the checkpoint file is missing or corrupted,
    start with an empty checkpoint instead of crashing.
    """
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return set(data)
        print("WARNING: Checkpoint format is invalid.")
        return set()
    except json.JSONDecodeError:
        print("=" * 60)
        print("WARNING: Checkpoint file is corrupted.")
        print("A new checkpoint will be created automatically.")
        print("=" * 60)
        # Rename the bad checkpoint instead of deleting it
        bad_file = CHECKPOINT_FILE + ".corrupted"
        try:
            os.replace(CHECKPOINT_FILE, bad_file)
            print(f"Corrupted checkpoint renamed to:\n{bad_file}")
        except Exception:
            pass
        return set()
    except Exception as e:
        print(f"Error reading checkpoint: {e}")
        return set()


def save_checkpoint(done_set):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(sorted(done_set), f)


# ====================================
# SINGLE-FILE WORKER
# ====================================

def process_one_file(args):
    """
    Processes a single .nc file and returns a (key, DataFrame|None, error|None) tuple.
    key = "grid_folder_name/file_name" used for checkpointing.
    """
    grid_folder_name, file_name, file_path = args
    key = f"{grid_folder_name}/{file_name}"

    try:
        # decode_times=False + drop_variables left at default keeps it simple;
        # mask_and_scale=True (default) is needed for fill-value handling via NaN already done manually below
        ds = xr.open_dataset(file_path, engine="netcdf4")

        lat, lon, date = np.nan, np.nan, pd.NaT
        try:
            lat = float(ds["LATITUDE"].values[0])
        except Exception:
            pass
        try:
            lon = float(ds["LONGITUDE"].values[0])
        except Exception:
            pass
        try:
            date = pd.to_datetime(ds["JULD"].values[0])
        except Exception:
            pass

        if "PRES_ADJUSTED" in ds.variables:
            pres = ds["PRES_ADJUSTED"].values[0, :]
        elif "PRES" in ds.variables:
            pres = ds["PRES"].values[0, :]
        else:
            ds.close()
            return key, None, None

        if "TEMP_ADJUSTED" in ds.variables:
            temp = ds["TEMP_ADJUSTED"].values[0, :]
        elif "TEMP" in ds.variables:
            temp = ds["TEMP"].values[0, :]
        else:
            ds.close()
            return key, None, None

        if "PSAL_ADJUSTED" in ds.variables:
            psal = ds["PSAL_ADJUSTED"].values[0, :]
        elif "PSAL" in ds.variables:
            psal = ds["PSAL"].values[0, :]
        else:
            psal = np.full_like(pres, np.nan, dtype=float)

        temp_qc = None
        psal_qc = None
        try:
            if "TEMP_ADJUSTED_QC" in ds.variables:
                temp_qc = ds["TEMP_ADJUSTED_QC"].values[0, :]
            elif "TEMP_QC" in ds.variables:
                temp_qc = ds["TEMP_QC"].values[0, :]
        except Exception:
            pass
        try:
            if "PSAL_ADJUSTED_QC" in ds.variables:
                psal_qc = ds["PSAL_ADJUSTED_QC"].values[0, :]
            elif "PSAL_QC" in ds.variables:
                psal_qc = ds["PSAL_QC"].values[0, :]
        except Exception:
            pass

        ds.close()

        pres = np.asarray(pres, dtype=float)
        temp = np.asarray(temp, dtype=float)
        psal = np.asarray(psal, dtype=float)

        pres[pres > 9990] = np.nan
        temp[temp > 9990] = np.nan
        psal[psal > 9990] = np.nan

        mask = np.isfinite(pres) & np.isfinite(temp)
        pres = pres[mask]
        temp = temp[mask]
        psal = psal[mask]

        if temp_qc is not None:
            temp_qc = np.asarray(temp_qc)[mask]
        if psal_qc is not None:
            psal_qc = np.asarray(psal_qc)[mask]

        n = len(pres)
        if n == 0:
            return key, None, None

        try:
            depth = -gsw.z_from_p(pres, lat)
        except Exception:
            depth = pres.copy()

        temp_min, temp_max, temp_mean = np.nanmin(temp), np.nanmax(temp), np.nanmean(temp)
        pres_min, pres_max, pres_mean = np.nanmin(pres), np.nanmax(pres), np.nanmean(pres)
        depth_min, depth_max, depth_mean = np.nanmin(depth), np.nanmax(depth), np.nanmean(depth)

        if np.isfinite(psal).any():
            psal_min, psal_max, psal_mean = np.nanmin(psal), np.nanmax(psal), np.nanmean(psal)
        else:
            psal_min = psal_max = psal_mean = np.nan

        df = pd.DataFrame({
            "grid_id":   grid_folder_name,
            "file_name": file_name,
            "date":      date,
            "latitude":  lat,
            "longitude": lon,
            "temp_min":   temp_min,
            "temp_max":   temp_max,
            "temp_mean":  temp_mean,
            "psal_min":   psal_min,
            "psal_max":   psal_max,
            "psal_mean":  psal_mean,
            "pres_min":   pres_min,
            "pres_max":   pres_max,
            "pres_mean":  pres_mean,
            "depth_min":  depth_min,
            "depth_max":  depth_max,
            "depth_mean": depth_mean,
            "num_levels": n,
            "pressure":    pres,
            "depth":       depth,
            "temperature": temp,
            "salinity":    psal,
        })

        if temp_qc is not None:
            df["temp_qc"] = temp_qc.astype(str)
        if psal_qc is not None:
            df["psal_qc"] = psal_qc.astype(str)

        return key, df, None

    except Exception as e:
        return key, None, str(e)


# ====================================
# MAIN
# ====================================

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(GRID_PARQUET_DIR, exist_ok=True)

    done_keys = load_checkpoint()
    print(f"Resuming: {len(done_keys)} files already processed previously (will be skipped).")

    files_processed = 0
    files_failed = 0
    files_skipped = 0

    for grid_id in sorted(southern_indian_ocean_grids):
        grid_folder_name = f"Grid_{grid_id}"
        grid_path = os.path.join(ROOT_FOLDER, grid_folder_name)

        if not os.path.isdir(grid_path):
            continue

        grid_out_file = os.path.join(GRID_PARQUET_DIR, f"{grid_folder_name}.parquet")

        # If this grid's parquet already exists and every .nc in the folder is
        # already in the checkpoint, skip the whole grid instantly.
        nc_files = [f for f in os.listdir(grid_path) if f.endswith(".nc")]
        tasks = []
        for file_name in nc_files:
            key = f"{grid_folder_name}/{file_name}"
            if key in done_keys:
                files_skipped += 1
                continue
            tasks.append((grid_folder_name, file_name, os.path.join(grid_path, file_name)))

        if not tasks:
            print(f"{grid_folder_name}: nothing new, skipping.")
            continue

        print(f"\nProcessing {grid_folder_name}: {len(tasks)} new files "
              f"({len(nc_files) - len(tasks)} already done)")

        grid_dfs = []

        with Pool(processes=N_WORKERS) as pool:
            for key, df, err in pool.imap_unordered(process_one_file, tasks, chunksize=8):
                if err is not None:
                    files_failed += 1
                    print(f"  ERROR: {key} -> {err}")
                    continue

                done_keys.add(key)

                if df is None:
                    # no usable data, still mark as done so we don't retry it
                    continue

                grid_dfs.append(df)
                files_processed += 1

                if files_processed % 200 == 0:
                    print(f"  Processed: {files_processed} files so far...")

        # merge new rows with any existing parquet for this grid (resume-safe)
        if grid_dfs:
            new_grid_df = pd.concat(grid_dfs, ignore_index=True)
            if os.path.exists(grid_out_file):
                old_grid_df = pd.read_parquet(grid_out_file, engine="pyarrow")
                new_grid_df = pd.concat([old_grid_df, new_grid_df], ignore_index=True)

            # Guard against duplicate rows if a checkpoint reset ever causes
            # the same (grid_id, file_name) to be reprocessed and re-merged.
            new_grid_df = new_grid_df.drop_duplicates(
                subset=["grid_id", "file_name", "pressure"], keep="last"
            )

            new_grid_df = new_grid_df.sort_values(["date", "file_name", "pressure"]).reset_index(drop=True)
            new_grid_df.to_parquet(grid_out_file, engine="pyarrow", index=False)

        # save checkpoint after every grid
        save_checkpoint(done_keys)

    print("\n============================================")
    print("DONE - Grid-wise parquet files created.")
    print("============================================")
    print(f"Files processed : {files_processed}")
    print(f"Files failed    : {files_failed}")
    print(f"Files skipped   : {files_skipped}")
    print(f"\nGrid-wise parquet files are saved in:")
    print(GRID_PARQUET_DIR)


if __name__ == "__main__":
    main()