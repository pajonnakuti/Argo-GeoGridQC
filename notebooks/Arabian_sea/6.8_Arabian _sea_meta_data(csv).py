import os
import xarray as xr
import numpy as np
import pandas as pd

ROOT_FOLDER = r"D:\INCOIS\Agro_project\data\raw\arabian_sea_nc_files"

records = []

files_processed = 0
files_failed = 0

for dirpath, _, filenames in os.walk(ROOT_FOLDER):

    for file in filenames:

        if not file.endswith(".nc"):
            continue

        file_path = os.path.join(dirpath, file)

        try:
            ds = xr.open_dataset(
                file_path,
                engine="netcdf4",
                decode_timedelta=False
            )

            grid_id = os.path.basename(dirpath).replace("Grid_", "")

            # -------------------------
            # LAT / LON
            # -------------------------
            lat = np.nan
            lon = np.nan

            if "LATITUDE" in ds:
                lat = float(ds["LATITUDE"].values[0])

            if "LONGITUDE" in ds:
                lon = float(ds["LONGITUDE"].values[0])

            # -------------------------
            # TEMPERATURE
            # -------------------------
            temp_mean = temp_min = temp_max = np.nan

            if "TEMP" in ds:
                temp = ds["TEMP"].values.flatten()
                temp = temp[~np.isnan(temp)]

                if len(temp) > 0:
                    temp_mean = np.mean(temp)
                    temp_min = np.min(temp)
                    temp_max = np.max(temp)

            # -------------------------
            # SALINITY
            # -------------------------
            psal_mean = psal_min = psal_max = np.nan

            if "PSAL" in ds:
                psal = ds["PSAL"].values.flatten()
                psal = psal[~np.isnan(psal)]

                if len(psal) > 0:
                    psal_mean = np.mean(psal)
                    psal_min = np.min(psal)
                    psal_max = np.max(psal)

            # -------------------------
            # PRESSURE (NO CONVERSION)
            # -------------------------
            max_depth = np.nan
            num_levels = 0

            if "PRES" in ds:
                pres = ds["PRES"].values.flatten()
                pres = pres[~np.isnan(pres)]

                if len(pres) > 0:
                    max_depth = np.max(pres)
                    num_levels = len(pres)

            # -------------------------
            # DATE
            # -------------------------
            profile_date = None

            if "JULD" in ds:
                try:
                    profile_date = str(ds["JULD"].values[0])
                except:
                    pass

            # -------------------------
            # STORE RECORD
            # -------------------------
            records.append({

                "file_name": file,
                "grid_id": grid_id,
                "latitude": lat,
                "longitude": lon,
                "date": profile_date,

                "temp_mean": temp_mean,
                "temp_min": temp_min,
                "temp_max": temp_max,

                "psal_mean": psal_mean,
                "psal_min": psal_min,
                "psal_max": psal_max,

                "max_depth": max_depth,
                "num_levels": num_levels

            })

            files_processed += 1

            if files_processed % 1000 == 0:
                print(f"Processed {files_processed} files")

            ds.close()

        except Exception as e:
            files_failed += 1
            print(f"Failed: {file}")

# -------------------------
# SAVE CSV
# -------------------------
df = pd.DataFrame(records)

output_file = r"D:\INCOIS\Agro_project\data\processed\arabian_sea_metadata.csv"

df.to_csv(output_file, index=False)

print("\n========== SUMMARY ==========")
print(f"Processed : {files_processed}")
print(f"Failed    : {files_failed}")
print(f"Rows Saved: {len(df)}")
print(f"Output    : {output_file}")
print("============================")