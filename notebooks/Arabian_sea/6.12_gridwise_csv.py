import os
import pandas as pd

# ==========================================================
# INPUT PARQUET FILE
# ==========================================================

INPUT_FILE = (
    r"D:\INCOIS\Agro_project\data\processed"
    r"\argo_zscore_results.parquet"
)

# ==========================================================
# OUTPUT FOLDER
# ==========================================================

OUTPUT_FOLDER = (
    r"D:\INCOIS\Agro_project\data"
    r"\Arabian_sea_gridwise_csv"
)

# ==========================================================
# CREATE OUTPUT DIRECTORY
# ==========================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("=" * 60)
print("ARABIAN SEA GRID-WISE CSV CREATION")
print("=" * 60)

# ==========================================================
# LOAD DATA
# ==========================================================

print("\nLoading parquet file...")
print(INPUT_FILE)

df = pd.read_parquet(INPUT_FILE)

print(f"\nRows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

# ==========================================================
# REMOVE UNWANTED COLUMNS
# ==========================================================

remove_columns = [
    "z_final_flag",
    "z_flag_label"
]

existing_columns = [
    col
    for col in remove_columns
    if col in df.columns
]

if existing_columns:
    df = df.drop(columns=existing_columns)

print("\nRemoved columns:")
for col in existing_columns:
    print(f"  {col}")

# ==========================================================
# CHECK GRID COLUMN
# ==========================================================

if "grid_id" not in df.columns:
    raise ValueError(
        "grid_id column not found in dataframe."
    )

# ==========================================================
# SAVE ONE CSV PER GRID
# ==========================================================

unique_grids = sorted(df["grid_id"].dropna().unique())

total_grids = len(unique_grids)

print(f"\nTotal grids found : {total_grids:,}")
print("\nSaving CSV files...\n")

for i, grid_id in enumerate(unique_grids, start=1):

    grid_df = df[df["grid_id"] == grid_id]

    output_file = os.path.join(
        OUTPUT_FOLDER,
        f"{grid_id}.csv"
    )

    grid_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"[{i}/{total_grids}] "
        f"Saved {grid_id}.csv "
        f"({len(grid_df):,} rows)"
    )

# ==========================================================
# SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"\nOutput Folder:")
print(OUTPUT_FOLDER)

print(f"\nTotal CSV Files Created : {total_grids:,}")

print("\nDone.")