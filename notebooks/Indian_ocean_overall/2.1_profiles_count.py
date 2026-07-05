import pandas as pd
import numpy as np

# Read master file
df = pd.read_csv(
    r"D:\INCOIS\Agro_project\data\processed\indian_ocean_master.csv"
)

# Grid settings
grid_size = 5

lon_min, lon_max = 20, 120
lat_min, lat_max = -70, 30

rows = int((lat_max - lat_min) / grid_size)
cols = int((lon_max - lon_min) / grid_size)

lon_lines = np.linspace(lon_min, lon_max, cols + 1)
lat_lines = np.linspace(lat_min, lat_max, rows + 1)

grid_ids = []

for lon, lat in zip(df["longitude"], df["latitude"]):

    col = np.searchsorted(
        lon_lines,
        lon,
        side="right"
    ) - 1

    row = np.searchsorted(
        lat_lines,
        lat,
        side="right"
    ) - 1

    if 0 <= col < cols and 0 <= row < rows:

        grid_id = (rows - row - 1) * cols + col + 1

    else:

        grid_id = -1

    grid_ids.append(grid_id)

# New column
df["Grid_ID"] = grid_ids

print(df[["file", "Grid_ID"]].head())

# Save
df.to_csv(
    r"D:\INCOIS\Agro_project\data\processed\profiles_with_grid.csv",
    index=False
)

print("profiles_with_grid.csv created successfully")

# Count profiles per grid
grid_counts = (
    df[df["Grid_ID"] != -1]
    .groupby("Grid_ID")
    .size()
    .reset_index(name="Profile_Count")
)

print("\nProfiles per Grid:")
print(grid_counts)

grid_counts.to_csv(
    r"D:\INCOIS\Agro_project\data\processed\grid_profile_counts.csv",
    index=False
)

print("grid_profile_counts.csv created successfully")
