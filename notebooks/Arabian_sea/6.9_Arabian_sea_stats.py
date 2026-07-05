import pandas as pd

# ===================================
# READ METADATA FILE
# ===================================

df = pd.read_csv(
    r"D:\INCOIS\Agro_project\data\processed\arabian_sea_metadata.csv"
)

# ===================================
# GRID STATISTICS
# ===================================

grid_stats = (
    df.groupby("grid_id")
    .agg(
        profile_count=("file_name", "count"),

        mean_temp=("temp_mean", "mean"),
        min_temp=("temp_min", "min"),
        max_temp=("temp_max", "max"),

        mean_psal=("psal_mean", "mean"),
        min_psal=("psal_min", "min"),
        max_psal=("psal_max", "max"),

        avg_depth=("max_depth", "mean"),
        max_depth=("max_depth", "max"),

        avg_levels=("num_levels", "mean"),
        max_levels=("num_levels", "max")
    )
    .reset_index()
)

# Round values
grid_stats = grid_stats.round(3)

# ===================================
# SAVE
# ===================================

output_file = (
    r"D:\INCOIS\Agro_project\data\processed\grid_statistics.csv"
)

grid_stats.to_csv(
    output_file,
    index=False
)

print("\nGrid Statistics Created Successfully\n")

print(grid_stats.head())

print("\nTotal Grids:", len(grid_stats))

print("\nSaved To:")
print(output_file)