import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# =========================
# READ DATA
# =========================

df = pd.read_csv(
    r"D:\INCOIS\Agro_project\data\processed\profiles_with_grid.csv"
)

##indian ocean boundaries

lon_min, lon_max = 20, 120
lat_min, lat_max = -70, 30

grid_size = 5

rows = int((lat_max - lat_min) / grid_size)
cols = int((lon_max - lon_min) / grid_size)

# Grid boundaries
lon_lines = np.linspace(
    lon_min,
    lon_max,
    cols + 1
)

lat_lines = np.linspace(
    lat_min,
    lat_max,
    rows + 1
)

##density calculation   

density = np.zeros(
    (rows, cols),
    dtype=int
)

for lon, lat in zip(
    df["longitude"],
    df["latitude"]
):

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
        density[row, col] += 1

# =========================
# SMOOTH DENSITY
# =========================

density_smooth = gaussian_filter(
    density,
    sigma=1
)

# Grid centres
lon_centers = (
    lon_lines[:-1] +
    lon_lines[1:]
) / 2

lat_centers = (
    lat_lines[:-1] +
    lat_lines[1:]
) / 2

Lon, Lat = np.meshgrid(
    lon_centers,
    lat_centers
)

# =========================
# CREATE MAP
# =========================

fig = plt.figure(figsize=(15, 10))

ax = plt.axes(
    projection=ccrs.PlateCarree()
)

ax.set_extent(
    [lon_min, lon_max, lat_min, lat_max]
)

ax.add_feature(
    cfeature.LAND,
    color="lightgray"
)

ax.coastlines()

mesh = ax.contourf(
    Lon,
    Lat,
    density_smooth,
    levels=30,
    cmap="turbo",
    transform=ccrs.PlateCarree()
)

# Grid lines
for lon in lon_lines:

    ax.plot(
        [lon, lon],
        [lat_min, lat_max],
        color="black",
        linewidth=0.4,
        transform=ccrs.PlateCarree()
    )

for lat in lat_lines:

    ax.plot(
        [lon_min, lon_max],
        [lat, lat],
        color="black",
        linewidth=0.4,
        transform=ccrs.PlateCarree()
    )

# Colorbar
cbar = plt.colorbar(
    mesh,
    ax=ax,
    pad=0.02
)

cbar.set_label(
    "Number of Profiles"
)

plt.title(
    "Indian Ocean Profile Density Map"
)

# =========================
# SAVE TO OUTPUTS
# =========================

output_folder = (
    r"D:\INCOIS\Agro_project\outputs\visualizations"
)

os.makedirs(
    output_folder,
    exist_ok=True
)

save_path = os.path.join(
    output_folder,
    f"density_map_{grid_size}deg.png"
)

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Map saved to:\n{save_path}")

# =========================
# STATISTICS
# =========================

print(
    "Minimum profiles in grid:",
    int(density.min())
)

print(
    "Maximum profiles in grid:",
    int(density.max())
)

plt.show()