import pandas as pd

ocean_df = pd.read_csv(
    r"D:\INCOIS\Agro_project\data\processed\indian_ocean_master.csv"
)

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Map boundaries
lon_min, lon_max = 20, 120
lat_min, lat_max = -70,30

g_size=int(input("enter size of grid"))
rows=(lat_max-lat_min)/g_size
cols=(lon_max-lon_min)/g_size
rows=int(rows)
cols=int(cols)

# Create grid lines
lon_lines = np.linspace(lon_min, lon_max, cols +1)
lat_lines = np.linspace(lat_min, lat_max, rows + 1)

# Create figure
fig = plt.figure(figsize=(20,15))

# Create map projection
ax = plt.axes(projection=ccrs.PlateCarree())

# Set map extent
ax.set_extent([lon_min, lon_max, lat_min, lat_max])

# Add ocean and land
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAND)

# Add coastlines
ax.coastlines()

# Draw vertical grid lines
for lon in lon_lines:
    ax.plot(
        [lon, lon],
        [lat_min, lat_max],
        color='black',
        transform=ccrs.PlateCarree()
    )

# Draw horizontal grid lines
for lat in lat_lines:
    ax.plot(
        [lon_min, lon_max],
        [lat, lat],
        color='black',
        transform=ccrs.PlateCarree()
    )

# Add grid IDs
grid_id = 1

for i in reversed(range(rows)):
    for j in range(cols):

        # Grid center
        x_center = (lon_lines[j] + lon_lines[j+1]) / 2
        y_center = (lat_lines[i] + lat_lines[i+1]) / 2

        # Add text label
        ax.text(
            x_center,
            y_center,
            f"{grid_id}",
            transform=ccrs.PlateCarree(),
            ha='center',
            fontsize=6,
            color='black'
        )

        grid_id += 1
ax.scatter(
    ocean_df['longitude'],
    ocean_df['latitude'],
    s=0.5,

    color='purple',
    transform=ccrs.PlateCarree()   
)

plt.savefig(
    r"D:\INCOIS\Agro_project\outputs\maps\indian_ocean_grid.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

