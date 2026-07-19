"""Generate static basemap PNG for the study domain."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

LON_MIN, LON_MAX = 20, 120
LAT_MIN, LAT_MAX = -70, 30
OUT = Path(__file__).resolve().parents[1] / "static" / "basemap.png"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 7.2), facecolor="#061018")
    if HAS_CARTOPY:
        ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.set_facecolor("#081820")
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#081820")
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#1a2830", edgecolor="#2a3a44")
        ax.coastlines(resolution="50m", color="#3a4f5c", linewidth=0.6)
        ax.gridlines(draw_labels=False, linewidth=0.35, color="#2a4050", alpha=0.5, linestyle="--")
    else:
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor("#081820")
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)

    tr = ccrs.PlateCarree() if HAS_CARTOPY else None
    for lon in np.arange(LON_MIN, LON_MAX + 0.1, 5):
        ax.plot([lon, lon], [LAT_MIN, LAT_MAX], color="#00f5d4", lw=0.25, alpha=0.35, transform=tr)
    for lat in np.arange(LAT_MIN, LAT_MAX + 0.1, 5):
        ax.plot([LON_MIN, LON_MAX], [lat, lat], color="#00f5d4", lw=0.25, alpha=0.35, transform=tr)

    ax.axis("off")
    fig.savefig(OUT, dpi=150, facecolor="#061018", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved {OUT} (cartopy={HAS_CARTOPY})")


if __name__ == "__main__":
    main()
