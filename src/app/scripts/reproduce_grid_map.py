"""
Reproduce the 20×20 Indian Ocean grid map (reference style) and enhanced QC-status version.

Run from app/:
    python scripts/reproduce_grid_map.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from backend.config import (
    COLS, DATA_PATH, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, MAPS_DIR,
    N_GRIDS, OUTPUT_DIR, PIPELINE_GRID_SUMMARY, ROWS, SIZE,
)
from backend.grid import all_grid_cells, grid_id_bbox

# Reference visual style (indian_ocean_grid.png)
LAND_COLOR = "#F2F1E1"
OCEAN_COLOR = "#A9C9E9"
POINT_COLOR = "#800080"
GRID_LINE = "#000000"
STATUS_COLORS = {
    "OK": "#4fd69c",
    "NO_DATA": "#8592a0",
    "PARTIAL": "#e8b84b",
    "FAILED": "#e5654e",
}


def load_obs_sample(max_points: int = 120_000) -> pd.DataFrame:
    import duckdb

    p = str(DATA_PATH).replace("'", "''")
    con = duckdb.connect()
    return con.execute(f"""
        SELECT lat, lon, grid_id
        FROM read_parquet('{p}')
        WHERE lat IS NOT NULL AND lon IS NOT NULL AND grid_id IS NOT NULL
        USING SAMPLE {max_points} ROWS
    """).fetchdf()


def load_grid_status() -> dict[int, str]:
    path = PIPELINE_GRID_SUMMARY
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {int(r.grid_id): str(r.grid_status) for _, r in df.iterrows()}


def draw_base_map(ax, obs: pd.DataFrame, title: str):
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=OCEAN_COLOR)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=LAND_COLOR, edgecolor="black", linewidth=0.3)
        ax.coastlines(resolution="50m", color="black", linewidth=0.4)
        tr = ccrs.PlateCarree()
        ax.scatter(obs["lon"], obs["lat"], c=POINT_COLOR, s=0.15, alpha=0.35, transform=tr, rasterized=True)
    except ImportError:
        ax.set_facecolor(OCEAN_COLOR)
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.scatter(obs["lon"], obs["lat"], c=POINT_COLOR, s=0.15, alpha=0.35, rasterized=True)

    for lon in np.arange(LON_MIN, LON_MAX + 0.1, SIZE):
        ax.plot([lon, lon], [LAT_MIN, LAT_MAX], color=GRID_LINE, lw=0.5, alpha=0.8)
    for lat in np.arange(LAT_MIN, LAT_MAX + 0.1, SIZE):
        ax.plot([LON_MIN, LON_MAX], [lat, lat], color=GRID_LINE, lw=0.5, alpha=0.8)

    for cell in all_grid_cells():
        bb = cell
        cx = (bb["lon_min"] + bb["lon_max"]) / 2
        cy = (bb["lat_min"] + bb["lat_max"]) / 2
        ax.text(cx, cy, str(bb["grid_id"]), fontsize=5, ha="center", va="center", color="black")

    ax.set_title(title, fontsize=12)


def draw_enhanced_map(ax, obs: pd.DataFrame, grid_status: dict[int, str], title: str):
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import cartopy

        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=OCEAN_COLOR)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=LAND_COLOR, edgecolor="black", linewidth=0.3)
        ax.coastlines(resolution="50m", color="black", linewidth=0.4)
        tr = ccrs.PlateCarree()
        ax.scatter(obs["lon"], obs["lat"], c=POINT_COLOR, s=0.12, alpha=0.25, transform=tr, rasterized=True)
    except ImportError:
        ax.set_facecolor(OCEAN_COLOR)
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.scatter(obs["lon"], obs["lat"], c=POINT_COLOR, s=0.12, alpha=0.25, rasterized=True)

    for gid in range(1, N_GRIDS + 1):
        bb = grid_id_bbox(gid)
        status = grid_status.get(gid, "NO_DATA")
        color = STATUS_COLORS.get(status, STATUS_COLORS["NO_DATA"])
        lw = 2.5 if status in ("FAILED", "PARTIAL") else 0.6
        rect_lons = [bb["lon_min"], bb["lon_max"], bb["lon_max"], bb["lon_min"], bb["lon_min"]]
        rect_lats = [bb["lat_min"], bb["lat_min"], bb["lat_max"], bb["lat_max"], bb["lat_min"]]
        ax.plot(rect_lons, rect_lats, color=color if status != "OK" else GRID_LINE, lw=lw, alpha=0.9)
        if status in ("FAILED", "PARTIAL"):
            cx = (bb["lon_min"] + bb["lon_max"]) / 2
            cy = (bb["lat_min"] + bb["lat_max"]) / 2
            ax.text(cx, cy, str(gid), fontsize=5, ha="center", va="center",
                    color="white", bbox=dict(boxstyle="round,pad=0.1", fc=color, alpha=0.85))

    ax.set_title(title, fontsize=12)


def make_interactive_html(obs: pd.DataFrame, grid_status: dict[int, str], out: Path):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  plotly not installed — skipping interactive map")
        return

    cells = all_grid_cells()
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=obs["lon"], lat=obs["lat"], mode="markers",
        marker=dict(size=1, color=POINT_COLOR, opacity=0.3),
        name="Observations", hoverinfo="skip",
    ))
    for cell in cells:
        gid = cell["grid_id"]
        status = grid_status.get(gid, "NO_DATA")
        fig.add_trace(go.Scattergeo(
            lon=[cell["lon_min"], cell["lon_max"], cell["lon_max"], cell["lon_min"], cell["lon_min"]],
            lat=[cell["lat_min"], cell["lat_min"], cell["lat_max"], cell["lat_max"], cell["lat_min"]],
            mode="lines",
            line=dict(width=2 if status in ("FAILED", "PARTIAL") else 0.5,
                      color=STATUS_COLORS.get(status, "#333")),
            fill="toself", fillcolor=STATUS_COLORS.get(status, "rgba(0,0,0,0)") if status != "OK" else "rgba(0,0,0,0)",
            opacity=0.5,
            name=f"GRID-{gid} ({status})",
            hovertemplate=f"GRID-{gid}<br>Status: {status}<extra></extra>",
        ))
    fig.update_geos(
        projection_type="natural earth",
        lonaxis_range=[LON_MIN, LON_MAX], lataxis_range=[LAT_MIN, LAT_MAX],
        showland=True, landcolor=LAND_COLOR, oceancolor=OCEAN_COLOR,
        showcountries=True, coastlinecolor="black",
    )
    fig.update_layout(title="Indian Ocean Grid — QC Status (interactive)", height=700)
    fig.write_html(str(out))


def main():
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading observation sample for map…")
    obs = load_obs_sample()
    grid_status = load_grid_status()

    # Static reference-style map
    try:
        import cartopy.crs as ccrs
        fig = plt.figure(figsize=(14, 11))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    except ImportError:
        fig, ax = plt.subplots(figsize=(14, 11))

    draw_base_map(ax, obs, f"Indian Ocean 20×20 Grid ({N_GRIDS} cells) — reproduced from reference")
    out_static = MAPS_DIR / "indian_ocean_grid.png"
    fig.savefig(out_static, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out_static}")

    # Copy to app static basemap
    import shutil
    shutil.copy(out_static, APP_ROOT / "static" / "basemap.png")

    # Enhanced map with QC status
    try:
        import cartopy.crs as ccrs
        fig2 = plt.figure(figsize=(14, 11))
        ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    except ImportError:
        fig2, ax2 = plt.subplots(figsize=(14, 11))

    draw_enhanced_map(ax2, obs, grid_status,
                      "Enhanced grid map — problem cells highlighted (FAILED/PARTIAL/NO_DATA)")
    # Legend
    for status, col in STATUS_COLORS.items():
        ax2.plot([], [], color=col, lw=3, label=status)
    ax2.legend(loc="lower left", fontsize=8)
    out_enh = MAPS_DIR / "indian_ocean_grid_enhanced.png"
    fig2.savefig(out_enh, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"  Saved {out_enh}")

    make_interactive_html(obs, grid_status, MAPS_DIR / "indian_ocean_grid_interactive.html")
    print("Done.")


if __name__ == "__main__":
    main()
