"""Grid geometry — matches grid_id numbering in 3.0_map_with profiles.py."""
from __future__ import annotations

import numpy as np

from .config import COLS, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, ROWS, SIZE


def grid_id_bbox(grid_id: int) -> dict:
    gid = int(grid_id)
    row_from_top = (gid - 1) // COLS
    col = (gid - 1) % COLS
    lat_max = LAT_MAX - row_from_top * SIZE
    lat_min = lat_max - SIZE
    lon_min = LON_MIN + col * SIZE
    lon_max = lon_min + SIZE
    return {
        "grid_id": gid,
        "row": row_from_top + 1,
        "col": col + 1,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_center": (lat_min + lat_max) / 2,
        "lon_center": (lon_min + lon_max) / 2,
    }


def latlon_to_grid_id(lat: float, lon: float) -> int | None:
    if not (LAT_MIN <= lat <= LAT_MAX) or not (LON_MIN <= lon <= LON_MAX):
        return None
    col = int(np.floor((lon - LON_MIN) / SIZE))
    row_from_top = int(np.floor((LAT_MAX - lat) / SIZE))
    col = min(max(col, 0), COLS - 1)
    row_from_top = min(max(row_from_top, 0), ROWS - 1)
    return row_from_top * COLS + col + 1


def all_grid_cells() -> list[dict]:
    return [grid_id_bbox(g) for g in range(1, ROWS * COLS + 1)]
