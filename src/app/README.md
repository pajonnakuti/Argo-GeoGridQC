# ARGO SENTINEL — Ocean QC Intelligence

Full-stack dashboard backed by **ALL_REGIONS_UNIFIED.parquet** and chunked RandomForest QC models.

| Station | Purpose |
|---------|---------|
| **MHOG Observatory** | Mesoscale Hydrographic Observing Grid — interactive Indian Ocean map |
| **QC Prediction Centre** | Predict `temp_qc` / `psal_qc`; highlights grid on map |

## Folder layout

```
app/
├── README.md
├── requirements.txt
├── run.py                         ← start the web app (auto port 8050–8059)
├── backend/
│   ├── main.py                    ← FastAPI API + serves frontend
│   ├── config.py                  ← paths & grid constants
│   ├── database.py                ← DuckDB queries + grid analysis
│   ├── cache.py                   ← disk cache for fast startup
│   ├── port_util.py               ← port conflict handling
│   ├── grid.py                    ← 5° grid geometry (20×20, −70°–30°N)
│   └── ml/
│       ├── preprocess.py
│       └── predictor.py
├── static/
│   ├── index.html
│   ├── css/styles.css
│   ├── js/app.js
│   └── basemap.png                ← generated on first run
└── scripts/
    ├── train_single_parquet_rf_chunked.py
    ├── build_model_bundles.py
    └── generate_basemap.py
```

## Prerequisites

- Python 3.10+
- `data/processed/Final parquet files/ALL_REGIONS_UNIFIED.parquet`
- Optional training data: `Indian_ocean/train_clean/`, `Indian_ocean/test_clean/`

## 1. Install

```powershell
cd D:\INCOIS\Agro_project\app
pip install -r requirements.txt
```

Optional (better coastlines on basemap):

```powershell
pip install cartopy
```

## 2. Train models (optional)

```powershell
python scripts/train_single_parquet_rf_chunked.py
python scripts/build_model_bundles.py
```

## 3. Run

```powershell
python run.py
```

Open the URL printed in the terminal → **Dive In ↓**

- Default port **8050**; if busy, next free port is used automatically.
- Startup is fast: grid stats load from `data/processed/.argo_cache/`.
- Full 237M-row refresh runs once in the background.

## Features

### MHOG Observatory (Mesoscale Hydrographic Observing Grid)

Indian Ocean basemap (20°–120°E, −70°–30°N, 5° cells, 20×20 = 400 grids).

**Observations mode**
- Click a grid → zoom in, mask other cells
- Argo / CTD / XBT counts with badges and pie chart
- Interactive profile points (toggle instruments)
- Gaussian-smoothed profile density heatmap

**Analysis mode**
- Choropleth by observation count, mean T/S, mean depth
- Thermohaline stats: mean / min / max, climatology ±σ
- Raw `temp_qc` and `psal_qc` distributions
- Model-predicted QC vs raw flags (sample agreement %)
- Depth bins, seasonal coverage, z-score summary
- Grid ID search

### QC Prediction Centre
- Manual entry or CSV/Parquet batch upload
- Chunked RandomForest (`temp_qc`, `psal_qc`) with confidence scores
- Resolved grid **highlighted on MHOG map** after prediction
- Falls back to z-score rules if RF bundles are missing

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/meta` | Dataset + grid + model stats |
| GET | `/api/models` | RF bundle status |
| GET | `/api/grids` | All grid cells for map |
| GET | `/api/grids/{id}` | Grid detail (instruments, QC) |
| GET | `/api/grids/{id}/analysis` | Full research analysis panel |
| GET | `/api/grids/{id}/profiles` | Sample profile points |
| GET | `/api/grids/{id}/heatmap` | Gaussian density heatmap |
| POST | `/api/qc/predict` | Single-profile QC prediction |
| POST | `/api/qc/batch` | Batch CSV/Parquet QC |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Port 8050 in use | `run.py` picks next port; or `netstat -ano \| findstr :8050` then `taskkill /PID <pid> /F` |
| Map empty on first load | Wait for background cache warmup (~1–2 min first time) |
| QC shows z-score fallback | Run `python scripts/build_model_bundles.py` |
| Slow restart | Cache at `data/processed/.argo_cache/grid_stats.parquet` |

## Grid numbering

Grid ID 1 = northwest corner (−70° to −65°N, 20°–25°E), increases east then south — matches `scripts/3.0_map_with profiles.py`.

## 5. QC full pipeline (maps, tables, 9 plots)

Runs diagnosis, prediction, and all deliverables from the project brief:

```powershell
cd D:\INCOIS\Agro_project\app
python scripts/qc_full_pipeline.py          # full run (~500k train sample, 200 obs/grid)
python scripts/qc_full_pipeline.py --quick  # fast test (50k train, 50 obs/grid)
```

**Outputs** (`outputs/qc_pipeline/`):

| File | Description |
|------|-------------|
| `grid_summary.csv` | 400 rows — per-grid status (OK/NO_DATA/PARTIAL/FAILED) |
| `master_results.parquet` | Observation sample with raw + predicted QC |
| `master_results_sample.csv` | First 100k rows for Excel |
| `plots/01_…09_*.png` | All 9 required figures |
| `pipeline_report.md` | Root-cause notes + metrics |
| `pipeline.log` | Per-grid errors (no silent drops) |

**Maps** (`outputs/maps/`):

| File | Description |
|------|-------------|
| `indian_ocean_grid_reference.png` | Your reference image (archived) |
| `indian_ocean_grid.png` | Reproduced static map (cream land, purple points) |
| `indian_ocean_grid_enhanced.png` | Problem grids highlighted |
| `indian_ocean_grid_interactive.html` | Plotly click map |

Regenerate map only:

```powershell
python scripts/reproduce_grid_map.py
```

The **MHOG Observatory** in the web app reads `grid_summary.csv` and highlights FAILED/PARTIAL grids in red.
