import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, no_update
import dash_leaflet as dl
import json

# ─────────────────────────────────────────────
# LOAD REAL DATA
# ─────────────────────────────────────────────
df = pd.read_csv("D:\\INCOIS\\Agro_project\\data\\processed\\grid_statistics.csv")
df["grid_id"] = df["grid_id"].astype(int)
data_dict = df.set_index("grid_id").to_dict(orient="index")
data_ids  = set(data_dict.keys())

# ─────────────────────────────────────────────
# GRID CONSTANTS
# ─────────────────────────────────────────────
LON_MIN, LON_MAX = 20, 120
LAT_MIN, LAT_MAX = -70, 30
G      = 5
COLS   = (LON_MAX - LON_MIN) // G   # 20
ROWS   = (LAT_MAX - LAT_MIN) // G   # 20

# ─────────────────────────────────────────────
# TEMPERATURE → HEX COLOUR
# ─────────────────────────────────────────────
STOPS = [
    (0.00, (5,   48,  97)),
    (0.20, (33,  102, 172)),
    (0.40, (103, 169, 207)),
    (0.55, (209, 229, 240)),
    (0.70, (253, 219, 199)),
    (0.85, (239, 138, 98)),
    (1.00, (178, 24,  43)),
]

def temp_to_hex(t, t_min=10, t_max=30):
    v = max(0, min(1, (t - t_min) / (t_max - t_min)))
    for i in range(1, len(STOPS)):
        if v <= STOPS[i][0]:
            t0, c0 = STOPS[i-1]
            t1, c1 = STOPS[i]
            f = (v - t0) / (t1 - t0)
            r = int(c0[0] + (c1[0] - c0[0]) * f)
            g = int(c0[1] + (c1[1] - c0[1]) * f)
            b = int(c0[2] + (c1[2] - c0[2]) * f)
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#b2182b"

# ─────────────────────────────────────────────
# BUILD GEOJSON GRID
# ─────────────────────────────────────────────
features = []
gid = 1
for row in range(ROWS):
    for col in range(COLS):
        lat = LAT_MAX - (row + 1) * G   # bottom of cell
        lon = LON_MIN + col * G

        has_data = gid in data_ids
        d        = data_dict.get(gid, {})
        color    = temp_to_hex(d["mean_temp"]) if has_data else "#cccccc"
        opacity  = 0.72 if has_data else 0.0

        features.append({
            "type": "Feature",
            "properties": {
                "grid_id":  gid,
                "lat":      lat,
                "lon":      lon,
                "has_data": has_data,
                "color":    color,
                "opacity":  opacity,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon,   lat],
                    [lon+G, lat],
                    [lon+G, lat+G],
                    [lon,   lat+G],
                    [lon,   lat],
                ]]
            }
        })
        gid += 1

geojson_data = {"type": "FeatureCollection", "features": features}

# ─────────────────────────────────────────────
# STYLE FUNCTION (client-side)
# ─────────────────────────────────────────────
style_func = """
function(feature) {
    return {
        color:       '#1a1a1a',
        weight:      0.7,
        opacity:     0.65,
        fillColor:   feature.properties.color,
        fillOpacity: feature.properties.opacity
    };
}
"""

highlight_func = """
function(feature) {
    return {
        color:       '#ffffff',
        weight:      2.0,
        opacity:     1,
        fillColor:   feature.properties.color,
        fillOpacity: feature.properties.has_data ? 0.92 : 0.15
    };
}
"""

# ─────────────────────────────────────────────
# DASH APP
# ─────────────────────────────────────────────
app = Dash(__name__)

app.layout = html.Div([

    # ── Header ──────────────────────────────
    html.Div([
        html.H1("Indian Ocean Grid Dashboard",
                style={"margin": 0, "fontSize": 16, "fontWeight": 500}),
        html.P("20°E–120°E  |  70°S–30°N  |  5°×5°  |  406 cells  |  26 with data",
               style={"margin": 0, "fontSize": 11, "color": "#888"})
    ], style={
        "padding": "8px 16px 6px",
        "borderBottom": "1px solid #ddd",
        "display": "flex",
        "alignItems": "center",
        "gap": 16
    }),

    # ── Body ─────────────────────────────────
    html.Div([

        # Map
        dl.Map(
            center=[0, 70], zoom=3,
            children=[
                dl.TileLayer(
                    url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
                    attribution="© OpenStreetMap contributors © CARTO",
                    subdomains="abcd", maxZoom=8
                ),
                dl.GeoJSON(
                    id="grid-layer",
                    data=geojson_data,
                    options=dict(style=style_func),
                    hoverStyle=dict(
                        weight=2, color="#ffffff", fillOpacity=0.9
                    ),
                    zoomToBounds=False,
                ),
                # Grid number labels rendered as markers
                dl.LayerGroup(id="labels-layer", children=[
                    dl.Marker(
                        position=[
                            f["properties"]["lat"] + G - 0.45,
                            f["properties"]["lon"] + 0.35
                        ],
                        icon=dict(
                            iconUrl="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7",
                            iconSize=[0, 0]
                        ),
                        children=dl.Tooltip(
                            str(f["properties"]["grid_id"]),
                            permanent=True,
                            direction="center",
                            className="grid-label"
                        )
                    )
                    for f in geojson_data["features"]
                ])
            ],
            style={"flex": 1, "height": "100%"},
            id="map"
        ),

        # Side panel
        html.Div([
            html.H3("Grid Information",
                    style={"fontSize": 10, "fontWeight": 500, "color": "#888",
                           "textTransform": "uppercase", "letterSpacing": "0.07em",
                           "marginBottom": 6}),
            html.Div("Click any grid cell to view oceanographic data",
                     id="panel",
                     style={"color": "#aaa", "fontSize": 12, "textAlign": "center",
                            "padding": "40px 10px", "lineHeight": 1.8}),
        ], style={
            "width": 220,
            "flexShrink": 0,
            "borderLeft": "1px solid #ddd",
            "padding": 12,
            "overflowY": "auto",
            "background": "#fff",
            "display": "flex",
            "flexDirection": "column",
            "gap": 8
        })

    ], style={"display": "flex", "flex": 1, "overflow": "hidden", "height": "calc(100vh - 45px)"})

], style={"fontFamily": "sans-serif", "height": "100vh", "display": "flex", "flexDirection": "column"})


# ─────────────────────────────────────────────
# CALLBACK — click → panel
# ─────────────────────────────────────────────
@app.callback(
    Output("panel", "children"),
    Input("grid-layer", "clickData")
)
def update_panel(click_data):
    if click_data is None:
        return "Click any grid cell to view oceanographic data"

    props   = click_data["properties"]
    grid_id = props["grid_id"]
    lat     = props["lat"]
    lon     = props["lon"]

    def lat_str(v):
        return f"{v}°N" if v >= 0 else f"{abs(v)}°S"

    info_box = html.Div([
        html.Div(f"Grid #{grid_id}", style={"fontSize": 22, "fontWeight": 500, "color": "#185FA5"}),
        html.Div([
            html.Span("Lat: ", style={"fontWeight": 500}),
            f"{lat_str(lat)} – {lat_str(lat + G)}",
            html.Br(),
            html.Span("Lon: ", style={"fontWeight": 500}),
            f"{lon}°E – {lon + G}°E",
            html.Br(),
            html.Span("Center: ", style={"fontWeight": 500}),
            f"{lat + G/2}°, {lon + G/2}°E",
        ], style={"fontSize": 10, "color": "#378ADD", "marginTop": 3, "lineHeight": 1.65})
    ], style={
        "background": "#ddeaf8", "borderRadius": 8,
        "padding": "10px 12px", "border": "0.5px solid #b5d0ee"
    })

    if grid_id not in data_ids:
        badge = html.Span("No oceanographic data",
                          style={"background": "#f5f5f5", "color": "#999",
                                 "fontSize": 9, "padding": "2px 7px",
                                 "borderRadius": 4, "fontWeight": 500})
        return html.Div([info_box, badge])

    d = data_dict[grid_id]

    def metric(label, value, unit=""):
        return html.Div([
            html.Div(label, style={"fontSize": 10, "color": "#999", "marginBottom": 1}),
            html.Div([
                html.Span(f"{value}", style={"fontSize": 15, "fontWeight": 500}),
                html.Span(f" {unit}", style={"fontSize": 10, "color": "#888"})
            ])
        ], style={"background": "#f5f5f5", "borderRadius": 7, "padding": "8px 10px"})

    def metric2(label1, val1, label2, val2):
        return html.Div([
            html.Div([
                html.Div(label1, style={"fontSize": 9, "color": "#aaa"}),
                html.Div(str(val1), style={"fontSize": 13, "fontWeight": 500})
            ], style={"background": "#f5f5f5", "borderRadius": 6, "padding": "7px 8px", "flex": 1}),
            html.Div([
                html.Div(label2, style={"fontSize": 9, "color": "#aaa"}),
                html.Div(str(val2), style={"fontSize": 13, "fontWeight": 500})
            ], style={"background": "#f5f5f5", "borderRadius": 6, "padding": "7px 8px", "flex": 1}),
        ], style={"display": "flex", "gap": 6})

    badge = html.Span(
        f"✓  Has data · {int(d['profile_count']):,} profiles",
        style={"background": "#e8f5e9", "color": "#2e7d32",
               "fontSize": 9, "padding": "2px 7px",
               "borderRadius": 4, "fontWeight": 500}
    )

    legend = html.Div([
        html.Div("Temperature colour scale", style={"fontSize": 9, "color": "#aaa", "marginBottom": 4}),
        html.Div(style={
            "height": 8, "borderRadius": 3,
            "background": "linear-gradient(to right,#053061,#2166ac,#4393c3,#92c5de,#f7f7f7,#f4a582,#d6604d,#b2182b)"
        }),
        html.Div([html.Span("10°C"), html.Span("20°C"), html.Span("30°C")],
                 style={"display": "flex", "justifyContent": "space-between",
                        "fontSize": 9, "color": "#aaa", "marginTop": 2})
    ], style={"marginTop": "auto", "paddingTop": 8, "borderTop": "0.5px solid #eee"})

    return html.Div([
        info_box,
        badge,
        metric("Temperature — mean", f"{d['mean_temp']:.3f}", "°C"),
        metric2("Min temp", f"{d['min_temp']:.3f} °C", "Max temp", f"{d['max_temp']:.3f} °C"),
        metric("Salinity — mean", f"{d['mean_psal']:.3f}", "PSU"),
        metric2("Min psal", f"{d['min_psal']:.3f}", "Max psal", f"{d['max_psal']:.3f}"),
        metric("Avg depth", f"{d['avg_depth']:.1f}", "m"),
        metric("Max depth", f"{d['max_depth']:.1f}", "m"),
        metric2("Avg levels", f"{d['avg_levels']:.1f}", "Max levels", str(int(d['max_levels']))),
        legend,
    ], style={"display": "flex", "flexDirection": "column", "gap": 7})


if __name__ == "__main__":
    app.run(debug=True)