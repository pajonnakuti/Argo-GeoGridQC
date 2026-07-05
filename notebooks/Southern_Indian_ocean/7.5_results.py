"""
Ocean QC Model Results — Interactive Dash Dashboard
Run:  python dashboard.py
Then open:  http://127.0.0.1:8050

Reads the master results CSV produced by the training pipeline
(all_grids_10model_results.csv) and gives you filterable, tabbed
views over grid / target / model performance.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import dcc, html, dash_table, Input, Output
import dash.dash_table.Format as dtf

# ── Load data ─────────────────────────────────────────────────────────────────
# Points at the "v2" results root from the leak-safe training run and the
# 10-model master CSV it produces in All_grids_models/.
CSV_PATH = (
    r"D:\INCOIS\Agro_project\results_southern_indian_ocean_v2"
    r"\All_grids_models\all_grids_10model_results.csv"
)
df = pd.read_csv(CSV_PATH)

MODELS  = df["model"].unique().tolist()
GRIDS   = sorted(df["grid_id"].unique().tolist())
TARGETS = df["target"].unique().tolist()

# 10-model palette (FAST_MODE model set: Hist/LightGBM/XGB/CatBoost/RF/ET/
# AdaBoost/MLP/LogReg/GaussianNB). Colors are assigned by name if present,
# any unrecognized model name still gets a color via the fallback cycle.
_BASE_COLORS = {
    "HistGradientBoosting": "#3b82f6",
    "LightGBM":             "#10b981",
    "XGBoost":               "#ef4444",
    "CatBoost":              "#8b5cf6",
    "RandomForest":          "#f97316",
    "ExtraTrees":            "#eab308",
    "AdaBoost":              "#06b6d4",
    "MLP":                   "#ec4899",
    "LogisticRegression":    "#84cc16",
    "GaussianNB":            "#94a3b8",
}
_FALLBACK_CYCLE = list(_BASE_COLORS.values())
MODEL_COLORS = {
    m: _BASE_COLORS.get(m, _FALLBACK_CYCLE[i % len(_FALLBACK_CYCLE)])
    for i, m in enumerate(MODELS)
}

METRICS = {
    "test_f1_macro":    "Test F1-Macro",
    "test_accuracy":    "Test Accuracy",
    "test_f1_weighted": "Test F1-Weighted",
    "val_f1_macro":     "Val F1-Macro",
    "val_accuracy":     "Val Accuracy",
    "train_time_s":     "Train Time (s)",
}

# ── Pre-compute best-per-grid-target ──────────────────────────────────────────
best_df = (
    df.loc[df.groupby(["grid_id", "target"])["test_f1_macro"].idxmax()]
    .reset_index(drop=True)
)

# ── Dark theme template ───────────────────────────────────────────────────────
DARK = dict(
    plot_bgcolor  = "#111827",
    paper_bgcolor = "#0d1120",
    font_color    = "#e2e8f0",
)

def dark_fig(fig):
    fig.update_layout(
        plot_bgcolor  = DARK["plot_bgcolor"],
        paper_bgcolor = DARK["paper_bgcolor"],
        font_color    = DARK["font_color"],
    )
    fig.update_xaxes(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f")
    fig.update_yaxes(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f")
    return fig

# ── App layout ────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="Ocean QC Dashboard",
    suppress_callback_exceptions=True,
)
server = app.server  # exposed for gunicorn/wsgi deployment if ever needed

HEADER = html.Div([
    html.Div([
        html.H1("🌊 Ocean QC Classification — Model Results Dashboard",
                style={"fontSize":"1.4rem","color":"#60a5fa","margin":0}),
        html.P("Southern Indian Ocean · Gridwise · Leak-Safe Split · "
               f"{len(MODELS)} Models · Targets: {', '.join(TARGETS)}",
               style={"color":"#94a3b8","fontSize":"0.8rem","margin":"4px 0 0"}),
    ]),
    html.Div(f"{len(df)} records · {len(GRIDS)} grids · {len(MODELS)} models",
             style={"background":"#1e3a6e","border":"1px solid #3b5998",
                    "borderRadius":"20px","padding":"6px 16px",
                    "fontSize":"0.75rem","color":"#93c5fd"}),
], style={"display":"flex","justifyContent":"space-between","alignItems":"center",
          "background":"linear-gradient(135deg,#0d1b3e,#1a2a5e,#0d1b3e)",
          "borderBottom":"2px solid #2a4080","padding":"18px 28px"})

def stat_card(val, label, color="#60a5fa"):
    return html.Div([
        html.Div(str(val), style={"fontSize":"1.6rem","fontWeight":700,"color":color}),
        html.Div(label,    style={"fontSize":"0.68rem","color":"#64748b",
                                  "textTransform":"uppercase","letterSpacing":"0.5px","marginTop":"2px"}),
    ], style={"background":"#111827","border":"1px solid #1e3a5f",
              "borderRadius":"10px","padding":"14px","textAlign":"center"})

avg_acc = round(df["test_accuracy"].mean(), 3)
avg_f1  = round(df["test_f1_macro"].mean(), 3)
win_counts = best_df["model"].value_counts()
top_model  = win_counts.idxmax() if len(win_counts) else "—"

STATS_BAR = html.Div([
    stat_card(len(GRIDS),    "Grids"),
    stat_card(len(MODELS),   "Models"),
    stat_card(avg_acc,       "Avg Test Accuracy", "#34d399"),
    stat_card(avg_f1,        "Avg Test F1-Macro", "#f97316"),
    stat_card(top_model,     "Most-Wins Model",   "#a78bfa"),
], style={"display":"grid","gridTemplateColumns":"repeat(5,1fr)","gap":"12px",
          "padding":"14px 24px","background":"#0d1120","borderBottom":"1px solid #1e2d4a"})

CONTROLS = html.Div([
    html.Div([
        html.Label("Target", style={"fontSize":"0.78rem","color":"#94a3b8"}),
        dcc.Dropdown(
            id="ctrl-target",
            options=[{"label":"Both","value":"all"}] +
                    [{"label":t,"value":t} for t in TARGETS],
            value="all", clearable=False,
            style={"minWidth":"130px","fontSize":"0.82rem"},
        ),
    ], style={"display":"flex","flexDirection":"column","gap":"4px"}),
    html.Div([
        html.Label("Primary Metric", style={"fontSize":"0.78rem","color":"#94a3b8"}),
        dcc.Dropdown(
            id="ctrl-metric",
            options=[{"label":v,"value":k} for k,v in METRICS.items()],
            value="test_f1_macro", clearable=False,
            style={"minWidth":"180px","fontSize":"0.82rem"},
        ),
    ], style={"display":"flex","flexDirection":"column","gap":"4px"}),
    html.Div([
        html.Label("Grid Filter", style={"fontSize":"0.78rem","color":"#94a3b8"}),
        dcc.Dropdown(
            id="ctrl-grid",
            options=[{"label":"All Grids","value":"all"}] +
                    [{"label":g,"value":g} for g in GRIDS],
            value="all", clearable=False,
            style={"minWidth":"140px","fontSize":"0.82rem"},
        ),
    ], style={"display":"flex","flexDirection":"column","gap":"4px"}),
    html.Div([
        html.Label("Model Filter", style={"fontSize":"0.78rem","color":"#94a3b8"}),
        dcc.Dropdown(
            id="ctrl-model",
            options=[{"label":"All Models","value":"all"}] +
                    [{"label":m,"value":m} for m in MODELS],
            value="all", clearable=False,
            style={"minWidth":"180px","fontSize":"0.82rem"},
        ),
    ], style={"display":"flex","flexDirection":"column","gap":"4px"}),
], style={"display":"flex","gap":"16px","alignItems":"flex-end","flexWrap":"wrap",
          "padding":"14px 24px","background":"#0d1120","borderBottom":"1px solid #1e2d4a"})

TABS = dcc.Tabs(id="tabs", value="overview", children=[
    dcc.Tab(label="📊 Overview",          value="overview"),
    dcc.Tab(label="🤖 Model Comparison",  value="models"),
    dcc.Tab(label="🗺 Grid Analysis",     value="grids"),
    dcc.Tab(label="🔥 Heatmap",           value="heatmap"),
    dcc.Tab(label="📋 Data Table",        value="table"),
], colors={"border":"#1e3a5f","primary":"#3b82f6","background":"#0d1120"},
   style={"borderBottom":"1px solid #1e3a5f"})

app.layout = html.Div([
    HEADER, STATS_BAR, CONTROLS, TABS,
    html.Div(id="tab-content", style={"padding":"20px 24px","background":"#0a0e1a","minHeight":"70vh"}),
], style={"background":"#0a0e1a","minHeight":"100vh","fontFamily":"Segoe UI,sans-serif","color":"#e0e6f0"})


# ── Shared filter helper ───────────────────────────────────────────────────────
def apply_filters(target, grid, model):
    d = df.copy()
    if target != "all": d = d[d["target"] == target]
    if grid   != "all": d = d[d["grid_id"] == grid]
    if model  != "all": d = d[d["model"]   == model]
    return d


# ── Tab router ────────────────────────────────────────────────────────────────
@app.callback(Output("tab-content","children"),
              Input("tabs","value"),
              Input("ctrl-target","value"),
              Input("ctrl-metric","value"),
              Input("ctrl-grid","value"),
              Input("ctrl-model","value"))
def render_tab(tab, target, metric, grid, model):
    fd = apply_filters(target, grid, model)
    if tab == "overview": return build_overview(fd, metric)
    if tab == "models":   return build_models(fd, metric)
    if tab == "grids":    return build_grids(fd, metric)
    if tab == "heatmap":  return build_heatmap(target)
    if tab == "table":    return build_table(fd)
    return html.Div("Select a tab")


# ── Card wrapper ──────────────────────────────────────────────────────────────
def card(title, children, col_span=1):
    style = {"background":"#111827","border":"1px solid #1e3a5f","borderRadius":"12px",
             "padding":"16px","gridColumn":f"span {col_span}"}
    return html.Div([
        html.H3(title, style={"fontSize":"0.85rem","color":"#93c5fd",
                               "fontWeight":600,"marginBottom":"12px"}),
        children,
    ], style=style)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
def build_overview(fd, metric):
    if fd.empty:
        return html.Div("No rows match the current filters.", style={"color":"#94a3b8"})

    # 1. Avg metric per model (bar)
    model_avgs = fd.groupby("model")[metric].mean().reset_index()
    fig_avg = px.bar(model_avgs, x="model", y=metric,
                     color="model", color_discrete_map=MODEL_COLORS,
                     text=metric, labels={"model":"", metric: METRICS[metric]})
    fig_avg.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_avg.update_layout(showlegend=False)
    dark_fig(fig_avg)

    # 2. Win count doughnut
    wins = best_df[best_df["grid_id"].isin(fd["grid_id"].unique())]
    wc = wins["model"].value_counts().reset_index()
    wc.columns = ["model","wins"]
    fig_wins = px.pie(wc, names="model", values="wins", hole=0.55,
                      color="model", color_discrete_map=MODEL_COLORS)
    fig_wins.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.15))
    dark_fig(fig_wins)

    # 3. Training time (horizontal bar)
    time_avgs = fd.groupby("model")["train_time_s"].mean().reset_index().sort_values("train_time_s")
    fig_time = px.bar(time_avgs, y="model", x="train_time_s", orientation="h",
                      color="model", color_discrete_map=MODEL_COLORS,
                      labels={"model":"","train_time_s":"Avg Train Time (s)"})
    fig_time.update_layout(showlegend=False)
    dark_fig(fig_time)

    # 4. Val vs Test F1 scatter
    fig_sc = px.scatter(fd, x="val_f1_macro", y="test_f1_macro",
                        color="model", color_discrete_map=MODEL_COLORS,
                        hover_data=["grid_id","target"],
                        labels={"val_f1_macro":"Val F1-Macro","test_f1_macro":"Test F1-Macro"})
    fig_sc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                     line=dict(color="#475569", dash="dash", width=1))
    dark_fig(fig_sc)

    grid_style = {"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px"}
    return html.Div([
        html.Div([
            card(f"📈 Avg {METRICS[metric]} by Model",       dcc.Graph(figure=fig_avg, style={"height":"300px"})),
            card("🏆 Model Win Count (Best per Grid×Target)", dcc.Graph(figure=fig_wins, style={"height":"300px"})),
            card("⏱ Avg Training Time by Model (s)",         dcc.Graph(figure=fig_time, style={"height":"300px"})),
            card("📉 Val F1-Macro vs Test F1-Macro",         dcc.Graph(figure=fig_sc,  style={"height":"300px"})),
        ], style=grid_style)
    ])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def build_models(fd, metric):
    if fd.empty:
        return html.Div("No rows match the current filters.", style={"color":"#94a3b8"})

    # Box plot: distribution per model
    fig_box = px.box(fd, x="model", y=metric, color="model",
                     color_discrete_map=MODEL_COLORS,
                     points="all", notched=False,
                     labels={"model":"", metric: METRICS[metric]})
    fig_box.update_layout(showlegend=False)
    dark_fig(fig_box)

    # Accuracy vs F1 bubble
    grp = fd.groupby("model").agg(
        acc=("test_accuracy","mean"),
        f1=("test_f1_macro","mean"),
        time=("train_time_s","mean"),
    ).reset_index()
    fig_bubble = px.scatter(grp, x="acc", y="f1", size="time", color="model",
                            color_discrete_map=MODEL_COLORS,
                            text="model", size_max=60,
                            labels={"acc":"Avg Test Accuracy","f1":"Avg Test F1-Macro","time":"Train Time"})
    fig_bubble.update_traces(textposition="top center")
    dark_fig(fig_bubble)

    # Overfitting gap (val − test)
    gap = fd.groupby("model").apply(
        lambda x: pd.Series({"gap": x["val_f1_macro"].mean() - x["test_f1_macro"].mean()})
    ).reset_index()
    gap["color"] = gap["gap"].apply(lambda g: "#f97316" if g > 0 else "#10b981")
    fig_gap = go.Figure(go.Bar(
        x=gap["model"], y=gap["gap"],
        marker_color=gap["color"].tolist(),
        text=[f"{v:.3f}" for v in gap["gap"]], textposition="outside",
    ))
    fig_gap.update_layout(title="Val F1 − Test F1 (positive = overfitting)", showlegend=False)
    dark_fig(fig_gap)

    # Radar chart — normalised averages across metrics
    radar_metrics = ["test_accuracy","test_f1_macro","test_f1_weighted","val_f1_macro","val_accuracy"]
    radar_labels  = ["Test Acc","Test F1m","Test F1w","Val F1m","Val Acc"]
    fig_radar = go.Figure()
    for m in MODELS:
        sub = fd[fd["model"]==m]
        if sub.empty: continue
        vals = [sub[c].mean() for c in radar_metrics]
        vals += [vals[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=radar_labels+[radar_labels[0]],
            name=m, line=dict(color=MODEL_COLORS.get(m,"#888")),
            fill="toself", opacity=0.35,
        ))
    fig_radar.update_layout(polar=dict(
        bgcolor="#111827",
        radialaxis=dict(visible=True,range=[0,1],gridcolor="#1e3a5f",color="#94a3b8"),
        angularaxis=dict(gridcolor="#1e3a5f",color="#94a3b8"),
    ), showlegend=True)
    dark_fig(fig_radar)

    grid_style = {"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px"}
    return html.Div([
        html.Div([
            card("📦 Distribution of " + METRICS[metric] + " per Model",
                 dcc.Graph(figure=fig_box,    style={"height":"320px"})),
            card("🫧 Accuracy vs F1 Bubble (size = Train Time)",
                 dcc.Graph(figure=fig_bubble, style={"height":"320px"})),
            card("📊 Overfitting Gap (Val F1 − Test F1)",
                 dcc.Graph(figure=fig_gap,    style={"height":"300px"})),
            card("🕸 Radar — Avg Metrics per Model",
                 dcc.Graph(figure=fig_radar,  style={"height":"300px"})),
        ], style=grid_style)
    ])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — GRID ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def build_grids(fd, metric):
    if fd.empty:
        return html.Div("No rows match the current filters.", style={"color":"#94a3b8"})

    grids_in = sorted(fd["grid_id"].unique())

    # Best metric per grid
    best_per_grid = fd.groupby("grid_id")[metric].max().reset_index().sort_values("grid_id")
    best_per_grid["color"] = best_per_grid[metric].apply(
        lambda v: "#10b981" if v > 0.4 else ("#f97316" if v > 0.3 else "#ef4444")
    )
    fig_best = go.Figure(go.Bar(
        x=best_per_grid["grid_id"], y=best_per_grid[metric],
        marker_color=best_per_grid["color"].tolist(),
        text=[f"{v:.3f}" for v in best_per_grid[metric]], textposition="outside",
    ))
    fig_best.update_layout(title=f"Best {METRICS[metric]} per Grid",
                           xaxis_title="", yaxis_range=[0,1], showlegend=False)
    dark_fig(fig_best)

    # Test rows per grid
    rows_per_grid = fd.groupby("grid_id")["test_rows"].first().reset_index().sort_values("grid_id")
    fig_rows = px.bar(rows_per_grid, x="grid_id", y="test_rows",
                      color_discrete_sequence=["#3b82f6"],
                      labels={"grid_id":"","test_rows":"Test Rows"})
    fig_rows.update_layout(showlegend=False); dark_fig(fig_rows)

    # Winner per grid (stacked)
    targets_in = fd["target"].unique().tolist()
    winner_data = []
    for g in grids_in:
        for t in targets_in:
            sub = fd[(fd["grid_id"]==g)&(fd["target"]==t)]
            if sub.empty: continue
            best_model = sub.loc[sub["test_f1_macro"].idxmax(), "model"]
            winner_data.append({"grid_id":g,"target":t,"model":best_model})
    wdf = pd.DataFrame(winner_data)
    win_counts_grid = wdf.groupby(["grid_id","model"]).size().reset_index(name="wins")
    fig_winner = px.bar(win_counts_grid, x="grid_id", y="wins", color="model",
                        color_discrete_map=MODEL_COLORS, barmode="stack",
                        labels={"grid_id":"","wins":"Wins","model":"Model"})
    dark_fig(fig_winner)

    # Grid vs target heatmap using metric
    pivot = fd.groupby(["grid_id","target"])[metric].max().reset_index()
    pivot_wide = pivot.pivot(index="grid_id", columns="target", values=metric)
    fig_ht = px.imshow(pivot_wide, color_continuous_scale="RdYlGn",
                       zmin=0, zmax=1,
                       labels=dict(color=METRICS[metric]),
                       aspect="auto")
    dark_fig(fig_ht)

    grid_style = {"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px"}
    return html.Div([
        html.Div([
            card(f"🗺 Best {METRICS[metric]} per Grid",
                 dcc.Graph(figure=fig_best,   style={"height":"300px"}), col_span=2),
            card("📦 Test Rows per Grid",
                 dcc.Graph(figure=fig_rows,   style={"height":"280px"})),
            card("🏆 Best Model per Grid × Target (stacked)",
                 dcc.Graph(figure=fig_winner, style={"height":"280px"})),
            card(f"🌡 Grid × Target {METRICS[metric]} Heatmap",
                 dcc.Graph(figure=fig_ht,     style={"height":"280px"}), col_span=2),
        ], style={**grid_style, "gridTemplateColumns":"1fr 1fr"})
    ])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — HEATMAP (Model × Grid)
# ─────────────────────────────────────────────────────────────────────────────
def build_heatmap(target):
    children = []
    tgts = TARGETS if target == "all" else [target]
    for t in tgts:
        sub = df[df["target"] == t]
        pivot = sub.groupby(["grid_id","model"])["test_f1_macro"].mean().reset_index()
        pivot_wide = pivot.pivot(index="grid_id", columns="model", values="test_f1_macro")
        cols_order = [m for m in MODELS if m in pivot_wide.columns]
        pivot_wide = pivot_wide[cols_order]

        fig = px.imshow(
            pivot_wide,
            color_continuous_scale="RdYlGn",
            zmin=0, zmax=1,
            text_auto=".3f",
            aspect="auto",
            labels=dict(color="Test F1-Macro", x="Model", y="Grid"),
            title=f"Test F1-Macro — {t}",
        )
        fig.update_xaxes(tickangle=-30)
        fig.update_layout(height=max(300, 40*len(pivot_wide)+100))
        dark_fig(fig)
        children.append(card(f"🔥 Heatmap: Grid × Model — {t}",
                             dcc.Graph(figure=fig)))
    return html.Div(children, style={"display":"flex","flexDirection":"column","gap":"16px"})


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — DATA TABLE
# ─────────────────────────────────────────────────────────────────────────────
def build_table(fd):
    display_cols = ["grid_id","target","model",
                    "test_accuracy","test_f1_macro","test_f1_weighted",
                    "val_accuracy","val_f1_macro",
                    "train_time_s","train_rows_resampled","test_rows"]
    display_cols = [c for c in display_cols if c in fd.columns]
    tbl_df = fd[display_cols].copy()
    for c in ["test_accuracy","test_f1_macro","test_f1_weighted",
              "val_accuracy","val_f1_macro"]:
        if c in tbl_df.columns:
            tbl_df[c] = tbl_df[c].round(4)

    columns = []
    for c in display_cols:
        col = {"name": c.replace("_"," ").title(), "id": c, "sortable": True}
        if tbl_df[c].dtype in [float, np.float64]:
            col["type"] = "numeric"
            col["format"] = dtf.Format(precision=4, scheme=dtf.Scheme.fixed)
        columns.append(col)

    # Conditional styles — green/red for key metrics
    cond_styles = []
    for c in ["test_f1_macro","test_accuracy"]:
        if c not in tbl_df.columns: continue
        cond_styles += [
            {"if": {"filter_query": f"{{{c}}} > 0.9", "column_id": c},
             "color": "#34d399", "fontWeight": "bold"},
            {"if": {"filter_query": f"{{{c}}} < 0.25", "column_id": c},
             "color": "#f87171"},
        ]

    return html.Div([
        card("📋 Full Results Table — sortable, filterable",
             dash_table.DataTable(
                 data=tbl_df.to_dict("records"),
                 columns=columns,
                 page_size=20,
                 sort_action="native",
                 filter_action="native",
                 export_format="csv",
                 export_headers="display",
                 style_table={"overflowX":"auto"},
                 style_header={
                     "background":"#1e293b","color":"#93c5fd",
                     "fontWeight":"600","fontSize":"0.78rem",
                     "border":"1px solid #1e3a5f",
                 },
                 style_cell={
                     "background":"#111827","color":"#cbd5e1",
                     "fontSize":"0.78rem","padding":"7px 10px",
                     "border":"1px solid #1e2d4a",
                 },
                 style_data_conditional=cond_styles + [
                     {"if":{"row_index":"odd"},"background":"#0f1a2e"},
                 ],
             ), col_span=1,
        ),
    ])


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🌊 Ocean QC Dashboard — starting...")
    print("  Open:  http://127.0.0.1:8050")
    print("="*60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=8050)