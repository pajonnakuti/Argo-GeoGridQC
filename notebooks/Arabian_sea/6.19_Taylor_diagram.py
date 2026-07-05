"""
Taylor Diagram — 6-Model QC Classification
Arabian Sea Gridwise · temp_qc & psal_qc targets
============================================================
Usage:
    python taylor_diagram_6models.py
    python taylor_diagram_6models.py --csv path/to/all_grids_6model_results.csv --out output.png
"""

import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ── CLI ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv", default=r"D:\INCOIS\Agro_project\results\All_grids_models\all_grids_6model_results.csv")
parser.add_argument("--out", default="taylor_diagram_6models.png")
args = parser.parse_args()

csv_path = Path(args.csv)
df = pd.read_csv(csv_path)

# Keep only the 6 original models (ignore any extras in the file)
SIX_MODELS = ["RandomForest", "ExtraTrees", "XGBoost",
               "CatBoost", "LightGBM", "LogisticRegression"]
df = df[df["model"].isin(SIX_MODELS)]

TARGETS = ["temp_qc", "psal_qc"]
TARGET_LABELS = {
    "temp_qc": "Temperature QC  (temp_qc)",
    "psal_qc": "Salinity QC  (psal_qc)",
}

MODEL_STYLE = {
    "RandomForest":       {"color": "#4f8ef7", "marker": "o", "family": "Tree ensemble"},
    "ExtraTrees":         {"color": "#7b6af0", "marker": "s", "family": "Tree ensemble"},
    "XGBoost":            {"color": "#e0855a", "marker": "^", "family": "Boosting"},
    "CatBoost":           {"color": "#e05a8a", "marker": "D", "family": "Boosting"},
    "LightGBM":           {"color": "#f0c040", "marker": "v", "family": "Boosting"},
    "LogisticRegression": {"color": "#5abcb9", "marker": "P", "family": "Linear"},
}

# ── Compute Taylor stats ──────────────────────────────────────
def compute_taylor_stats(df, target):
    """
    Reference per grid = best test_f1_macro across all 6 models for that grid.
    Each model is then compared to this reference across all 26 grids:
      - correlation   : how similarly a model ranks grids vs the reference
      - norm_std      : model std / reference std  (1.0 = perfect spread match)
      - norm_rmsd     : centred RMSD / reference std  (0.0 = perfect)
      - mean_f1       : mean test_f1_macro (absolute performance)
    """
    sub = df[df["target"] == target].copy()
    ref_series = sub.groupby("grid_id")["test_f1_macro"].max()

    stats = {}
    for model, grp in sub.groupby("model"):
        grp   = grp.set_index("grid_id")
        shared = grp.index.intersection(ref_series.index)
        pred  = grp.loc[shared, "test_f1_macro"].values
        ref   = ref_series.loc[shared].values

        std_ref  = ref.std(ddof=1)
        std_pred = pred.std(ddof=1)
        corr     = float(np.corrcoef(ref, pred)[0, 1])
        crmsd    = np.sqrt(np.mean(
            ((pred - pred.mean()) - (ref - ref.mean())) ** 2
        ))
        stats[model] = dict(
            corr      = corr,
            norm_std  = std_pred / std_ref if std_ref > 0 else 0,
            norm_rmsd = crmsd    / std_ref if std_ref > 0 else 0,
            mean_f1   = pred.mean(),
        )
    return stats

# ── Draw one Taylor panel ─────────────────────────────────────
def draw_taylor(ax, stats, title, max_norm_std=1.6):
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, max_norm_std + 0.15)
    ax.set_ylim(-0.05, max_norm_std + 0.60)   # extra headroom for stats table
    ax.axis("off")

    BG   = "#0f1117"
    GRID = "#2e3349"
    TEXT = "#c8cce0"
    REF  = "#4fc86a"

    ax.set_facecolor(BG)

    # ── Correlation radial lines ────────────────────────────
    corr_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                  0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    for r in corr_ticks:
        theta = np.arccos(r)
        x1 = np.cos(theta) * (max_norm_std + 0.05)
        y1 = np.sin(theta) * (max_norm_std + 0.05)
        lw = 1.2 if r in (0.0, 0.5, 0.9) else 0.7
        ls = "-"  if r in (0.0, 0.5, 0.9) else "--"
        ax.plot([np.cos(theta)*0.02, x1],
                [np.sin(theta)*0.02, y1],
                color=GRID, lw=lw, ls=ls, zorder=1)
        # label at outer rim
        lr = max_norm_std + 0.13
        label = f"{r:.2f}".rstrip("0").rstrip(".") if r < 0.99 else "0.99"
        ax.text(np.cos(theta)*lr, np.sin(theta)*lr, label,
                ha="center", va="center", fontsize=8, color=TEXT, zorder=5)

    # "Correlation" arc label
    arc_r = max_norm_std + 0.30
    ax.text(arc_r*np.cos(np.pi/4), arc_r*np.sin(np.pi/4),
            "Correlation", ha="center", va="center",
            fontsize=9, color=TEXT, fontweight="bold",
            rotation=-45, zorder=5)

    # ── Standard deviation arcs ─────────────────────────────
    theta_arc = np.linspace(0, np.pi/2, 300)
    for s in np.arange(0.2, max_norm_std + 0.01, 0.2):
        is_ref = abs(s - 1.0) < 0.01
        ax.plot(s*np.cos(theta_arc), s*np.sin(theta_arc),
                color=REF if is_ref else GRID,
                lw=1.5 if is_ref else 0.6, zorder=1)
        ax.text(s, -0.06, f"{s:.1f}",
                ha="center", va="top", fontsize=8,
                color=REF if is_ref else TEXT)

    ax.text(max_norm_std/2, -0.17,
            "Normalised Standard Deviation",
            ha="center", va="top", fontsize=9,
            color=TEXT, fontweight="bold")

    # ── RMSE contours (centred around REF at x=1,y=0) ──────
    theta_c = np.linspace(0, np.pi, 400)
    for rmsd in np.arange(0.1, max_norm_std, 0.1):
        xs = 1 + rmsd * np.cos(theta_c)
        ys = rmsd * np.sin(theta_c)
        mask = (xs >= 0) & (ys >= 0) & \
               (np.sqrt(xs**2 + ys**2) <= max_norm_std + 0.05)
        if mask.sum() < 5:
            continue
        key = round(rmsd, 1) in (0.2, 0.4, 0.6)
        ax.plot(xs[mask], ys[mask],
                color="#7090c0", lw=1.0 if key else 0.5,
                ls=":", alpha=0.6 if key else 0.28, zorder=2)
        if key:
            mid = mask.nonzero()[0][len(mask.nonzero()[0])//2]
            ax.text(xs[mid], ys[mid], f"{rmsd:.1f}",
                    fontsize=7, color="#8090b0",
                    ha="center", va="center",
                    bbox=dict(fc=BG, ec="none", pad=1), zorder=4)

    ax.text(1.38, 0.50, "RMSE", ha="center", va="center",
            fontsize=8, color="#8090b0", style="italic", zorder=5)

    # ── Reference star ──────────────────────────────────────
    ax.plot(1, 0, marker="*", ms=15,
            color=REF, markeredgecolor="white", markeredgewidth=0.8,
            zorder=8)
    ax.text(1.05, -0.09, "REF", fontsize=8,
            color=REF, ha="left", va="top", fontweight="bold")

    # ── Model dots ──────────────────────────────────────────
    plotted = []
    for model in SIX_MODELS:          # fixed order so colours are consistent
        if model not in stats:
            continue
        s     = stats[model]
        style = MODEL_STYLE[model]
        theta = np.arccos(np.clip(s["corr"], -1, 1))
        x = s["norm_std"] * np.cos(theta)
        y = s["norm_std"] * np.sin(theta)
        size = 90 + s["mean_f1"] * 650
        ax.scatter(x, y, s=size,
                   color=style["color"], marker=style["marker"],
                   edgecolors="white", linewidths=0.8,
                   zorder=9, alpha=0.95)
        short = model.replace("LogisticRegression", "LR")
        ax.annotate(short, (x, y),
                    xytext=(6, 5), textcoords="offset points",
                    fontsize=8, color=style["color"],
                    fontweight="bold", zorder=10)
        plotted.append((model, style, s))

    # ── Stats table above the diagram ───────────────────────
    col_x  = [0.00, 0.28, 0.48, 0.64, 0.80]
    row_h  = 0.060
    tbl_y  = max_norm_std + 0.52
    headers = ["Model", "Corr", "Norm σ", "RMSE", "F1 mean"]
    for hdr, cx in zip(headers, col_x):
        ax.text(cx * (max_norm_std + 0.15), tbl_y,
                hdr, fontsize=7.5, color="#7b82a0",
                fontweight="bold", va="top")

    # horizontal rule
    ax.axhline(tbl_y - 0.01, xmin=0, xmax=1,
               color=GRID, lw=0.6, zorder=0)

    for ri, (model, style, s) in enumerate(
            sorted(plotted, key=lambda x: -x[2]["corr"])):
        ry = tbl_y - (ri + 1) * row_h
        short = model.replace("LogisticRegression", "LogReg")
        vals  = [short,
                 f"{s['corr']:.3f}",
                 f"{s['norm_std']:.3f}",
                 f"{s['norm_rmsd']:.3f}",
                 f"{s['mean_f1']:.4f}"]
        colors = [style["color"]] + ["#c8cce0"] * 4
        for v, cx, vc in zip(vals, col_x, colors):
            ax.text(cx * (max_norm_std + 0.15), ry,
                    v, fontsize=7.5, color=vc, va="top")

    ax.set_title(title, fontsize=12, color=TEXT,
                 fontweight="bold", pad=10)

# ── Figure ────────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#0f1117",
    "text.color":       "#c8cce0",
    "font.family":      "DejaVu Sans",
})

fig, axes = plt.subplots(1, 2, figsize=(22, 11))
fig.patch.set_facecolor("#0f1117")

for ax, target in zip(axes, TARGETS):
    draw_taylor(ax, compute_taylor_stats(df, target), TARGET_LABELS[target])

# ── Legend ────────────────────────────────────────────────────
handles = []
for m in SIX_MODELS:
    s = MODEL_STYLE[m]
    handles.append(mpl.lines.Line2D(
        [], [], color=s["color"], marker=s["marker"], ms=9,
        linestyle="None", markeredgecolor="white", markeredgewidth=0.5,
        label=f"{m}  [{s['family']}]"
    ))
handles += [
    mpl.lines.Line2D([], [], color="#4fc86a", marker="*", ms=11,
                     linestyle="None", markeredgecolor="white",
                     markeredgewidth=0.5,
                     label="Reference — best per grid"),
    mpl.patches.Patch(color="none",
                      label="Dot size ∝ mean test F1-macro"),
]
fig.legend(handles=handles, loc="lower center", ncol=4,
           fontsize=9.5, framealpha=0.15,
           edgecolor="#2e3349", facecolor="#181c26",
           labelcolor="#c8cce0",
           bbox_to_anchor=(0.5, -0.03))

fig.suptitle(
    "Taylor Diagram — 6-Model QC Classification · Arabian Sea (26 grids)\n"
    "Reference = best test F1-macro per grid  ·  targets: temp_qc & psal_qc",
    fontsize=13, fontweight="bold", color="#e2e5f0", y=1.005,
)

plt.tight_layout(rect=[0, 0.06, 1, 1])
out_path = Path(args.out)
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#0f1117")
print(f"Saved → {out_path}")