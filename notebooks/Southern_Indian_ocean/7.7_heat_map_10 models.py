"""
Heatmap of model performance — ACCURACY ONLY (grid x model).

Excludes f1_macro / f1_weighted columns, per request.
Produces one heatmap per target (temp_qc, psal_qc):
    rows    = grid_id
    columns = model
    color   = test_accuracy

Run this AFTER the main pipeline has produced:
    <OUTPUT_DIR>/all_grids_10model_results.csv
(OUTPUT_DIR = RESULTS_ROOT/All_grids_models in the training script)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# CONFIG — point this at your results CSV
# ------------------------------------------------------------
RESULTS_CSV = (
    r"D:\INCOIS\Agro_project\results_southern_indian_ocean_v2"
    r"\All_grids_models\all_grids_10model_results.csv"
)

# Only accuracy — no macro/weighted F1.
METRIC = "test_accuracy"

OUTPUT_DIR = os.path.dirname(RESULTS_CSV)


def main():
    df = pd.read_csv(RESULTS_CSV)

    # Drop the excluded columns entirely so they can never leak into a plot
    df = df.drop(columns=[c for c in df.columns if "f1_macro" in c or "f1_weighted" in c],
                 errors="ignore")

    if METRIC not in df.columns:
        raise ValueError(f"Column '{METRIC}' not found in {RESULTS_CSV}")

    for target in sorted(df["target"].unique()):
        sub = df[df["target"] == target]

        # pivot: rows = grid_id, cols = model, values = metric
        pivot = sub.pivot_table(
            index="grid_id", columns="model", values=METRIC, aggfunc="mean"
        )

        # order columns by average performance (best model on the right)
        pivot = pivot[pivot.mean(axis=0).sort_values().index]

        height = max(6, 0.35 * len(pivot))   # scale figure with number of grids
        plt.figure(figsize=(12, height))

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            vmin=0,
            vmax=1,
            linewidths=0.4,
            linecolor="white",
            cbar_kws={"label": METRIC},
        )

        plt.title(f"{METRIC} by grid and model — target: {target}",
                  fontsize=14, fontweight="bold")
        plt.xlabel("Model")
        plt.ylabel("Grid ID")
        plt.tight_layout()

        out_path = os.path.join(OUTPUT_DIR, f"heatmap_{METRIC}_{target}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"✅ Saved heatmap → {out_path}")


if __name__ == "__main__":
    main()