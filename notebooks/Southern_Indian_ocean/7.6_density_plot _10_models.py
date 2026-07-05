"""
Density (KDE) curve of model performance — ACCURACY ONLY.

Deliberately excludes f1_macro / f1_weighted columns, per request.
Plots the distribution of test_accuracy (and optionally val_accuracy)
across all grids, one density curve per model, so you can compare
how consistently each classifier performs.

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

# Which accuracy column(s) to plot. Only accuracy — no macro/weighted F1.
METRICS_TO_PLOT = ["test_accuracy", "val_accuracy"]

# Optionally restrict to one target ("temp_qc" or "psal_qc"), or None for both combined
TARGET_FILTER = None  # e.g. "temp_qc"

OUTPUT_DIR = os.path.dirname(RESULTS_CSV)


def main():
    df = pd.read_csv(RESULTS_CSV)

    # Drop the excluded columns entirely so they can never leak into a plot
    df = df.drop(columns=[c for c in df.columns if "f1_macro" in c or "f1_weighted" in c],
                 errors="ignore")

    if TARGET_FILTER is not None:
        df = df[df["target"] == TARGET_FILTER]

    sns.set_style("whitegrid")

    for metric in METRICS_TO_PLOT:
        if metric not in df.columns:
            print(f"⚠ Column '{metric}' not found — skipping")
            continue

        plt.figure(figsize=(10, 6))

        models = sorted(df["model"].unique())
        palette = sns.color_palette("tab10", n_colors=len(models))

        for model_name, color in zip(models, palette):
            subset = df.loc[df["model"] == model_name, metric].dropna()
            if len(subset) < 2:
                continue  # KDE needs at least 2 points
            sns.kdeplot(
                subset,
                label=model_name,
                color=color,
                fill=True,
                alpha=0.15,
                linewidth=2,
                clip=(0, 1),
            )

        title_target = f" — target: {TARGET_FILTER}" if TARGET_FILTER else " — both targets combined"
        plt.title(f"Density of {metric}{title_target}", fontsize=14, fontweight="bold")
        plt.xlabel(metric)
        plt.ylabel("Density")
        plt.xlim(0, 1)
        plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()

        out_path = os.path.join(OUTPUT_DIR, f"density_{metric}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"✅ Saved density plot → {out_path}")


if __name__ == "__main__":
    main()