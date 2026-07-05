import os
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION  — must match your original script paths
# ============================================================

TRAIN_FOLDER = (
    r"D:\INCOIS\Agro_project\data"
    r"\Arabian_sea_gridwise_split\train"
)
TEST_FOLDER  = (
    r"D:\INCOIS\Agro_project\data"
    r"\Arabian_sea_gridwise_split\test"
)
RESULTS_ROOT = r"D:\INCOIS\Agro_project\results"
OUTPUT_DIR   = os.path.join(RESULTS_ROOT, "All_grids_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VAL_SIZE     = 0.15
RANDOM_STATE = 42
RESUME       = True
N_WORKERS    = max(1, os.cpu_count() // 2)

# New models being added in this script
NEW_MODEL_NAMES = ["HistGradientBoosting", "MLP", "SVM", "GradientBoosting"]
N_NEW_MODELS    = len(NEW_MODEL_NAMES)
N_TARGETS       = 2
EXPECTED_NEW_ROWS_PER_GRID = N_NEW_MODELS * N_TARGETS  # 8

# ============================================================
# FEATURES & TARGETS  — same as original
# ============================================================

FEATURE_COLS = [
    "latitude", "longitude",
    "depth", "depth_bin", "pressure",
    "temperature", "salinity",
    "month", "season",
    "temp_grid_mean", "temp_grid_std",
    "sal_grid_mean",  "sal_grid_std",
    "n_obs",
    "temp_zscore", "sal_zscore",
    "temp_zscore_abs", "sal_zscore_abs",
    "temp_z_flag", "sal_z_flag",
]

TARGET_COLS = ["temp_qc", "psal_qc"]

# ============================================================
# IMPORTS
# ============================================================

from sklearn.ensemble        import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neural_network  import MLPClassifier
from sklearn.svm             import SVC
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score, f1_score, confusion_matrix

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

# ============================================================
# MODEL FACTORY  — 4 new models only
# ============================================================

def build_new_models():
    return {
        "HistGradientBoosting": {
            "model": HistGradientBoostingClassifier(
                max_iter=300, max_depth=8, learning_rate=0.05,
                random_state=RANDOM_STATE, class_weight="balanced",
            ),
            "scaled": False,
        },
        "MLP": {
            "model": MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu", solver="adam",
                max_iter=200, random_state=RANDOM_STATE,
                early_stopping=True, validation_fraction=0.1,
            ),
            "scaled": True,
        },
        "SVM": {
            "model": SVC(
                kernel="rbf", C=1.0, gamma="scale",
                class_weight="balanced", random_state=RANDOM_STATE,
            ),
            "scaled": True,
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, random_state=RANDOM_STATE,
            ),
            "scaled": False,
        },
    }

# ============================================================
# HELPERS
# ============================================================

def evaluate_quiet(y_true, y_pred):
    acc    = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_wt  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return acc, f1_mac, f1_wt


def grid_output_dir(grid_id):
    d = os.path.join(RESULTS_ROOT, grid_id)
    os.makedirs(os.path.join(d, "confusion_matrices"), exist_ok=True)
    return d

# ============================================================
# SINGLE-GRID WORKER
# ============================================================

def process_grid_new_models(grid_id):
    """
    Runs only the 4 new models for one grid.
    Appends results to the existing per-grid CSV and returns rows.
    """

    train_csv = os.path.join(TRAIN_FOLDER, f"{grid_id}_train.csv")
    test_csv  = os.path.join(TEST_FOLDER,  f"{grid_id}_test.csv")

    grid_dir     = grid_output_dir(grid_id)
    grid_cm_dir  = os.path.join(grid_dir, "confusion_matrices")
    results_path = os.path.join(grid_dir, f"{grid_id}_6model_results.csv")
    done_marker  = os.path.join(grid_dir, ".done_new_models")

    skipped_log  = []
    new_rows     = []

    # ── RESUME CHECK ─────────────────────────────────────
    # Skip this grid if new-model results already exist for it
    if RESUME and os.path.exists(done_marker):
        try:
            existing = pd.read_csv(results_path)
            already_done = existing[existing["model"].isin(NEW_MODEL_NAMES)]
            if len(already_done) >= EXPECTED_NEW_ROWS_PER_GRID:
                return grid_id, already_done.to_dict("records"), [], True
        except Exception:
            pass

    if not os.path.exists(test_csv):
        return grid_id, [], [{"grid_id": grid_id, "reason": "no test file"}], False

    # ── LOAD DATA ────────────────────────────────────────
    try:
        train_full_df = pd.read_csv(train_csv)
        test_df       = pd.read_csv(test_csv)
    except Exception as e:
        return grid_id, [], [{"grid_id": grid_id, "reason": f"load error: {e}"}], False

    feat_cols  = [c for c in FEATURE_COLS if c in train_full_df.columns]
    X_all_raw  = train_full_df[feat_cols].copy()
    X_test_raw = test_df[feat_cols].copy()

    for col in feat_cols:
        med = X_all_raw[col].median()
        X_all_raw[col]  = X_all_raw[col].fillna(med)
        X_test_raw[col] = X_test_raw[col].fillna(med)

    grid_t0 = time.time()

    # ── PER TARGET ───────────────────────────────────────
    for target in TARGET_COLS:

        if target not in train_full_df.columns or target not in test_df.columns:
            continue

        y_all_full  = train_full_df[target].copy()
        y_test_full = test_df[target].copy()
        valid_all   = y_all_full.notna()
        valid_test  = y_test_full.notna()

        X_all = X_all_raw[valid_all].reset_index(drop=True)
        y_all = y_all_full[valid_all].reset_index(drop=True)
        X_te  = X_test_raw[valid_test].reset_index(drop=True)
        y_te  = y_test_full[valid_test].reset_index(drop=True)

        if y_all.nunique() < 2 or len(X_all) < 20 or len(X_te) < 5:
            continue

        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_all, y_all, test_size=VAL_SIZE,
                stratify=y_all, random_state=RANDOM_STATE,
            )
        except ValueError:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_all, y_all, test_size=VAL_SIZE,
                random_state=RANDOM_STATE,
            )

        X_tr  = X_tr.reset_index(drop=True);  y_tr  = y_tr.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True); y_val = y_val.reset_index(drop=True)

        print(f"\n  TARGET: {target}  |  Train={len(X_tr):,}  "
              f"Val={len(X_val):,}  Test={len(X_te):,}  "
              f"Classes={sorted(y_all.unique().tolist())}")

        # ── SMOTE (same as original) ──────────────────────
        if HAS_SMOTE:
            class_counts   = y_tr.value_counts()
            min_class_size = class_counts.min()
            k_neighbors    = max(1, min(5, min_class_size - 1))

            if min_class_size > 1 and len(class_counts) > 1:
                try:
                    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
                    X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
                    X_tr_res = pd.DataFrame(X_tr_res, columns=X_tr.columns)
                    y_tr_res = pd.Series(y_tr_res)
                    print(f"    ✅ SMOTE: {len(X_tr):,} → {len(X_tr_res):,} rows "
                          f"(k_neighbors={k_neighbors})")
                except Exception as e:
                    print(f"    ⚠ SMOTE failed ({e}) — using original training data")
                    X_tr_res, y_tr_res = X_tr, y_tr
            else:
                print(f"    ⚠ SMOTE skipped — a class has too few samples")
                X_tr_res, y_tr_res = X_tr, y_tr
        else:
            X_tr_res, y_tr_res = X_tr, y_tr

        # ── SCALING ──────────────────────────────────────
        scaler      = StandardScaler()
        X_tr_res_sc = scaler.fit_transform(X_tr_res)
        X_val_sc    = scaler.transform(X_val)
        X_te_sc     = scaler.transform(X_te)

        le = LabelEncoder()
        le.fit(y_all)

        # ── 4 NEW MODELS ─────────────────────────────────
        models = build_new_models()

        for model_name, cfg in models.items():

            model      = cfg["model"]
            use_scaled = cfg.get("scaled", False)

            Xtr_fit  = X_tr_res_sc if use_scaled else X_tr_res.values
            Xval_fit = X_val_sc    if use_scaled else X_val.values
            Xte_fit  = X_te_sc     if use_scaled else X_te.values

            try:
                t0 = time.time()
                model.fit(Xtr_fit, y_tr_res.values)
                train_time = time.time() - t0

                y_val_pred = model.predict(Xval_fit)
                val_acc, val_f1_mac, val_f1_wt = evaluate_quiet(y_val, y_val_pred)

                y_te_pred = model.predict(Xte_fit)
                te_acc, te_f1_mac, te_f1_wt = evaluate_quiet(y_te, y_te_pred)

                # Confusion matrix
                labels_sorted = sorted(y_all.unique().tolist())
                cm = confusion_matrix(y_te, y_te_pred, labels=labels_sorted)
                cm_df = pd.DataFrame(
                    cm,
                    index=[f"true_{l}" for l in labels_sorted],
                    columns=[f"pred_{l}" for l in labels_sorted],
                )
                cm_df.to_csv(
                    os.path.join(grid_cm_dir,
                                 f"{grid_id}_{target}_{model_name}_cm.csv")
                )

                print(f"    {model_name:<22} "
                      f"val_f1m={val_f1_mac:.3f}  test_f1m={te_f1_mac:.3f}  "
                      f"test_acc={te_acc:.3f}  ({train_time:.1f}s)")

                new_rows.append({
                    "grid_id"             : grid_id,
                    "target"              : target,
                    "model"               : model_name,
                    "val_accuracy"        : round(val_acc,    4),
                    "val_f1_macro"        : round(val_f1_mac, 4),
                    "val_f1_weighted"     : round(val_f1_wt,  4),
                    "test_accuracy"       : round(te_acc,     4),
                    "test_f1_macro"       : round(te_f1_mac,  4),
                    "test_f1_weighted"    : round(te_f1_wt,   4),
                    "train_time_s"        : round(train_time, 2),
                    "train_rows_resampled": len(Xtr_fit),
                    "val_rows"            : len(X_val),
                    "test_rows"           : len(X_te),
                })

            except Exception as e:
                print(f"    ❌ {model_name} FAILED on {grid_id}/{target}: {e}")
                skipped_log.append({
                    "grid_id": grid_id, "target": target,
                    "model": model_name, "reason": str(e),
                })

    grid_elapsed = time.time() - grid_t0
    print(f"\n  ⏱  Grid {grid_id} done in {grid_elapsed/60:.1f} min")

    # ── APPEND TO EXISTING PER-GRID CSV ──────────────────
    new_df = pd.DataFrame(new_rows)
    if os.path.exists(results_path):
        existing_df = pd.read_csv(results_path)
        # Drop any previous runs of these new models before appending
        existing_df = existing_df[~existing_df["model"].isin(NEW_MODEL_NAMES)]
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_csv(results_path, index=False)
    print(f"  ✅ Appended → {results_path}  (total rows: {len(combined_df)})")

    # Update best model CSV for this grid
    if len(combined_df) > 0:
        grid_best = combined_df.loc[
            combined_df.groupby("target")["test_f1_macro"].idxmax()
        ]
        grid_best.to_csv(
            os.path.join(grid_dir, f"{grid_id}_best_model.csv"), index=False
        )

    # Write done marker for new models
    with open(done_marker, "w") as f:
        f.write(f"completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"new_rows_saved={len(new_df)}\n")

    return grid_id, new_rows, skipped_log, False


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_files = sorted(Path(TRAIN_FOLDER).glob("*_train.csv"))
    GRID_IDS    = [f.stem.replace("_train", "") for f in train_files]

    print("=" * 70)
    print("  4 NEW MODELS — APPENDING TO EXISTING RESULTS")
    print("  (HistGradientBoosting, MLP, SVM, GradientBoosting)")
    print("=" * 70)
    print(f"\nGrids found : {len(GRID_IDS)}")
    print(f"Workers     : {N_WORKERS}")

    if not HAS_SMOTE:
        print("\n⚠ imbalanced-learn not installed — SMOTE disabled\n")

    all_new_results = []
    skipped_log     = []
    global_t0       = time.time()

    # ── PARALLEL EXECUTION ───────────────────────────────
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_grid_new_models, gid): gid for gid in GRID_IDS}

        for i, future in enumerate(as_completed(futures), start=1):
            grid_id = futures[future]
            try:
                gid, new_rows, skipped, was_skipped = future.result()
                all_new_results.extend(new_rows)
                skipped_log.extend(skipped)

                status = "⏩ SKIP" if was_skipped else f"✅ done ({len(new_rows)} new rows)"
                elapsed = time.time() - global_t0
                print(f"  [{i:>3}/{len(GRID_IDS)}]  {gid:<25} {status}"
                      f"  |  total so far: {elapsed/60:.1f} min")

            except Exception as exc:
                print(f"  ❌ {grid_id} raised: {exc}")
                skipped_log.append({"grid_id": grid_id, "reason": str(exc)})

    total_elapsed = time.time() - global_t0
    print(f"\n{'='*70}")
    print(f"  ALL GRIDS COMPLETE  —  total time: {total_elapsed/60:.1f} min")
    print(f"{'='*70}")

    # ── REBUILD MASTER CSV FROM ALL PER-GRID FILES ───────
    # Reload all per-grid CSVs (original 6 + new 4 models)
    all_records = []
    for gid in GRID_IDS:
        per_grid_csv = os.path.join(RESULTS_ROOT, gid, f"{gid}_6model_results.csv")
        if os.path.exists(per_grid_csv):
            all_records.append(pd.read_csv(per_grid_csv))

    if all_records:
        master_df = pd.concat(all_records, ignore_index=True)
    else:
        master_df = pd.DataFrame()

    out_csv = os.path.join(OUTPUT_DIR, "all_grids_6model_results.csv")
    master_df.to_csv(out_csv, index=False)
    print(f"\n✅ Master results updated → {out_csv}  ({len(master_df)} total rows)")

    skipped_df = (
        pd.DataFrame(skipped_log) if skipped_log
        else pd.DataFrame(columns=["grid_id", "reason"])
    )
    skipped_df.to_csv(os.path.join(OUTPUT_DIR, "skipped_log.csv"), index=False)

    # ── SUMMARY ──────────────────────────────────────────
    if len(master_df) > 0:
        best_rows = (
            master_df
            .loc[master_df.groupby(["grid_id", "target"])["test_f1_macro"].idxmax()]
            .sort_values(["grid_id", "target"])
        )
        best_rows.to_csv(
            os.path.join(OUTPUT_DIR, "best_model_per_grid_target.csv"), index=False
        )

        print("\n" + "=" * 70)
        print("  BEST MODEL PER GRID PER TARGET  (original 6 + new 4)")
        print("=" * 70)
        print(best_rows[[
            "grid_id", "target", "model",
            "test_accuracy", "test_f1_macro", "test_f1_weighted"
        ]].to_string(index=False))

        print("\n" + "=" * 70)
        print("  MODEL WIN COUNT")
        print("=" * 70)
        print(best_rows["model"].value_counts().to_string())

        print("\n" + "=" * 70)
        print("  AVERAGE TEST F1-MACRO PER MODEL  (all 10 models)")
        print("=" * 70)
        print(
            master_df.groupby("model")[
                ["test_accuracy", "test_f1_macro", "test_f1_weighted"]
            ].mean().sort_values("test_f1_macro", ascending=False)
            .round(4).to_string()
        )

    print(f"\n{'='*70}")
    print(f"  ✅ DONE — {len(GRID_IDS)} grids | 4 new models | 2 targets")
    print(f"{'='*70}")