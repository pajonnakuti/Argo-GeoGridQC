import os
import pandas as pd
import numpy as np
import glob
import warnings

warnings.filterwarnings('ignore')

# ==========================================================
# 1. CONFIGURATION
# ==========================================================
GRID_FOLDER  = r"D:\INCOIS\Agro_project\data\Arabian_sea_gridwise_csv"
REPORT_DIR   = r"D:\INCOIS\Agro_project\data\processed"
REPORT_FILE  = os.path.join(REPORT_DIR, "grid_validation_report.txt")

os.makedirs(REPORT_DIR, exist_ok=True)

# ==========================================================
# PHYSICAL RANGE DEFINITIONS (Single source of truth)
# Must match exactly what apply_zscore_flags() uses
# ==========================================================
TEMP_MIN = -2
TEMP_MAX =  45
SAL_MIN  =  5
SAL_MAX  =  45
BAD_FLAG_THRESHOLD = 3   # flags >= 3 are considered BAD

# ==========================================================
# 2. FLAG LOGIC CHECKER
# Must mirror apply_zscore_flags() exactly
# ==========================================================
def expected_flag(z_value):
    """
    Returns the expected flag for a given Z-score.
    Mirrors zscore_to_flag() in apply_zscore_flags().
    """
    if pd.isna(z_value):
        return 9
    abs_z = abs(z_value)
    if abs_z <= 2.0:
        return 1
    elif abs_z <= 3.0:
        return 2
    elif abs_z <= 4.0:
        return 3
    else:
        return 4


# ==========================================================
# 3. PER-GRID VALIDATION
# ==========================================================
def validate_grid(df, grid_id):
    """
    Runs 3 checks on a single grid dataframe.
    Returns a status dict.
    """
    status = {
        "grid_id"  : grid_id,
        "rows"     : len(df),
        "math"     : "PASSED",
        "logic"    : "PASSED",
        "physical" : "PASSED",
        "errors"   : []
    }

    # ----------------------------------------------------------
    # CHECK 1: MATHEMATICAL (Z-score recalculation)
    # ----------------------------------------------------------
    required_math_cols = [
        'temperature', 'temp_grid_mean',
        'temp_grid_std', 'temp_zscore'
    ]
    if all(c in df.columns for c in required_math_cols):
        std_safe   = df['temp_grid_std'].replace(0, np.nan)
        recalc_z   = (
            (df['temperature'] - df['temp_grid_mean']) / std_safe
        )
        diff       = (df['temp_zscore'] - recalc_z).abs()
        max_diff   = diff.max()

        if max_diff > 0.01:
            status["math"] = "FAILED"
            status["errors"].append(
                f"Math: Z-score mismatch (max diff={max_diff:.4f})"
            )
    
    # Salinity math check
    required_sal_cols = [
        'salinity', 'sal_grid_mean',
        'sal_grid_std', 'sal_zscore'
    ]
    if all(c in df.columns for c in required_sal_cols):
        std_safe   = df['sal_grid_std'].replace(0, np.nan)
        recalc_z   = (
            (df['salinity'] - df['sal_grid_mean']) / std_safe
        )
        diff       = (df['sal_zscore'] - recalc_z).abs()
        max_diff   = diff.max()

        if max_diff > 0.01:
            status["math"] = "FAILED"
            status["errors"].append(
                f"Math: Salinity Z-score mismatch (max diff={max_diff:.4f})"
            )

    # ----------------------------------------------------------
    # CHECK 2: LOGICAL (flags match Z-score thresholds)
    # Physical overrides are allowed — skip logic check for
    # physically impossible rows (they are correctly set to 4)
    # ----------------------------------------------------------
    if 'temp_zscore' in df.columns and 'temp_z_flag' in df.columns:

        # Identify physically impossible rows (override is correct)
        temp_physically_bad = (
            (df['temperature'] > TEMP_MAX) |
            (df['temperature'] < TEMP_MIN)
        )

        # Only check logic on rows that are NOT physically impossible
        df_logic = df[~temp_physically_bad].copy()

        if not df_logic.empty:
            df_logic['expected_flag'] = df_logic['temp_zscore'].apply(
                expected_flag
            )
            logic_errors = df_logic[
                df_logic['temp_z_flag'] != df_logic['expected_flag']
            ]
            if not logic_errors.empty:
                status["logic"] = "FAILED"
                status["errors"].append(
                    f"Logic: {len(logic_errors)} rows have incorrect flags"
                )

    # Salinity logic check
    if 'sal_zscore' in df.columns and 'sal_z_flag' in df.columns:

        sal_physically_bad = (
            (df['salinity'] > SAL_MAX) |
            (df['salinity'] < SAL_MIN)
        )

        df_sal_logic = df[~sal_physically_bad].copy()

        if not df_sal_logic.empty:
            df_sal_logic['expected_sal_flag'] = df_sal_logic['sal_zscore'].apply(
                expected_flag
            )
            sal_logic_errors = df_sal_logic[
                df_sal_logic['sal_z_flag'] != df_sal_logic['expected_sal_flag']
            ]
            if not sal_logic_errors.empty:
                status["logic"] = "FAILED"
                status["errors"].append(
                    f"Logic: {len(sal_logic_errors)} salinity rows have incorrect flags"
                )

    # ----------------------------------------------------------
    # CHECK 3: PHYSICAL RANGE
    # Impossible values must be flagged BAD (flag >= 3)
    # Check temp and salinity SEPARATELY with their OWN flags
    # ----------------------------------------------------------

    # Temperature physical check
    if 'temperature' in df.columns and 'temp_z_flag' in df.columns:
        temp_impossible     = (
            (df['temperature'] > TEMP_MAX) |
            (df['temperature'] < TEMP_MIN)
        )
        temp_not_flagged    = temp_impossible & (
            df['temp_z_flag'] < BAD_FLAG_THRESHOLD
        )
        uncaught_temp       = temp_not_flagged.sum()
    else:
        uncaught_temp = 0

    # Salinity physical check — uses sal_z_flag NOT temp_z_flag
    if 'salinity' in df.columns and 'sal_z_flag' in df.columns:
        sal_impossible      = (
            (df['salinity'] > SAL_MAX) |
            (df['salinity'] < SAL_MIN)
        )
        sal_not_flagged     = sal_impossible & (
            df['sal_z_flag'] < BAD_FLAG_THRESHOLD   # ← KEY FIX
        )
        uncaught_sal        = sal_not_flagged.sum()
    else:
        uncaught_sal = 0

    total_uncaught = uncaught_temp + uncaught_sal

    if total_uncaught > 0:
        status["physical"] = "FAILED"
        detail_parts = []
        if uncaught_temp > 0:
            detail_parts.append(f"temp={uncaught_temp}")
        if uncaught_sal > 0:
            detail_parts.append(f"sal={uncaught_sal}")
        status["errors"].append(
            f"Physical: {total_uncaught} impossible values were NOT flagged as BAD "
            f"({', '.join(detail_parts)})"
        )

    return status


# ==========================================================
# 4. MAIN RUNNER
# ==========================================================
def run_smart_verification():
    print("=" * 60)
    print("INCOIS ARGO QC: SMART VERIFICATION SCRIPT")
    print("=" * 60)
    print(f"\nPhysical ranges used:")
    print(f"  Temperature : [{TEMP_MIN}, {TEMP_MAX}]")
    print(f"  Salinity    : [{SAL_MIN},  {SAL_MAX}]")
    print(f"  BAD flag    : >= {BAD_FLAG_THRESHOLD}")

    csv_files   = sorted(glob.glob(os.path.join(GRID_FOLDER, "*.csv")))
    total_files = len(csv_files)

    if total_files == 0:
        print(f"\n❌ No CSV files found in:\n{GRID_FOLDER}")
        return

    print(f"\nTotal CSV files found: {total_files:,}")
    print("\nRunning validation...\n")

    results      = []
    failed_list  = []

    for i, file_path in enumerate(csv_files, start=1):
        file_name  = os.path.basename(file_path)
        grid_id    = file_name.replace(".csv", "")

        try:
            df     = pd.read_csv(file_path)
            status = validate_grid(df, grid_id)
            results.append(status)

            overall = (
                status["math"]     == "PASSED" and
                status["logic"]    == "PASSED" and
                status["physical"] == "PASSED"
            )
            if not overall:
                failed_list.append(status)

        except Exception as e:
            err_status = {
                "grid_id"  : grid_id,
                "rows"     : 0,
                "math"     : "ERROR",
                "logic"    : "ERROR",
                "physical" : "ERROR",
                "errors"   : [f"File read error: {str(e)}"]
            }
            results.append(err_status)
            failed_list.append(err_status)

        if i % 50 == 0 or i == total_files:
            passed_so_far = i - len(failed_list)
            print(f"  [{i:>4}/{total_files}] "
                  f"Passed={passed_so_far} | Failed={len(failed_list)}")

    # ----------------------------------------------------------
    # SAVE REPORT
    # ----------------------------------------------------------
    passed_count = total_files - len(failed_list)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:

        f.write("INCOIS ARGO SMART VALIDATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Physical ranges:\n")
        f.write(f"  Temperature : [{TEMP_MIN}, {TEMP_MAX}]\n")
        f.write(f"  Salinity    : [{SAL_MIN}, {SAL_MAX}]\n")
        f.write(f"  BAD flag    : >= {BAD_FLAG_THRESHOLD}\n\n")

        f.write(f"Total Grids : {total_files}\n")
        f.write(f"Passed      : {passed_count}\n")
        f.write(f"Failed      : {len(failed_list)}\n\n")

        if failed_list:
            f.write("FAILED GRIDS:\n")
            f.write("-" * 40 + "\n")
            for fg in failed_list:
                for err in fg["errors"]:
                    f.write(f"Grid {fg['grid_id']} FAILED: {err}\n")
        else:
            f.write(
                "✅ SUCCESS: All grids passed all QC checks.\n"
            )

        # Full results table
        f.write("\n\nFULL RESULTS TABLE\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"{'Grid':<15} {'Rows':>8} "
            f"{'Math':>8} {'Logic':>8} {'Physical':>10}\n"
        )
        f.write("-" * 60 + "\n")
        for r in results:
            f.write(
                f"{r['grid_id']:<15} {r['rows']:>8,} "
                f"{r['math']:>8} {r['logic']:>8} {r['physical']:>10}\n"
            )

    print(f"\n{'='*60}")
    print(f"VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total  : {total_files}")
    print(f"Passed : {passed_count}")
    print(f"Failed : {len(failed_list)}")
    print(f"\nReport saved to:\n{REPORT_FILE}")

    if failed_list:
        print(f"\nFailed grids summary:")
        for fg in failed_list:
            print(f"  {fg['grid_id']}: {' | '.join(fg['errors'])}")


# ==========================================================
# 5. ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    run_smart_verification()