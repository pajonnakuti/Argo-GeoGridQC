#================================================
# Stage 1: Data Loading → Z-Score
#================================================

import numpy as np
import pandas as pd
import warnings
import os

warnings.filterwarnings('ignore')

print("="*60)
print("ARGO QC — Z-SCORE PIPELINE STARTING")
print("="*60)

#================================================
# STEP 1: LOAD YOUR DATAFRAME
#================================================

def load_and_verify(df):
    """
    Verify your dataframe and print basic info.
    """
    print("\n--- STEP 1: LOADING & VERIFYING DATA ---")

    print(f"Shape          : {df.shape}")
    print(f"Rows           : {df.shape[0]:,}")
    print(f"Columns        : {df.shape[1]}")

    print(f"\nColumn names:\n{list(df.columns)}")

    print(f"\nNull counts:\n{df.isnull().sum()}")

    print(f"\nFirst 3 rows:")
    print(df.head(3))

    print(f"\nData types:")
    print(df.dtypes)

    print("\n✅ Data loaded successfully")

    return df

#================================================
# STEP 2: ADD DATE FEATURES
#================================================

def add_date_features(df):
    """
    Extract month, year, season from date column.

    Season mapping:
    DJF = 1 → Dec, Jan, Feb (Winter)
    MAM = 2 → Mar, Apr, May (Spring)
    JJA = 3 → Jun, Jul, Aug (Summer/Monsoon)
    SON = 4 → Sep, Oct, Nov (Autumn)
    """
    print("\n--- STEP 2: ADDING DATE FEATURES ---")

    # Convert to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Extract month and year
    df['month'] = df['date'].dt.month
    df['year']  = df['date'].dt.year

    # Map month to season
    def get_season(month):
        if month in [12, 1, 2]:
            return 1       # DJF Winter
        elif month in [3, 4, 5]:
            return 2       # MAM Spring
        elif month in [6, 7, 8]:
            return 3       # JJA Summer
        else:
            return 4       # SON Autumn

    df['season'] = df['month'].apply(get_season)

    # Print summary
    print(f"Date range   : {df['date'].min()} to {df['date'].max()}")
    print(f"Month range  : {df['month'].min()} to {df['month'].max()}")
    print(f"Year range   : {df['year'].min()} to {df['year'].max()}")

    print(f"\nSeason distribution:")
    season_map = {1:'DJF-Winter', 2:'MAM-Spring',
                  3:'JJA-Summer', 4:'SON-Autumn'}
    for s, count in df['season'].value_counts().sort_index().items():
        print(f"  Season {s} ({season_map[s]}): {count:,} rows")

    print("\n✅ Date features added")
    print(f"New columns added: month, year, season")

    return df

#================================================
# STEP 3: ADD DEPTH BINS
#================================================

def add_depth_bins(df):
    """
    Bin depths into standard oceanographic levels.
    Each measurement gets assigned to nearest
    standard depth bin.
    """
    print("\n--- STEP 3: ADDING DEPTH BINS ---")

    # Standard depth bin edges
    bins = [
        0, 10, 20, 30, 50, 75,
        100, 125, 150, 200, 300,
        400, 500, 700, 1000,
        1500, 2000, 9999
    ]

    # Labels = the representative depth for each bin
    labels = [
        0, 10, 20, 30, 50, 75,
        100, 125, 150, 200, 300,
        400, 500, 700, 1000,
        1500, 2000
    ]

    df['depth_bin'] = pd.cut(
        df['depth'],
        bins   = bins,
        labels = labels,
        right  = True
    ).astype(float)

    # Print summary
    print(f"Depth range  : {df['depth'].min():.2f}m"
          f" to {df['depth'].max():.2f}m")

    print(f"\nDepth bin distribution:")
    depth_counts = df['depth_bin'].value_counts().sort_index()
    for depth, count in depth_counts.items():
        print(f"  {depth:6.0f}m : {count:,} rows")

    nan_depth = df['depth_bin'].isna().sum()
    if nan_depth > 0:
        print(f"\n⚠ WARNING: {nan_depth:,} rows"
              f" outside depth bin range")

    print("\n✅ Depth bins added")
    print(f"New column added: depth_bin")

    return df

#================================================
# STEP 4: CALCULATE GRID STATISTICS
#================================================

def calculate_grid_statistics(df):
    """
    For each unique combination of:
    (grid_id, depth_bin, season)

    Calculate:
    - mean, std, median of temperature
    - mean, std, median of salinity
    - quartiles (Q1, Q3)
    - count of observations

    This becomes your in-house climatology.
    """
    print("\n--- STEP 4: CALCULATING GRID STATISTICS ---")

    print("Grouping by grid_id + depth_bin + season...")
    print("(This may take a moment for 18M rows)")

    grid_stats = df.groupby(
        ['grid_id', 'depth_bin', 'season'],
        observed = True
    ).agg(
        # Temperature statistics
        temp_grid_mean   = ('temperature', 'mean'),
        temp_grid_std    = ('temperature', 'std'),
        temp_grid_median = ('temperature', 'median'),
        temp_grid_q1     = ('temperature', lambda x: x.quantile(0.25)),
        temp_grid_q3     = ('temperature', lambda x: x.quantile(0.75)),
        temp_grid_min    = ('temperature', 'min'),
        temp_grid_max    = ('temperature', 'max'),

        # Salinity statistics
        sal_grid_mean    = ('salinity', 'mean'),
        sal_grid_std     = ('salinity', 'std'),
        sal_grid_median  = ('salinity', 'median'),
        sal_grid_q1      = ('salinity', lambda x: x.quantile(0.25)),
        sal_grid_q3      = ('salinity', lambda x: x.quantile(0.75)),
        sal_grid_min     = ('salinity', 'min'),
        sal_grid_max     = ('salinity', 'max'),

        # Count
        n_obs            = ('temperature', 'count')
    ).reset_index()

    # Print summary
    print(f"\nTotal unique groups : {len(grid_stats):,}")
    print(f"(grid_id x depth_bin x season combinations)")

    print(f"\nGroups with < 10 observations:")
    low_obs = (grid_stats['n_obs'] < 10).sum()
    print(f"  {low_obs:,} groups"
          f" ({100*low_obs/len(grid_stats):.1f}%)")

    print(f"\nSample grid statistics:")
    print(grid_stats.head(5).to_string())

    print(f"\nGrid statistics summary:")
    print(grid_stats[[
        'temp_grid_mean', 'temp_grid_std',
        'sal_grid_mean',  'sal_grid_std',
        'n_obs'
    ]].describe().round(3))

    print("\n✅ Grid statistics calculated")

    return grid_stats

#================================================
# STEP 5: MERGE STATISTICS INTO MAIN DATAFRAME
#================================================

def merge_grid_statistics(df, grid_stats):
    """
    Join grid statistics back into the main dataframe.
    Each row gets the mean/std of its grid cell + depth bin + season.
    """
    print("\n--- STEP 5: MERGING GRID STATISTICS ---")

    rows_before = len(df)

    df = df.merge(
        grid_stats,
        on  = ['grid_id', 'depth_bin', 'season'],
        how = 'left'
    )

    rows_after = len(df)

    print(f"Rows before merge : {rows_before:,}")
    print(f"Rows after merge  : {rows_after:,}")

    if rows_before != rows_after:
        print(f"⚠ WARNING: Row count changed after merge!")
    else:
        print(f"✅ Row count preserved")

    # Check how many rows got stats
    matched = df['temp_grid_mean'].notna().sum()
    unmatched = df['temp_grid_mean'].isna().sum()

    print(f"\nRows matched to grid stats : {matched:,}")
    print(f"Rows unmatched             : {unmatched:,}")

    if unmatched > 0:
        print(f"\nUnmatched rows sample:")
        print(df[df['temp_grid_mean'].isna()][[
            'grid_id', 'depth_bin', 'season',
            'depth', 'temperature', 'salinity'
        ]].head(5))

    print("\n✅ Grid statistics merged")

    return df

#================================================
# STEP 6: CALCULATE Z-SCORES
#================================================

def calculate_zscores(df):
    """
    Calculate Z-scores for temperature and salinity.
    Formula: Z = (value - grid_mean) / grid_std
    """
    print("\n--- STEP 6: CALCULATING Z-SCORES ---")

    # Replace zero std with NaN to avoid division by zero
    temp_std_safe = df['temp_grid_std'].replace(0, np.nan)
    sal_std_safe  = df['sal_grid_std'].replace(0, np.nan)

    # Calculate Z-scores
    df['temp_zscore'] = (
        (df['temperature'] - df['temp_grid_mean'])
        / temp_std_safe
    )

    df['sal_zscore'] = (
        (df['salinity'] - df['sal_grid_mean'])
        / sal_std_safe
    )

    # Absolute Z-scores (magnitude only)
    df['temp_zscore_abs'] = df['temp_zscore'].abs()
    df['sal_zscore_abs']  = df['sal_zscore'].abs()

    # Print statistics
    print(f"\nTemperature Z-score stats:")
    print(df['temp_zscore'].describe().round(4))

    print(f"\nSalinity Z-score stats:")
    print(df['sal_zscore'].describe().round(4))

    # Count anomalies at different thresholds
    print(f"\nAnomaly counts at different Z thresholds:")
    for threshold in [2, 3, 4, 5]:
        t_anom = (df['temp_zscore_abs'] > threshold).sum()
        s_anom = (df['sal_zscore_abs']  > threshold).sum()
        print(f"  |Z| > {threshold} : "
              f"temp={t_anom:,} "
              f"({100*t_anom/len(df):.2f}%)  |  "
              f"sal={s_anom:,} "
              f"({100*s_anom/len(df):.2f}%)")

    print("\n✅ Z-scores calculated")
    return df

#================================================
# STEP 7: APPLY Z-SCORE FLAGS
#================================================

def apply_zscore_flags(df):
    print("\n--- STEP 7: APPLYING Z-SCORE FLAGS & RANGE CHECK ---")

    # 1. Standard Z-Score Logic
    def zscore_to_flag(z):
        if pd.isna(z): return 9
        abs_z = abs(z)
        if abs_z <= 2.0: return 1
        elif abs_z <= 3.0: return 2
        elif abs_z <= 4.0: return 3
        else: return 4

    df['temp_z_flag'] = df['temp_zscore'].apply(zscore_to_flag)
    df['sal_z_flag'] = df['sal_zscore'].apply(zscore_to_flag)

    # 2. PHYSICAL RANGE OVERRIDE (CRITICAL)
    # Force Flag 4 if values are physically impossible
    temp_bad = (df['temperature'] > 45) | (df['temperature'] < -2)
    sal_bad = (df['salinity'] > 45) | (df['salinity'] < 5)
    
    df.loc[temp_bad, 'temp_z_flag'] = 4
    df.loc[sal_bad, 'sal_z_flag'] = 4

    # 3. Final Flag calculation
    df['z_final_flag'] = df[['temp_z_flag', 'sal_z_flag']].max(axis=1)
    
    print(f"✅ Physical Check: Flagged {temp_bad.sum()} Temp and {sal_bad.sum()} Sal points as BAD.")
    return df
#================================================
# STEP 8: Z-SCORE SUMMARY REPORT
#================================================

def zscore_summary_report(df):
    """
    Print and save summary of Z-score QC results.
    """
    print("\n--- STEP 8: Z-SCORE SUMMARY REPORT ---")
    print("="*60)

    total = len(df)
    print(f"\nOVERALL SUMMARY")
    print(f"Total observations : {total:,}")

    # Per grid summary
    print(f"\nPER GRID SUMMARY (Top 10 grids by bad %)")
    grid_summary = df.groupby('grid_id').agg(
        total_obs    = ('z_final_flag', 'count'),
        good_count   = ('z_final_flag', lambda x: (x==1).sum()),
        bad_count    = ('z_final_flag', lambda x: (x>=3).sum()),
        mean_temp_z  = ('temp_zscore_abs', 'mean'),
        mean_sal_z   = ('sal_zscore_abs',  'mean'),
        max_temp_z   = ('temp_zscore_abs', 'max'),
        max_sal_z    = ('sal_zscore_abs',  'max'),
    ).reset_index()

    grid_summary['bad_pct'] = (100 * grid_summary['bad_count'] / grid_summary['total_obs']).round(2)
    print(grid_summary.sort_values('bad_pct', ascending=False).head(10).to_string())

    # Per depth summary
    print(f"\nPER DEPTH SUMMARY")
    depth_summary = df.groupby('depth_bin').agg(
        total_obs   = ('z_final_flag', 'count'),
        bad_count   = ('z_final_flag', lambda x: (x>=3).sum()),
        mean_temp_z = ('temp_zscore_abs', 'mean'),
        mean_sal_z  = ('sal_zscore_abs',  'mean'),
    ).reset_index()
    depth_summary['bad_pct'] = (100 * depth_summary['bad_count'] / depth_summary['total_obs']).round(2)
    print(depth_summary.sort_values('depth_bin').to_string())

    # Per season summary
    print(f"\nPER SEASON SUMMARY")
    season_map = {1:'DJF-Winter', 2:'MAM-Spring', 3:'JJA-Summer', 4:'SON-Autumn'}
    season_summary = df.groupby('season').agg(
        total_obs   = ('z_final_flag', 'count'),
        bad_count   = ('z_final_flag', lambda x: (x>=3).sum()),
        mean_temp_z = ('temp_zscore_abs', 'mean'),
        mean_sal_z  = ('sal_zscore_abs',  'mean'),
    ).reset_index()
    season_summary['season_name'] = season_summary['season'].map(season_map)
    season_summary['bad_pct'] = (100 * season_summary['bad_count'] / season_summary['total_obs']).round(2)
    print(season_summary.to_string())

    # TOP ANOMALIES (MEMORY SAFE SAVING)
    print(f"\nSaving worst anomalies...")
    
    sal_cols = ['grid_id', 'file_name', 'date', 'latitude', 'longitude', 'depth', 'salinity', 'sal_grid_mean', 'sal_grid_std', 'sal_zscore', 'sal_z_flag', 'sal_zscore_abs']
    sal_cols = [c for c in sal_cols if c in df.columns]
    df[sal_cols].sort_values('sal_zscore_abs', ascending=False).head(100).to_csv("worst_salinity_anomalies.csv", index=False)
    print("Saved: worst_salinity_anomalies.csv")

    temp_cols = ['grid_id', 'file_name', 'date', 'latitude', 'longitude', 'depth', 'temperature', 'temp_grid_mean', 'temp_grid_std', 'temp_zscore', 'temp_z_flag', 'temp_zscore_abs']
    temp_cols = [c for c in temp_cols if c in df.columns]
    df[temp_cols].sort_values('temp_zscore_abs', ascending=False).head(100).to_csv("worst_temperature_anomalies.csv", index=False)
    print("Saved: worst_temperature_anomalies.csv")

    return grid_summary, depth_summary

#================================================
# STEP 9: SAVE RESULTS
#================================================

def save_zscore_results(df, grid_summary, depth_summary):
    """
    Save all Z-score results to files.
    """
    print("\n--- STEP 9: SAVING RESULTS ---")

    output_cols = [
        'grid_id', 'file_name', 'date', 'latitude', 'longitude',
        'depth', 'depth_bin', 'pressure', 'temperature', 'salinity',
        'month', 'year', 'season', 'temp_grid_mean', 'temp_grid_std',
        'sal_grid_mean',  'sal_grid_std', 'n_obs', 'temp_zscore', 'sal_zscore',
        'temp_zscore_abs', 'sal_zscore_abs', 'temp_z_flag', 'sal_z_flag',
        'z_final_flag', 'z_flag_label', 'temp_qc', 'psal_qc'
    ]
    output_cols = [c for c in output_cols if c in df.columns]

    df[output_cols].to_parquet('argo_zscore_results.parquet', index=False)
    print(f"✅ Saved: argo_zscore_results.parquet")

    grid_summary.to_csv('grid_zscore_summary.csv', index=False)
    print(f"✅ Saved: grid_zscore_summary.csv")

    depth_summary.to_csv('depth_zscore_summary.csv', index=False)
    print(f"✅ Saved: depth_zscore_summary.csv")

    # Save bad observations only
    bad_obs = df[df['z_final_flag'] >= 3]
    bad_obs.to_csv('bad_observations_zscore.csv', index=False)
    print(f"✅ Saved: bad_observations_zscore.csv ({len(bad_obs):,} rows)")

    return df

#================================================
# MAIN — RUN ALL STEPS
#================================================

def run_zscore_pipeline(df):
    """
    Run complete Z-score QC pipeline.
    """
    print("\n" + "="*60)
    print("STARTING Z-SCORE PIPELINE")
    print("="*60)

    df = load_and_verify(df)
    df = add_date_features(df)
    df = add_depth_bins(df)
    grid_stats = calculate_grid_statistics(df)
    df = merge_grid_statistics(df, grid_stats)
    df = calculate_zscores(df)
    df = apply_zscore_flags(df)
    
    grid_sum, depth_sum = zscore_summary_report(df)
    df = save_zscore_results(df, grid_sum, depth_sum)

    print("\n" + "="*60)
    print("✅ Z-SCORE PIPELINE COMPLETE")
    print(f"Final dataframe shape: {df.shape}")
    print("="*60)

    return df, grid_stats

#================================================
# ENTRY POINT
#================================================

if __name__ == "__main__":
    # Update this path for your environment
    INPUT_FILE = r"D:\INCOIS\Agro_project\data\processed\arabian_sea_df.parquet"

    print("\nLoading dataframe...")
    print(INPUT_FILE)

    if os.path.exists(INPUT_FILE):
        df = pd.read_parquet(INPUT_FILE)
        print("\nData Loaded Successfully")
        print(f"Shape : {df.shape}")

        df_with_zscores, grid_stats = run_zscore_pipeline(df)

        grid_stats.to_parquet("grid_statistics.parquet", index=False)
        print("\nSaved: grid_statistics.parquet")

        print("\nFinal columns in dataframe:\n")
        for col in df_with_zscores.columns:
            print(col)

        print("\n====================================")
        print("PIPELINE FINISHED SUCCESSFULLY")
        print("====================================")
    else:
        print(f"File not found: {INPUT_FILE}")