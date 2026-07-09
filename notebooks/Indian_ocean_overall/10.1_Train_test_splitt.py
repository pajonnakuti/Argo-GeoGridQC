"""
split_gridwise_chunked_resume_parallel.py
==========================================
Parallel + resume version of split_gridwise_chunked.py.

Since PASS 1 (partitioning by grid_id) already completed, this script
SKIPS pass 1 entirely and goes straight to PASS 2 (train/test split per
grid), picking up only the grids that haven't been written yet -- and
does the per-grid work across multiple CPU cores.

Resume logic (unchanged from the serial resume script):
  - Before processing a grid_id, check if both
    train_<grid_id>.csv and test_<grid_id>.csv already exist in OUTPUT_DIR.
  - If both exist, skip it (already done from a previous run).
  - Otherwise, process it (in a worker process).

Parallelism:
  - Each grid is independent (read parquet -> split -> write 2 csvs), so
    grids are farmed out to a process pool via
    concurrent.futures.ProcessPoolExecutor.
  - N_WORKERS defaults to os.cpu_count() (all cores). Lower it if you
    want to leave headroom for other work on the machine.
  - Each worker does its own already_done() check right before writing,
    so re-running this script concurrently with itself, or re-running it
    after a partial run, is still safe.

Progress is printed one line per grid as results come back (flushed
immediately), so you can tail the log.

IMPORTANT: This script does NOT delete PARTITION_DIR at the end, in case
you need to resume again. Delete it manually once you've confirmed all
grids are written.

Run:
    python split_gridwise_chunked_resume_parallel.py
    python split_gridwise_chunked_resume_parallel.py --workers 8
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ======================= CONFIG =======================
PARTITION_DIR  = "/home/incois/PAJO/pplWorks/geogrid/Indian_ocean/_grid_partitions_tmp"
OUTPUT_DIR     = "/home/incois/PAJO/pplWorks/geogrid/Indian_ocean/grid_splits"

TEST_SIZE            = 0.2      # 80/20 train/test split, per grid
RANDOM_STATE         = 42       # base seed; each grid gets a derived, distinct seed
MIN_ROWS_PER_GRID    = 10       # skip grids too small to split meaningfully
N_WORKERS_DEFAULT    = os.cpu_count() or 4
# ========================================================


def already_done(grid_id: str) -> bool:
    train_path = os.path.join(OUTPUT_DIR, f"train_{grid_id}.csv")
    test_path = os.path.join(OUTPUT_DIR, f"test_{grid_id}.csv")
    return os.path.exists(train_path) and os.path.exists(test_path)


def _seed_for_grid(grid_id: str) -> int:
    # Deterministic, distinct seed per grid, derived from the base seed
    # so results are reproducible regardless of processing order.
    h = abs(hash((RANDOM_STATE, grid_id))) % (2**32 - 1)
    return h


def process_one_grid(args):
    """
    Runs in a worker process. Returns a dict describing what happened,
    so the main process can print progress and build the skip log.
    """
    i, total, grid_id, grid_path = args

    # Re-check here too: another run / worker may have written it already.
    if already_done(grid_id):
        return {"i": i, "total": total, "grid_id": grid_id, "status": "already_done"}

    df = pq.read_table(grid_path).to_pandas()
    n = len(df)

    if n < MIN_ROWS_PER_GRID:
        return {"i": i, "total": total, "grid_id": grid_id, "status": "too_few_rows", "n_rows": n}

    rng = np.random.RandomState(_seed_for_grid(grid_id))
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * TEST_SIZE)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    if len(train_idx) == 0:
        return {"i": i, "total": total, "grid_id": grid_id, "status": "empty_train", "n_rows": n}

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    train_path = os.path.join(OUTPUT_DIR, f"train_{grid_id}.csv")
    test_path = os.path.join(OUTPUT_DIR, f"test_{grid_id}.csv")

    # Write to temp names then rename, so a killed process never leaves a
    # half-written train_/test_ file that already_done() would trust.
    tmp_train = train_path + ".tmp"
    tmp_test = test_path + ".tmp"
    train_df.to_csv(tmp_train, index=False)
    test_df.to_csv(tmp_test, index=False)
    os.replace(tmp_train, train_path)
    os.replace(tmp_test, test_path)

    return {
        "i": i, "total": total, "grid_id": grid_id, "status": "done",
        "n_train": len(train_df), "n_test": len(test_df),
    }


def pass2_split_each_grid_resume_parallel(n_workers: int):
    if not os.path.isdir(PARTITION_DIR):
        print(f"ERROR: partition dir not found: {PARTITION_DIR}")
        print("Pass 1 output is missing -- can't resume pass 2 without it.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    grid_dirs = sorted(
        d for d in os.listdir(PARTITION_DIR)
        if d.startswith("grid_id=") and os.path.isdir(os.path.join(PARTITION_DIR, d))
    )
    total = len(grid_dirs)
    print(f"PASS 2 (resume, parallel x{n_workers}): found {total} grid partition(s) in {PARTITION_DIR}")

    # Preserve any skipped_grids.csv from a prior run so we don't lose that log
    skipped_path = os.path.join(OUTPUT_DIR, "skipped_grids.csv")
    skipped = []
    if os.path.exists(skipped_path):
        try:
            skipped = pd.read_csv(skipped_path).to_dict("records")
        except Exception:
            skipped = []

    # Cheap pre-filter in the main process: skip already-done grids up
    # front so we don't even hand them to a worker / spin up a process.
    tasks = []
    already_written = 0
    for i, d in enumerate(grid_dirs, 1):
        grid_id = d.split("=", 1)[1]
        if already_done(grid_id):
            already_written += 1
            print(f"[{i}/{total}] grid_id={grid_id}: SKIP (already written)", flush=True)
            continue
        grid_path = os.path.join(PARTITION_DIR, d)
        tasks.append((i, total, grid_id, grid_path))

    newly_written = 0

    if tasks:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(process_one_grid, t): t for t in tasks}
            for fut in as_completed(futures):
                r = fut.result()
                status = r["status"]
                i, tot, grid_id = r["i"], r["total"], r["grid_id"]

                if status == "already_done":
                    already_written += 1
                    print(f"[{i}/{tot}] grid_id={grid_id}: SKIP (already written)", flush=True)
                elif status == "too_few_rows":
                    skipped.append({"grid_id": grid_id, "n_rows": r["n_rows"], "reason": "too_few_rows"})
                    print(f"[{i}/{tot}] grid_id={grid_id}: SKIP (only {r['n_rows']} rows < {MIN_ROWS_PER_GRID})", flush=True)
                elif status == "empty_train":
                    skipped.append({"grid_id": grid_id, "n_rows": r["n_rows"], "reason": "empty_train_after_split"})
                    print(f"[{i}/{tot}] grid_id={grid_id}: SKIP (empty train after split)", flush=True)
                elif status == "done":
                    newly_written += 1
                    print(f"[{i}/{tot}] grid_id={grid_id}: DONE (train={r['n_train']}, test={r['n_test']})", flush=True)

    print(
        f"\nPASS 2 (resume, parallel) done. "
        f"{already_written} already had files, {newly_written} newly written, "
        f"{len(skipped)} total skipped."
    )

    if skipped:
        pd.DataFrame(skipped).drop_duplicates(subset=["grid_id"]).to_csv(skipped_path, index=False)
        print(f"Skipped grids logged to: {skipped_path}")


def main():
    parser = argparse.ArgumentParser(description="Parallel resume pass-2 grid splitter")
    parser.add_argument(
        "--workers", type=int, default=N_WORKERS_DEFAULT,
        help=f"Number of worker processes (default: {N_WORKERS_DEFAULT}, i.e. all CPU cores)",
    )
    args = parser.parse_args()

    pass2_split_each_grid_resume_parallel(args.workers)
    print(f"\nAll done (resume, parallel run). train_<grid>.csv / test_<grid>.csv files are in: {OUTPUT_DIR}")
    print("Once you've confirmed everything looks right, you can manually delete:")
    print(f"  {PARTITION_DIR}")


if __name__ == "__main__":
    main()