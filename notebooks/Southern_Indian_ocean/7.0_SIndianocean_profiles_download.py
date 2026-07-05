import pandas as pd
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, local as thread_local
import time

# ====================================
# FILE PATHS
# ====================================

CSV_FILE    = r"D:\INCOIS\Agro_project\data\processed\profiles_with_grid.csv"
ROOT_FOLDER = r"D:\INCOIS\Agro_project\data\raw\southern_indian_ocean_nc_files"
DONE_LOG    = r"D:\INCOIS\Agro_project\data\southern_indian_ocean_download_log.txt"
FAILED_LOG  = r"D:\INCOIS\Agro_project\data\southern_indian_ocean_failed_log.txt"
BASE_URL    = "https://data-argo.ifremer.fr/dac/"

# ====================================
# TUNING
# ====================================

MAX_WORKERS = 100
CHUNK_SIZE  = 4 * 1024 * 1024
TIMEOUT     = 30

# ====================================
# SOUTHERN INDIAN OCEAN GRIDS
# ====================================
southern_indian_ocean_grids = [
    151,
    165, 166, 167, 168, 169, 170, 171,
    184, 185, 187, 188, 189, 190, 191,
    204, 205, 206, 207, 208, 209, 210, 211,
    223, 224, 225, 226, 227, 228, 229, 230, 231,
    243, 244, 245, 246, 247, 248, 249, 250, 251,
    261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271,
    281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291,
    301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311,
    321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331,
    341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351,
    361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371,
    381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391,
    
]

# ====================================
# READ + FILTER
# ====================================

df = pd.read_csv(CSV_FILE, usecols=["Grid_ID", "file"])
df = df[df["Grid_ID"].isin(set(southern_indian_ocean_grids))]
df = df[df["Grid_ID"] != -1].reset_index(drop=True)

print(f"\nTotal Profiles : {len(df)}")
print(f"Total Grids    : {df['Grid_ID'].nunique()}")

# ====================================
# PRE-CREATE FOLDERS
# ====================================

os.makedirs(ROOT_FOLDER, exist_ok=True)
for gid in df["Grid_ID"].unique():
    os.makedirs(os.path.join(ROOT_FOLDER, f"Grid_{gid}"), exist_ok=True)

# ====================================
# LOAD DONE LOG
# ====================================

if os.path.exists(DONE_LOG):
    with open(DONE_LOG, "r") as f:
        done_set = set(line.strip() for line in f if line.strip())
else:
    done_set = set()

print(f"Already downloaded : {len(done_set)} files")

# ====================================
# BUILD WORK LIST
# ====================================

def already_on_disk(grid_id, file_path):
    """Check if file exists on disk under either R or D prefix."""
    folder   = os.path.join(ROOT_FOLDER, f"Grid_{grid_id}")
    filename = os.path.basename(file_path)

    if os.path.exists(os.path.join(folder, filename)):
        return True

    if filename.startswith("R"):
        alt = "D" + filename[1:]
    elif filename.startswith("D"):
        alt = "R" + filename[1:]
    else:
        alt = None

    if alt and os.path.exists(os.path.join(folder, alt)):
        return True

    return False

records = [
    (int(row.Grid_ID), row.file)
    for row in df.itertuples(index=False)
    if row.file not in done_set
    and not already_on_disk(int(row.Grid_ID), row.file)
]

print(f"Files to download  : {len(records)}")
print(f"Already skipped    : {len(df) - len(records)}\n")

if not records:
    print("Nothing to download. All files already present.")
    raise SystemExit(0)

# ====================================
# THREAD-SAFE LOGGERS
# ====================================

done_lock   = Lock()
failed_lock = Lock()

def log_done(file_path: str):
    with done_lock:
        with open(DONE_LOG, "a") as f:
            f.write(file_path + "\n")

def log_failed(file_path: str, reason: str):
    with failed_lock:
        with open(FAILED_LOG, "a") as f:
            f.write(f"{reason}|{file_path}\n")

# ====================================
# PER-THREAD SESSION
# ====================================

_local = thread_local()

def get_session() -> requests.Session:
    if not hasattr(_local, "session"):
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=4,
            pool_maxsize=8,
            max_retries=0
        )
        s.mount("https://", adapter)
        _local.session = s
    return _local.session

# ====================================
# SMART DOWNLOAD
# ====================================

def try_download(session, url, save_path) -> bool:
    """Attempt one download with one retry on network error."""
    tmp_path = save_path + ".tmp"
    for attempt in range(2):
        try:
            with session.get(url, stream=True, timeout=TIMEOUT) as r:
                if r.status_code != 200:
                    return False        # 404 / 403 etc — don't retry
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
            os.replace(tmp_path, save_path)
            return True
        except Exception:
            time.sleep(0.5 * attempt)  # 0s on first fail, 0.5s before retry
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
    return False

def download_profile(args):
    grid_id, file_path = args

    folder   = os.path.join(ROOT_FOLDER, f"Grid_{grid_id}")
    filename = os.path.basename(file_path)
    dir_part = os.path.dirname(file_path)
    session  = get_session()

    # --- Attempt 1: original filename ---
    url       = BASE_URL + file_path
    save_path = os.path.join(folder, filename)

    if try_download(session, url, save_path):
        log_done(file_path)
        return "downloaded"

    # --- Attempt 2: swap R ↔ D prefix ---
    if filename.startswith("R"):
        alt_filename = "D" + filename[1:]
    elif filename.startswith("D"):
        alt_filename = "R" + filename[1:]
    else:
        alt_filename = None

    if alt_filename:
        alt_file_path = dir_part + "/" + alt_filename
        alt_url       = BASE_URL + alt_file_path
        alt_save_path = os.path.join(folder, alt_filename)

        if try_download(session, alt_url, alt_save_path):
            log_done(file_path)
            return "downloaded_alt"

    # --- Both failed ---
    log_failed(file_path, "404_both")
    return "failed"

# ====================================
# PARALLEL DOWNLOAD
# ====================================

counters   = {"downloaded": 0, "downloaded_alt": 0, "failed": 0}
total      = len(records)
start_time = time.time()

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

    futures = {executor.submit(download_profile, rec): rec for rec in records}

    for future in as_completed(futures):
        result = future.result()
        counters[result] = counters.get(result, 0) + 1

        done_count = sum(counters.values())
        elapsed    = time.time() - start_time
        rate       = done_count / elapsed if elapsed > 0 else 0
        remaining  = (total - done_count) / rate if rate > 0 else 0
        ok         = counters["downloaded"] + counters["downloaded_alt"]

        print(
            f"\r  {done_count}/{total} | "
            f"OK: {ok} (alt: {counters['downloaded_alt']}) | "
            f"Failed: {counters['failed']} | "
            f"Rate: {rate:.1f} f/s | "
            f"ETA: {remaining/60:.0f} min   ",
            end="", flush=True
        )

print()

# ====================================
# SUMMARY
# ====================================

elapsed = time.time() - start_time
ok      = counters["downloaded"] + counters["downloaded_alt"]

print("\n========== SUMMARY ==========")
print(f"Downloaded (original) : {counters['downloaded']}")
print(f"Downloaded (R↔D swap) : {counters['downloaded_alt']}")
print(f"Total downloaded      : {ok}")
print(f"Failed (both 404)     : {counters['failed']}")
print(f"Total time            : {elapsed/60:.1f} min")
print(f"Rate                  : {ok/elapsed:.2f} files/sec")
if counters["failed"]:
    print(f"\nFailed paths saved to:\n{FAILED_LOG}")
    print("These files may have been deleted from the Argo server entirely.")
print("==============================")