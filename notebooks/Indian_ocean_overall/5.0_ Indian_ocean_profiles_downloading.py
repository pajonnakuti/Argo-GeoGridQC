import requests
import os
import pandas as pd

done_log = r"D:\INCOIS\Agro_project\data\downloaded_log.txt"

if os.path.exists(done_log):
    with open(done_log, "r") as f:
        done_set = set(line.strip() for line in f)
else:
    done_set = set()

df = pd.read_csv(
    r"D:\INCOIS\Agro_project\data\processed\profiles_with_grid.csv"
)

df = df[
    (df["Grid_ID"] >= 201) &
    (df["Grid_ID"] <= 400)
]

df = df[df["Grid_ID"] != -1]

base_url = "https://data-argo.ifremer.fr/dac/"

root_folder = r"D:\INCOIS\Agro_project\data\raw\nc_files"

os.makedirs(root_folder, exist_ok=True)

downloaded = 0
skipped = 0
failed = 0

for _, row in df.iterrows():

    grid_id = int(row["Grid_ID"])      # ✅ FIX 1 (you missed this)
    file_path = row["file"]

    # resume check
    if file_path in done_set:
        skipped += 1
        continue

    url = base_url + file_path         # ✅ FIX 2 (you never defined url)

    grid_folder = os.path.join(
        root_folder,
        f"Grid_{grid_id}"
    )

    os.makedirs(grid_folder, exist_ok=True)

    filename = os.path.basename(file_path)

    save_path = os.path.join(grid_folder, filename)

    if os.path.exists(save_path):
        skipped += 1
        continue

    try:
        r = requests.get(url, timeout=60)

        if r.status_code == 200:

            with open(save_path, "wb") as f:
                f.write(r.content)

            downloaded += 1

            # log progress
            with open(done_log, "a") as f:
                f.write(file_path + "\n")

            print(f"Downloaded | Grid {grid_id} | {filename}")

        else:
            failed += 1
            print(f"Failed ({r.status_code}) | {filename}")

    except Exception as e:
        failed += 1
        print(f"Error | {filename}")
        print(e)

print("\n========== SUMMARY ==========")
print(f"Downloaded : {downloaded}")
print(f"Skipped    : {skipped}")
print(f"Failed     : {failed}")
print("============================")