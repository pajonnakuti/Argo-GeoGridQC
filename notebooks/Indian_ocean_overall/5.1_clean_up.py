import os
import hashlib

root_folder = r"D:\INCOIS\Agro_project\data\raw\nc_files"

def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

seen = {}
duplicates = []

for dirpath, _, filenames in os.walk(root_folder):
    for file in filenames:
        if file.endswith(".nc"):
            full_path = os.path.join(dirpath, file)

            h = file_hash(full_path)

            if h in seen:
                duplicates.append(full_path)
            else:
                seen[h] = full_path

# delete duplicates
for f in duplicates:
    os.remove(f)
    print("Deleted duplicate:", f)

print("Total duplicates removed:", len(duplicates))