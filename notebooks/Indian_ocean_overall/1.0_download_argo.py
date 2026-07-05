##Grid-based Machine Learning Quality Control Framework for Argo Oceanographic Observations 

import requests

url = "https://data-argo.ifremer.fr/ar_index_global_prof.txt"

r = requests.get(url, stream=True)

with open(
    r"D:\INCOIS\Agro_project\data\raw\ar_index_global_prof.txt",
    "wb"
) as f:
    for chunk in r.iter_content(chunk_size=1024*1024):
        f.write(chunk)

print("Download Complete")

