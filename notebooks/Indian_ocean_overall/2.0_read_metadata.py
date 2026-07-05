import pandas as pd

df = pd.read_csv(r"D:\INCOIS\Agro_project\data\raw\ar_index_global_prof.txt", comment="#")
print(df.head())

date=df['date'].values
print(date)
longitude = df['longitude'].values
print(longitude)
latitude=df['latitude'].values
print(latitude)
institution=df['institution'].values
print(institution)
ocean=df['ocean'].values
print(ocean)

ocean_df=df[df['ocean']=='I']
print(ocean_df.head())
print(ocean_df['ocean'])


#append only new profiles
import os
import pandas as pd

master_file = r"D:\INCOIS\Agro_project\data\processed\indian_ocean_master.csv"

if not os.path.exists(master_file):

    ocean_df.to_csv(master_file, index=False)

    print("Master file created")
    print(f"Profiles in master: {len(ocean_df)}")

else:

    master_df = pd.read_csv(master_file)

    new_profiles = ocean_df[
        ~ocean_df["file"].isin(master_df["file"])
    ]

    updated_master = pd.concat(
        [master_df, new_profiles],
        ignore_index=True
    )

    updated_master.to_csv(master_file, index=False)

    print(f"Profiles already in master: {len(master_df)}")
    print(f"New profiles added today: {len(new_profiles)}")
    print(f"Updated master total: {len(updated_master)}")