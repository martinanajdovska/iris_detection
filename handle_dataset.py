import pandas as pd
import os
from sklearn.model_selection import train_test_split

df = pd.read_csv("openEDS.csv")

# ignore images with closed eye
df = df[df['check'] != 0]
df.to_csv("openEDS-edited.csv")

df = pd.read_csv("openEDS-edited.csv")
df['folder'] = df['png_path'].apply(lambda x: x.split('/')[2])


def select_random_images(group):
    return group.sample(n=50, random_state=42)


df_random = df.groupby('folder', group_keys=False).apply(select_random_images).reset_index(drop=True)
df_random = df_random.drop(df.columns[[0, -1]], axis=1)
df_random.to_csv("selected-images.csv",index=False)

df = pd.read_csv("selected-images.csv")
df['png_path'] = df['png_path'].apply(lambda x: x.replace('\\', '/'))
existing_files = set(df['png_path'].values)

root_folder = "./openEDS"

for root, dirs, files in os.walk(root_folder):
    for file in files:
        file_path = os.path.join(root, file).replace('\\', '/')
        file_check = file_path.replace("npy","png")
        if file_check not in existing_files:
            os.remove(file_path)

