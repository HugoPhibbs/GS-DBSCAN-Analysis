import os
import pandas as pd
import polars as pl
import numpy as np

PAMAP_DIR = "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/pamap/raw/"
PAMAP_HANDLED_DIR = "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/pamap/handled"
PAMAP_DF_NAME = "pamap_df.parquet"

PAMAP_F32_PATH = "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/pamap/handled/pamap_f32.bin"
PAMAP_F16_PATH = "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/pamap/handled/pamap_f16.bin"

PAMAP_LABELS_PATH = "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/pamap/handled/pamap_labels.bin"

def write_df():
    file_dirs = os.listdir(PAMAP_DIR)

    df_list = []

    for dir in file_dirs:
        this_dir = os.path.join(PAMAP_DIR, dir)
        pamap_files = os.listdir(this_dir)
        pamap_files = [os.path.join(this_dir, f) for f in pamap_files]

        for f in pamap_files:
            df = pd.read_csv(f, sep=" ", header=None)
            df_list.append(df)

    df = pd.concat(df_list)

    df.columns = [str(col_idx) for col_idx in range(len(df.columns))]

    df.to_parquet(os.path.join(PAMAP_HANDLED_DIR, PAMAP_DF_NAME), engine="fastparquet")

    print(df.head())

def write_arrays():
    df_path = os.path.join(PAMAP_HANDLED_DIR, PAMAP_DF_NAME)
    df = pl.read_parquet(df_path)
    # df = df.drop_nulls(?)

    labels = df["1"]
    df = df.drop(["0", "1", "2", "index"]) # drop timestamp, activity_id and heart rate

    np_df = df.to_numpy()
    np_labels = labels.to_numpy()

    np_df.astype(np.float32).tofile(PAMAP_F32_PATH)
    np_df.astype(np.float16).tofile(PAMAP_F16_PATH)

    np_labels.astype(np.int8).tofile(PAMAP_LABELS_PATH)






if __name__ == "__main__":
    write_arrays()
