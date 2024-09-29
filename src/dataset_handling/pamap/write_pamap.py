import os
import pandas as pd
import polars as pl
import numpy as np
from tqdm import tqdm

PAMAP_DIR = "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/pamap/raw/"
PAMAP_HANDLED_DIR = "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/pamap/handled"
PAMAP_DF_NAME = "pamap_df.parquet"

PAMAP_F32_PATH = "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/pamap/handled/pamap_f32.bin"
PAMAP_F16_PATH = "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/pamap/handled/pamap_f16.bin"

PAMAP_LABELS_PATH = "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/pamap/handled/pamap_labels.bin"

PAMAP_SAMPLES_DIR_F32 = "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/pamap/handled/samples/f32"
PAMAP_SAMPLES_DIR_F16 = "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/pamap/handled/samples/f16"

PAMAP_SAMPLE_SIZES = [
    100_000,
    250_000,
    500_000,
    1_000_000,
    1_500_000,
    2_500_000,
    3_000_000,
]


PAMAP_SAMPLES_PATHS_F32 = {size: os.path.join(PAMAP_SAMPLES_DIR_F32, f"pamap_sample_n{size}_f32.bin") for size in
                           PAMAP_SAMPLE_SIZES}
PAMAP_SAMPLES_PATHS_F16 = {size: os.path.join(PAMAP_SAMPLES_DIR_F16, f"pamap_sample_n{size}_f16.bin") for size in
                           PAMAP_SAMPLE_SIZES}

PAMAP_SAMPLES_LABELS_PATHS_F32 = {size: os.path.join(PAMAP_SAMPLES_DIR_F32, f"pamap_sample_n{size}_f32_labels.bin") for size in
                           PAMAP_SAMPLE_SIZES}
PAMAP_SAMPLES_LABELS_PATHS_F16 = {size: os.path.join(PAMAP_SAMPLES_DIR_F16, f"pamap_sample_n{size}_f16_labels.bin") for size in
                           PAMAP_SAMPLE_SIZES}

PAMAP_SAMPLES_PATHS_DICT = {
    "f32": {"data": PAMAP_SAMPLES_PATHS_F32, "labels": PAMAP_SAMPLES_LABELS_PATHS_F32},
    "f16": {"data": PAMAP_SAMPLES_PATHS_F16, "labels": PAMAP_SAMPLES_LABELS_PATHS_F16}
}

PAMAP_PATHS_DICT = {
    "f32": PAMAP_F32_PATH,
    "f16": PAMAP_F16_PATH,
}


PAMAP_D = 51
PAMAP_N = 3850505


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
    df = df.drop(["0", "1", "2", "index"])  # drop timestamp, activity_id and heart rate

    np_df = df.to_numpy()
    np_labels = labels.to_numpy()

    np_df.astype(np.float32).tofile(PAMAP_F32_PATH)
    np_df.astype(np.float16).tofile(PAMAP_F16_PATH)

    np_labels.astype(np.int8).tofile(PAMAP_LABELS_PATH)


def write_samples(dtype=np.float32):
    pamap_array = np.fromfile(PAMAP_F32_PATH, dtype=np.float32)
    pamap_array = pamap_array.astype(np.float16)
    pamap_array = pamap_array.reshape(-1, PAMAP_D)

    pamap_labels = np.fromfile(PAMAP_LABELS_PATH, dtype=np.int8)

    if dtype == np.float32:
        samples_paths_dict = PAMAP_SAMPLES_PATHS_F32
        samples_labels_paths_dict = PAMAP_SAMPLES_LABELS_PATHS_F32

    elif dtype == np.float16:
        samples_paths_dict = PAMAP_SAMPLES_PATHS_F16
        samples_labels_paths_dict = PAMAP_SAMPLES_LABELS_PATHS_F16

    else:
        raise Exception("Invalid dtype")

    for size in tqdm(PAMAP_SAMPLE_SIZES, desc="Writing samples"):
        this_path = samples_paths_dict[size]
        sample_indices = np.random.choice(len(pamap_array), size=size, replace=False)

        sample = pamap_array[sample_indices]
        sample_labels = pamap_labels[sample_indices]

        sample.astype(dtype).tofile(this_path)
        sample_labels.astype(np.int8).tofile(samples_labels_paths_dict[size])


if __name__ == "__main__":
    write_samples(dtype=np.float16)
