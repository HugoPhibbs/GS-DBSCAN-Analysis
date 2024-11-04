import numpy as np
from tqdm import tqdm
import src.experiment.sample_utils as su

PAMAP_DATA_DIR = f"/home/hphi344/Documents/GS-DBSCAN-Analysis/data/pamap"

PAMAP_DIR = f"{PAMAP_DATA_DIR}/raw/"
PAMAP_HANDLED_DIR = f"{PAMAP_DATA_DIR}/handled"
PAMAP_DF_NAME = "pamap_df.parquet"

PAMAP_F32_PATH = f"{PAMAP_DATA_DIR}/handled/pamap_f32.bin"
PAMAP_F16_PATH = f"{PAMAP_DATA_DIR}/handled/pamap_f16.bin"

PAMAP_LABELS_PATH = f"{PAMAP_DATA_DIR}/handled/pamap_labels.bin"

PAMAP_SAMPLES_DIR = f"{PAMAP_DATA_DIR}/handled/samples"

PAMAP_SAMPLE_SIZES = [
    50_000,
    100_000,
    200_000,
    300_000,
    400_000,
    600_000,
    800_000,
    1_000_000,
    1_200_000,
    1_400_000,
    1_600_000,
    1_770_131,
]

PAMAP_PATHS_DICT = {
    "f32": PAMAP_F32_PATH,
    "f16": PAMAP_F16_PATH,
}

PAMAP_SAMPLES_PATHS_DICT = su.get_sample_dict("pamap", PAMAP_SAMPLE_SIZES, PAMAP_SAMPLES_DIR)


PAMAP_DIM = 51
PAMAP_N = 1770131


def process_pamap2(dtype="f32"):
    arr = np.loadtxt(rf"{PAMAP_HANDLED_DIR}/pamap2_X_no_0", dtype=np.float64)

    arr.astype(np.float32 if dtype=="f32" else np.float16).tofile(PAMAP_F32_PATH if dtype=="f32" else PAMAP_F16_PATH)

    labels = np.loadtxt(rf"{PAMAP_HANDLED_DIR}/pamap2_y_no_0_1770131_51", dtype=np.float64)

    labels.astype(np.int8).tofile(PAMAP_LABELS_PATH)


def write_samples(dtype=np.float32, dtype_str="f32"):
    pamap_array = np.fromfile(PAMAP_F32_PATH, dtype=np.float32)
    pamap_array = pamap_array.astype(np.float16)
    pamap_array = pamap_array.reshape(-1, PAMAP_DIM)

    pamap_labels = np.fromfile(PAMAP_LABELS_PATH, dtype=np.int8)

    samples_paths_dict = PAMAP_SAMPLES_PATHS_DICT[dtype_str]["data"]
    samples_labels_paths_dict = PAMAP_SAMPLES_PATHS_DICT[dtype_str]["labels"]

    for size in tqdm(PAMAP_SAMPLE_SIZES, desc="Writing samples"):
        this_path = samples_paths_dict[size]
        sample_indices = np.random.choice(len(pamap_array), size=size, replace=False)

        sample = pamap_array[sample_indices]
        sample_labels = pamap_labels[sample_indices]

        sample.astype(dtype).tofile(this_path)
        sample_labels.astype(np.int8).tofile(samples_labels_paths_dict[size])


if __name__ == "__main__":
    process_pamap2("f16")
    write_samples(np.float16, "f16")
    print("Done!")