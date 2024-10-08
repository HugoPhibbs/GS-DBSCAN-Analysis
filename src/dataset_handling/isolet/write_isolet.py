from ucimlrepo import fetch_ucirepo
import numpy as np

ISOLET_DIR = r"/home/hphi344/Documents/GS-DBSCAN-Analysis/data/isolet"

ISOLET_N = 7797
ISOLET_DIM = 617


def write_isolet(dtype="f16"):
    # fetch dataset
    isolet = fetch_ucirepo(id=54)

    # data (as pandas dataframes)
    X = isolet.data.features
    labels = isolet.data.targets

    # Write to binary arr with numpy
    labels_arr = labels["class"].to_numpy()
    X_arr = X.to_numpy()

    np_dtype = np.float32 if dtype == "f32" else np.float16

    # Write to binary files
    labels_arr.astype(np.uint8).tofile(f"{ISOLET_DIR}/isolet_labels_{dtype}.bin")
    X_arr.astype(np_dtype).tofile(f"{ISOLET_DIR}/isolet_data_{dtype}.bin")

    # metadata
    print(isolet.metadata)

    # variable information
    print(isolet.variables)


if __name__ == "__main__":
    write_isolet("f32")
