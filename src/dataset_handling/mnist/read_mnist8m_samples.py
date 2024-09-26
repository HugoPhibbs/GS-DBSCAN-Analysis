import os
import numpy as np
import matplotlib.pyplot as plt

SAMPLES_DIR = "/data/mnist8m/samples"
REPO_DIR = "/home/hphi344/Documents/GS-DBSCAN-Analysis"

def read_samples(dtype="f32"):
    labels = []
    samples = []

    data_dir = f"{REPO_DIR}/{SAMPLES_DIR}/data/{dtype}"
    labels_dir = f"{REPO_DIR}/{SAMPLES_DIR}/labels/{dtype}"

    data_files = os.listdir(data_dir)
    label_files = os.listdir(labels_dir)

    print(data_files)
    print(label_files)

    for file in data_files:
        file_path = f"{data_dir}/{file}"
        if dtype == "f32":
            arr = np.fromfile(file_path, dtype=np.float32)
        else:
            arr = np.fromfile(file_path, dtype=np.float16)
        samples.append(arr)

    for file in label_files:
        file_path = f"{labels_dir}/{file}"
        arr = np.fromfile(file_path, dtype=np.uint8)
        labels.append(arr)

    return samples, labels


# samples, labels = read_samples()

f16_50k = np.fromfile("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist8m/samples/data/f16/mnist8m_f16_sample_50000.bin", dtype=np.float16)

f16_50k = f16_50k.reshape(50000, 784)

print(f16_50k.shape)


plt.imshow(f16_50k[0].reshape(28, 28), cmap="gray")
