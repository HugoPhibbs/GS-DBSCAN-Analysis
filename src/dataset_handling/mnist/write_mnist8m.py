import numpy as np
import src.experiment.sample_utils as su
import os
import shutil

REPO_DIR = "/home/hphi344/Documents/GS-DBSCAN-Analysis"

MNIST_8M_FILE_PATH = f"{REPO_DIR}/data/mnist8m/mnist8m_f32_X.bin"
MNIST_8M_LABELS_FILE_NAME = f"{REPO_DIR}/data/mnist8m/mnist8m_y_8100000_784.bin"
N_EXPERIMENT_VALUES = [70_000, 100_000, 150_000, 200_000, 300_000, 500_000, 1_000_000, 2_000_000, 3_000_000,
                       5_000_000, 8_000_000]

MNIST_8M_SAMPLES_DIR = "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist8m/samples"

MNIST_8M_SAMPLES_DICT = su.get_sample_dict("mnist8m", sample_n_vals=N_EXPERIMENT_VALUES,
                                           sample_dir=MNIST_8M_SAMPLES_DIR)

MNIST_8M_SIZE = 8_100_000


def write_8m_txt_to_binary():
    mnist8m = np.loadtxt(f"{REPO_DIR}/data/mnist8m/mnist8m_X.txt")
    mnist8m /= 255.0
    mnist8m = mnist8m.astype(np.float32)
    mnist8m.tofile(MNIST_8M_FILE_PATH)
    mnist8m_labels = np.loadtxt(f"{REPO_DIR}/data/mnist8m/mnist8m_y_8100000_784.txt")
    mnist8m_labels = mnist8m_labels.astype(np.uint8)
    mnist8m_labels.tofile(MNIST_8M_LABELS_FILE_NAME)


# TODO, note that the below code is now invalid, it writes to a different directory structure than that expected by the sample utils module.
# But I'm keeping it here anyway - the samples are already written and I ceebs changing it.

def generate_8m_samples(type_str):
    mnist_8m = np.fromfile(MNIST_8M_FILE_PATH, dtype=np.float32)
    mnist_8m = mnist_8m.reshape((MNIST_8M_SIZE, 784))

    mnist_8m_labels = np.fromfile(MNIST_8M_LABELS_FILE_NAME, dtype=np.uint8)

    print("Read mnist8m binary")

    for n in N_EXPERIMENT_VALUES:
        print(f"curr n: {n}")
        sample_filename = f"{REPO_DIR}/data/mnist8m/samples/{type_str}/mnist8m_sample_n{n}_{type_str}.bin"
        sample_labels_filename = f"{REPO_DIR}/data/mnist8m/samples/{type_str}/mnist8m_sample_n{n}_{type_str}_labels.bin"

        sample_indices = np.random.choice(MNIST_8M_SIZE, n, replace=False)
        mnist_8m_sample = mnist_8m[sample_indices]
        mnist_8m_sample_labels = mnist_8m_labels[sample_indices]

        mnist_8m_sample = mnist_8m_sample.astype(np.float32 if type_str == "f32" else np.float16)

        mnist_8m_sample.tofile(sample_filename)
        mnist_8m_sample_labels.tofile(sample_labels_filename)

#
# def get_8m_sample_filenames(type_str=TYPE_STR):
#     sample_filenames_list = []
#     sample_labels_filenames_list = []
#
#     for n in N_EXPERIMENT_VALUES:
#         sample_filename = f"{REPO_DIR}/data/mnist8m/samples/data/mnist8m_{type_str}_sample_{n}.bin"
#         sample_labels_filename = f"{REPO_DIR}/data/mnist8m/samples/labels/mnist8m_sample_labels_{n}.bin"
#
#         sample_filenames_list.append(sample_filename)
#         sample_labels_filenames_list.append(sample_labels_filename)
#
#     return sample_filenames_list, sample_labels_filenames_list


# generate_8m_samples(type_str="f16")
# generate_8m_samples(type_str="f32")
# write_8m_txt_to_binary()

if __name__ == '__main__':
    generate_8m_samples("f16")
    pass
