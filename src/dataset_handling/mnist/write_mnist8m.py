import numpy as np

REPO_DIR = "/home/hphi344/Documents/GS-DBSCAN-Analysis"

TYPE_STR = "f32"

ROW_MAJOR_8M_FILE_NAME = f"{REPO_DIR}/data/mnist8m/mnist8m_f32_X.bin"
ROW_MAJOR_8M_LABELS_FILE_NAME = f"{REPO_DIR}/data/mnist8m/mnist8m_y_8100000_784.bin"
N_EXPERIMENT_VALUES = [50_000, 70_000, 100_000, 150_000, 200_000, 300_000, 500_000, 1_000_000, 2_000_000, 3_000_000,
                       5_000_000, 8_000_000]

MNIST_8M_SIZE = 8_100_000


def write_8m_txt_to_binary():
    mnist8m = np.loadtxt(f"{REPO_DIR}/data/mnist8m/mnist8m_X.txt")
    mnist8m /= 255.0
    mnist8m = mnist8m.astype(np.float32)
    mnist8m.tofile(ROW_MAJOR_8M_FILE_NAME)
    mnist8m_labels = np.loadtxt(f"{REPO_DIR}/data/mnist8m/mnist8m_y_8100000_784.txt")
    mnist8m_labels = mnist8m_labels.astype(np.uint8)
    mnist8m_labels.tofile(ROW_MAJOR_8M_LABELS_FILE_NAME)


def generate_8m_samples(type_str=TYPE_STR):
    mnist_8m = np.fromfile(ROW_MAJOR_8M_FILE_NAME, dtype=np.float32)
    mnist_8m = mnist_8m.reshape((MNIST_8M_SIZE, 784))

    mnist_8m_labels = np.fromfile(ROW_MAJOR_8M_LABELS_FILE_NAME, dtype=np.uint8)

    print("Read mnist8m binary")

    for n in N_EXPERIMENT_VALUES:
        print(f"curr n: {n}")
        sample_filename = f"{REPO_DIR}/data/mnist8m/samples/data/{type_str}/mnist8m_{type_str}_sample_{n}.bin"
        sample_labels_filename = f"{REPO_DIR}/data/mnist8m/samples/labels/{type_str}/mnist8m_for{type_str}_sample_labels_{n}.bin"

        sample_indices = np.random.choice(MNIST_8M_SIZE, n, replace=False)
        mnist_8m_sample = mnist_8m[sample_indices]
        mnist_8m_sample_labels = mnist_8m_labels[sample_indices]

        mnist_8m_sample = mnist_8m_sample.astype(np.float32 if type_str == "f32" else np.float16)

        mnist_8m_sample.tofile(sample_filename)
        mnist_8m_sample_labels.tofile(sample_labels_filename)


def get_8m_sample_filenames(type_str=TYPE_STR):
    sample_filenames_list = []
    sample_labels_filenames_list = []

    for n in N_EXPERIMENT_VALUES:
        sample_filename = f"{REPO_DIR}/data/mnist8m/samples/data/mnist8m_{type_str}_sample_{n}.bin"
        sample_labels_filename = f"{REPO_DIR}/data/mnist8m/samples/labels/mnist8m_sample_labels_{n}.bin"

        sample_filenames_list.append(sample_filename)
        sample_labels_filenames_list.append(sample_labels_filename)

    return sample_filenames_list, sample_labels_filenames_list


# generate_8m_samples(type_str="f16")
# generate_8m_samples(type_str="f32")
# write_8m_txt_to_binary()
