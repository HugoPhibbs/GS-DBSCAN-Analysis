import numpy as np
import tensorflow as tf
import csv

MNIST_DATA_DIR = "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist"

COL_MAJOR_FILENAME = f'{MNIST_DATA_DIR}/mnist_images_col_major.bin'
ROW_MAJOR_FILENAME = f'{MNIST_DATA_DIR}/mnist_images_row_major.bin'

COL_MAJOR_FILENAME_F16 = f'{MNIST_DATA_DIR}/mnist_images_col_major_f16.bin'
ROW_MAJOR_FILENAME_F16 = f'{MNIST_DATA_DIR}/mnist_images_row_major_f16.bin'

LABEL_FILENAME = f'{MNIST_DATA_DIR}/mnist_labels.bin'
LABEL_FILENAME_F16 = f'{MNIST_DATA_DIR}/mnist_labels_f16.bin'


def write_mnist_to_binary(shuffle=True, dtype='float32'):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.astype(dtype)
    test_images = test_images.astype(dtype)

    # Combine training and test mnist8m, then shuffle them
    all_images = np.concatenate([train_images, test_images], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)

    indices = np.arange(all_images.shape[0])

    if shuffle:
        np.random.shuffle(indices)

    all_images = all_images[indices]
    all_labels = all_labels[indices]
    all_images_flattened = all_images.reshape(all_images.shape[0], -1)

    if dtype == 'float16':
        col_major_filename = COL_MAJOR_FILENAME_F16
        row_major_filename = ROW_MAJOR_FILENAME_F16
        label_filename = LABEL_FILENAME_F16
    else:
        col_major_filename = COL_MAJOR_FILENAME
        row_major_filename = ROW_MAJOR_FILENAME
        label_filename = LABEL_FILENAME

    all_images_flattened.tofile(row_major_filename)

    all_images_flattened.T.tofile(col_major_filename)

    all_labels.tofile(label_filename)

    return (row_major_filename, col_major_filename, label_filename), all_images, all_labels


write_mnist_to_binary(dtype='float16')
