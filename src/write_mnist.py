import numpy as np
import tensorflow as tf
import csv

COL_MAJOR_FILENAME = '/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist_images_col_major.bin'
ROW_MAJOR_FILENAME = '/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist_images_row_major.bin'
LABEL_FILENAME = '/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist_labels.bin'


def write_mnist_to_binary(shuffle=True):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    # Combine training and test datasets, then shuffle them
    all_images = np.concatenate([train_images, test_images], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)

    indices = np.arange(all_images.shape[0])

    if shuffle:
        np.random.shuffle(indices)

    all_images = all_images[indices]
    all_labels = all_labels[indices]
    all_images_flattened = all_images.reshape(all_images.shape[0], -1)

    col_major_filename = COL_MAJOR_FILENAME
    row_major_filename = ROW_MAJOR_FILENAME
    label_filename = LABEL_FILENAME

    all_images_flattened.tofile(row_major_filename)

    all_images_flattened.T.tofile(col_major_filename)

    all_labels.tofile(label_filename)

    return (row_major_filename, col_major_filename, label_filename), all_images, all_labels
