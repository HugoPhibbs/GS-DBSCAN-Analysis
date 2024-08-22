import numpy as np
import tensorflow as tf
import csv

# 1. Load both the training and test MNIST datasets
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

# Combine training and test datasets, then shuffle them
all_images = np.concatenate([train_images, test_images], axis=0)
all_labels = np.concatenate([train_labels, test_labels], axis=0)

indices = np.arange(all_images.shape[0])
np.random.shuffle(indices)
all_images = all_images[indices]
all_labels = all_labels[indices]

# Reshape the images to 2D arrays (each image is 28x28, flatten it to a 784-dimensional vector)
all_images_flattened = all_images.reshape(all_images.shape[0], -1)

# # 2. Write the flattened images to CSV files in row-major order (default)
# with open('../data/mnist_images_row_major.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     flat = all_images_flattened.flatten()
#     for i in range(len(flat)):
#         val = flat[i]
#         writer.writerow([val])
#
# # 3. Write the flattened images to CSV files in column-major order
# with open('../data/mnist_images_col_major.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     flat = all_images_flattened.T.flatten()
#     for i in range(len(flat)):
#         val = flat[i]
#         writer.writerow([val])
#
# # 4. Write the labels to a separate CSV file (as a single column)
# with open('../data/mnist_labels.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     for label in all_labels:
#         writer.writerow([label])
#
# print("MNIST data has been written to 'mnist_images_row_major.csv', 'mnist_images_col_major.csv', and 'mnist_labels.csv'.")

# # 2. Write the flattened images to a binary file in row-major order (default)
all_images_flattened.tofile('../data/mnist_images_row_major.bin')

# 3. Write the flattened images to a binary file in column-major order
all_images_flattened.T.tofile('../data/mnist_images_col_major.bin')

# # 4. Write the labels to a separate binary file
#     all_labels.tofile(file)
#
#