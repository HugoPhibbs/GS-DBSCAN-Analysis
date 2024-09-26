import numpy as np

mnist_f16 = np.fromfile("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist/mnist_images_row_major_f16.bin", dtype=np.float16)

print(mnist_f16.shape)

mnist_f16 = mnist_f16.reshape(70_000, 28, 28)

print(mnist_f16.shape)

