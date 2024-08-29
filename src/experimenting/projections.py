import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import normalize

# Load the MNIST dataset (both train and test combined)
mnist = fetch_openml('mnist_784')

# Combine train and test data
X = mnist['data']

# Cast to float before normalizing
X = X.astype(np.float64)

# Normalize using L2 norm
X_normalized = normalize(X, norm='l2', axis=1)

# Generate a random projection matrix of size (784, 1024)
D = 1024
random_projection_matrix = np.random.randn(784, D)

# Perform projections using matrix multiplication
projections = np.dot(X_normalized, random_projection_matrix)

# Normalize the projections by the square root of D
# projections /= np.sqrt(D)

# Print the first 10 elements and 10 columns of the projections
print(projections[:10, :10])
