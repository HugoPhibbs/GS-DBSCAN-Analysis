from sklearn.datasets import fetch_openml
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype('float32')

# Normalize each row (i.e., each sample) to have unit norm
X_normalized = normalize(X, norm='l2', axis=1)

# Compute cosine distances (DBSCAN uses distance, not similarity)
cosine_dist_matrix = cosine_distances(X_normalized)

# Fit DBSCAN using the precomputed cosine distance matrix
db = DBSCAN(eps=0.11, min_samples=10, metric='precomputed').fit(cosine_dist_matrix)

# Get the number of core points
num_core_points = len(db.core_sample_indices_)

print(f"Number of core points: {num_core_points}")
