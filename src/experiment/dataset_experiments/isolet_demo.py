import time
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt

X = np.fromfile("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/isolet/isolet_data_f16.bin", dtype=np.float16)
X = X.reshape(-1, 617)

y_true = np.fromfile("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/isolet/isolet_labels_f16.bin", dtype=np.uint8)

# Assume X is the flattened ISOLET dataset and y_true contains the true labels
# Normalizing the dataset by L2 norm across each row
X_normalized = normalize(X, norm='l2', axis=1)

# Parameters for DBSCAN
min_samples = 50  # Keep this constant
eps_values = np.linspace(0.01, 0.2, 20)  # Range of epsilon values to test

nmi_scores = []


start = time.time()

dbscan = DBSCAN(eps=0.12, min_samples=min_samples, metric='cosine', n_jobs=-1)
labels = dbscan.fit_predict(X_normalized)

end = time.time()

print(f"Time taken: {end - start} seconds")

nmi = normalized_mutual_info_score(y_true, labels, average_method='arithmetic')

print(f"NMI: {nmi}")


# # Run DBSCAN for each epsilon value and compute NMI
# for eps in eps_values:
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
#     labels = dbscan.fit_predict(X_normalized)
    
#     # Calculate NMI using the true labels
#     nmi = normalized_mutual_info_score(y_true, labels, average_method='arithmetic')
#     nmi_scores.append(nmi)

# # Plotting the NMI scores against epsilon values
# plt.plot(eps_values, nmi_scores, marker='o')
# plt.xlabel('Epsilon (eps)')
# plt.ylabel('Normalized Mutual Information (NMI)')
# plt.title('DBSCAN Performance on ISOLET Dataset (Cosine Distance)')
# plt.grid(True)
# plt.savefig('isolet_dbscan.png', dpi=300)
