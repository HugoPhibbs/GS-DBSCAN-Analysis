# import time
# import numpy as np
# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import normalize
# from sklearn.metrics import normalized_mutual_info_score
#
# X = np.fromfile("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist/mnist_images_row_major.bin", dtype=np.float32)
# X = X.reshape(-1, 784)
#
# y_true = np.fromfile("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist/mnist_labels.bin", dtype=np.uint8)
#
# X_normalized = normalize(X, norm='l2', axis=1)
#
# # Parameters for DBSCAN
# min_samples = 50
#
# start = time.time()
#
# dbscan = DBSCAN(eps=0.11, min_samples=min_samples, metric='cosine', algorithm="brute", n_jobs=-1)
# labels = dbscan.fit_predict(X_normalized)
#
# end = time.time()
#
# print(f"Time taken: {end - start} seconds")
#
# nmi = normalized_mutual_info_score(y_true, labels, average_method='arithmetic')
#
# print(f"NMI: {nmi}")


import numpy as np
import math
import timeit
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score


def run_scikit_dbscan():

    X = np.fromfile("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist/mnist_images_row_major.bin", dtype=np.float32)
    X = X.reshape(-1, 784)
    y = np.fromfile("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist/mnist_labels.bin", dtype=np.uint8)
    print("Finish reading data for dbscan")

    eps = 0.11
    # eps = math.sqrt(2 * eps)
    # eps = 12000
    min_samples = 50
    t1 = timeit.default_timer()
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', n_jobs=-1)
    y_pred = db.fit_predict(X)
    t2 = timeit.default_timer()
    print('DBSCAN Time: {}'.format(t2 - t1))

    np.savetxt('dbscan_L0_Eps_0003_MinPts_50', y_pred, delimiter=',')

    print("Number of core points found by Euclidean DBSCAN: {}".format(len(db.core_sample_indices_)))
    print("Acc: Adj. Rand Index Score: %f." % adjusted_rand_score(y_pred, y))
    print("Acc: Adj. Mutual Info Score: %f." % adjusted_mutual_info_score(y_pred, y))
    print("Acc: NMI %f." % normalized_mutual_info_score(y_pred, y))

if __name__ == '__main__':
    run_scikit_dbscan()
