import subprocess
import pandas as pd
import os
from dotenv import load_dotenv
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import write_mnist as wm

load_dotenv()


def run_gs_dbscan(datasetFilename, outFile, n, d, D, minPts, k, m, eps, alpha, distancesBatchSize, distanceMetric,
                  clusterBlockSize, clusterOnCpu=False, needToNormalize=True, print_cmd=False):
    run_cmd = [
        "../GS-DBSCAN-CPP/build-release/GS-DBSCAN",
        "--datasetFilename", datasetFilename,
        "--outFile", outFile,
        "--n", str(n),
        "--d", str(d),
        "--D", str(D),
        "--minPts", str(minPts),
        "--k", str(k),
        "--m", str(m),
        "--eps", str(eps),
        "--alpha", str(alpha),
        "--distancesBatchSize", str(distancesBatchSize),
        "--distanceMetric", distanceMetric,
        "--clusterBlockSize", str(clusterBlockSize),
        "--clusterOnCpu", str(int(clusterOnCpu)),
        "--needToNormalize", str(int(needToNormalize))
    ]

    print("Running GS-DBSCAN\n")

    if print_cmd:
        print(run_cmd)

    result = subprocess.run(run_cmd, capture_output=True, text=True)

    print("Standard Output:\n", result.stdout)

    if result.returncode == 0:
        print("Execution successful")
    else:
        print(f"Execution failed with return code {result.returncode}, error message: {result.stderr}")

    return result


def print_results(results_df, nmi):
    times = results_df['times'][0]
    keys = list(times.keys())
    values = list(times.values())
    print("Times...")
    for i in range(len(keys)):
        print(f"{keys[i]}: {(values[i]) / 1000000:.2f}")

    print("\n")

    print("Number of clusters: ", results_df['numClusters'][0])

    print("\n")

    print("NMI: ", nmi)

    print("\n")

    # print("Number of core points: ", find_num_core_points(results_df['typeLabels'][0]))


def find_num_core_points(typeLabels):
    count = np.sum(np.array(typeLabels) == 1)
    return count


def read_results(results_file):
    return pd.read_json(results_file)


def calculate_nmi(labels_true, labels_pred):
    return normalized_mutual_info_score(labels_true, labels_pred)


# (_, col_major_filename, _), all_images, true_labels = wm.write_mnist_to_binary(shuffle=True)


true_labels = np.fromfile("./data/mnist_labels.bin", dtype=np.uint8)

col_major_filename = wm.COL_MAJOR_FILENAME

run_gs_dbscan(col_major_filename,
              outFile="results.json",
              n=70000,
              d=784,
              D=1024,
              minPts=50,
              k=2,
              m=2000,
              eps=0.11,
              alpha=1.2,
              distancesBatchSize=-1,
              distanceMetric="COSINE",
              clusterBlockSize=256,
              clusterOnCpu=True,
              needToNormalize=True,
              print_cmd=True)

results_df = read_results("results.json")

nmi = calculate_nmi(true_labels, results_df['clusterLabels'][0])

print_results(results_df, nmi)
