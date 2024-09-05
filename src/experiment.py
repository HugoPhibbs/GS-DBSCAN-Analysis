import subprocess
import pandas as pd
import os
from dotenv import load_dotenv
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import src.write_mnist as wm
import matplotlib.pyplot as plt

load_dotenv()

REPO_DIR = "/home/hphi344/Documents/GS-DBSCAN-Analysis"

def results_file_name(n, d, D, minPts, k, m, eps, alpha, distance_batch_size, distanceMetric,
                      clusterBlockSize, clusterOnCpu=False, needToNormalize=True):
    return f"{REPO_DIR}/results/batch_experiments/results_n{n}_d{d}_D{D}_mp{minPts}_k{k}_m{m}_e{eps}_a{alpha}_dbs{distance_batch_size}_dm{distanceMetric}_cbs{clusterBlockSize}_coc{int(clusterOnCpu)}_nn{int(needToNormalize)}.json"


def run_batch_experiments(distance_batch_sizes=None, n=70000, d=784, D=1024, minPts=50, k=5, m=50, eps=0.11, alpha=1.2, 
                          distanceMetric="COSINE", clusterBlockSize=256, clusterOnCpu=True, needToNormalize=True, 
                          print_cmd=True, datasetFilename=wm.COL_MAJOR_FILENAME, write_to_pickle=False):
    
    if distance_batch_sizes is None:
        distance_batch_sizes = [10, 100, 250, 500, 1000, 2000, 5000]

    # Now run the experiments

    out_files = [results_file_name(n, d, D, minPts, k, m, eps, alpha, distancesBatchSize, distanceMetric,
                                   clusterBlockSize, clusterOnCpu=False, needToNormalize=True) for distancesBatchSize in
                 distance_batch_sizes]
    results_df = None

    sucessful_batch_sizes = []

    for i in range(len(distance_batch_sizes)):
        distancesBatchSize = distance_batch_sizes[i]
        this_out_file = out_files[i]

        try:
            run_gs_dbscan(datasetFilename, this_out_file, n, d, D, minPts, k, m, eps, alpha, distancesBatchSize,
                      distanceMetric,
                      clusterBlockSize, clusterOnCpu, needToNormalize, print_cmd)
            
        except Exception as e:
            print(f"Failed to run experiment with distance batch size {distancesBatchSize}")
            print(e)
            print('Exiting Loop')
            break
            
        results_df = read_results(this_out_file, results_df)

        sucessful_batch_sizes.append(distancesBatchSize)

    results_df = add_nmi_to_results(results_df, get_mnist_labels())

    results_df["batchSize"] = sucessful_batch_sizes

    if write_to_pickle:
        results_df.to_pickle(f"{REPO_DIR}/results/batch_experiments/results_df.pkl")

    return results_df


def run_gs_dbscan(datasetFilename, outFile, n, d, D, minPts, k, m, eps, alpha, distancesBatchSize, distanceMetric,
                  clusterBlockSize, clusterOnCpu=False, needToNormalize=True, print_cmd=False):
    run_cmd = [
        f"{REPO_DIR}/../GS-DBSCAN-CPP/build-release/GS-DBSCAN",
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
        raise Exception(f"Execution failed with return code {result.returncode}, error message: {result.stderr}")

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


def read_results(results_file, curr_df=None):
    new_df = pd.read_json(results_file)
    if curr_df is None:
        return new_df
    return pd.concat([curr_df, new_df])


def calculate_nmi(labels_true, labels_pred):
    return normalized_mutual_info_score(labels_true, labels_pred)


def add_nmi_to_results(results_df, true_labels):
    nmi_vals = []

    for i in range(len(results_df)):
        nmi = calculate_nmi(true_labels, results_df['clusterLabels'].iloc[i])
        nmi_vals.append(nmi)

    results_df['nmi'] = nmi_vals

    return results_df


def get_mnist_labels():
    return np.fromfile(f"{REPO_DIR}/data/mnist_labels.bin", dtype=np.uint8)

def run_complete_sdbscan_pipeline(outFile="results.json", n=70000, d=784, D=1024, minPts=50, k=5, m=50, eps=0.11, alpha=1.2, 
                                  distancesBatchSize=-1, distanceMetric="COSINE", clusterBlockSize=256, clusterOnCpu=True, 
                                  needToNormalize=True, print_cmd=True, datasetFilename=wm.COL_MAJOR_FILENAME):
    # (_, col_major_filename, _), all_images, true_labels = wm.write_mnist_to_binary(shuffle=True)

    true_labels = get_mnist_labels()

    run_gs_dbscan(datasetFilename,
                  outFile=outFile,
                  n=n,
                  d=d,
                  D=D,
                  minPts=minPts,
                  k=k,
                  m=m,
                  eps=eps,
                  alpha=alpha,
                  distancesBatchSize=distancesBatchSize,
                  distanceMetric=distanceMetric,
                  clusterBlockSize=clusterBlockSize,
                  clusterOnCpu=clusterOnCpu,
                  needToNormalize=needToNormalize,
                  print_cmd=print_cmd)

    results_df = read_results(outFile)

    nmi = calculate_nmi(true_labels, results_df['clusterLabels'][0])

    print_results(results_df, nmi)