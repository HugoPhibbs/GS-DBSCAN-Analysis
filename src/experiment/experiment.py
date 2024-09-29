import os.path
import subprocess
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import src.dataset_handling.mnist.write_mnist as wm
import src.dataset_handling.mnist.write_mnist8m as wm8

load_dotenv()

REPO_DIR = "/"

EXEC_PATH = "/home/hphi344/Documents/GS-DBSCAN-CPP/build-release/GS-DBSCAN"


@dataclass
class RunParams:
    d: int
    minPts: int
    k: int
    m: int
    eps: float
    alpha: float
    distancesBatchSize: int
    distanceMetric: str
    clusterBlockSize: int
    miniBatchSize: int
    ABatchSize: int
    BBatchSize: int
    normBatchSize: int
    sigmaEmbed: int = 1
    datasetDType: str = "f32"
    n: int = -1
    D: int = -1
    datasetFilename: str = None
    clusterOnCpu: bool = True
    needToNormalize: bool = True
    verbose: bool = True
    useBatchDbscan: bool = False
    useBatchABMatrices: bool = False
    useBatchNorm: bool = False
    timeIt: bool = True
    print_cmd: bool = True
    labels_filename: str = None

    def __str__(self):
        filename = os.path.splitext(os.path.basename(self.datasetFilename))[0]
        return f"{filename}_{self.n}_{self.d}_{self.D}_{self.minPts}_{self.k}_{self.m}_{self.eps}_{self.alpha}_{self.distancesBatchSize}_{self.distanceMetric}_{self.clusterBlockSize}_{self.clusterOnCpu}_{self.needToNormalize}_{self.verbose}_{self.useBatchDbscan}_{self.useBatchABMatrices}_{self.useBatchNorm}_{self.timeIt}_{self.print_cmd}"

    @property
    def outputFilename(self, base_dir="results", comments=""):
        return f"{base_dir}/results_{comments}_{str(self)}.json"


# def run_batch_experiments(distance_batch_sizes=None, n=70000, d=784, D=1024, minPts=50, k=5, m=50, eps=0.11, alpha=1.2,
#                           distanceMetric="COSINE", clusterBlockSize=256, clusterOnCpu=True, needToNormalize=True,
#                           print_cmd=True, datasetFilename=wm.ROW_MAJOR_FILENAME, write_to_pickle=False):
#     if distance_batch_sizes is None:
#         distance_batch_sizes = [10, 100, 250, 500, 1000, 2000, 5000]
#
#     # Now run the experiments
#
#     out_files = [results_file_name(n, d, D, minPts, k, m, eps, alpha, distancesBatchSize, distanceMetric,
#                                    clusterBlockSize, clusterOnCpu=False, needToNormalize=True) for distancesBatchSize in
#                  distance_batch_sizes]
#     results_df = None
#
#     sucessful_batch_sizes = []
#
#     for i in range(len(distance_batch_sizes)):
#         distancesBatchSize = distance_batch_sizes[i]
#         this_out_file = out_files[i]
#
#         try:
#             run_gs_dbscan(datasetFilename, this_out_file, n, d, D, minPts, k, m, eps, alpha, distancesBatchSize,
#                           distanceMetric,
#                           clusterBlockSize, clusterOnCpu, needToNormalize, print_cmd)
#
#         except Exception as e:
#             print(f"Failed to run experiment with distance batch size {distancesBatchSize}")
#             print(e)
#             print('Exiting Loop')
#             break
#
#         results_df = read_results(this_out_file, results_df)
#
#         sucessful_batch_sizes.append(distancesBatchSize)
#
#     results_df = add_nmi_to_results(results_df, get_mnist_labels())
#
#     results_df["batchSize"] = sucessful_batch_sizes
#
#     if write_to_pickle:
#         results_df.to_pickle(f"{REPO_DIR}/results/batch_experiments/results_df.pkl")
#
#     return results_df


def run_gs_dbscan(params: RunParams):
    run_cmd = [
        EXEC_PATH,
        "--datasetFilename", params.datasetFilename,
        "--outputFilename", params.outputFilename,
        "--n", str(params.n),
        "--d", str(params.d),
        "--D", str(params.D),
        "--minPts", str(params.minPts),
        "--k", str(params.k),
        "--m", str(params.m),
        "--eps", str(params.eps),
        "--alpha", str(params.alpha),
        "--distancesBatchSize", str(params.distancesBatchSize),
        "--distanceMetric", params.distanceMetric,
        "--clusterBlockSize", str(params.clusterBlockSize),
        "--datasetDType", str(params.datasetDType),
        "--miniBatchSize", str(params.miniBatchSize),
        "--ABatchSize", str(params.ABatchSize),
        "--BBatchSize", str(params.BBatchSize),
        "--normBatchSize", str(params.normBatchSize),
        "--sigmaEmbed", str(params.sigmaEmbed)
    ]

    if params.clusterOnCpu:
        run_cmd.append("--clusterOnCpu")

    if params.needToNormalize:
        run_cmd.append("--needToNormalize")

    if params.verbose:
        run_cmd.append("--verbose")

    if params.useBatchDbscan:
        run_cmd.append("--useBatchClustering")

    if params.useBatchABMatrices:
        run_cmd.append("--useBatchABMatrices")

    if params.useBatchNorm:
        run_cmd.append("--useBatchNorm")

    if params.timeIt:
        run_cmd.append("--timeIt")

    print("Running GS-DBSCAN\n")

    if params.print_cmd:
        print(" ".join(run_cmd))

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


def read_results(results_file):
    return pd.read_json(results_file)


def calculate_nmi(labels_true, labels_pred):
    return normalized_mutual_info_score(labels_true, labels_pred)


def add_nmi_to_results_labels_list(results_df, true_labels_list):
    nmi_vals = []

    for i in range(len(results_df)):
        nmi = calculate_nmi(true_labels_list[i], results_df['clusterLabels'].iloc[i])
        nmi_vals.append(nmi)

    results_df['nmi'] = nmi_vals

    return results_df


def add_nmi_to_results(results_df, true_labels):
    nmi_vals = []

    for i in range(len(results_df)):
        nmi = calculate_nmi(true_labels, results_df['clusterLabels'].iloc[i])
        nmi_vals.append(nmi)

    results_df['nmi'] = nmi_vals

    return results_df


def get_mnist_labels(file=f"{REPO_DIR}/data/mnist/mnist_labels.bin"):
    return np.fromfile(file, dtype=np.uint8)


def run_complete_sdbscan_pipeline(params: RunParams) -> pd.DataFrame | None:
    try:
        run_gs_dbscan(params)

    except Exception as e:
        print(f"Failed to run experiment")
        print(e)
        print('Exiting Loop')
        return None

    results_df = read_results(params.outputFilename)

    nmi = -1

    if params.labels_filename is not None:
        true_labels = get_mnist_labels(params.labels_filename)

        nmi = calculate_nmi(true_labels, results_df['clusterLabels'][0])

    print_results(results_df, nmi)

    return results_df

# run_n_size_experiments(
#     k=2,
#     m=2000
# )
