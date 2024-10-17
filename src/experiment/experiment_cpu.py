import copy
import os
import subprocess
from dataclasses import dataclass
import pandas as pd

from src.experiment.experiment_utils import calculate_nmi

import numpy as np
import sDbscan

REPO_PATH = r"/home/hphi344/Documents/GS-DBSCAN-Analysis"
CPU_RESULTS_PATH = f"{REPO_PATH}/results/cpu_results"


@dataclass
class CpuRunParams:
    d: int
    eps: float
    minPts: int
    D: int
    k: int
    m: int
    numThreads: int
    dist: str
    n: int = -1
    datasetFilename: str = None
    verbose: bool = False
    output_file: str = "y_dbscan"
    file_type: str = "txt"
    labels_filename: str = None
    labels_file_type: str = "txt"


def run_cpu_sdbscan(run_params: CpuRunParams):
    if run_params.file_type == "txt":
        dataset = np.loadtxt(run_params.file_path)
    else:
        # Assume binary
        dataset = np.fromfile(run_params.file_path, dtype=np.float32).reshape(run_params.n, run_params.d)

    dataset_t = np.transpose(dataset)

    dbs = sDbscan.sDbscan(run_params.n, run_params.d)

    dbs.set_params(run_params.D, run_params.k, run_params.m, run_params.dist, 1024, 2600, 0.4, 0.01, 0,
                   run_params.verbose,
                   run_params.numThreads, 1, run_params.output_file)

    dbs.fit_sDbscan(dataset_t, run_params.eps, run_params.minPts)

    result_labels = None

    out_file_name = os.path.basename(run_params.output_file)

    times_dict = {}

    for file in os.listdir(CPU_RESULTS_PATH):
        if file.startswith(out_file_name):
            file_path = os.path.join(CPU_RESULTS_PATH, file)

            if 'times' in file:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        split_line = line.split(": ")
                        metric = split_line[0]

                        times_dict[metric] = int(split_line[1])

                    times_dict["overall"] = sum(times_dict.values())
            else:
                result_labels = np.loadtxt(file_path)

    nmi = None

    if run_params.labels_file_path:
        if run_params.labels_file_type == "txt":
            true_labels = np.loadtxt(run_params.labels_file_path)
        else:
            true_labels = np.fromfile(run_params.labels_file_path, dtype=np.uint8)
        # Calculate the clustering metrics
        nmi = calculate_nmi(true_labels, result_labels)
        print(f"NMI: {nmi}")

    results_df = pd.DataFrame({
        "params": [copy.deepcopy(run_params.__dict__)],
        "nmi": [nmi],
        "numClusters": [len(np.unique(result_labels))],
        "times": [times_dict]
    })

    return results_df


if __name__ == "__main__":
    datasetFilename = rf"{REPO_PATH}/data/mnist8m/samples/f32/mnist8m_sample_n70000_f32.bin"
    output_file_path = rf"{CPU_RESULTS_PATH}/mnist8m_70k_dbscan_test"
    labels_filename = rf"{REPO_PATH}/data/mnist8m/samples/f32/mnist8m_sample_n70000_f32_labels.bin"

    n = 70_000
    d = 784
    eps = 0.11
    minPts = 50
    k = 5
    m = 50
    numThreads = 64
    dist = "Cosine"

    run_params = CpuRunParams(d, eps, minPts, 1024, k, m, numThreads, dist, n, datasetFilename, True,
                              output_file=output_file_path, file_type="bin", labels_filename=labels_filename,
                              labels_file_type="bin")

    results_df = run_cpu_sdbscan(run_params)
