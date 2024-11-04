import os.path
import re
import subprocess
from dataclasses import dataclass
import copy

from tqdm import tqdm

import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
import numpy as np

REPO_DIR = "/home/hphi344/Documents/GS-DBSCAN-Analysis"

EXEC_PATH = "/home/hphi344/Documents/GS-DBSCAN-CPP/build-release/GS-DBSCAN"


@dataclass
class RunParams:
    d: int
    minPts: int
    alpha: float
    distancesBatchSize: int
    distanceMetric: str
    clusterBlockSize: int
    miniBatchSize: int
    ABatchSize: int
    BBatchSize: int
    normBatchSize: int
    eps: float = -1
    sigmaEmbed: int = 1
    datasetDType: str = "labels"
    k: int = -1
    m: int = -1
    n: int = -1
    D: int = -1
    datasetFilename: str = None
    labels_filename: str = None
    clusterOnCpu: bool = True
    needToNormalize: bool = True
    verbose: bool = True
    useBatchDbscan: bool = False
    useBatchABMatrices: bool = False
    useBatchNorm: bool = False
    timeIt: bool = True
    print_cmd: bool = True
    ignoreAdjListSymmetry: bool = False
    outputfile_dir: str = None

    def __str__(self):
        filename = os.path.splitext(os.path.basename(self.datasetFilename))[0]
        return f"{filename}_n{self.n}_d{self.d}_D{self.D}_mp{self.minPts}_k{self.k}_m{self.m}_e{self.eps:.4f}"

    @property
    def outputfile_path(self, base_dir=None, comments=""):
        if self.outputfile_dir is None:
            base_dir = os.path.join(REPO_DIR, "results")
        else:
            base_dir = self.outputfile_dir

        return f"{base_dir}/results_{comments}_{str(self)}.json"


def run_gs_dbscan(params: RunParams):
    run_cmd = [
        EXEC_PATH,
        "--datasetFilename", params.datasetFilename,
        "--outputFilename", params.outputfile_path,
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

    if params.ignoreAdjListSymmetry:
        run_cmd.append("--ignoreAdjListSymmetry")

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


def read_json_results(results_file):
    return pd.read_json(results_file)

def process_results(results_file, labels_filename, params=None):
    results_df = read_json_results(results_file)

    nmi = -1

    if params is not None:
        results_df["params"] = [copy.deepcopy(params.__dict__)]

    if labels_filename is not None:
        true_labels = get_mnist_labels(labels_filename)

        nmi = calculate_nmi(true_labels, results_df['clusterLabels'][0])

        results_df["nmi"] = nmi
    
    return results_df

def calculate_nmi(labels_true, labels_pred):
    return normalized_mutual_info_score(labels_true, labels_pred)

def get_mnist_labels(file=f"{REPO_DIR}/data/mnist/mnist_labels.bin"):
    return np.fromfile(file, dtype=np.uint8)


def run_complete_sdbscan_pipeline(params: RunParams, results_parquet_name = None):
    try:
        run_gs_dbscan(params)

    except Exception as e:
        print(f"Failed to run experiment")
        print(e)
        print('Exiting Loop')
        return None
    
    results_df = process_results(params.outputfile_path, params.labels_filename, params)

    if results_parquet_name is not None:
        results_df.to_parquet(results_parquet_name)

    print_results(results_df, results_df["nmi"])

    return results_df

def run_k_m_experiments(k_m_vals, params, parquet_name=None):
    results_df_list = []

    for k, m in k_m_vals:
        params.k = k
        params.m = m

        this_result_df = run_complete_sdbscan_pipeline(params)

        results_df_list.append(this_result_df)

    results_df = pd.concat(results_df_list)

    if parquet_name is not None:
        results_df.to_parquet(parquet_name)

    return results_df

def process_k_m_results(results_dir, labels_path, clean=False):
    all_results_dfs = []

    for file in os.listdir(results_dir):

        if file.endswith(".json"):
            file_path = os.path.join(results_dir, file)

            matches = re.findall(r'k(\d+)_m(\d+)', file)
            
            k = int(matches[0][0])
            m = int(matches[0][1])

            if k > m:
                continue

            this_df = process_results(file_path, labels_path)

            this_df['k'] = k
            this_df['m'] = m

            all_results_dfs.append(this_df)

    all_results_df = pd.concat(all_results_dfs)

    if clean:
        all_results_df.drop(columns=["clusterLabels"], inplace=True)
        all_results_df.sort_values(by="k", ascending=False, inplace=True)
    
    return all_results_df

def get_k_m_experiments_table(results_df):
    overall_times = results_df['times'].apply(lambda x: x['overall']).values
    overall_times = np.array(overall_times) / 1000000
    overall_times = overall_times.round(2)

    nmi_vals = results_df['nmi'].values
    nmi_vals = nmi_vals.round(2)

    table_df = pd.DataFrame({
        "k": [k for k in results_df["k"]],
        "m": [m for m in results_df["m"]],
        "2km" : [2*k*m for k, m in zip(results_df["k"], results_df["m"])],
        "Time (s)": overall_times,
        "NMI": nmi_vals
    })

    table_df.sort_values(by=["2km", "Time (s)"], inplace=True)

    # Convert the DataFrame to LaTeX format
    latex_table = table_df.to_latex(index=False, column_format='c'*len(table_df.columns), escape=False, float_format="%.2f")

    return latex_table

def run_eps_experiments(eps_vals, params, parquet_name=None):
    results_df_list = []

    for eps in tqdm(eps_vals, "Running eps experiments"):
        params.eps = eps

        this_result_df = run_complete_sdbscan_pipeline(params)

        results_df_list.append(this_result_df)

    results_df = pd.concat(results_df_list)

    if parquet_name is not None:
        results_df.to_parquet(parquet_name)

    return results_df


# run_n_size_experiments(
#     k=2,
#     m=2000
# )
