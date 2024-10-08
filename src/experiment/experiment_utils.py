import os.path
import subprocess
from dataclasses import dataclass
import copy

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import normalized_mutual_info_score
import numpy as np

load_dotenv()

REPO_DIR = "/home/hphi344/Documents/GS-DBSCAN-Analysis"

EXEC_PATH = "/home/hphi344/Documents/GS-DBSCAN-CPP/build-release/GS-DBSCAN"


@dataclass
class RunParams:
    d: int
    minPts: int
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

    def __str__(self):
        filename = os.path.splitext(os.path.basename(self.datasetFilename))[0]
        return f"{filename}_n{self.n}_d{self.d}_D{self.D}_mp{self.minPts}_k{self.k}_m{self.m}_e{self.eps}"

    @property
    def outputFilename(self, base_dir=os.path.join(REPO_DIR, "results"), comments=""):
        return f"{base_dir}/results_{comments}_{str(self)}.json"


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


def read_results(results_file):
    return pd.read_json(results_file)


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

    results_df = read_results(params.outputFilename)

    nmi = -1

    results_df["params"] = [copy.deepcopy(params.__dict__)]

    if params.labels_filename is not None:
        true_labels = get_mnist_labels(params.labels_filename)

        nmi = calculate_nmi(true_labels, results_df['clusterLabels'][0])

        results_df["nmi"] = nmi

    if results_parquet_name is not None:
        results_df.to_parquet(results_parquet_name)

    print_results(results_df, nmi)

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

def get_k_m_experiements_table(k_m_vals, results_df):
    overall_times = results_df['times'].apply(lambda x: x['overall']).values
    overall_times = np.array(overall_times) / 1000000

    nmi_vals = results_df['nmi'].values

    table_df = pd.DataFrame({
        "m": [m for _, m in k_m_vals],
        "k": [k for k, _ in k_m_vals],
        "Time (s)": overall_times,
        "NMI": nmi_vals
    })

    # Convert the DataFrame to LaTeX format
    latex_table = table_df.to_latex(index=False, column_format='c'*len(table_df.columns), escape=False)

    return latex_table


# run_n_size_experiments(
#     k=2,
#     m=2000
# )
