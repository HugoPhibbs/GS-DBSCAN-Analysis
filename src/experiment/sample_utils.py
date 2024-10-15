import numpy as np

import src.experiment.experiment_utils as exp
import pandas as pd
import os
import matplotlib.pyplot as plt


def run_sample_experiments(params: exp.RunParams, results_dir: str, sample_sizes: list,
                           sample_paths_dict=None,
                           sample_experiment_name="sample_experiements") -> pd.DataFrame:
    results_df_list = []

    for n in sample_sizes:
        params.n = n
        this_result_df = run_sample(params, sample_paths_dict)
        this_result_df.drop("clusterLabels", axis=1, inplace=True)  # Drop the cluster labels to save space
        results_df_list.append(this_result_df)

    results_df = pd.concat(results_df_list)
    results_df.to_parquet(os.path.join(results_dir, sample_experiment_name + ".parquet"))

    return results_df


def plot_sample_time_results(sample_results_df, save_path=None, title="Sample size vs total runtime"):
    n_vals = [params["n"] for params in sample_results_df["params"]]
    time_vals = np.array([times["overall"] for times in sample_results_df["times"]]) / 1e6

    fig, ax = plt.subplots()
    ax.plot(n_vals, time_vals)
    ax.set_title(title)
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Time (s)")

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    # Return the figure and axes for further manipulation or display in the notebook
    return fig, ax


def plot_sample_nmi_results(sample_results_df, save_path=None, title="Sample size vs NMI"):
    n_vals = [params["n"] for params in sample_results_df["params"]]
    nmi_vals = sample_results_df["nmi"]

    fig, ax = plt.subplots()
    ax.plot(n_vals, nmi_vals)
    ax.set_title(title)
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("NMI")

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    return fig, ax

def plot_sample_time_breakdown(sample_results_df, save_file=None, title="Sample size vs runtime breakdown"):
    n_vals = [params["n"] for params in sample_results_df["params"]]

    times_overall_vals =  np.array([times["overall"] for times in sample_results_df["times"]]) / 1e6

    distances_vals = np.array([times["totalTimeDistances"] for times in sample_results_df["times"]]) / 1e6
    copy_convert_vals = np.array([times["copyingAndConvertData"] for times in sample_results_df["times"]]) / 1e6
    process_adj_list_vals = np.array([times["processAdjacencyList"] for times in sample_results_df["times"]]) / 1e6
    AB_matrices_vals = np.array([times["constructABMatrices"] for times in sample_results_df["times"]]) / 1e6
    adj_list_vals = np.array([times["adjList"] for times in sample_results_df["times"]]) / 1e6
    clustering_vals = np.array([times["formClusters"] for times in sample_results_df["times"]]) / 1e6
    normalize_vals = np.array([times["normalise"] for times in sample_results_df["times"]]) / 1e6

    remainder_vals  = times_overall_vals - distances_vals - copy_convert_vals - process_adj_list_vals - AB_matrices_vals - clustering_vals - adj_list_vals - normalize_vals

    fig, ax = plt.subplots()

    ax.stackplot(n_vals, distances_vals, AB_matrices_vals, adj_list_vals, process_adj_list_vals, copy_convert_vals, clustering_vals, normalize_vals, remainder_vals)

    ax.set_title(title)
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Time (s)")
    ax.legend(["Distances", "AB Matrices", "Adj List", "Process Adj List", "Copy and Convert", "Forming Clusters", "Other"], loc="upper left")

    if save_file is not None:
        plt.savefig(f"plots/{save_file}", dpi=300)

    return fig, ax

def plot_sample_time_breakdown_perc(sample_results_df, save_file=None, title="Sample Size vs Runtime Breakdown (Percentage)"):
    n_vals = [params["n"] for params in sample_results_df["params"]]

    times_overall_vals = np.array([times["overall"] for times in sample_results_df["times"]]) / 1e6

    # Calculate individual times in seconds
    distances_vals = np.array([times["totalTimeDistances"] for times in sample_results_df["times"]]) / 1e6
    copy_convert_vals = np.array([times["copyingAndConvertData"] for times in sample_results_df["times"]]) / 1e6
    process_adj_list_vals = np.array([times["processAdjacencyList"] for times in sample_results_df["times"]]) / 1e6
    AB_matrices_vals = np.array([times["constructABMatrices"] for times in sample_results_df["times"]]) / 1e6
    adj_list_vals = np.array([times["adjList"] for times in sample_results_df["times"]]) / 1e6
    clustering_vals = np.array([times["formClusters"] for times in sample_results_df["times"]]) / 1e6
    normalize_vals = np.array([times["normalise"] for times in sample_results_df["times"]]) / 1e6

    remainder_vals = times_overall_vals - (distances_vals + copy_convert_vals + process_adj_list_vals +
                                           AB_matrices_vals + adj_list_vals + clustering_vals + normalize_vals)

    # Calculate percentage contributions of each component
    distances_pct = (distances_vals / times_overall_vals) * 100
    copy_convert_pct = (copy_convert_vals / times_overall_vals) * 100
    process_adj_list_pct = (process_adj_list_vals / times_overall_vals) * 100
    AB_matrices_pct = (AB_matrices_vals / times_overall_vals) * 100
    adj_list_pct = (adj_list_vals / times_overall_vals) * 100
    clustering_pct = (clustering_vals / times_overall_vals) * 100
    normalize_pct = (normalize_vals / times_overall_vals) * 100
    remainder_pct = (remainder_vals / times_overall_vals) * 100

    # Plotting the percentages
    fig, ax = plt.subplots()

    ax.plot(n_vals, distances_pct, label="Distances")
    ax.plot(n_vals, AB_matrices_pct, label="AB Matrices")
    ax.plot(n_vals, adj_list_pct, label="Adj List")
    ax.plot(n_vals, process_adj_list_pct, label="Process Adj List")
    ax.plot(n_vals, copy_convert_pct, label="Copy and Convert")
    ax.plot(n_vals, clustering_pct, label="Forming Clusters")
    ax.plot(n_vals, normalize_pct, label="Normalize")
    ax.plot(n_vals, remainder_pct, label="Other")

    ax.set_title(title)
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Percentage of Total Runtime (%)")
    ax.legend(loc="upper left")

    if save_file is not None:
        plt.savefig(f"plots/{save_file}", dpi=300)

    return fig, ax

def run_sample(params: exp.RunParams, sample_paths_dict) -> pd.DataFrame:
    data_path = sample_paths_dict[params.datasetDType]["data"][params.n]
    labels_path = sample_paths_dict[params.datasetDType]["labels"][params.n]

    params.labels_filename = labels_path
    params.datasetFilename = data_path

    return exp.run_complete_sdbscan_pipeline(params)


def get_sample_dict(dataset_name: str, sample_n_vals: list, sample_dir: str, dtypes: tuple = ("f16", "f32")):
    sample_paths_dict = {}  # dtype->data/labels->n->path

    for dtype in dtypes:
        sample_paths_dict[dtype] = {"data": {}, "labels": {}}
        data_dict = sample_paths_dict[dtype]["data"]
        labels_dict = sample_paths_dict[dtype]["labels"]

        for n in sample_n_vals:
            data_dict[n] = os.path.join(sample_dir, dtype, f"{dataset_name}_sample_n{n}_{dtype}.bin")
            labels_dict[n] = os.path.join(sample_dir, dtype, f"{dataset_name}_sample_n{n}_{dtype}_labels.bin")

    return sample_paths_dict
