import numpy as np

import src.experiment.experiment_utils as exp
import pandas as pd
import os
import matplotlib.pyplot as plt

import src.experiment.experiment_cpu as exp_cpu


def run_sample_experiments(params: exp.RunParams, sample_sizes: list,
                           sample_paths_dict=None,
                           sample_experiment_name="sample_experiments", parquet_dir=None) -> pd.DataFrame:
    results_df_list = []

    for n in sample_sizes:
        params.n = n
        set_params_labels_and_data_paths(sample_paths_dict, params, params.n, params.datasetDType)
        this_result_df = exp.run_complete_sdbscan_pipeline(params)
        this_result_df.drop("clusterLabels", axis=1, inplace=True)  # Drop the cluster labels to save space
        results_df_list.append(this_result_df)

    results_df = pd.concat(results_df_list)

    if parquet_dir is not None:
        results_df.to_parquet(os.path.join(parquet_dir, sample_experiment_name + ".parquet"))

    return results_df


def set_params_labels_and_data_paths(sample_paths_dict, params, n, dataset_dtype="f32"):
    data_path = sample_paths_dict[dataset_dtype]["data"][n]
    labels_path = sample_paths_dict[dataset_dtype]["labels"][n]

    params.labels_filename = labels_path
    params.datasetFilename = data_path

    return data_path, labels_path


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


def run_sample_experiments_cpu(cpu_params: exp_cpu.CpuRunParams, results_dir: str, sample_sizes: list,
                               sample_paths_dict=None,
                               sample_experiment_name="sample_experiments_cpu") -> pd.DataFrame:
    results_df_list = []

    for n in sample_sizes:
        cpu_params.n = n

        cpu_params.output_file = os.path.join(results_dir, f"cpu_{sample_experiment_name}_n{n}")

        set_params_labels_and_data_paths(sample_paths_dict, cpu_params, cpu_params.n, "f32")

        this_result_df = exp_cpu.run_cpu_sdbscan(cpu_params)

        results_df_list.append(this_result_df)

    results_df = pd.concat(results_df_list)
    results_df.to_parquet(os.path.join(results_dir, sample_experiment_name + ".parquet"))

    return results_df



def plot_sample_time_results(sample_results_df, save_path=None, title="Sample size vs total runtime", fig=None, ax=None, label="sDBSCAN"):
    n_vals = [params["n"] for params in sample_results_df["params"]]
    time_vals = np.array([times["overall"] for times in sample_results_df["times"]]) / 1e6

    if fig is None and ax is None:
        fig, ax = plt.subplots()

    line, = ax.plot(n_vals, time_vals, label=label)
    ax.set_title(title)
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Time (s)")

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    # Return the figure, axes, and line for further manipulation or display in the notebook
    return fig, ax, line

def plot_compare_cpu_gpu_time(results_gpu_df, results_cpu_df, save_path=None, title="Sample size vs runtime", add_speedup=False):
    fig, ax, line_gpu = plot_sample_time_results(results_gpu_df, save_path=None, title=title, label="CUDA-sDBSCAN")
    fig, ax, line_cpu = plot_sample_time_results(results_cpu_df, save_path=None, title=title, ax=ax, fig=fig, label="CPU-sDBSCAN")

    if add_speedup:
        n_vals = [params["n"] for params in results_gpu_df["params"]]
        time_vals_gpu = np.array([times["overall"] for times in results_gpu_df["times"]]) / 1e6
        time_vals_cpu = np.array([times["overall"] for times in results_cpu_df["times"]]) / 1e6

        print(time_vals_cpu)
        print(time_vals_gpu)

        speedup_vals = time_vals_cpu / time_vals_gpu

        ax2 = ax.twinx()
        line_speedup, = ax2.plot(n_vals, speedup_vals, color="red", linestyle="--", label="Speedup")
        print(f"Average Speedup: {np.mean(speedup_vals)}")

        ax2.set_ylabel("Speedup")
        ax2.set_ylim(0, np.max(speedup_vals) + 1)

        # Collect handles and labels for both axes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="lower right")
    else:
        ax.legend([line_gpu, line_cpu], ["CUDA-sDBSCAN", "CPU sDBSCAN"], loc="lower right")

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_compare_cpu_gpu_nmi(results_gpu_df, results_cpu_df, title ="Sample size vs NMI", save_path=None):
    fig, ax = plot_sample_nmi_results(results_gpu_df, save_path=None)
    plot_sample_nmi_results(results_cpu_df, title=title, save_path=None, ax=ax, fig=fig)
    plt.legend(["CUDA-sDBSCAN", "CPU sDBSCAN"])

    if save_path is not None:
        plt.savefig(save_path, dpi=300)


def plot_sample_nmi_results(sample_results_df, save_path=None, title="Sample size vs NMI", fig=None, ax=None, **kwargs):
    n_vals = [params["n"] for params in sample_results_df["params"]]
    nmi_vals = sample_results_df["nmi"]

    if fig is None and ax is None:
        fig, ax = plt.subplots()
    ax.plot(n_vals, nmi_vals, **kwargs)
    ax.set_title(title)
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("NMI")
    ax.set_ylim(0, 0.5)
    # ax.lines[0].set_linestyle(line_style)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    return fig, ax


def plot_sample_time_breakdown(sample_results_df, save_file=None, title="Sample size vs runtime breakdown"):
    n_vals = [params["n"] for params in sample_results_df["params"]]

    times_overall_vals = np.array([times["overall"] for times in sample_results_df["times"]]) / 1e6

    distances_vals = np.array([times["totalTimeDistances"] for times in sample_results_df["times"]]) / 1e6
    copy_convert_vals = np.array([times["copyingAndConvertData"] for times in sample_results_df["times"]]) / 1e6
    process_adj_list_vals = np.array([times["processAdjacencyList"] for times in sample_results_df["times"]]) / 1e6
    AB_matrices_vals = np.array([times["constructABMatrices"] for times in sample_results_df["times"]]) / 1e6
    adj_list_vals = np.array([times["adjList"] for times in sample_results_df["times"]]) / 1e6
    clustering_vals = np.array([times["formClusters"] for times in sample_results_df["times"]]) / 1e6
    normalize_vals = np.array([times["normalise"] for times in sample_results_df["times"]]) / 1e6

    remainder_vals = times_overall_vals - distances_vals - copy_convert_vals - process_adj_list_vals - AB_matrices_vals - clustering_vals - adj_list_vals - normalize_vals

    fig, ax = plt.subplots()

    ax.stackplot(n_vals, distances_vals, AB_matrices_vals, adj_list_vals, process_adj_list_vals, copy_convert_vals,
                 clustering_vals, normalize_vals, remainder_vals)

    ax.set_title(title)
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Time (s)")
    ax.legend(
        ["Distances", "AB Matrices", "Adj List", "Process Adj List", "Copy and Convert", "Forming Clusters", "Other"],
        loc="upper left")

    if save_file is not None:
        plt.savefig(f"plots/{save_file}", dpi=300)

    return fig, ax


def plot_sample_time_breakdown_perc(sample_results_df, save_file=None,
                                    title="Sample Size vs Runtime Breakdown (Percentage)", ymax=105):
    n_vals = [params["n"] for params in sample_results_df["params"]]

    times_overall_vals = np.array([times["overall"] for times in sample_results_df["times"]]) / 1e6

    # Calculate individual times in seconds
    distances_vals = np.array([times["totalTimeDistances"] for times in sample_results_df["times"]]) / 1e6
    copy_convert_vals = np.array([times["copyingAndConvertData"] for times in sample_results_df["times"]]) / 1e6
    process_adj_list_vals = np.array([times["processAdjacencyList"] for times in sample_results_df["times"]]) / 1e6
    AB_matrices_vals = np.array([times["constructABMatrices"] for times in sample_results_df["times"]]) / 1e6

    remainder_vals = times_overall_vals - (distances_vals + copy_convert_vals + process_adj_list_vals +
                                           AB_matrices_vals)

    # Calculate percentage contributions of each component
    distances_pct = (distances_vals / times_overall_vals) * 100
    copy_convert_pct = (copy_convert_vals / times_overall_vals) * 100
    process_adj_list_pct = (process_adj_list_vals / times_overall_vals) * 100
    AB_matrices_pct = (AB_matrices_vals / times_overall_vals) * 100
    remainder_pct = (remainder_vals / times_overall_vals) * 100

    # Plotting the percentages
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(n_vals, distances_pct, label="Distance Calculations")
    ax.plot(n_vals, AB_matrices_pct, label="Create AB Matrices")
    ax.plot(n_vals, process_adj_list_pct, label="Ensure Adjacency List Symmetry")
    ax.plot(n_vals, copy_convert_pct, label="Copy dataset from CPU to GPU")
    ax.plot(n_vals, remainder_pct, label="Other (incl. Forming Clusters)")

    ax.set_title(title)
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Percentage of Total Runtime (%)")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_yticks(range(0, 110, 10))
    ax.set_ylim(0, ymax)

    plt.subplots_adjust(right=0.5)
    plt.tight_layout()

    if save_file is not None:
        plt.savefig(f"plots/{save_file}", dpi=300)

    return fig, ax


if __name__ == "__main__":
    n = 70_000
    d = 784
    eps = 0.11
    minPts = 50
    k = 5
    m = 50
    numThreads = 64
    dist = "Cosine"

    run_params = exp_cpu.CpuRunParams(d, eps, minPts, 1024, k, m, numThreads, dist, True,
                                      output_file=output_file_path, file_type="bin", labels_filename=labels_filename,
                                      labels_file_type="bin")
