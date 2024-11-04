import os
import pandas as pd

import src.dataset_handling.pamap.write_pamap as wp
import src.experiment.sample_utils as su

import src.experiment.experiment_utils as exp
import src.experiment.experiment_cpu as exp_cpu

PAMAP_RESULTS_DIR = "/home/hphi344/Documents/GS-DBSCAN-Analysis/results/pamap/samples/raw"


def run_pamap_normal(params: exp.RunParams, dtype="f32") -> pd.DataFrame:
    pamap_path = wp.PAMAP_PATHS_DICT[dtype]

    labels_path = wp.PAMAP_LABELS_PATH

    params.labels_filename = labels_path
    params.datasetFilename = pamap_path
    params.n = wp.PAMAP_N
    params.d = wp.PAMAP_DIM

    return exp.run_complete_sdbscan_pipeline(params)

def run_pamap_normal_cpu(params: exp_cpu.CpuRunParams, dtype="f32", parquet_path=None) -> pd.DataFrame:
    pamap_path = wp.PAMAP_PATHS_DICT[dtype]

    labels_path = wp.PAMAP_LABELS_PATH

    params.labels_filename = labels_path
    params.datasetFilename = pamap_path
    params.n = wp.PAMAP_N
    params.d = wp.PAMAP_DIM

    return exp_cpu.run_cpu_sdbscan(params, parquet_path)


def run_pamap_samples(params: exp.RunParams, parquet_name) -> pd.DataFrame:
    results_df = su.run_sample_experiments(params, wp.PAMAP_SAMPLE_SIZES,
                                           wp.PAMAP_SAMPLES_PATHS_DICT, parquet_name, parquet_dir="/home/hphi344/Documents/GS-DBSCAN-Analysis/results/pamap/samples/parquets")
    return results_df


def run_pamap_samples_cpu(params: exp_cpu.CpuRunParams, sample_results_df_name, results_subdir=None, max_n=float("inf")):
    results_dir = PAMAP_RESULTS_DIR

    if results_subdir:
        results_dir = os.path.join(results_dir, results_subdir)

    n_vals = [val for val in wp.PAMAP_SAMPLE_SIZES if val <= max_n]

    results_df = su.run_sample_experiments_cpu(params, results_dir, n_vals, wp.PAMAP_SAMPLES_PATHS_DICT, sample_results_df_name)
    return results_df


if __name__ == "__main__":
    params = exp.RunParams(d=wp.PAMAP_DIM, D=1024, minPts=50, k=5, m=50, eps=0.04, alpha=1.2,
                           distancesBatchSize=1000, distanceMetric="COSINE",
                           clusterBlockSize=256, clusterOnCpu=True, needToNormalize=True, print_cmd=True,
                           verbose=False, useBatchDbscan=True, timeIt=True, useBatchABMatrices=True,
                           useBatchNorm=True,
                           datasetDType="f16", ABatchSize=10_000, BBatchSize=28, miniBatchSize=10_000, normBatchSize=10_000, ignoreAdjListSymmetry=True)


    results = run_pamap_samples(params)

    # results_df = pd.read_parquet("/home/hphi344/Documents/GS-DBSCAN-Analysis/results/pamap/samples/pamap_sample_experiments.parquet")
    #
    # su.plot_sample_time_results(results_df)