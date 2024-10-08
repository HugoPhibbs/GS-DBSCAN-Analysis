import pandas as pd

import src.dataset_handling.pamap.write_pamap as wp
import src.experiment.sample_utils as su

import src.experiment.experiment_utils as exp

PAMAP_RESULTS_DIR = "/home/hphi344/Documents/GS-DBSCAN-Analysis/results/pamap/samples/"


def run_pamap_normal(params: exp.RunParams, dtype="f32") -> pd.DataFrame:
    pamap_path = wp.PAMAP_PATHS_DICT[dtype]

    labels_path = wp.PAMAP_LABELS_PATH

    params.labels_filename = labels_path
    params.datasetFilename = pamap_path
    params.n = wp.PAMAP_N
    params.D = wp.PAMAP_DIM

    return exp.run_complete_sdbscan_pipeline(params)


def run_pamap_samples(params: exp.RunParams, parquet_name) -> pd.DataFrame:
    results_df = su.run_sample_experiments(params, PAMAP_RESULTS_DIR, wp.PAMAP_SAMPLE_SIZES,
                                           wp.PAMAP_SAMPLES_PATHS_DICT, parquet_name)
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