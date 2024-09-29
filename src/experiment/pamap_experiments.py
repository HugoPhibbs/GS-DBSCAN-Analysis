import pandas as pd

import src.dataset_handling.pamap.write_pamap as wp
import src.experiment.sample_utils as su

import src.experiment.experiment as exp

PAMAP_RESULTS_DIR = "/home/hphi344/Documents/GS-DBSCAN-Analysis/results/pamap/samples/"


def run_pamap_normal(params: exp.RunParams, dtype="f32") -> pd.DataFrame:
    pamap_path = wp.PAMAP_PATHS_DICT[dtype]

    labels_path = wp.PAMAP_LABELS_PATH

    params.labels_filename = labels_path
    params.datasetFilename = pamap_path
    params.n = wp.PAMAP_N
    params.D = wp.PAMAP_D

    return exp.run_complete_sdbscan_pipeline(params)


def run_pamap_samples(params: exp.RunParams) -> pd.DataFrame:
    results_df = su.run_sample_experiments(params, PAMAP_RESULTS_DIR, wp.PAMAP_SAMPLE_SIZES,
                                           wp.PAMAP_SAMPLES_PATHS_DICT, "pamap_sample_experiments")
    return results_df
