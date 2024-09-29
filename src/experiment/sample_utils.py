import src.experiment.experiment as exp
import pandas as pd
import os


def run_sample_experiments(params: exp.RunParams, results_dir: str, sample_sizes: list,
                           sample_paths_dict=None,
                           sample_experiment_name="sample_experiements") -> pd.DataFrame:
    results_df_list = []

    for n in sample_sizes:
        params.n = n
        this_result_df = run_sample(params, sample_paths_dict)
        this_result_df.drop(["clusterLabels"])  # Drop the cluster labels to save space
        results_df_list.append(this_result_df)

    results_df = pd.concat(results_df_list)
    results_df.to_parquet(os.path.join(results_dir, sample_experiment_name + ".parquet"))

    return results_df


def run_sample(params: exp.RunParams, sample_paths_dict) -> pd.DataFrame:
    data_path = sample_paths_dict[params.datasetDType]["data"]
    labels_path = sample_paths_dict[params.datasetDType]["labels"]

    params.labels_filename = labels_path
    params.datasetFilename = data_path

    return exp.run_complete_sdbscan_pipeline(params)
