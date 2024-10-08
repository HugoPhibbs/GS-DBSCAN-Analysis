from src.experiment.experiment_utils import run_complete_sdbscan_pipeline
import src.dataset_handling.mnist.write_mnist8m as wm8
from src.experiment.experiment_utils import run_gs_dbscan, read_results, REPO_DIR
import src.experiment.sample_utils as su
import src.experiment.experiment_utils as exp

MNIST_RESULTS_DIR = "/home/hphi344/Documents/GS-DBSCAN-Analysis/results/mnist/samples/"

def run_mnist_samples(params: exp.RunParams, sample_results_df_name):
    results_df = su.run_sample_experiments(params, MNIST_RESULTS_DIR, wm8.N_EXPERIMENT_VALUES, wm8.MNIST_8M_SAMPLES_DICT, sample_results_df_name)
    return results_df

if __name__ == '__main__':

    # def run_n_size_experiments(n_values=wm8.N_EXPERIMENT_VALUES, d=784, D=1024, minPts=50, k=5, m=50,
    #                            eps=0.11, alpha=1.2, distancesBatchSize=200,
    #                            distanceMetric="COSINE", clusterBlockSize=256, clusterOnCpu=True, needToNormalize=True,
    #                            print_cmd=True, write_to_pickle=False):
    #     results_df = None
    #
    #     sample_labels_list = []
    #
    #     sample_filenames_list, sample_labels_filenames_list = wm8.get_8m_sample_filenames()
    #
    #     for i in range(len(n_values)):
    #         n = n_values[i]
    #
    #         sample_filename = sample_filenames_list[i]
    #         out_file = f"{REPO_DIR}/results/n_experiments/mnist8m_sample_results_{n}.json"
    #
    #         try:
    #             run_gs_dbscan(sample_filename, out_file, n, d, D, minPts, k, m, eps, alpha, distancesBatchSize,
    #                           distanceMetric,
    #                           clusterBlockSize, clusterOnCpu, needToNormalize, print_cmd)
    #
    #         except Exception as e:
    #             print(f"Failed to run experiment with n {n}")
    #             print(e)
    #             print('Exiting Loop')
    #             break
    #
    #         results_df = read_results(out_file, results_df)
    #
    #     add_nmi_to_results_labels_list(results_df, sample_labels_list)
    #
    #     if write_to_pickle:
    #         results_df.to_pickle(f"{REPO_DIR}/results/n_experiments/results_df.pkl")


    # run_batch_experiments(distance_batch_sizes=[125], k=2, m=2000)

    # run_complete_sdbscan_pipeline("results.json", n=70_000, d=784, D=1024, minPts=50, k=2, m=2000, eps=0.11, alpha=1.2,
    #                               distancesBatchSize=100, distanceMetric="COSINE",
    #                               clusterBlockSize=256, clusterOnCpu=True, needToNormalize=True, print_cmd=True,
    #                               verbose=True, useBatchDbscan=True, timeIt=True, useBatchABMatrices=False)

    # run_complete_sdbscan_pipeline("results.json",
    #                               datasetFilename="/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist8m/samples/data/f16/mnist8m_f16_sample_5000000.bin",
    #                               labels_filename="/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist8m/samples/labels/f16/mnist8m_forf16_sample_labels_5000000.bin",
    #                               n=5_000_000, d=784, D=1024, minPts=50, k=5, m=100, eps=0.16, alpha=1.2,
    #                               distancesBatchSize=100, distanceMetric="COSINE",
    #                               clusterBlockSize=256, clusterOnCpu=True, needToNormalize=True, print_cmd=True,
    #                               verbose=True, useBatchDbscan=True, timeIt=True, useBatchABMatrices=False,
    #                               useBatchNorm=True,
    #                               dataset_dtype="f16", ABatchSize=100_000, miniBatchSize=100_000, normBatchSize=100_000, BBatchSize=32)

    # params = exp.RunParams(
    #     datasetFilename="/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist/mnist_images_row_major_f16.bin",
    #     labels_filename="/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist/mnist_labels_f16.bin",
    #     n=70_000, d=784, D=1024, minPts=50, k=2, m=2000, eps=0.11, alpha=1.2,
    #     distancesBatchSize=125, distanceMetric="COSINE",
    #     clusterBlockSize=256, clusterOnCpu=True, needToNormalize=True, print_cmd=True,
    #     verbose=True, useBatchDbscan=True, timeIt=True, useBatchABMatrices=False,
    #     useBatchNorm=True,
    #     datasetDType="f16", ABatchSize=10_000, BBatchSize=10_000, miniBatchSize=10_000, normBatchSize=10_000, ignoreAdjListSymmetry=False)
    #
    # run_complete_sdbscan_pipeline(params)

    params = exp.RunParams(d=784, D=1024, minPts=50, k=2, m=2000, eps=0.04, alpha=1.2,
                           distancesBatchSize=1000, distanceMetric="COSINE",
                           clusterBlockSize=256, clusterOnCpu=True, needToNormalize=True, print_cmd=True,
                           verbose=False, useBatchDbscan=True, timeIt=True, useBatchABMatrices=True,
                           useBatchNorm=True,
                           datasetDType="f16", ABatchSize=10_000, BBatchSize=28, miniBatchSize=10_000, normBatchSize=10_000, ignoreAdjListSymmetry=True)

    results = run_mnist_samples(params)

# run_complete_sdbscan_pipeline("results.json",
    #                               datasetFilename="/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist/mnist_images_row_major.bin",
    #                               labels_filename="/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist/mnist_labels.bin",
    #                               n=70_000, d=784, D=1024, minPts=50, k=5, m=50, eps=0.11, alpha=1.2,
    #                               clusterBlockSize=256, clusterOnCpu=True, needToNormalize=True, print_cmd=True,
    #                               verbose=True, useBatchDbscan=False, timeIt=True, useBatchABMatrices=False,
    #                               useBatchNorm=True,
    #                               dataset_dtype="labels", ABatchSize=10_000, miniBatchSize=10_000, normBatchSize=10_000)

    # run_complete_sdbscan_pipeline("results.json",
    #                               datasetFilename="/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist8m/samples/data/mnist8m_f16_sample_50000.bin",
    #                               labels_filename="/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist8m/samples/labels/mnist8m_sample_labels_50000.bin",
    #                               n=50_000, d=784, D=1024, minPts=50, k=2, m=2000, eps=0.11, alpha=1.2,
    #                               distancesBatchSize=100, distanceMetric="COSINE",
    #                               clusterBlockSize=256, clusterOnCpu=True, needToNormalize=True, print_cmd=True,
    # #                               verbose=True, useBatchDbscan=True, timeIt=True, useBatchABMatrices=False, dataset_dtype="f16")
    #
    # run_complete_sdbscan_pipeline("results.json",
    #                               datasetFilename="/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist/mnist_images_row_major.bin",
    #                               labels_filename="/data/mnist/mnist_labels.bin",
    #                               n=70_000, d=784, D=1024, minPts=50, k=2, m=2000, eps=1350, alpha=1.2,
    #                               clusterBlockSize=256, clusterOnCpu=True, needToNormalize=False, print_cmd=True,
    #                               distanceMetric="L2",
    #                               verbose=True, useBatchDbscan=False, timeIt=True, useBatchABMatrices=False,
    #                               useBatchNorm=True,
    #                               dataset_dtype="labels", ABatchSize=10_000, miniBatchSize=10_000, normBatchSize=10_000, sigmaEmbed=2700)