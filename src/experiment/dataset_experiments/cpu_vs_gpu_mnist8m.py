from src.experiment.experiment_cpu import CpuRunParams
import src.experiment.dataset_experiments.mnist_experiments as m_exp
import src.experiment.experiment_utils as exp


def compare_cpu_to_gpu_mnist8m(k, m, eps, minPts=50, dist="Cosine", D=1024):
    d = 784
    numThreads = 64

    run_params_cpu = CpuRunParams(d, eps, minPts, D, k, m, numThreads, dist, datasetFilename=None, verbose=True,
                                output_file=None, file_type="bin", labels_filename=None,
                                labels_file_type="bin")

    results_cpu_df = m_exp.run_mnist_samples_cpu(run_params_cpu, sample_results_df_name=f"mnist_samples_cpu_results_k{k}_m{m}_eps{eps}")
    
    params = exp.RunParams(d=d, D=D, minPts=minPts, k=k, m=m, eps=eps, alpha=1.2,
                        distancesBatchSize=100, distanceMetric="COSINE",
                        clusterBlockSize=256, clusterOnCpu=True, needToNormalize=True, print_cmd=True,
                        verbose=False, useBatchDbscan=True, timeIt=True, useBatchABMatrices=True,
                        useBatchNorm=True,
                        datasetDType="f32", ABatchSize=10_000, BBatchSize=28, miniBatchSize=10_000, normBatchSize=10_000, ignoreAdjListSymmetry=False)


    results_gpu_df = m_exp.run_mnist_samples(params, sample_results_df_name=f"mnist_samples_gpu_results_k{k}_m{m}_eps{eps}")

    return results_cpu_df, results_gpu_df