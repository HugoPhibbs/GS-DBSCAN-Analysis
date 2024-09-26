from src.experiment.experiment import run_complete_sdbscan_pipeline


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

# run_complete_sdbscan_pipeline("results.json",
#                               datasetFilename="/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist/mnist_images_row_major_f16.bin",
#                               labels_filename="/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist/mnist_labels_f16.bin",
#                               n=70_000, d=784, D=1024, minPts=50, k=2, m=2000, eps=0.15, alpha=1.2,
#                               distancesBatchSize=100, distanceMetric="COSINE",
#                               clusterBlockSize=256, clusterOnCpu=True, needToNormalize=True, print_cmd=True,
#                               verbose=True, useBatchDbscan=True, timeIt=True, useBatchABMatrices=False,
#                               useBatchNorm=True,
#                               dataset_dtype="f16", ABatchSize=10_000, miniBatchSize=10_000, normBatchSize=10_000)

# run_complete_sdbscan_pipeline("results.json",
#                               datasetFilename="/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist/mnist_images_row_major.bin",
#                               labels_filename="/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist/mnist_labels.bin",
#                               n=70_000, d=784, D=1024, minPts=50, k=5, m=50, eps=0.11, alpha=1.2,
#                               clusterBlockSize=256, clusterOnCpu=True, needToNormalize=True, print_cmd=True,
#                               verbose=True, useBatchDbscan=False, timeIt=True, useBatchABMatrices=False,
#                               useBatchNorm=True,
#                               dataset_dtype="f32", ABatchSize=10_000, miniBatchSize=10_000, normBatchSize=10_000)

# run_complete_sdbscan_pipeline("results.json",
#                               datasetFilename="/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist8m/samples/data/mnist8m_f16_sample_50000.bin",
#                               labels_filename="/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist8m/samples/labels/mnist8m_sample_labels_50000.bin",
#                               n=50_000, d=784, D=1024, minPts=50, k=2, m=2000, eps=0.11, alpha=1.2,
#                               distancesBatchSize=100, distanceMetric="COSINE",
#                               clusterBlockSize=256, clusterOnCpu=True, needToNormalize=True, print_cmd=True,
#                               verbose=True, useBatchDbscan=True, timeIt=True, useBatchABMatrices=False, dataset_dtype="f16")

run_complete_sdbscan_pipeline("results.json",
                              datasetFilename="/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist/mnist_images_row_major.bin",
                              labels_filename="/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist/mnist_labels.bin",
                              n=70_000, d=784, D=1024, minPts=50, k=2, m=2000, eps=1350, alpha=1.2,
                              clusterBlockSize=256, clusterOnCpu=True, needToNormalize=False, print_cmd=True,
                              distanceMetric="L2",
                              verbose=True, useBatchDbscan=False, timeIt=True, useBatchABMatrices=False,
                              useBatchNorm=True,
                              dataset_dtype="f32", ABatchSize=10_000, miniBatchSize=10_000, normBatchSize=10_000, sigmaEmbed=2700)