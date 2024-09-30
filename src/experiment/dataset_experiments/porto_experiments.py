from src.experiment.experiment_utils import run_complete_sdbscan_pipeline
from src.dataset_handling.spatial_datasets.handle_porto import SAMPLE_S640_F32_PATH, S640_SAMPLE_SIZE, SAMPLE_S80_F32_PATH, S80_SAMPLE_SIZE

EPSILON = 0.008

run_complete_sdbscan_pipeline(
    datasetFilename=SAMPLE_S640_F32_PATH,
    outFile="/home/hphi344/Documents/GS-DBSCAN-Analysis/src/experiment/results_porto_s640.json",
    n=S640_SAMPLE_SIZE,
    d=2,
    D=1024,
    k=4,
    m=160,
    minPts=8,
    needToNormalize=False,
    eps=EPSILON,
    distanceMetric="L2",
    distancesBatchSize=100,
    sigmaEmbed=2 * EPSILON,
    useBatchNorm=False,
    useBatchABMatrices=True,
    useBatchDbscan=True,
    ABatchSize=400000,
    BBatchSize=64,
    miniBatchSize=100000,
    dataset_dtype="f32",
    timeIt=True
)

# run_complete_sdbscan_pipeline(
#     datasetFilename=SAMPLE_S80_F32_PATH,
#     outFile="/home/hphi344/Documents/GS-DBSCAN-Analysis/src/experiment/results_porto_s80.json",
#     n=S80_SAMPLE_SIZE,
#     d=2,
#     D=1024,
#     k=4,
#     m=160,
#     minPts=8,
#     needToNormalize=False,
#     eps=EPSILON,
#     distanceMetric="L2",
#     distancesBatchSize=100,
#     sigmaEmbed=2 * EPSILON,
#     useBatchNorm=False,
#     useBatchABMatrices=True,
#     useBatchDbscan=True,
#     ABatchSize=400000,
#     BBatchSize=64,
#     miniBatchSize=100000,
#     dataset_dtype="f32",
#     timeIt=True
# )