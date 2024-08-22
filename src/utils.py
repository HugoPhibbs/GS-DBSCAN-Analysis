import subprocess
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()


def run_gs_dbscan(datasetFilename, outFile, n, d, D, minPts, k, m, eps, alpha, distancesBatchSize, distanceMetric,
                  clusterBlockSize):
    run_cmd = [
        "../GS-DBSCAN-CPP/build/sDbscan",
        "--datasetFilename", datasetFilename,
        "--outFile", outFile,
        "--n", str(n),
        "--d", str(d),
        "--D", str(D),
        "--minPts", str(minPts),
        "--k", str(k),
        "--m", str(m),
        "--eps", str(eps),
        "--alpha", str(alpha),
        "--distancesBatchSize", str(distancesBatchSize),
        "--distanceMetric", distanceMetric,
        "--clusterBlockSize", str(clusterBlockSize)
    ]

    result = subprocess.run(run_cmd, capture_output=True, text=True)

    print("Standard Output:\n", result.stdout)

    if result.returncode == 0:
        print("Execution successful")
    else:
        print(f"Execution failed with return code {result.returncode}, error message: {result.stderr}")

    return result


def read_results(results_file):
    return pd.read_json(results_file)


run_gs_dbscan("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist_images_col_major.bin",
              outFile="results.json",
              n=70000,
              d=784,
              D=1024,
              minPts=50,
              k=5,
              m=50,
              eps=0.11,
              alpha=1.2,
              distancesBatchSize=-1,
              distanceMetric="L2",
              clusterBlockSize=256)
