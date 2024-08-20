import subprocess


def run_gs_dbscan(datasetFilename, outFile, n, d, D, minPts, k, m, eps, distanceMetric, clusterBlockSize, timeIt):
    # Build the command list
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
        "--distanceMetric", distanceMetric,
        "--clusterBlockSize", str(clusterBlockSize),
        "--timeIt", str(timeIt)
    ]

    # Run the command
    result = subprocess.run(run_cmd, capture_output=True, text=True)

    # Print standard output and standard error
    print("Standard Output:\n", result.stdout)

    # Check return code to verify execution success
    if result.returncode == 0:
        print("Execution successful")
    else:
        print(f"Execution failed with return code {result.returncode}, error message: {result.stderr}")

    return result

run_gs_dbscan("test_dataset.csv", "results.json", 100, 2, 3, 4, 5, 6, 0.5, "euclidean", 256, 1)