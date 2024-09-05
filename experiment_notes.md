# Experimentation Notes

* All of the packages we compare to are low dimensional

## Where to find packages

#### F-DBSCAN
* Run via the ArborX library. The executable is located at:
`/home/hphi344/Documents/GS-DBSCAN-Exp-Repos/ArborX/build/benchmarks/dbscan/ArborX_Benchmark_DBSCAN.exe`

* See DBSCAN README, here:
`/home/hphi344/Documents/GS-DBSCAN-Exp-Repos/ArborX/benchmarks/dbscan/README.md`

* Repo link: https://github.com/arborx/ArborX
* Reads from a binary file, the format is shown in the dbscan README.

#### Cuda-DClust+
* Run from directory at `/home/hphi344/Documents/GS-DBSCAN-Exp-Repos/fast-cuda-gpu-dbscan`
* You will need to change the `common.h` file from the top level directory to adjust the algo parameters
* Repo link https://github.com/l3lackcurtains/fast-cuda-gpu-dbscan
* With datasets, wants them in a .txt format. The repo already gives us a bunch of datasets to run with, so can just use these

#### G-DBSCAN
* Run from directory at `/home/hphi344/Documents/GS-DBSCAN-Exp-Repos/fast-cuda-gpu-dbscan/gdbscan`
* Change the `main.c` file from this directory for algo parameters
* Repo link https://github.com/l3lackcurtains/fast-cuda-gpu-dbscan/tree/main/gdbscan
* Same dataset comments as Cuda-DClust+

##### RT-DBSCAN
* Run from directory at `/home/hphi344/Documents/GS-DBSCAN-Exp-Repos/fast-cuda-gpu-dbscan/OWLRayTracing`
* Haven't yet been able to build with cmake - waiting on a GH issue. Once this has been done, 
* Repo link https://github.com/vani-nag/OWLRayTracing/tree/rt-dbscan

* The RT-DBSCAN run instructions can be found here: https://github.com/vani-nag/OWLRayTracing/tree/rt-dbscan/samples/cmdline/s02-rtdbscan
* It's as simple a running an executable. Reads in a txt as an input file (assume one line per row), see https://github.com/vani-nag/OWLRayTracing/blob/rt-dbscan/samples/cmdline/s02-rtdbscan/hostCode.cpp