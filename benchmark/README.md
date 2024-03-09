# Running the benchmarks
This table indicates the modules that you must load in order to ensure that
all the code compiles and executes correctly.

|Benchmark (for all files in folder) | Modules|
|:---|  ---:|
|CUDA| ```CUDA``` ```OpenMPI```|
|MPI_1NODE|```OpenMPI```|
|MPI_distributed_matrix|```OpenMPI```|
|OpenMP|```OpenBLAS``` ```intel```|
|IO|```OpenMPI```|
|SpeedUpTemplates|***ND***: build the project|
