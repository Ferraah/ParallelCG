#!/bin/bash

srun -N 1 -n 1 -c 256  ./test/mpi_openmp >> mpi_openmp_combinations.txt 

srun -N 1 -n 2 -c 128 ./test/mpi_openmp >> mpi_openmp_combinations.txt 

srun -N 1 -n 4 -c 64 ./test/mpi_openmp >> mpi_openmp_combinations.txt 

srun -N 1 -n 8 -c 32 ./test/mpi_openmp >> mpi_openmp_combinations.txt 

srun -N 1 -n 16 -c 16 ./test/mpi_openmp >> mpi_openmp_combinations.txt 

srun -N 1 -n 32 -c 8 ./test/mpi_openmp >> mpi_openmp_combinations.txt 

srun -N 1 -n 64 -c 4 ./test/mpi_openmp >> mpi_openmp_combinations.txt 

srun -N 1 -n 128 -c 2 ./test/mpi_openmp  >> mpi_openmp_combinations.txt 

srun -N 1 -n 256 -c 1 ./test/mpi_openmp >> mpi_openmp_combinations.txt 