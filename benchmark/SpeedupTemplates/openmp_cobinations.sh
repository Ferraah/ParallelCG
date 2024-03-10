#!/bin/bash

srun -N 1 -n 1 -c 256  ./test/openmp >> openmp_combinations.txt 
srun -N 1 -n 1 -c 128 ./test/openmp >> openmp_combinations.txt 
srun -N 1 -n 1 -c 64 ./test/openmp >> openmp_combinations.txt 
srun -N 1 -n 1 -c 32 ./test/openmp >> openmp_combinations.txt 
srun -N 1 -n 1 -c 16 ./test/openmp >> openmp_combinations.txt 
srun -N 1 -n 1 -c 8 ./test/openmp >> openmp_combinations.txt 
srun -N 1 -n 1 -c 4 ./test/openmp >> openmp_combinations.txt 
srun -N 1 -n 1 -c 2 ./test/openmp >> openmp_combinations.txt 
srun -N 1 -n 1 -c 1 ./test/openmp >> openmp_combinations.txt 
