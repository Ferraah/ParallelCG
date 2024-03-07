#!/bin/bash

# Script for benchmarking the OpenMP+OpenBLAS execution of the cg-algorithm. The matrix size stays constant (10000 x 10000) and the number of threads vary.
# The C++ script writes the results to a temporary file which then gets read out.
# The benchmarking results are stored in cg_bench_omp_blas_fixM.txt

# filename
filename='cg_bench_omp_blas_fixM.txt'

# Number of threads
ts=(1 2 4 6 8 10 12 14 16)
# ts=(8)

module load OpenBLAS
module load intel
icpx -O2 src/cg_omp_blas.cpp -o cg_omp_blas -lopenblas -fopenmp
#g++ -O3 -fopenmp src/cg_timed_ompV02.cpp -o cg_timed_ompV02
# icpx -O2 or g++
./random_spd_system.sh 10000 io/matrix.bin io/rhs.bin
echo "threads   matrix_size   time" > $filename

for i in "${ts[@]}"
    do
    export OMP_NUM_THREADS=$i
    ./cg_omp_blas io/matrix.bin io/rhs.bin io/sol.bin
    temp_path='io/temp.txt'
    n=1
    while read line; do
    echo "RESULTS FOR THIS ITERATION: $line"
    echo "$i  $line" >> $filename
    n=$((n+1))
    done < $temp_path
done