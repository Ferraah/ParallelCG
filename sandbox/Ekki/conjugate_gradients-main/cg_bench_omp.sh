#!/bin/bash

# Script for benchmarking the openMP execution of the cg-algorithm
# The C++ script writes the results to a temporary file which then gets read out.
# The benchmarking results are stored in cg_bench_omp.txt

# Matrix sizes
# ms=(1 5 10 50 100 500 1000 5000 10000 50000 100000)
ms=(1 5 10 50 100 500 1000 5000 10000)

module load intel
# export OMP_NUM_THREADS=4 # for example
icpx -O2 -fopenmp src/cg_timed_omp.cpp -o cg_timed_omp

echo "matrix_size   time" > cg_bench_omp.txt

for i in "${ms[@]}"
    do
    ./random_spd_system.sh $i io/matrix.bin io/rhs.bin
    ./cg_timed_omp io/matrix.bin io/rhs.bin io/sol.bin
    temp_path='io/temp.txt'
    n=1
    while read line; do
    echo "RESULTS FOR THIS ITERATION: $line"
    echo "$line" >> cg_bench_omp.txt
    n=$((n+1))
    done < $temp_path
done