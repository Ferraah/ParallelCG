#!/bin/bash

# Script for benchmarking the openMP execution of the cg-algorithm. The matrix size stays constant (10000 x 10000) and the number of threads vary.
# The C++ script writes the results to a temporary file which then gets read out.
# The benchmarking results are stored in cg_bench_omp_Xthreads.txt

# Number of threads
ts=(1 2 3 4 5 6 7 8 9 10)

module load intel
icpx -O2 -fopenmp src/cg_timed_omp.cpp -o cg_timed_omp
./random_spd_system.sh 10000 io/matrix.bin io/rhs.bin
echo "threads   matrix_size   time" > cg_bench_omp_fixM.txt

for i in "${ts[@]}"
    do
    export OMP_NUM_THREADS=$i # for example
    ./cg_timed_omp io/matrix.bin io/rhs.bin io/sol.bin
    temp_path='io/temp.txt'
    n=1
    while read line; do
    echo "RESULTS FOR THIS ITERATION: $line"
    echo "$i  $line" >> cg_bench_omp_fixM.txt
    n=$((n+1))
    done < $temp_path
done