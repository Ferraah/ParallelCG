#!/bin/bash

# Script for benchmarking the MPI+OpenMP execution of the cg-algorithm. The matrix size stays constant (10000 x 10000) and the number of processes and threads vary.
# The C++ script writes the results to a temporary file which then gets read out.
# The benchmarking results are stored in cg_bench_mpi_omp_*.txt

# filename
filename='cg_bench_mpi_omp_fixM.txt'

# Number of processes
ps=(1 2 4 8 16 32 64 128 256)
# Number of threads
ts=(1 2 4 8 16)

module load OpenMPI
mpic++ -o cg_mpi_omp src/cg_mpi_omp.cpp -fopenmp
./random_spd_system.sh 10000 io/matrix.bin io/rhs.bin
echo "processes   threads    matrix_size   time" > $filename

for p in "${ps[@]}"
    do
    for t in "${ts[@]}"
        do
        export OMP_NUM_THREADS=$t
        srun -n $p ./cg_mpi_omp io/matrix.bin io/rhs.bin io/sol.bin
        temp_path='io/temp.txt'
        n=1
        while read line; do
        echo "RESULTS FOR THIS ITERATION: $line"
        echo "$p    $t    $line" >> $filename
        n=$((n+1))
        done < $temp_path
    done 
done