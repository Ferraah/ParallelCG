#!/bin/bash

# Script for benchmarking the MPI execution of the cg-algorithm. The matrix size stays constant (10000 x 10000) and the number of processes vary.
# The C++ script writes the results to a temporary file which then gets read out.
# The benchmarking results are stored in cg_bench_mpi_*.txt

# filename
filename='cg_bench_mpi_fixM.txt'

# Number of processes
ps=(1 2 4 8 16 32 64 128 256)
# ps=(16)

module load OpenMPI
mpic++ -o cg_timed_mpi src/cg_timed_mpi.cpp 
./random_spd_system.sh 10000 io/matrix.bin io/rhs.bin
echo "processes   matrix_size   time" > $filename

for i in "${ps[@]}"
    do
    srun -n $i ./cg_timed_mpi io/matrix.bin io/rhs.bin io/sol.bin 
    temp_path='io/temp.txt'
    n=1
    while read line; do
    echo "RESULTS FOR THIS ITERATION: $line"
    echo "$i  $line" >> $filename
    n=$((n+1))
    done < $temp_path
done