#!/bin/bash

output="results.txt"

# compile the executables

echo compiling...
g++ parallel_io.cpp utils.cpp -o parallel -lmpi -O3
g++ serial_io.cpp utils.cpp -o serial -lmpi -O3


echo "  dimensions: 1 4 8 16 32 64 128 256 512 1024"
echo ""
echo ""

for size in 100 500 1000 5000 10000 20000 30000 40000 50000 60000; do
    path_to_mat="/project/home/p200301/tests/matrix$size.bin"
    path_to_vec="/project/home/p200301/tests/rhs$size.bin"

    time_output=$(./serial $path_to_mat $path_to_vec)
    printf "%6d |" $size 
    printf "%7.0f " $time_output
    for processes in 4 8 16 32 64 128 256 512 1024; do
        if [ $processes -le $size ]; then
            time_output=$(srun -n $processes ./parallel $path_to_mat $path_to_vec)
            printf "%7.0f " $time_output
        else
            printf "        " # Placeholder for empty cells
        fi
    done
    echo " "
done
