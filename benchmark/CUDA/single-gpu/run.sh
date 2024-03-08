#!/bin/bash


# Compilation
echo "Compiling the executable..."
nvcc CudaSingleGPU.cpp CUBLAS_single.cu utils.cpp -I/ -lcublas -lmpi -O3

# Number of iterations for benchmarking
num_iter=8

# Output file
output_file="benchmark.txt"

# Path to directory containing matrices and vectors
path_to_dir="/project/home/p200301/tests/"

# Loop over different matrix sizes
for matrix_size in 100 500 1000 5000 10000 20000 30000 40000 50000; do
    echo "========== Matrix Size: $matrix_size =========="
    echo "========== Matrix Size: $matrix_size ==========" >> "$output_file"

    matrix_file="$path_to_dir/matrix${matrix_size}.bin"
    vector_file="$path_to_dir/rhs${matrix_size}.bin"

    min_time=0
    echo "Accessing directory $path_to_dir for matrix of size $matrix_size"
    
    # Execute program multiple times and record minimum time
    for ((iter=0; iter<=$num_iter; iter++)); do
        time_output=$(./a.out "$matrix_file" "$vector_file" $matrix_size)
        if [ "$iter" -eq 0 ]; then
            min_time="$time_output"
        elif [ "$time_output" -lt "$min_time" ]; then
            min_time="$time_output"
        fi
    done

    # Log minimum time for the matrix size
    echo "Minimum Time: $min_time ms" >> "$output_file"
    echo "Minimum Time: $min_time ms"
done

echo Finished!
rm a.out