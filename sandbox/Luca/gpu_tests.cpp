#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cassert>
#include <chrono>
#include <ctime>
#include <climits>
#include "gpu_tests.hpp"

#define SIZE 1<<30
#define NUM_ROWS 4096 * 10
#define NUM_COLS NUM_ROWS
#define MAX_ERR 1e-2


void test_dot_product()
{
    double* a = (double*) malloc(sizeof(double) * SIZE);
    double* b = (double*) malloc(sizeof(double) * SIZE);

    std::cout << "Allocated arrays of " << (SIZE) << " doubles" << std::endl;
    double dot = 0.0; 
    for (unsigned int i = 0; i < SIZE; i++)
    {
        a[i] = 1.0;
        b[i] = 1.0;
    }
    
    // Test for increasing number of sizes
    for (unsigned int size = 256; size <= 1<<30; size <<=1)
    {
        std::cout << "====== SIZE: " << size << " ======" << std::endl;

        for (unsigned int i = 0; i < size; i++)
        {
            a[i] = 1.0;
            b[i] = 1.0;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        dot = vec_dot_func_optimized(a, b, size);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "Computed dot product: " << std::setprecision(15) << dot << std::endl;
        std::cout << "Expected dot product: " << size << std::endl;
        std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " milliseconds" << std::endl;
        std::cout << "=============" << std::endl;
        assert(dot == size && "ERROR IN THE COMPUTATION OF THE DOT PRODUCT");
    }
}

void test_matrix_vector_product()
{

    unsigned int num_rows = NUM_ROWS;
    unsigned int num_cols = NUM_COLS;
    std::cout << "Using " << num_rows << " rows and cols" << std::endl;
    
    std::cout << "Allocating results for gpu" << std::endl;
    double* gpu_res = new double[num_rows];
    std::cout << "Allocating results for opt gpu" << std::endl;
    double* gpu_opt_res = new double[num_rows];
    std::cout << "Allocating results for opt gpu tiled" << std::endl;
    double* gpu_opt_tiled_res = new double[num_rows];

    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "Allocating matrix " << std::endl;
    double* A = new double[num_rows * num_cols];

    std::cout << "Allocating vector " << std::endl;
    double* x = new double[num_rows];

    std::cout << "Allocating results for cpu" << std::endl;
    double* cpu_res = new double[num_rows];

    // ======================= BUILDING THE MATRIX =======================
    std::cout << "Building the matrix...";
    for(unsigned int i = 0; i < num_rows; ++i)
    {
        for(unsigned int j = 0; j < num_cols; ++j)
        {
            A[i * num_cols + j] = rand()/(INT_MAX * 1.0);
        } 
        x[i] = rand() / (INT_MAX * 1.0);
    }
    std::cout << "done." << std::endl;


    // ======================= SERIAL COMPUTATION =======================

    std::cout << "serial CPU computation..." << std::endl;
    // computing the product


    double accumulator = 0.0;
    for(unsigned int r = 0; r < num_rows; ++r)
    {
        accumulator = 0.0;
        for (unsigned int c = 0; c < num_cols; ++c)
        {
            accumulator += A[r * num_cols + c] * x[c];
        }
        cpu_res[r] = accumulator;

    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "done" << std::endl;

    // ======================= GPU COMPUTATION, ONLY USE OF GLOBAL MEMORY =======================
    // THE USE OF THE GLOBAL MEMORY HAS THE LIMITATION OF HAVING SLOW MEMORY ACCESS

    auto t3 = std::chrono::high_resolution_clock::now();
    A_times_x_func(A, x, gpu_res, num_rows, num_cols);
    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout << "Checking the result..." << std::endl;
    for (unsigned int i = 0; i < num_rows; ++i)
    {
        // std::cout << gpu_res[i] << " " << cpu_res[i] << std::endl;
        assert("Error doing calculation" && (std::abs(gpu_res[i] - cpu_res[i]) < MAX_ERR));
    }
    std::cout << " Correct!" << std::endl;

    // ======================= GPU COMPUTATION, ALL THE VECTOR IN EACH BLOCK. =======================
    // THIS APPROACH HAS THE SIZE OF THE SHARED MEMORY OF EACH BLOCK BEING A LIMITATION

    // auto t5 = std::chrono::high_resolution_clock::now();
    // A_times_x_func_opt(A, x, gpu_opt_res, num_rows, num_cols);
    // auto t6 = std::chrono::high_resolution_clock::now();
    // std::cout << "Checking the result..." << std::endl;
    // for (unsigned int i = 0; i < num_rows; ++i)
    // {
    //     // std::cout << gpu_res[i] << " " << cpu_res[i] << std::endl;
    //     assert("Error doing calculation (gpu-optimized)" && std::abs(gpu_opt_res[i] - cpu_res[i]) < MAX_ERR);
    // }
    // std::cout << " Correct!" << std::endl;


    // ======================= GPU COMPUTATION, CHOOSE MAXIMUM VECTOR CHUNK IN SHARED MEMORY =======================
    // THIS APPROACH SHOULD BE THE MOST SCALABLE: THEORETICALLY WE USE THE SHARED MEMORY AND AT THE SAME TIME WE USE
    // STORE THE PARTIAL RESULTS. THE COMPUTATION IS TILED

    auto t7 = std::chrono::high_resolution_clock::now();
    A_times_x_func_opt_tiled(A, x, gpu_opt_tiled_res, num_rows, num_cols);
    auto t8 = std::chrono::high_resolution_clock::now();
    std::cout << "Checking the result..." << std::endl;
    for (unsigned int i = 0; i < num_rows; ++i)
    {
        std::cout << "i:" << i << " " << gpu_opt_tiled_res[i] << " " << cpu_res[i] << std::endl;
        assert("Error doing calculation (gpu-tiled)" && std::abs(gpu_opt_tiled_res[i] - cpu_res[i]) < MAX_ERR);
    }
    std::cout << " Correct!" << std::endl;

    std::cout << "Times (containing memory allocation)" << std::endl;
    std::cout << "Square matrix of size " << num_rows << std::endl; 
    std::cout << "CPU Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " milliseconds" << std::endl;
    std::cout << "GPU Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count() << " milliseconds" << std::endl;
    // std::cout << "GPU Time with shared: " << std::chrono::duration_cast<std::chrono::milliseconds>(t6-t5).count() << " milliseconds" << std::endl;
    std::cout << "GPU Time with tiled shared: " << std::chrono::duration_cast<std::chrono::milliseconds>(t8-t7).count() << " milliseconds" << std::endl;


    delete [] A;
    delete [] x;
    delete [] cpu_res;
    delete [] gpu_res;
    delete [] gpu_opt_res;
    delete [] gpu_opt_tiled_res;

}

void test_matrix_vector_product_single_node_distributed_mpi()
{
    return;
}

int main()
{
    // test_dot_product();   
    test_matrix_vector_product();
    // test_matrix_vector_product_single_node_distributed_mpi()
    return 0;
}