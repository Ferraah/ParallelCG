#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cassert>
#include <chrono>
#include "gpu_tests.hpp"

#define SIZE 1<<30
#define NUM_ROWS 1<<10
#define NUM_COLS NUM_ROWS
#define MAX_ERR 1e-5


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
    double* A;
    double* x;
    double* cpu_res;
    double* gpu_res;

    unsigned int num_rows = NUM_ROWS;
    unsigned int num_cols;



    // computing the product
    std::cout << "serial CPU computation...";
    unsigned int begin_row_index;
    unsigned int end_row_index;

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
    std::cout << "done";

    A_times_x_func(A, x, gpu_res, num_rows, num_cols);

    std::cout << "Checking the result...";
    for (unsigned int i = 0; i < num_rows; ++i)
    {
        assert("Error doing calculation" && std::abs(gpu_res[i] - cpu_res[i]) < MAX_ERR);
    }
    std::cout << " Correct!" << std::endl;
}

int main()
{
    // test_dot_product();   
    test_matrix_vector_product();

    return 0;
}