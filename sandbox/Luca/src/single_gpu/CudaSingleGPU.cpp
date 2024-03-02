#include <iostream>
#include "single_gpu_header.hpp"
#include "CUBLAS_single.hpp"
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <random>
#include <climits>

#define SIZE 22500


double dot(const double * x, const double * y, size_t size)
{
    double result = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        result += x[i] * y[i];
    }
    return result;
}



void axpby(double alpha, const double * x, double beta, double * y, size_t size)
{
    // y = alpha * x + beta * y

    for(size_t i = 0; i < size; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}



void gemv(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols) 
{
    // y = alpha * A * x + beta * y;

    for(size_t r = 0; r < num_rows; r++)
    {
        double y_val = 0.0;
        for(size_t c = 0; c < num_cols; c++)
        {
            y_val += alpha * A[r * num_cols + c] * x[c];
        }
        y[r] = beta * y[r] + y_val;
    }
}


int main()
{

    std::cout << "Allocating CPU memory" << std::endl;
    double* A = new double[SIZE * SIZE];
    double* x = new double[SIZE];
    double* b = new double[SIZE];

    std::cout << "Building matrix and vector" << std::endl;
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    // Fill the upper triangular part with random values
    std::cout << "Generating symmetric matrix" << std::endl;
    for (int i = 0; i < SIZE; ++i) {
        for (int j = i; j < SIZE; ++j) {
            A[i * SIZE + j] = dis(gen);
            A[j * SIZE + i] = A[i * SIZE + j]; // Make the matrix symmetric
        }
        b[i] = dis(gen);
    }
    // Make the matrix diagonally dominant to ensure positive definiteness
    std::cout << "Making it diagonally dominant" << std::endl;
    for (int i = 0; i < SIZE; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < SIZE; ++j) {
            row_sum += std::abs(A[i * SIZE + j]);
        }
        A[i * SIZE + i] += row_sum + 0.1; // Add a small positive diagonal shift
    }

    // for (int i = 0; i < SIZE; ++i)
    // {
    //     for (int j = 0; j < SIZE; ++j)
    //     {
    //         std::cout << A[i * SIZE + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    std::cout << "Calling conjugate gradient method" << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    conjugate_gradient(A, x, b, SIZE, 10, 1e-3);
    auto t2 = std::chrono::high_resolution_clock::now();


    auto t3 = std::chrono::high_resolution_clock::now();

    double alpha, beta, bb, rr, rr_new;
    int size = SIZE;
    double * r = new double[size];
    double * p = new double[size];
    double * Ap = new double[size];
    int num_iters;
    int max_iters = 1000;
    double rel_error = 1e-3;

    for(size_t i = 0; i < size; i++)
    {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
    }

    bb = dot(b, b, size);
    rr = bb;
    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        gemv(1.0, A, p, 0.0, Ap, size, size);
        alpha = rr / dot(p, Ap, size);
        axpby(alpha, p, 1.0, x, size);
        axpby(-alpha, Ap, 1.0, r, size);
        rr_new = dot(r, r, size);
        std::cout << rr_new << std::endl;
        beta = rr_new / rr;
        rr = rr_new;

        if(std::sqrt(rr / bb) < rel_error) { break; }
        axpby(1.0, r, beta, p, size);
    }

    delete[] r;
    delete[] p;
    delete[] Ap;

    if(num_iters <= max_iters)
    {
        printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
    }
    auto t4 = std::chrono::high_resolution_clock::now();

    auto t5 = std::chrono::high_resolution_clock::now();
    conjugate_gradient_blas(A, x, b, SIZE, 10, 1e-3);
    auto t6 = std::chrono::high_resolution_clock::now();
    std::cout << "Dense Matrix of size: " << SIZE << std::endl;
    std::cout << "CPU serial  : " << std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count() << " milliseconds" << std::endl;
    std::cout << "GPU vanilla : " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " milliseconds" << std::endl;
    std::cout << "GPU blas    : " << std::chrono::duration_cast<std::chrono::milliseconds>(t6-t5).count() << " milliseconds" << std::endl;

}