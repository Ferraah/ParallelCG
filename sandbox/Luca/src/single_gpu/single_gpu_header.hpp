#ifndef CUDA_SINGLE_GPU_HPP
#define CUDA_SINGLE_GPU_HPP

#include <iostream>

    /**
     * @brief Iteratively solves the conjugate gradient method on a single GPU.
     * 
     * @param A System matrix
     * @param x Unknowns vector
     * @param b Right hand side vector
     * @param size Size of the matrix
     * @param max_iters Maximum number of allowed iteration for convergence
     * @param rel_error Maximum allowed relative error
     */
    void conjugate_gradient(const double * A, double * x, const double * b, size_t size, int max_iters, double rel_error);


#endif // CUDA_SINGLE_GPU_HPP