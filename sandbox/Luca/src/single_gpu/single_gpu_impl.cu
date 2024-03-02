// Conjugate gradient function
#include "single_gpu_header.hpp"

// Cuda kernels
#include "single_gpu_kernels.cuh"

void conjugate_gradient(const double * A, double * x, const double * b, size_t size, int max_iters, double rel_error)
{
    // Declare the vectors used on the CPU

    double dot_temp;

    const unsigned int dot_product_threads = 256;
    const unsigned int dot_product_blocks = (size-1)/dot_product_threads + 1;
    const unsigned int gemv_tile_width = 16;
    const unsigned int gemv_blocks = (size - 1)/ gemv_tile_width + 1;
    const unsigned int gemv_threads = gemv_tile_width;
    const unsigned int axpby_threads = 256;
    const unsigned int axpby_blocks = (size-1)/axpby_threads + 1;
    const unsigned int matrix_bytes = sizeof(double) * size * size;
    const unsigned int vector_bytes = sizeof(double) * size;

    double* partial_dot = new double[dot_product_blocks];

    // First instantiate the vectors in the GPU memory
    double* dev_A;
    double* dev_b;
    double* dev_x;
    double* dev_Ap;
    double* dev_r;
    double* dev_p;
    double* dev_partial_dot;


    cudaMalloc(&dev_A, matrix_bytes);
    cudaMalloc(&dev_b, vector_bytes);
    cudaMalloc(&dev_x, vector_bytes);
    cudaMalloc(&dev_Ap, vector_bytes);
    cudaMalloc(&dev_r, vector_bytes);
    cudaMalloc(&dev_p, vector_bytes);
    cudaMalloc(&dev_partial_dot, sizeof(double) * dot_product_blocks);


    // And copy data into the global memory of the GPU
    cudaMemcpy(dev_A, A, matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, vector_bytes, cudaMemcpyHostToDevice);
    cudaMemset(dev_x, 0, vector_bytes);
    cudaMemcpy(dev_r, b, vector_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_p, b, vector_bytes, cudaMemcpyHostToDevice);


    int num_iters;
    double alpha, beta, bb, rr, rr_new;

    // compute bb
    num_iters = 1;
    dot<<<dot_product_blocks, dot_product_threads, dot_product_threads * sizeof(double)>>>(
        dev_partial_dot,
        dev_b,
        dev_b,
        size
    );
    cudaDeviceSynchronize();

    cudaMemcpy(partial_dot, dev_partial_dot, sizeof(double) * dot_product_blocks, cudaMemcpyDeviceToHost);
    bb = 0.0;
    
    #pragma unroll
    for (unsigned int i = 0; i < dot_product_blocks; ++i)
    {
        bb = bb + partial_dot[i];
    }
    rr = bb;

    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        // Compute A*p
        gemv<<<gemv_blocks, gemv_threads>>>(
            dev_A, 
            dev_p,
            dev_Ap, 
            size,
            size
        );
        cudaDeviceSynchronize();


        // Compute the dot product (p, Ap)
        dot<<<dot_product_blocks, dot_product_threads, dot_product_threads * sizeof(double)>>>(
            dev_partial_dot,
            dev_p,
            dev_Ap,
            size
        );
        cudaDeviceSynchronize();
        cudaMemcpy(partial_dot, dev_partial_dot, sizeof(double) * dot_product_blocks, cudaMemcpyDeviceToHost);
        dot_temp = 0.0;
        for (unsigned int i = 0; i < dot_product_blocks; ++i)
        {
            dot_temp = dot_temp + partial_dot[i];
        }
        alpha = rr / dot_temp;

        // Compute the correction of x with the residual

        axpby<<<axpby_blocks, axpby_threads>>>(
            alpha,
            dev_p,
            1.0,
            dev_x,
            size
        );
        cudaDeviceSynchronize();

        // Compute the new residual vector

        axpby<<<axpby_blocks, axpby_threads>>>(
            -alpha, 
            dev_Ap, 
            1.0, 
            dev_r, 
            size
        );  
        cudaDeviceSynchronize();

        // Compute the new norm of the residual
        dot<<<dot_product_blocks, dot_product_threads, dot_product_threads * sizeof(double)>>>(
            dev_partial_dot,
            dev_r,
            dev_r,
            size
        );
        cudaDeviceSynchronize();
        cudaMemcpy(partial_dot, dev_partial_dot, sizeof(double) * dot_product_blocks, cudaMemcpyDeviceToHost);
        dot_temp = 0.0;
        for (unsigned int i = 0; i < dot_product_blocks; ++i)
        {
            dot_temp = dot_temp + partial_dot[i];
        }

        rr_new = dot_temp;
        beta = rr_new / rr;
        rr = rr_new;

        if(std::sqrt(rr / bb) < rel_error) { break; }
        axpby<<<axpby_blocks, axpby_threads>>>(
            1.0,
            dev_r,
            beta,
            dev_p, 
            size
        );
        cudaDeviceSynchronize();
    }

    if(num_iters <= max_iters)
    {
        printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
    }
    cudaMemcpy(x, dev_x, sizeof(double) * size, cudaMemcpyDeviceToHost);
    // All Cuda free that will eventually come

}


