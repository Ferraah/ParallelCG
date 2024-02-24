// Include (.cu) with the implementation of the kernel
#include "cuda_kernels.cuh"

// Include (.hpp) used to link with the .cpp file
#include "gpu_tests.hpp"

void daxpy_func(double a, double* x, double* y, unsigned int size)
{
    // just compute the stride and 
    // call the kernel.

    unsigned int num_blocks = 25;
    unsigned int thread_per_blocks = 512;

    // allocate device memory
    double* dev_x;
    cudaMalloc(&dev_x, size * sizeof(double));
    double* dev_y;
    cudaMalloc(&dev_y, size * sizeof(double));

    cudaMemcpy(dev_x, x, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, size * sizeof(double), cudaMemcpyHostToDevice);
    std::cout << "Calling the kernel" << std::endl;

    double c = 1.0;

    saxpy_kernel<<<num_blocks, thread_per_blocks>>>(c, dev_x, dev_y, size);
    cudaDeviceSynchronize();
    std::cout << "Kernel returned" << std::endl;
    cudaMemcpy(x, dev_x, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dev_x);
    cudaFree(dev_y);
}

double vec_dot_func_optimized(double* a, double* b, unsigned int size)
{
    unsigned int num_blocks = 25;
    unsigned int threads_per_block = 1024;

    // allocate device memory
    double* dev_a;
    cudaMalloc(&dev_a, size * sizeof(double));
    double* dev_b;
    cudaMalloc(&dev_b, size * sizeof(double));
   
    double* res;
    res = (double*) malloc(num_blocks * sizeof(double));
    double* dev_res;
    cudaMalloc(&dev_res, num_blocks * sizeof(double));

    std::cout << std::endl;
    // copy data to the device
    cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(double), cudaMemcpyHostToDevice);

    vec_dot_kernel_optimized<<<num_blocks, threads_per_block, threads_per_block * sizeof(double)>>>(dev_a, dev_b, size, dev_res);
    cudaDeviceSynchronize();

    cudaMemcpy(res, dev_res, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);
    double dot_product = 0;

    for(unsigned int i = 0; i < num_blocks; i++)
    {
        dot_product = dot_product + res[i];
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_res);

    free(res);

    return dot_product;
}

void A_times_x_func(double* A,
    double* x, 
    double* res, 
    unsigned int num_rows, 
    unsigned int num_cols)
{
    unsigned int num_blocks = 25;
    unsigned int num_threads = 1024;
    
    double* dev_A;
    double* dev_x;

    double* dev_res;

    // allocating all the vector on the gpu

    unsigned int total_elems = num_rows * num_cols;

    std::cout << "Allocating gpu memory...";

    cudaMalloc(&dev_A, total_elems * sizeof(double));
    cudaMalloc(&dev_x, (num_rows) * sizeof(double));
    cudaMalloc(&dev_res, (num_rows) * sizeof(double));

    std::cout << "done" << std::endl;
    // copying data from cpu to gpu

    std::cout << "Copying data towards gpu...";
    cudaMemcpy(dev_A, A, total_elems * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x, x, num_rows * sizeof(double), cudaMemcpyHostToDevice);
    std::cout << "done" << std::endl;


    unsigned int num_blocks = 3;
    unsigned int threads_per_block = 1024;

    std::cout << "Calling the kernel...";
    A_times_x_kernel<<<num_blocks, threads_per_block>>>(
        dev_A,
        dev_x,
        dev_res,
        num_rows,
        num_cols,
    )
    cudaDeviceSynchronize();
    std::cout << "done";

    cudaMemcpy(res, dev_res, num_rows * sizeof(double), cudaMemcpyDeviceToHost);
    
}

