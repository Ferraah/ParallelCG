// Include (.cu) with the implementation of the kernel
#include "vec_sum.cuh"

// Include (.hpp) used to link with the .cpp file
#include "gpu_tests.hpp"

void vec_sum_func(double* a, double* b, unsigned int size)
{
    // just compute the stride and 
    // call the kernel.

    unsigned int num_blocks = 25;
    unsigned int thread_per_blocks = 512;

    // allocate device memory
    double* dev_a;
    cudaMalloc(&dev_a, size * sizeof(double));
    double* dev_b;
    cudaMalloc(&dev_b, size * sizeof(double));

    cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(double), cudaMemcpyHostToDevice);
    std::cout << "Calling the kernel" << std::endl;
    vec_sum_kernel<<<num_blocks, thread_per_blocks>>>(dev_a, dev_b, size);
    cudaDeviceSynchronize();
    std::cout << "Kernel returned" << std::endl;
    cudaMemcpy(a, dev_a, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dev_a);
    cudaFree(dev_b);
}

double vec_dot_func(const double* a, const double* b, const unsigned int size)
{
    unsigned int num_blocks = 4;
    unsigned int threads_per_block = 256;

    // allocate device memory
    std::cout << "Allocating memory on the device" << std::endl;
    double* dev_a;
    cudaMalloc(&dev_a, size * sizeof(double));
    double* dev_b;
    cudaMalloc(&dev_b, size * sizeof(double));
   
    double* res;
    res = (double*) malloc(num_blocks * sizeof(double));
    double* dev_res;
    cudaMalloc(&dev_res, num_blocks * sizeof(double));

    for(unsigned int i = 0; i < size; i++)
    {
        std::cout << a[i];
    }

    for(unsigned int i = 0; i < size; i++)
    {
        std::cout << b[i];
    }

    // copy data to the device
    cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(double), cudaMemcpyHostToDevice);

    std::cout << "Calling the kernel" << std::endl;
    vec_dot_kernel<<<num_blocks, threads_per_block, threads_per_block * sizeof(double)>>>(a, b, size, res, threads_per_block);
    cudaDeviceSynchronize();

    cudaMemcpy(res, dev_res, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "Computing the reduction" << std::endl;
    
    double dot_product = 0;

    for(unsigned int i = 0; i < num_blocks; i++)
    {
        std::cout << res[i] << std::endl;
        dot_product = dot_product + res[i];
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_res);

    free(res);

    return dot_product;
}