// Include (.cu) with the implementation of the kernel
#include "vec_sum.cuh"

// Include (.hpp) used to link with the .cpp file
#include "gpu_tests.hpp"

#include <iostream>
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