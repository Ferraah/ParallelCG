#ifndef CUDA_TEST_VEC_SUM
#define CUDA_TEST_VEC_SUM

#include <cuda.h>
#include <stdio.h>

/**
* More of a proof of concept:
* It computes the sum between two vectors of numbers: no shared
* memory, no optimizations.
*/
__global__ void vec_sum_kernel(double* a, double* b, const unsigned int dim)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    unsigned int stride = blockDim.x * gridDim.x;
    for(unsigned int pos = id; pos < dim; pos = pos + stride)
        a[pos] = a[pos] + b[pos];
}

__global__ void vec_dot_kernel(
                            const double* a, 
                            const double* b, 
                            const unsigned int dim, 
                            double* res,
                            const unsigned int threadsPerBlock)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // This must be defined externally as the total number of threads per blocks.
    extern __shared__ double partial_product[];

    // First initialize the array of partial sums
    partial_product[threadIdx.x] = 0.0;
    __syncthreads();

    // Then each thread accumulates the partial scalar product in the shared memory
    for (unsigned int pos = id; pos < dim; pos = pos + stride)
    {   
        partial_product[threadIdx.x] += a[pos] * b[pos];
        printf("%f\n", partial_product[threadIdx.x]);
    }
    __syncthreads();

    // TODO: implement the parallel reduction in the shared memory
    // Finally, the first thread computes the reduction.
    if (threadIdx.x == 0)
    {
        double sum = 0.0;
        for (unsigned int i = 0; i < threadsPerBlock; i++)
        {
            sum = sum + partial_product[i];
        }
        // And last but not least add the final sum over the block to the
        // element at index equal to the block id to the position.
        res[blockIdx.x] = 500;
    }
}
#endif // CUDA_TEST_VEC_SUM
