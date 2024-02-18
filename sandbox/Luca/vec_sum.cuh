#ifndef CUDA_TEST_VEC_SUM
#define CUDA_TEST_VEC_SUM

/**
* More of a proof of concept:
* It computes the sum between two vectors of numbers: no shared
* memory, no optimizations.
*/
__global__ void vec_sum_kernel(double* a, double* b, unsigned int dim)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    unsigned int stride = blockDim.x * gridDim.x;
    
    for(unsigned int pos = id; pos < dim; pos = pos + stride)
        a[pos] = a[pos] + b[pos];
}

#endif // CUDA_TEST_VEC_SUM