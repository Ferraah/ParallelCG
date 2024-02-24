#ifndef CUDA_TEST_KERNELS
#define CUDA_TEST_KERNELS

#include <cuda.h>
#include <stdio.h>

/**
* Computes the sum of two vectos, a*x + y 
* saving the result in x
*/
__global__ void daxpy_kernel(
    double a,
    double* x,
    double* y,
    const unsigned int dim)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;    
    unsigned int stride = blockDim.x * gridDim.x;

    for(unsigned int pos = id; pos < dim; pos = pos + stride)
    {
        x[pos] = a * x[pos] + y[pos];
    }
}

/**
* Compute a dot product between two double arrays
* with the use of shared memory. Optimized to perform reductions in parallel.
*/
__global__ void vec_dot_kernel_optimized(
    double* a, 
    double* b, 
    unsigned int dim, 
    double* res)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int local_id = threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int threads_per_block = blockDim.x;
    
    // This must be defined externally as the total number of threads per blocks.
    extern __shared__ double partial_product[];
    
    // First initialize the array of partial sums
    partial_product[local_id] = 0.0;
    __syncthreads();
    
    // Then each thread accumulates the partial scalar product in the shared memory
    for (unsigned int pos = id; pos < dim; pos = pos + stride)
    {   
        partial_product[local_id] += a[pos] * b[pos];
    }
    __syncthreads();
    
    for(unsigned int s = threads_per_block/2; s > 0; s >>= 1)
    {
        if(local_id < s)
        {
            partial_product[local_id] += partial_product[local_id + s];
        }
        __syncthreads();
    }

    // Then, as the parallel reduction accumulates the result on the
    // left of the array, the result is the first element of the shared memory vector
    res[blockIdx.x] = partial_product[0];
}


/**
* Compute the matrix-vector product (on the right). The matrix
* is linearized. This first kernel is rather simple: each row is mapped
* to a specific thread
* 
*/
__global__ void A_times_x_kernel(
    double* A,
    double* x,
    double* res,
    unsigned int num_rows,
    unsigned int num_cols
)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int local_id = threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // The idea here is that each thread of each block is assigned to a specific row of
    // the matrix.
    for (unsigned int row = tid; row < num_rows; row += stride)
    {
        double accumulator = 0.0;
        // And here there is no use of shared memory.
        for (unsigned int col = 0; col < num_cols; ++col)
        {
            accumulator += A[row * num_cols + col] * x[col];
        }
        // Finally, store the result in the corresponsing element
        // of the vector.
        res[row] = accumulator;
    }
}

/**
* Compute the matrix-vector product (on the right). The matrix
* is linearized. This version is the optimized kernel, which uses
* shared memory and optimized gpu programming techniques
* 
*/
__global__ void A_times_x_kernel_opt(
    double* A,
    double* x,
    double* res,
    unsigned int num_rows,
    unsigned int num_cols
)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int local_id = threadIdx.x;


}
#endif // CUDA_TEST_VEC_SUM
    