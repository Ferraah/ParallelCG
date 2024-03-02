#ifndef CUDA_SINGLE_GPU_CUH
#define CUDA_SINGLE_GPU_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16


/**
 * @author Luca Guffanti (luca2.guffanti@mail.polimi.it)
 * @brief This file contains the code of the kernels
 * used to implement a conjugate gradient solver on a single A100 Nvidia GPU
 */

/**
* @brief Computes the element-wise sum of two vectors of floating point numbers,
* with vectors optionally scaled by a factor, and saves the result in the second vector
* @param a scaling factor of the first vector
* @param x first term of the sum
* @param b scaling factor of the second vector
* @param y second term of the sum
* @param size number of elements of the vectors
*
*
* In order to favor a coalesced access to the global memory each thread is associated to a 
* non-contiguous set of elements which depends on the unique thread id and on the stride constant. 
* The stride constant maps a single thread to various elements, and is needed when the total number of
* elements in the vectors is greater than the total number of allocated threads.
*
*/ 
__global__ void axpby(const double a, double* x, const double b, double* y, const unsigned int size)
{
const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = tid; i < size; i += stride)
    y[i] = a * x[i] + b * y[i];
}


/**
* @brief Computes the dot product between two vectors, x and y.
* 
* @param res Vector containing all partial dot products computed by each block of threads
* @param x first term of the dot product
* @param y second term of the dot product
* @param size size of each single vector
*
* In order to get good performance, this kernel makes extensive use of the shared memory available to
* each thread block. Each thread is associated to a set of non-contiguous elements via a stride term in order
* have a coalesced memory access, and the product of each pair of elements, which is an intermediate result, is stored
* in a vector present in shared memory. Once the first set of computation is complement, i.e. once all the partial products
* have been computed and the shared memory updated (exclusive writes on shared memory) with the accumulation of the partial products,
* a second intermediate result is computed by reducing the shared memory vector in parallel with the tree-like approach.
* Last but not least the result of this operation is moved to the result vector (which has a number of elements equal 
* to the total number of blocks), and it's up to the caller to compute a final reduction, yielding the dot product.
*  
* @note As the kernel allocates a number of shared memory bytes equal to the size of a double times the total number of
* active threads, the number of threads per block that can be activated is strictly implementation related, and depends on the
* capability of each GPU. Nonetheless, a reduced number of available threads can be balanced by the allocation of more blocks.
*/
__global__ void dot(double* res, double* x, double* y, const unsigned int size)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int local_id = threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int threads_per_block = blockDim.x;
    
    extern __shared__ double partial_product[];
    
    // First initialize the array of partial sums. No loops as the shared memory array has
    // a number of elements equal to the number of active threads.
    partial_product[local_id] = 0.0;
    __syncthreads();
    
    // Then each thread accumulates the partial scalar product in the shared memory
    for (unsigned int pos = tid; pos < size; pos = pos + stride)
    {   
        partial_product[local_id] += x[pos] * y[pos];
    }
    __syncthreads();
    
    // Finally the threads in a block execute a parallel reduction
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
    if (local_id == 0)
        res[blockIdx.x] = partial_product[0];
}


/**
 * @brief Computes the matrix-vector product with tiling.
 * 
 * @param A the matrix to be multiplied
 * @param x the vector to be multiplied
 * @param res the result of the multiplication
 * @param m number of rows of the matrix
 * @param n number of columns of the matrix
 *
 *
 * The GPU implementation of the GEMV algorithm makes extensive use of the shared memory.
 * More specifically, the matrix is subdivided into square tiles of size TILE_WIDTH which are associated
 * to entire blocks of threads. Each block therefore loads a small sub-matrix and a set of rows of the vector
 * into the device shared memory and computes a partial result. Each thread is a block is associated to a specific
 * row in a tile.
 *
 */
__global__ void gemv(
    double* A,
    double* x,
    double* res,
    unsigned int m,
    unsigned int n
)
{
    // Shared memory vectors. The first is used to load 
    // the matrix, while the second stores the vector
    __shared__ double shared_matrix[TILE_WIDTH][TILE_WIDTH];
    __shared__ double shared_vector[TILE_WIDTH];

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int row = bx * TILE_WIDTH + tx;

    double partial = 0.0;
    // Iterating through all the tiles into which the matrix is subdivided.
    for (unsigned int p = 0; p < (n-1)/TILE_WIDTH + 1; ++p)
    {
        // We start by loading the vector x.
        // The matrix may not be square and also it may happen
        // that the dimension does not divide completely the size of each
        // tile => we introduce padding with 0s.
        if (row < m && p * TILE_WIDTH + tx < n)
        {
            shared_vector[tx] = x[p * TILE_WIDTH + tx];
        }
        else
        {
            shared_vector[tx] = 0.0; 
        }
        __syncthreads();
        // Then we load the shared matrix. The idea here is the same.
        
        if (row < m)
        {
            for (int c = 0; c < TILE_WIDTH && p * TILE_WIDTH + c < n; ++c)
            {
                shared_matrix[tx][c] = A[row * n + p * TILE_WIDTH + c];
            }
            __syncthreads();

            // Finally we compute the partial results
            for (int i = 0; i < TILE_WIDTH && p * TILE_WIDTH + i < n; ++i)
            {
                partial += shared_matrix[tx][i] * shared_vector[i];
            }

        }        
        __syncthreads();
        // And move on to loading the next tile.
    }

    if (row < m)
    {
        res[row] = partial;
    }

}

#endif // CUDA_SINGLE_GPU_CUH