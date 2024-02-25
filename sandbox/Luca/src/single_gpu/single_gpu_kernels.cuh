#include <cuda.h>
#include <cuda_runtime.h>

/**
 * @author Luca Guffanti (luca2.guffanti@mail.polimi.it)
 * @brief This file contains the code of the kernels
 * used to implement a conjugate gradient solver on a single A100 Nvidia GPU
 */

/**
* @brief Computes the element-wise sum of two vectors of floating point numbers,
* with vectors optionally scaled by a factor, and saves the result in the first vector
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
__global__ void axpy(const double a, double* x, const double b, double* y, const unsigned int size)
{
const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = tid; i < size; i += stride)
    x[i] = a * x[i] + b * y[i];
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
    for (unsigned int pos = tid; pos < dim; pos = pos + stride)
    {   
        partial_product[local_id] += a[pos] * b[pos];
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