
#include "CUBLAS_CG.hpp"

namespace cgcore {
    
    void CUBLAS_CG::run(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error) const
    {
        // Handle to the library
        cublasHandle_t handle = NULL;
        cublasCreate(&handle);

        // Dimensions of the matrix as described by the API
        const int m = size;
        const int n = size;
        // Extension of the leading dimension in the memory layout of A
        const int lda = m;

        // Constants used by the problem
        double alpha, beta, bb, rr, rr_new;
        int num_iters;


        // Vectors used on the GPU. Respectively,
        // dev_A   : linearized matrix
        // dev_x   : solution vector
        // dev_b   : known values vector
        // dev_r   : residual vector
        // dev_p   : temporary conjugate direction vector
        // dev_Ap  : temporary vector storing A * p

        double* dev_A;
        double* dev_x;
        double* dev_b;
        double* dev_r;
        double* dev_p;
        double* dev_Ap;

        // Allocating vectors on GPU
        const unsigned int matrix_bytes = m * n * sizeof(double);
        const unsigned int vector_bytes = m * sizeof(double);

        cudaMalloc(&dev_A, matrix_bytes);
        cudaMalloc(&dev_x, vector_bytes);
        cudaMalloc(&dev_b, vector_bytes);
        cudaMalloc(&dev_r, vector_bytes);
        cudaMalloc(&dev_p, vector_bytes);
        cudaMalloc(&dev_Ap, vector_bytes);

        // Transferring data to the GPU
        cudaMemcpy(dev_A, A, matrix_bytes, cudaMemcpyHostToDevice);
        cudaMemset(dev_x, 0, vector_bytes);
        cudaMemcpy(dev_b, b, vector_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_r, b, vector_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_p, b, vector_bytes, cudaMemcpyHostToDevice);

        // Auxiliary variables required by CUBLAS to call the API functions.
        // CUBLAS, in fact, requires the user to provide pointers to variables in which constants are stored.
        double one = 1.0;
        double zero = 0.0;
        double negative_alpha;
        double den;

        cublasDdot(handle, m, dev_b, 1, dev_b, 1, &bb);    
        rr = bb;
        for(num_iters = 1; num_iters <= max_iters; num_iters++)
        {

            cublasDgemv(handle, CUBLAS_OP_N, m, n, &one, dev_A, m, dev_p, 1, &zero, dev_Ap, 1);
            cublasDdot(handle, m, dev_p, 1, dev_Ap, 1, &den);
            alpha = rr / den;

            cublasDaxpy(handle, m, &alpha, dev_p, 1, dev_x, 1);
            

            negative_alpha = -alpha;
            cublasDaxpy(handle, m, &negative_alpha, dev_Ap, 1, dev_r, 1);

            cublasDdot(handle, m, dev_r, 1, dev_r, 1, &rr_new);

            beta = rr_new / rr;
            rr = rr_new;

            if(std::sqrt(rr / bb) < rel_error) { break; }

            // CUBLAS does not provide an implementation of the Daxpby algorithm,
            // so it's necessary to first rescale the vector and then apply Daxpy.
            // Even though these two calls could be merged, implementation specific
            // optimization employed by CUBLAS still outperform custom implementations.

            cublasDscal(handle, m, &beta, dev_p, 1);
            cublasDaxpy(handle, m, &one, dev_r, 1, dev_p, 1);
        }

        // As the algorithm terminates, if convergence is reached, the result is brought back to
        // the host. Otherwise, the x vector is set to be the vector of all zeroes.
        if(num_iters <= max_iters)
        {
            printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
            cudaMemcpy(x, dev_x, vector_bytes, cudaMemcpyDeviceToHost);
        }
        else
        {
            printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
            memset(x, 0, vector_bytes);
        }

        // Last but not least, the device memory is freed.
        cudaFree(dev_A);
        cudaFree(dev_x);
        cudaFree(dev_b);
        cudaFree(dev_r);
        cudaFree(dev_p);
        cudaFree(dev_Ap);
    }
}





