#ifndef CUDA_MPI_HPP
#define CUDA_MPI_HPP

// === ABSTRACT STRATEGY
#include "../CGStrategy.hpp"

// === STL INCLUDES ===
#include <cstdlib>
#include <iostream>
#include <cassert>
#include <unistd.h>
#include <stdint.h>

// === CUDA INCLUDES ===
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// === MPI INCLUDES ===
#include <mpi.h>


namespace cgcore
{
    class CUDA_MPI : public CGStrategy
    {
        public:
            // Functions to map the MPI rank to the available GPU
            static uint64_t getHostHash(const char* string);
            static void getHostName(char* hostname, int maxlen);

            void run(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error) const;
            
        private:
            void conjugate_gradient(const double* distr_A, double* x, const double* b, unsigned int rows, unsigned int cols, int* rows_per_process, int* displacements, const int max_iter, const double rel_err) const;
    };

};


#endif // CUDA_MPI