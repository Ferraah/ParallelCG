#ifndef CUDA_MPI
#define CUDA_MPI

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

        virtual void run(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error) const;
        
        private:
        void conjugate_gradient(const double* distr_A, double* x, double* b, const unsigned int rows, const unsigned int cols, const int* rows_per_process, const int* displacements, const int max_iter, const double rel_err);
    }

};


#endif // CUDA_MPI