#ifndef CUBLAS_HPP
#define CUBLAS_HPP

// === ABSTRACT STRATEGY
#include "../CGStrategy.hpp"

// === STL INCLUDES ===
#include <cstdlib>
#include <iostream>

// === CUDA INCLUDES ===
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>



namespace cgcore
{
    class CUBLAS_CG : public CGStrategy{
        public:
            virtual void run(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error) const;
    };
}

#endif // CUBLAS_HPP