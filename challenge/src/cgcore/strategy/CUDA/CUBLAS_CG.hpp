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
            void run(const double * , const double * , double * , size_t , int , double ) const ;
    };
}

#endif // CUBLAS_HPP