#ifndef OPENMP_BLAS_HPP
#define OPENMP_BLAS_HPP 

#include "../CGStrategy.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include <omp.h>
#include <numeric>
#include <cblas.h>

namespace cgcore{
    
    class OpenMP_BLAS_CG : public CGStrategy{
        public: 
            void run(const double * , const double * , double * , size_t , int , double ) const ;

        //private: // For debugging 
            double dot(const double * x, const double * y, size_t size) const;
            void axpby(double alpha, const double * x, double beta, double * y, size_t size) const;
            void gemv(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols) const;
            void conjugate_gradient(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error) const;

    };

}

#endif