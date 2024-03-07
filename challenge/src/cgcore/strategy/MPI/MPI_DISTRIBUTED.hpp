#ifndef MPI_DISTR_HPP
#define MPI_DISTR_HPP 

#include "../CGStrategy.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <cstring>
#include <mpi.h>

namespace cgcore{
    
    class MPI_DISTRIBUTED : public CGStrategy{
        public: 
            void run(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error) const ;

        private:
            double dot(const double * x, const double * y, size_t size) const;
            void axpby(double alpha, const double * x, double beta, double * y, size_t size) const;
            void gemv(double alpha, const double * A, const double * x, double beta, double *& y, size_t num_rows, size_t num_cols, int displ) const;
            void conjugate_gradient(const double * A, const double * b, double * x, size_t rows, size_t cols, int* rows_per_process, int* displacements, int max_iters, double rel_error) const;

    };

}

#endif