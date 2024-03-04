#ifndef MPI_HPP
#define MPI_HPP 

#include "../CGStrategy.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include <mpi.h>

namespace cgcore{
    
    class MPI_CG : public CGStrategy{
        public: 
            void run(const double * , const double * , double * , size_t , int , double ) const ;

        //private: // For debugging 
            double dot(const double * x, const double * y, size_t size) const;
            void axpby(double alpha, const double * x, double beta, double * y, size_t size) const;
            void gemv(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols, int num_processes, int my_rank, int * displacements, int * counts) const;
            void conjugate_gradient(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error) const;

    };

}

#endif