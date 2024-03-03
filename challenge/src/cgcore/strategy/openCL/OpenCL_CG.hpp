#ifndef OPENCLSTRATEGY_HPP 
#define OPENCLSTRATEGY_HPP 

#include "../CGStrategy.hpp"
#include "../../utils/utils.hpp"

#include <cmath>
#include <stdio.h>
#include <CL/cl.h>
#include <cassert>
#include "OpenCLUtils.hpp"

namespace cgcore{
    
    class OpenCL_CG: public CGStrategy{
        public:
            OpenCL_CG();
            void run(const double * , const double * , double * , size_t , int , double ) const;

        //private: Commented for benchamrking 
            double dot(cl_kernel kernel, const cl_mem &dA, const cl_mem &dB, size_t size) const;
            void vec_sum(cl_kernel kernel, double alpha, const cl_mem &dX, double beta, const cl_mem &dY, size_t size) const;
            void matrix_vector_mul(cl_kernel kernel, const cl_mem &dA,const cl_mem &dB,const cl_mem &dC, size_t size) const;
            void conjugate_gradient(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error) const;
            void load_matrix_to_device( const cl_mem &dA, const double * A, size_t n) const;
            void load_vector_to_device( const cl_mem &dB, const double * b, size_t n) const;
            void load_vector_to_host( double * b, const cl_mem &dB, size_t n) const;
            cl_context context;
            cl_command_queue command_queue;
            cl_program program;
            cl_device_id device;

    };

}

#endif