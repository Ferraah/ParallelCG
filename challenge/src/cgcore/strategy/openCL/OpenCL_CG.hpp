#ifndef OPENCLSTRATEGY_HPP 
#define OPENCLSTRATEGY_HPP 

#include "../CGStrategy.hpp"
#include "../../utils/utils.hpp"

#include <cmath>
#include <stdio.h>
#include <CL/cl.h>
#include <cassert>
#include "OpenCLUtils.hpp"

#define CHECK_OPENCL_ERROR(err, msg) \
    do { \
        if (err == NULL) { \
            std::cerr << "OpenCL error (" << err << "): " << msg << std::endl; \
        } \
    } while (0)

namespace cgcore{
    
    class OpenCL_CG: public CGStrategy{
        public:
            OpenCL_CG();
            void run(const double * , const double * , double * , size_t , int , double ) const;

        //private: Commented for benchamrking 
            double dot(cl_kernel kernel, const cl_mem &dA, const cl_mem &dB, const cl_mem &dC,const double * x, const double * y, size_t size) const;
            void axpby(double alpha, const double * x, double beta, double * y, size_t size) const;
            void matrix_vector_mul(cl_kernel kernel, const cl_mem &dA,const cl_mem &dB,const cl_mem &dS,const double * A, double * x, double * y, size_t size) const;
            void conjugate_gradient(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error) const;
            void load_matrix_to_device( const cl_mem &dA, const double * A, size_t n) const;
            cl_context context;
            cl_command_queue command_queue;
            cl_program program;
            cl_device_id device;

    };

}

#endif