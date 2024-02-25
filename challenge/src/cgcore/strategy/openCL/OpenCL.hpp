#ifndef OPENCLSTRATEGY_HPP 
#define OPENCLSTRATEGY_HPP 

#include "../CGStrategy.hpp"
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
    
    class OpenCL: public CGStrategy{
        public:
            OpenCL(){

                const char *kernel_abs_path = "/home/users/u101373/ParallelCG/challenge/src/cgcore/strategy/openCL/kernels.cl";
                context = OpenCLUtils::CreateContext();
                CHECK_OPENCL_ERROR(context, "Error context creation");
                command_queue = OpenCLUtils::CreateCommandQueue(context, &device);
                CHECK_OPENCL_ERROR(command_queue, "Error command queue creation");
                program = OpenCLUtils::CreateProgram(context, device, kernel_abs_path); 
                CHECK_OPENCL_ERROR(program, "Error program creation");

            };

            void run(const double * , const double * , double * , size_t , int , double ) const;

        private: 
            double dot(double * x, double * y, size_t size) const;
            void axpby(double alpha, double * x, double beta, double * y, size_t size) const;
            void gemv(double alpha, double * A, double * x, double beta, double * y, size_t num_rows, size_t num_cols) const;
            void conjugate_gradient(double * A, double * b, double * x, size_t size, int max_iters, double rel_error) const;
            cl_context context;
            cl_command_queue command_queue;
            cl_program program;
            cl_device_id device;

    };

}

#endif