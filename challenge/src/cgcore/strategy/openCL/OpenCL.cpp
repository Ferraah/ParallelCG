
#include "OpenCL.hpp"

#define CHECK_OPENCL_KERNEL_ARGS_ERROR(err, msg) \
    do { \
        if (err != CL_SUCCESS) { \
            std::cerr << "OpenCL error (" << err << "): " << msg << std::endl; \
            /*Cleanup(context, commandQueue, program, kernel, memObjects);*/ \
            return 1; \
        } \
    } while (0)


#define CHECK_OPENCL_ERROR(err, msg) \
    do { \
        if (err == NULL) { \
            std::cerr << "OpenCL error (" << err << "): " << msg << std::endl; \
            /*Cleanup(context, commandQueue, program, kernel, memObjects);*/ \
            return 1; \
        } \
    } while (0)

namespace cgcore{
    

    double OpenCL::dot(double * x, double * y, size_t size) const 
    {
        cl_kernel kernel = 0;
        cl_mem dA, dB, dC;
        cl_int err_num = 0;

        kernel = clCreateKernel(program, "dot_product", NULL);
        CHECK_OPENCL_ERROR(kernel, "Error kernel creation");

        size_t local_work_size = 5;
        size_t num_work_groups = size/local_work_size;

        assert(size % local_work_size == 0);

        double *hC = new double[local_work_size];

        dA = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            sizeof(double)*size, NULL, NULL);
        dB = clCreateBuffer(context, CL_MEM_READ_ONLY , 
            sizeof(double)*size, NULL, NULL);
        dC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
            num_work_groups*sizeof(double), NULL, NULL);

        CHECK_OPENCL_ERROR(dA, "Error dA creation");
        CHECK_OPENCL_ERROR(dB, "Error dB creation");
        CHECK_OPENCL_ERROR(dC, "Error dC creation");

        err_num = clEnqueueWriteBuffer(command_queue, dA, CL_FALSE, 0, size*sizeof(double), x, 0, NULL, NULL);
        err_num = clEnqueueWriteBuffer(command_queue, dB, CL_FALSE, 0, size*sizeof(double), y, 0, NULL, NULL);

        CHECK_OPENCL_KERNEL_ARGS_ERROR(err_num, "Error hA creation");
        CHECK_OPENCL_KERNEL_ARGS_ERROR(err_num, "Error hB creation");


        err_num = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA);
        err_num |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB);
        err_num |= clSetKernelArg(kernel, 2, size*sizeof(double), NULL);
        err_num |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &dC);

        CHECK_OPENCL_KERNEL_ARGS_ERROR(err_num, "Kernel arg errors");
        err_num = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &size, &local_work_size, 0, NULL, NULL);
        CHECK_OPENCL_KERNEL_ARGS_ERROR(err_num, "Kernel error.");

        double result = 0.0;
        err_num = clEnqueueReadBuffer(command_queue, dC, CL_TRUE, 0, num_work_groups*sizeof(double), 
            hC, 0, NULL, NULL);

        double result_cl = 0; 
        for( int i = 0; i < num_work_groups; i++ )
        {
            result_cl += hC[ i ];
        }

        CHECK_OPENCL_KERNEL_ARGS_ERROR(err_num, "Result read error.");

        clReleaseMemObject(dA);
        clReleaseMemObject(dB);
        clReleaseMemObject(dC);

        for(size_t i = 0; i < size; i++)
        {
            result += x[i] * y[i];
        }

        return result;
    }

    void OpenCL::axpby(double alpha, double * x, double beta, double * y, size_t size) const
    {
        // y = alpha * x + beta * y

        for(size_t i = 0; i < size; i++)
        {
            y[i] = alpha * x[i] + beta * y[i];
        }
    }



    void OpenCL::gemv(double alpha, double * A, double * x, double beta, double * y, size_t num_rows, size_t num_cols) const
    {
        // y = alpha * A * x + beta * y;

        for(size_t r = 0; r < num_rows; r++)
        {
            double y_val = 0.0;
            for(size_t c = 0; c < num_cols; c++)
            {
                y_val += alpha * A[r * num_cols + c] * x[c];
            }
            y[r] = beta * y[r] + y_val;
        }
    }


    /**
     * 
    */ 
    void OpenCL::conjugate_gradient(double * A, double * b, double * x, size_t size, int max_iters, double rel_error) const
    {


        double alpha, beta, bb, rr, rr_new;
        double * r = new double[size];
        double * p = new double[size];
        double * Ap = new double[size];
        int num_iters;

        for(size_t i = 0; i < size; i++)
        {
            x[i] = 0.0;
            r[i] = b[i];
            p[i] = b[i];
        }

        bb = dot(b, b, size);
        rr = bb;
        for(num_iters = 1; num_iters <= max_iters; num_iters++)
        {
            gemv(1.0, A, p, 0.0, Ap, size, size);
            alpha = rr / dot(p, Ap, size);
            axpby(alpha, p, 1.0, x, size);
            axpby(-alpha, Ap, 1.0, r, size);
            rr_new = dot(r, r, size);
            std::cout << r[0] << std::endl;
            beta = rr_new / rr;
            rr = rr_new;

            if(std::sqrt(rr / bb) < rel_error) { break; }
            axpby(1.0, r, beta, p, size);
        }

        delete[] r;
        delete[] p;
        delete[] Ap;

        if(num_iters <= max_iters)
        {
            printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
        }
        else
        {
            printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
        }
    }

    void OpenCL::run(const double * A , const double * b, double * x, size_t size, int max_iter, double res_error) const {
        // @TODO: const problems
        double *A2, *b2;
        conjugate_gradient(A2, b2, x, size, max_iter, res_error);
    }
    
}
