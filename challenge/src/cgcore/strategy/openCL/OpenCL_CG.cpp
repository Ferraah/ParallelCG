
#include "OpenCL_CG.hpp"
#define CHECK_OPENCL_ERROR(err, msg) \
    do { \
        if (err != CL_SUCCESS) { \
            std::cerr << "OpenCL error (" << err << "): " << msg << std::endl; \
            /*Cleanup(context, commandQueue, program, kernel, memObjects);*/ \
        } \
    } while (0)


#define CHECK_NOT_NULL(err, msg) \
    do { \
        if (err == NULL) { \
            std::cerr << "OpenCL error (" << err << "): " << msg << std::endl; \
            /*Cleanup(context, commandQueue, program, kernel, memObjects);*/ \
        } \
    } while (0)
    

namespace cgcore{
    
    /**
     * @param kernel The kernel associated to the method
     * @param dA The reference to the first vector in device memory 
     * @param dB The reference to the first vector in device memory 
     * @param size The size of the vectors 
    */
    double OpenCL_CG::dot(cl_kernel kernel, const cl_mem &dA, const cl_mem &dB, size_t size) const 
    {
        cl_mem dC;
        cl_int err_num = 0;

        size_t default_local_work_size = 1 << 10;
        size_t local_work_size = size < default_local_work_size ? size : default_local_work_size;
        
        // Round up division
        size_t num_work_groups = (size + (local_work_size - 1)) / local_work_size;
        size_t global_work_size = num_work_groups * local_work_size; 
        size_t vector_size = sizeof(double)*size;
        size_t c_size = sizeof(double)*num_work_groups;

        double *hC = new double[num_work_groups];

        dC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
            c_size, NULL, NULL);

        CHECK_NOT_NULL(dC, "Dot: Error dC creation");

                err_num = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA);
        err_num |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB);
        err_num |= clSetKernelArg(kernel, 2, local_work_size*sizeof(double), NULL);
        err_num |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &dC);
        err_num |= clSetKernelArg(kernel, 4, sizeof(int), &size);


        CHECK_OPENCL_ERROR(err_num, "Dot: Kernel arg errors");

        err_num = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        CHECK_OPENCL_ERROR(err_num, "Dot: Kernel error.");

        OpenCLUtils::Wait(command_queue);

        err_num = clEnqueueReadBuffer(command_queue, dC, CL_FALSE, 0, c_size, 
            hC, 0, NULL, NULL);

        OpenCLUtils::Wait(command_queue);
        double result_cl = 0; 
        for( int i = 0; i < num_work_groups; i++ )
        {
            result_cl += hC[ i ];
        }

        CHECK_OPENCL_ERROR(err_num, "Dot: Result read error.");

        clReleaseMemObject(dC);

        return result_cl;
    }

    /**
     * Sum two vector and replace the result as the first addend.
     * @param kernel OpenCL kernel method 
     * @param alpha
     * @param dX Reference to first vector in device memory
     * @param beta 
     * @param dY Reference to second vector in device memory
     * @param size Size of the two vectors
    */
    void OpenCL_CG::vec_sum(cl_kernel kernel, double alpha, const cl_mem  &dX, double beta, const cl_mem &dY, size_t size) const
    {

        cl_int err_num = 0;

        size_t default_local_work_size = 1 << 10;
        size_t local_work_size = size < default_local_work_size ? size : default_local_work_size;
        
        // Round up division
        size_t num_work_groups = (size + (local_work_size - 1)) / local_work_size;
        size_t global_work_size = num_work_groups * local_work_size; 

        err_num = clSetKernelArg(kernel, 0, sizeof(double), &alpha);
        err_num |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dX);
        err_num |= clSetKernelArg(kernel, 2, sizeof(double), &beta);
        err_num |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &dY);
        err_num |= clSetKernelArg(kernel, 4, sizeof(int), &size);

        CHECK_OPENCL_ERROR(err_num, "Mvm: Kernel arg errors");

        err_num = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);

        CHECK_OPENCL_ERROR(err_num, "Mvm: Kernel error.");

        OpenCLUtils::Wait(command_queue);
    }


    void OpenCL_CG::load_matrix_to_device(const cl_mem &dA, const double *A, size_t n) const{
        cl_int err_num = clEnqueueWriteBuffer(command_queue, dA, CL_TRUE, 0, sizeof(double)*n*n, A, 0, NULL, NULL);
        CHECK_OPENCL_ERROR(err_num,"laod_matrix_to_device: Write buffer dA error");
    }

    void OpenCL_CG::load_vector_to_device(const cl_mem &dB, const double *b, size_t n) const{
        cl_int err_num = clEnqueueWriteBuffer(command_queue, dB, CL_TRUE, 0, sizeof(double)*n, b, 0, NULL, NULL);
        CHECK_OPENCL_ERROR(err_num,"load_vector_to_device: Write buffer dB error");
    }

    void OpenCL_CG::load_vector_to_host(double *b, const cl_mem &dB, size_t n) const{
        cl_int err_num = clEnqueueReadBuffer(command_queue, dB, CL_TRUE, 0, sizeof(double)*n, b, 0, NULL, NULL);
        CHECK_OPENCL_ERROR(err_num,"read_vector_to_host: error");
    }

    /**
     * Run the matrix vector multiplication on the selected device. 
     * @param kernel the kernel associated to the method
     * @param dA the device pointer to the matrix A, already loaded in device memory. 
     * @param dB the device pointer to the vector b, already loaded in device memory.
     * @param dC the device pointer to the result c, in device memory.
     * 
     * @TODO: Remove A parameter
    */
    void OpenCL_CG::matrix_vector_mul(cl_kernel kernel, const cl_mem &dA,const cl_mem &dB,const cl_mem &dC, size_t size) const
    {
        cl_int err_num = 0;

        size_t local_work_size = 1 << 10;
        
        // Round up division
        size_t num_work_groups = (size + (local_work_size -1)) / local_work_size;
        size_t global_work_size = num_work_groups * local_work_size; 

        size_t matrix_size = sizeof(double)*size*size;
        size_t vector_size = sizeof(double)*size;

        // Matrix has been already loaded

        err_num = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA);
        err_num |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB);
        err_num |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &dC);
        err_num |= clSetKernelArg(kernel, 3, sizeof(int), &size);
        err_num |= clSetKernelArg(kernel, 4, sizeof(int), &size);

        CHECK_OPENCL_ERROR(err_num, "Mvm: Kernel arg errors");

        err_num = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);

        CHECK_OPENCL_ERROR(err_num, "Mvm: Kernel error.");

        OpenCLUtils::Wait(command_queue);

    }


    OpenCL_CG::OpenCL_CG(){

        cl_int err_num; 
        const char * source_path = "/home/users/u101373/ParallelCG/challenge/src/cgcore/strategy/openCL/kernels.cl";
        const char * binary_path = "kernels.cl.bin";


        OpenCLUtils::InitializePlatforms(device, context, command_queue);
        OpenCLUtils::InitializeProgram(source_path, binary_path, program, device, context);

        assert(device != NULL);
        assert(context != NULL);
        assert(command_queue != NULL);
        assert(program != NULL);
    }

    /**
     * Main method for the strategy 
    */ 
    void OpenCL_CG::conjugate_gradient(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error) const
    {

        // Prepare the kernels from loaded program 
        cl_kernel dot_kernel = clCreateKernel(program, "dot_product", NULL);
        cl_kernel mvm_kernel = clCreateKernel(program, "A_times_x_kernel", NULL);
        cl_kernel vs_kernel = clCreateKernel(program, "vec_sum", NULL);
        assert(dot_product != NULL);
        assert(mvm_product != NULL);
        assert(vs_product != NULL);
        
        // 

        // Allocate buffers in device memory
        size_t matrix_size = sizeof(double)*size*size;
        size_t vector_size = sizeof(double)*size;

        // Device memory --------------------------------------- 

        // Matrix A
        cl_mem dA = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            matrix_size , NULL, NULL);

        // Generic vectors of size len
        cl_mem d_Ad = clCreateBuffer(context, CL_MEM_READ_ONLY , 
            vector_size , NULL, NULL);
        cl_mem d_d = clCreateBuffer(context, CL_MEM_READ_ONLY , 
            vector_size , NULL, NULL);
        cl_mem d_r = clCreateBuffer(context, CL_MEM_READ_ONLY , 
            vector_size , NULL, NULL);
        cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_ONLY , 
            vector_size , NULL, NULL);
        cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY , 
            vector_size , NULL, NULL);

        // ----------------------------------------------------

        double alpha, beta;
        double rr, bb;  // To check relative error

        // Starting conditions
        double * r = new double[size];
        double * d = new double[size];

        int num_iters;

        for(size_t i = 0; i < size; i++)
        {
            x[i] = 0.0;
            r[i] = b[i];
            d[i] = b[i];
        }

        // One time only
        load_matrix_to_device(dA, A, size);

        // To check residual  
        load_vector_to_device(d_b, b, size);

        bb = dot(dot_kernel, d_b, d_b, size);

        load_vector_to_device(d_d, d, size);
        load_vector_to_device(d_r, r, size);
        load_vector_to_device(d_x, x, size);

        for(num_iters = 1; num_iters <= max_iters; num_iters++){

            // Calculating A*d
            matrix_vector_mul(mvm_kernel, dA, d_d, d_Ad, size);

            // Calculating alpha = d*r / (Ad * d)
            alpha = dot(dot_kernel, d_d, d_r, size)/(double)dot(dot_kernel, d_Ad, d_d, size);

            // Updating x along d
            vec_sum(vs_kernel, alpha, d_d, 1.0, d_x, size);

            // Updating r
            vec_sum(vs_kernel, -alpha, d_Ad, 1.0, d_r, size);

            // Calculating Beta
            beta = dot(dot_kernel, d_Ad, d_r, size)/(double) dot(dot_kernel, d_Ad, d_d, size);

            // Updating d
            vec_sum(vs_kernel, 1.0, d_r, -beta, d_d, size);

            // Checking residual conditions
            rr = dot(dot_kernel, d_r, d_r, size);
            if(std::sqrt(rr / bb) < rel_error) { break; }

        }

        load_vector_to_host(x, d_x, size);

        delete [] d;
        delete [] r;

        if(num_iters <= max_iters)
        {
            printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
        }
        else
        {
            printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
        }

        clReleaseMemObject(dA);
        clReleaseMemObject(d_x);
        clReleaseMemObject(d_r);
        clReleaseMemObject(d_d);
        clReleaseMemObject(d_b);
        clReleaseMemObject(d_Ad);

    }

    /**
     * Common strategy interface.
    */
    void OpenCL_CG::run(const double * A , const double * b, double * x, size_t size, int max_iter, double res_error) const {
        conjugate_gradient(A, b, x, size, max_iter, res_error);
    }
    
}
