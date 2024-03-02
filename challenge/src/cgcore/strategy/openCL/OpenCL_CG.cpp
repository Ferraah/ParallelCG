
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
    

    double OpenCL_CG::dot(cl_kernel kernel, const cl_mem &dA, const cl_mem &dB,const cl_mem &dS,const double * x, const double * y, size_t size) const 
    {
        cl_mem dC;
        cl_int err_num = 0;

        size_t default_local_work_size = 64;
        size_t local_work_size = size < default_local_work_size ? size : default_local_work_size;
        

        
        // Round up division
        size_t num_work_groups = (size + (local_work_size - 1)) / local_work_size;
        size_t global_work_size = num_work_groups * local_work_size; 

        size_t vector_size = sizeof(double)*size;
        size_t c_size = sizeof(double)*num_work_groups;

        double *hC = new double[num_work_groups];

        dC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
            c_size, NULL, NULL);

//        double *hC = new double[size];

        //dC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
        //        vector_size, NULL, NULL);

        CHECK_NOT_NULL(dC, "Dot: Error dC creation");

        err_num = clEnqueueWriteBuffer(command_queue, dA, CL_FALSE, 0, vector_size, x, 0, NULL, NULL);
        CHECK_OPENCL_ERROR(err_num,"Dot: Write buffer dA error");
        err_num = clEnqueueWriteBuffer(command_queue, dB, CL_FALSE, 0, vector_size, y, 0, NULL, NULL);
        CHECK_OPENCL_ERROR(err_num,"Dot: Write buffer dB error");
        err_num = clEnqueueWriteBuffer(command_queue, dS, CL_FALSE, 0, sizeof(size_t), &size, 0, NULL, NULL);
        CHECK_OPENCL_ERROR(err_num,"Dot: Write buffer dS error");

        OpenCLUtils::Wait(command_queue);


        err_num = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA);
        err_num |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB);
        err_num |= clSetKernelArg(kernel, 2, vector_size, NULL);
        err_num |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &dC);
        err_num |= clSetKernelArg(kernel, 4, sizeof(size_t), &dS);

        /*
        err_num = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA);
        err_num |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB);
        err_num |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &dC);
        err_num |= clSetKernelArg(kernel, 3, sizeof(size_t), &dS);
        */ 

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
        //std::cout << result_cl << std::endl;

        //utils::print_matrix(x, 1, size);
        //std::cout << result_cl << std::endl;

        CHECK_OPENCL_ERROR(err_num, "Dot: Result read error.");

        clReleaseMemObject(dC);

        return result_cl;
    }

    void OpenCL_CG::axpby(double alpha, const double * x, double beta, double * y, size_t size) const
    {
                // y = alpha * x + beta * y

        for(size_t i = 0; i < size; i++)
        {
            y[i] = alpha * x[i] + beta * y[i];
        }
    }


    void OpenCL_CG::load_matrix_to_device(const cl_mem &dA, const double *A, size_t n) const{
        cl_int err_num = clEnqueueWriteBuffer(command_queue, dA, CL_FALSE, 0, sizeof(double)*n*n, A, 0, NULL, NULL);
        CHECK_OPENCL_ERROR(err_num,"laod_matrix_to_device: Write buffer dA error");
    }

    /**
     * Run the matrix vector multiplication on the selected device. 
     * 
     * @TODO: Remove A parameter
    */
    void OpenCL_CG::matrix_vector_mul(cl_kernel kernel, const cl_mem &dA,const cl_mem &dB,const cl_mem &dC,const double * A, double * x, double * y, size_t size) const
    {
        cl_int err_num = 0;

        /*
        size_t max_local_size = 0; 
        clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_local_size, NULL);
        std::cout << "Max: " << max_local_size << std::endl; 
        */ 
       
        size_t local_work_size = 1 << 10;
        
        // Round up division
        size_t num_work_groups = (size + (local_work_size -1)) / local_work_size;
        size_t global_work_size = num_work_groups * local_work_size; 

        size_t matrix_size = sizeof(double)*size*size;
        size_t vector_size = sizeof(double)*size;

        // Matrix has been already loaded

        err_num = clEnqueueWriteBuffer(command_queue, dB, CL_FALSE , 0, vector_size , x, 0, NULL, NULL);
        CHECK_OPENCL_ERROR(err_num,"Mvm: Write buffer dB error");

        OpenCLUtils::Wait(command_queue);
        

        err_num = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA);
        err_num |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB);
        err_num |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &dC);
        err_num |= clSetKernelArg(kernel, 3, sizeof(int), &size);
        err_num |= clSetKernelArg(kernel, 4, sizeof(int), &size);

        CHECK_OPENCL_ERROR(err_num, "Mvm: Kernel arg errors");

        err_num = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);

        CHECK_OPENCL_ERROR(err_num, "Mvm: Kernel error.");

        OpenCLUtils::Wait(command_queue);

        err_num = clEnqueueReadBuffer(command_queue, dC, CL_TRUE, 0, vector_size, 
           y, 0, NULL, NULL);


        CHECK_OPENCL_ERROR(err_num, "Mvm: Result read error.");

    }

    OpenCL_CG::OpenCL_CG(){

        cl_int err_num; 
        const char * source_path = "/home/users/u101373/ParallelCG/challenge/src/cgcore/strategy/openCL/kernels.cl";
        const char * binary_path = "kernels.cl.bin";

        //Create device and context 
        OpenCLUtils::CreateContext(device, context);
            
        // Create a context  
        context = clCreateContext(0, 1, &device, NULL, NULL, &err_num);

        CHECK_NOT_NULL(context, "Context is null.");

        // Create a command queue 
        command_queue = clCreateCommandQueue(context, device, 0, &err_num);
    
        CHECK_NOT_NULL(command_queue, "Command queue is null.");

        OpenCLUtils::InitializeProgram(source_path, binary_path, program, device, context);
 
        CHECK_NOT_NULL(program, "Program is null.");
    }
    /**
     * 
    */ 
    void OpenCL_CG::conjugate_gradient(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error) const
    {

        
        assert(program != NULL);

        cl_kernel dot_kernel = clCreateKernel(program, "dot_product", NULL);
        CHECK_NOT_NULL(dot_kernel, "Error dot product kernel creation");
        cl_kernel gemv_kernel = clCreateKernel(program, "A_times_x_kernel", NULL);
        CHECK_NOT_NULL(gemv_kernel, "Error matrix-vector mult. kernel creation");

        // Kernel buffers ------------------------------
        size_t matrix_size = sizeof(double)*size*size;
        size_t vector_size = sizeof(double)*size;

        cl_mem dA_dot = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            matrix_size , NULL, NULL);
        cl_mem dB_dot = clCreateBuffer(context, CL_MEM_READ_ONLY , 
            vector_size , NULL, NULL);
        cl_mem dS_dot = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
            sizeof(size_t), NULL, NULL);

        cl_mem dA_mvm = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            matrix_size, NULL, NULL);
        cl_mem dB_mvm  = clCreateBuffer(context, CL_MEM_READ_ONLY , 
            vector_size, NULL, NULL);
        cl_mem dC_mvm  = clCreateBuffer(context, CL_MEM_READ_ONLY , 
            vector_size, NULL, NULL);

        // -------------------------------------------- 

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

        bb = dot(dot_kernel, dA_dot, dB_dot, dS_dot, b, b, size);
    
        rr = bb;
        for(num_iters = 1; num_iters <= max_iters; num_iters++)
        {
            matrix_vector_mul(gemv_kernel, dA_mvm, dB_mvm, dC_mvm, A, p, Ap, size);
            alpha = rr / dot(dot_kernel, dA_dot, dB_dot, dS_dot, p, Ap, size);
            axpby(alpha, p, 1.0, x, size);
            axpby(-alpha, Ap, 1.0, r, size);
            rr_new = dot(dot_kernel, dA_dot, dB_dot, dS_dot, r, r, size);
            beta = rr_new / rr;
            rr = rr_new;

            if(std::sqrt(rr / bb) < rel_error) { break; }
            axpby(1.0, r, beta, p, size);
        }

        delete[] r;
        delete[] p;
        delete[] Ap;

        clReleaseMemObject(dA_dot);
        clReleaseMemObject(dB_dot);
        clReleaseMemObject(dS_dot);
        clReleaseMemObject(dA_mvm);
        clReleaseMemObject(dB_mvm);
        clReleaseMemObject(dC_mvm);

        if(num_iters <= max_iters)
        {
            printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
        }
        else
        {
            printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
        }
    }

    void OpenCL_CG::run(const double * A , const double * b, double * x, size_t size, int max_iter, double res_error) const {
        conjugate_gradient(A, b, x, size, max_iter, res_error);
    }
    
}
