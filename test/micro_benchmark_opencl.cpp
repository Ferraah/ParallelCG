#include "../challenge/include/cg/cgcore.hpp"
#include <cassert>
#include <ctime>

#define CHECK_OPENCL_ERROR(err, msg) \
    do { \
        if (err != CL_SUCCESS) { \
            std::cerr << "OpenCL error (" << err << "): " << msg << std::endl; \
            /*Cleanup(context, commandQueue, program, kernel, memObjects);*/ \
        } \
    } while (0)

using namespace cgcore;


double norm(double *a, size_t n){
    double result = 0;
    for(size_t i = 0; i<n; i++){
        result += a[i];
    }
    return result;
}


int main(int argc, char ** argv){

    double *a, *b, o_result, s_result; 
    size_t len = 1000;
    utils::create_vector(a, len, 1);
    utils::create_vector(b, len, 1);
    

    OpenCL_CG o_strategy;
    Sequential s_strategy;

    assert(o_strategy.program != NULL);
    assert(o_strategy.context != NULL);

    // Kernel buffers ------------------------------
    size_t vector_size = sizeof(double)*len;
    cl_mem dA_dot = clCreateBuffer(o_strategy.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 
        vector_size, NULL, NULL);
    cl_mem dB_dot = clCreateBuffer(o_strategy.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 
        vector_size , NULL, NULL);
    cl_mem dS_dot = clCreateBuffer(o_strategy.context, CL_MEM_WRITE_ONLY, 
        sizeof(size_t), NULL, NULL);
    // -------------------------------------------- 

    cl_kernel dot_kernel = clCreateKernel(o_strategy.program, "dot_product", NULL);
    assert(dot_kernel != NULL);

    cl_int err_num;

    err_num = clEnqueueWriteBuffer(o_strategy.command_queue, dA_dot, CL_TRUE, 0, vector_size, a, 0, NULL, NULL);
    CHECK_OPENCL_ERROR(err_num,"Dot: Write buffer dA error");
    err_num = clEnqueueWriteBuffer(o_strategy.command_queue, dB_dot, CL_TRUE, 0, vector_size, b, 0, NULL, NULL);
    CHECK_OPENCL_ERROR(err_num,"Dot: Write buffer dB error");




    std::clock_t start, end;
    double o_time, s_time;

    start = std::clock();
    o_result = o_strategy.dot(dot_kernel, dA_dot, dB_dot, len);
    end = std::clock(); 
    o_time = double(end-start)/CLOCKS_PER_SEC;

    start= std::clock(); 
    s_result = s_strategy.dot(a, b, len);
    end = std::clock(); 
    s_time = double(end-start)/CLOCKS_PER_SEC;

    std::cout << "Error: " << o_result - s_result << std::endl;
    std::cout << "OpenCL: " << o_time << std::endl;
    std::cout << "Sequential: " << s_time << std::endl;
    std::cout << "Speedup: " << s_time/(double)o_time << std::endl;

    delete [] a;
    delete [] b;

    // Matrix vector multiplication 

    // Kernel buffers ------------------------------
    vector_size = sizeof(double)*len;
    size_t matrix_size = sizeof(double)*len*len;
    cl_mem dA_mvm = clCreateBuffer(o_strategy.context, CL_MEM_READ_ONLY, 
        matrix_size , NULL, NULL);
    cl_mem dB_mvm = clCreateBuffer(o_strategy.context, CL_MEM_READ_ONLY , 
        vector_size , NULL, NULL);
    cl_mem dC_mvm = clCreateBuffer(o_strategy.context, CL_MEM_WRITE_ONLY, 
        vector_size, NULL, NULL);
    // -------------------------------------------- 

    cl_kernel mvm_kernel = clCreateKernel(o_strategy.program, "A_times_x_kernel", NULL);
    assert(mvm_kernel != NULL);

    double *a2, *b2;
    utils::create_matrix(a2, len, len, 1);
    utils::create_vector(b2, len, 1);
    
    double *o_result2 = new double[len];
    double *s_result2 = new double[len];

    // OPENCL ---------------------------------- 

    // Loading A into device 
    o_strategy.load_matrix_to_device(dA_mvm, a2, len);

    // Loading b into device
    err_num = clEnqueueWriteBuffer(o_strategy.command_queue, dB_mvm, CL_FALSE , 0, vector_size , b2, 0, NULL, NULL);
    OpenCLUtils::Wait(o_strategy.command_queue);

    start = std::clock();
    o_strategy.matrix_vector_mul(mvm_kernel, dA_mvm, dB_mvm, dC_mvm, len);
    end = std::clock(); 

    err_num = clEnqueueReadBuffer(o_strategy.command_queue, dC_mvm, CL_TRUE, 0, vector_size, 
        o_result2, 0, NULL, NULL);

    o_time = double(end-start)/CLOCKS_PER_SEC;

    // ---------------------------------------- 

    // SEQUENTIAL ---------------------------------- 

    start= std::clock(); 
    s_strategy.gemv(1.0, a2, b2, 0.0, s_result2, len, len);
    end = std::clock(); 
    s_time = double(end-start)/CLOCKS_PER_SEC;

    std::cout << "Error 2: " << norm(o_result2, len) - norm(s_result2, len) << std::endl;
    std::cout << "OpenCL 2: " << o_time << std::endl;
    std::cout << "Sequential 2: " << s_time << std::endl;
    std::cout << "Speedup 2: " << s_time/(double)o_time << std::endl;
    delete [] a2;
    delete [] b2;
    delete [] o_result2;
    delete [] s_result2;

    // VECTOR SUM --------------------------------

    double *a3, *b3;
    double *o_result3 = new double[len];
    double *s_result3 = new double[len];

    utils::create_vector(a3, len, 1.0);
    utils::create_vector(b3, len, 1.0);

    // Kernel buffers ------------------------------
    cl_mem dA_vs = clCreateBuffer(o_strategy.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 
        vector_size, NULL, NULL);
    cl_mem dB_vs = clCreateBuffer(o_strategy.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 
        vector_size , NULL, NULL);
    // -------------------------------------------- 

    cl_kernel vs_kernel = clCreateKernel(o_strategy.program, "vec_sum", NULL);
    assert(vs_kernel != NULL);

    // Loading into device
    err_num = clEnqueueWriteBuffer(o_strategy.command_queue, dA_vs, CL_FALSE , 0, vector_size , a3, 0, NULL, NULL);
    err_num |= clEnqueueWriteBuffer(o_strategy.command_queue, dB_vs, CL_FALSE , 0, vector_size , b3, 0, NULL, NULL);
    OpenCLUtils::Wait(o_strategy.command_queue);


    double o_time3, s_time3;

    start = std::clock();
    o_strategy.vec_sum(vs_kernel, 1.0, dA_vs, 1.0, dB_vs, len); 
    end = std::clock(); 
    o_time3 = double(end-start)/CLOCKS_PER_SEC;

    start= std::clock(); 
    s_strategy.axpby(1.0, a3, 1.0, b3, len);
    s_result3 = b3;
    end = std::clock(); 
    s_time3 = double(end-start)/CLOCKS_PER_SEC;

    err_num = clEnqueueReadBuffer(o_strategy.command_queue, dB_vs, CL_TRUE, 0, vector_size, 
        o_result3, 0, NULL, NULL);

    std::cout << "Error: " << norm(o_result3, len) - norm(s_result3, len) << std::endl;
    std::cout << "OpenCL: " << o_time3 << std::endl;
    std::cout << "Sequential: " << s_time3 << std::endl;
    std::cout << "Speedup: " << s_time3/(double)o_time3 << std::endl;

    delete [] a3;
    delete [] b3;

    return 0;
}