#include "../challenge/include/cg/cgcore.hpp"
#include <cassert>
#include <ctime>


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
    size_t len = 1 << 13;
    utils::create_vector(a, len, 1);
    utils::create_vector(b, len, 2);
    

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
    CHECK_OPENCL_ERROR(dot_kernel, "Error dot product kernel creation");


    std::clock_t start, end;
    double o_time, s_time;

    /*
    start = std::clock();
    o_result = o_strategy.dot(dot_kernel, dA_dot, dB_dot, dS_dot, a, b, len);
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
    */
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
    CHECK_OPENCL_ERROR(mvm_kernel , "Error dot product kernel creation");

    double *a2, *b2;
    utils::create_matrix(a2, len, len, 1);
    utils::create_vector(b2, len, 1);
    
    double *o_result2 = new double[len];
    double *s_result2 = new double[len];

    // Loading A into device 
    o_strategy.load_matrix_to_device(dA_mvm, a2, len);

    start = std::clock();
    o_strategy.matrix_vector_mul(mvm_kernel, dA_mvm, dB_mvm, dC_mvm , a2, b2, o_result2, len);
    
    end = std::clock(); 
    o_time = double(end-start)/CLOCKS_PER_SEC;

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


    //solver.get_timer().print_last_formatted() ;
    
    return 0;
}