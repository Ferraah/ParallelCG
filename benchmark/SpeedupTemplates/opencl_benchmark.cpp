#include "benchmark_template.hpp"
#include "../../challenge/include/cg/cgcore.hpp"

using namespace cgcore;

// Checks how the solver behave on a fixed amount of processing units
// and increasing size of the problem
int main(int argc, char **argv){

    std::string data_folder = "/project/home/p200301/tests/";
    int sizes[] = {1000, 5000, 10000, 20000, 40000, 70000};
    
    for(int size : sizes){
        const char* size_str = std::to_string(size).c_str();
        benchmark_cg<OpenCL_CG, Sequential>(argc, argv, 
            (data_folder+"matrix"+size_str+".bin").c_str(),
            (data_folder+"rhs"+size_str+".bin").c_str(),
            "../io/banchmark_su_opencl.txt", 
            false, false);
    }

    return 0;
}