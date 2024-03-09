#include "benchmark_template.hpp"
#include "../../challenge/include/cg/cgcore.hpp"

using namespace cgcore;

// Checks how the solver behave on a fixed amount of processing units
// and increasing size of the problem
int main(int argc, char **argv){

    MPI_Init(&argc, &argv);

    std::string data_folder = "/project/home/p200301/tests/";
    int sizes[] = {100, 500, 1000, 5000, 10000, 20000, 30000, 40000};
    
    for(int size : sizes){
        const char* size_str = std::to_string(size).c_str();
        benchmark_cg<Sequential, Sequential>(argc, argv, 
            (data_folder+"matrix"+size_str+".bin").c_str(),
            (data_folder+"rhs"+size_str+".bin").c_str(),
            "../io/banchmark_sequential.txt", 
            false);
    }

    MPI_Finalize();
    return 0;
}