// From https://docs.lxp.lu/hpc/compiling/

//
// Execute with: g++ -o helloworld_omp helloworld_OMP.cpp -fopenmp
//               export OMP_NUM_THREADS=8
//               ./helloworld_omp
//

#include <iostream>
#include <omp.h> 

int main(void) 
{
    // Beginning of parallel region 
    #pragma omp parallel 
    { 
       std::cout << "Hello World... from thread = " << omp_get_thread_num() << std::endl;
    } 

    #pragma omp parallel
    {
        if (omp_get_thread_num() == 0){
            std::cout << "Max. thread nums = " << omp_get_max_threads() << std::endl;
        }
    }
    // Ending of parallel region 
    return 0;
}
