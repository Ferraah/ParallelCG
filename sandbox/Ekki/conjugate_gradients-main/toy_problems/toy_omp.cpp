// From https://docs.lxp.lu/hpc/compiling/

//
// Execute with: g++ -o helloworld_omp helloworld_OMP.cpp -fopenmp
//               export OMP_NUM_THREADS=8
//               ./helloworld_omp
//

#include <iostream>
#include <omp.h> 
#include <time.h>

// benchmarking: measuring time
double wall_time(){
  struct timespec t;
  clock_gettime (CLOCK_MONOTONIC, &t);
  return 1.*t.tv_sec + 1.e-9*t.tv_nsec;
}

int main(void) 
{
    // Beginning of parallel region 
    #pragma omp parallel 
    { 
       std::cout << "Hello World... from thread = " << omp_get_thread_num() << std::endl;
    }

    // Performing some calculation
    int num_steps = 1000000000;
    // int i = 0;
    double t_comp = - wall_time();
    // #pragma omp parallel private(i)
    // {
    //     int id = omp_get_thread_num();
    //     int numthreads = omp_get_num_threads();
    //     int start = id*num_steps/numthreads;
    //     int finish = (id+1)*num_steps/numthreads;
    //     double x;
    //     double partial_sum = 0;

    //     for (i=start; i<finish ; i++){
    //         x = (i+0.5)*step;
    //         partial_sum += + 4.0/(1.0+x*x);
    //     }
    //     #pragma omp atomic
    //     sum += partial_sum;
    // }
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < 1000000000; i++){
        sum += i;
    }

    t_comp += wall_time();

    #pragma omp parallel
    {
        if (omp_get_thread_num() == 0){
            std::cout << "Max. thread nums = " << omp_get_max_threads() << std::endl;
            std::cout << "TIME = " << t_comp <<std::endl;
        }
    }

    // Ending of parallel region 
    return 0;
}
