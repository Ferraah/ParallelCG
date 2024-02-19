#include <iostream>
#include <cstdlib>
#include "gpu_tests.hpp"

#define SIZE 1<<10

int main()
{
    double* a = (double*) malloc(sizeof(double) * SIZE);
    double* b = (double*) malloc(sizeof(double) * SIZE);

    for (unsigned int i = 0; i < SIZE; i++)
    {
        a[i] = 1.0;
        b[i] = 6.0;
    }
    std::cout << "Allocated arrays " << (SIZE) << std::endl;
    vec_sum_func(a, b, SIZE);
    std::cout << "Computed sums" << std::endl;
    for (unsigned int i = 0; i < SIZE; i++)
    {
        std::cout << a[i];
    }
    std::cout << std::endl;

    double dot = 0.0; 
    for (unsigned int i = 0; i < SIZE; i++)
    {
        a[i] = 1.0;
        b[i] = 1.0;
    }

    std::cout << "Reinitialized arrays " << (SIZE) << std::endl;
    dot = vec_dot_func(a, b, SIZE);
    std::cout << "Computed dot product: " << dot << std::endl;
    // for (unsigned int i = 0; i < SIZE; i++)
    // {
    //     std::cout << a[i] << std::endl;
    // }
    return 0;
}