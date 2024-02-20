#include <iostream>
#include <cstdlib>
#include <cassert>
#include "gpu_tests.hpp"

#define SIZE 1<<30

int main()
{
    double* a = (double*) malloc(sizeof(double) * SIZE);
    double* b = (double*) malloc(sizeof(double) * SIZE);

    std::cout << "Allocated arrays of " << (SIZE) << " doubles" << std::endl;
    double dot = 0.0; 
    for (unsigned int i = 0; i < SIZE; i++)
    {
        a[i] = 1.0;
        b[i] = 2.0;
    }

    std::cout << "Reinitialized arrays " << (SIZE) << std::endl;
    dot = vec_dot_func(a, b, SIZE);
    std::cout << "Computed dot product: " << dot << std::endl;
    assert(dot/2 == SIZE);
    return 0;
}