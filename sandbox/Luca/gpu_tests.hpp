#ifndef GPU_TESTS_HPP
#define GPU_TESTS_HPP

#include <iostream>
#include "omp.h"

void vec_sum_func(double* a, double* b, unsigned int size);
double vec_dot_func(double* a, double* b, unsigned int size);

#endif // GPU_TESTS_HPP