#ifndef GPU_TESTS_HPP
#define GPU_TESTS_HPP

#include <iostream>


// ==== FUNCTIONS ON A SINGLE GPU ====
void saxpy_func(double a, double* x, double* y, unsigned int size);
double vec_dot_func_optimized(double* a, double* b, unsigned int size);
void A_times_x_func(double* A, double* x, double* vec, unsigned int num_rows, unsigned int num_cols);

// ==== FUNCTIONS ON ALL THE GPU OF A NODE ====

#endif // GPU_TESTS_HPP