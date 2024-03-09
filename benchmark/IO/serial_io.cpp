#include <iostream>
#include <chrono>
#include "utils.hpp"


using namespace utils;
int main(int argc, char** argv)
{
    double* A;
    double* b;
    size_t rows;
    size_t cols;
    size_t vcols;

    auto t1 = std::chrono::high_resolution_clock::now();
    read_matrix_from_file(argv[1], A, rows, cols);
    read_vector_from_file(argv[2], b, rows);
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << std::endl;

}