
#include <iostream>
#include <mpi.h>
#include "utils.hpp"

int main(int argc, char **argv){
    double *matrix;
    double *matrix_partial;
    size_t n;
    utils::read_matrix_from_file("../assets/matrix.bin", &matrix, n, n);
    utils::read_matrix_rows("../assets/matrix.bin",&matrix_partial, 0, 10, n);
    std::cout << std::endl;  
    utils::print_matrix(matrix,n,n);
    std::cout << std::endl;  
    utils::print_matrix(matrix_partial,2,n);
    return 0;
}