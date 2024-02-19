
#include <iostream>
#include <mpi.h>
#include "utils.hpp"

int main(int argc, char **argv){

    MPI_Init(&argc, &argv);
    int rank, size; 

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double *matrix;
    double *vector;
    double *matrix_partial;
    size_t n, m;
    size_t starting_row, chunk_size;
    char * matrix_path = "../assets/matrix.bin";
    utils::read_matrix_dims(matrix_path, n, m);

    // Equal distribution
    assert(n % size == 0);

    chunk_size = n/size;
    starting_row = chunk_size * rank;

    //utils::read_matrix_from_file(matrix_path, &matrix, n, n);
    //utils::read_vector_from_file("../assert/rhs.bin", &vector, n);
    utils::read_matrix_rows("../assets/matrix.bin", &matrix_partial, starting_row, chunk_size, n);
    //std::cout << std::endl;  
    //utils::print_matrix(matrix,n,n);
    //std::cout << std::endl;  
    //utils::print_matrix(vector,n,1);
    std::cout << std::endl;  
    utils::print_matrix(matrix_partial,chunk_size,n);

    //MPI_Finalize();
    return 0;
}