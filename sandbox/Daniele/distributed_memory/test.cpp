// 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
#include <iostream>
#include <mpi.h>
#include "utils.hpp"


void matrix_vector_prod(double * &matrix, double * &vector, double * &result, size_t n, size_t m){
    result = new double[n];

    for(size_t r = 0; r<n; r++){
        for(size_t c = 0; c<m; c++){
            result[r] = 0;
            #pragma omp for reduction(+ : result[r])
            for(size_t k = 0; k<m; k++){
                result[r] += matrix[r*m + k] * vector[k];
            }  
        }
    }
}

int main(int argc, char **argv){

    MPI_Init(&argc, &argv);
    int rank, size; 

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double *matrix;
    double *vector;
    double *partial_matrix;
    double *partial_res;
    double *res;

    size_t n, m;
    n = m = 6;
    size_t starting_row, chunk_size;
    char * matrix_path = "../assets/matrix.bin";
    //utils::read_matrix_dims(matrix_path, n, m);

    // Equal distribution
    //assert(n % size == 0);
    size_t max_chunk_size = 4;
    if(rank != (size-1))
        chunk_size = max_chunk_size;
    else
        chunk_size = n - (size - 1)*max_chunk_size;

    starting_row = chunk_size * rank;

    //utils::read_matrix_from_file(matrix_path, &matrix, n, n);
    //utils::read_vector_from_file(matrix_path, vector, n);
    //utils::read_matrix_rows(matrix_path, matrix_partial, starting_row, chunk_size, n);
    utils::create_vector(vector, m, 1);
    utils::create_matrix(partial_matrix, chunk_size, m, 1);

    std::cout << "Allocated on rank " << rank << std::endl;
    //utils::print_matrix(partial_matrix , chunk_size, m);

    MPI_Barrier(MPI_COMM_WORLD);
    //matrix_vector_prod(partial_matrix, vector, partial_res, chunk_size, m);
    std::cout << "Product on rank " << rank << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    //utils::print_matrix(partial_res,1,chunk_size);

    delete [] partial_matrix;
    delete [] vector;
    //delete [] partial_res;

    std::cout << "Deleted on rank " << rank << std::endl;
    MPI_Finalize();

    return 0;
}