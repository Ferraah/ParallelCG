#include <iostream>
#include <mpi.h>
#include "utils.hpp"
#include <chrono>

using namespace utils::mpi;
int main(int argc, char** argv)
{
    double* A;
    double* b;
    size_t rows;
    size_t cols;
    size_t vcols;

    int* rows_per_process;
    int* displacements;

    int rank;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto t1 = std::chrono::high_resolution_clock::now();
    mpi_distributed_read_matrix(argv[1], A, rows, cols, rows_per_process, displacements);
    mpi_distributed_read_all_vector(argv[2], b, rows, vcols, rows_per_process, displacements);
    MPI_Barrier(MPI_COMM_WORLD);
    auto t2 = std::chrono::high_resolution_clock::now();

    if (rank == 0)
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << std::endl;

    MPI_Finalize();
}