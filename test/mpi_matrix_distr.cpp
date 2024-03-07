#include "../challenge/include/cg/cgcore.hpp"

using namespace cgcore;

int main(int argc, char ** argv){

    MPI_Init(&argc, &argv);
    int rank, num_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    double *matrix;
    double *vector;
    double *x;
    size_t n, m ; 
    size_t v_cols;
    int max_iter = 1000;
    double res = 1.e-6;

    CGSolver<MPI_DISTRIBUTED> solver;

    const char *m_path =  "../test/assets/matrix_10000.bin";
    const char *rhs_path =  "../test/assets/rhs_10000.bin";

    int* rows_per_process;
    int* displacements;

    utils::mpi::mpi_distributed_read_matrix(m_path , matrix, m, n, rows_per_process, displacements);
    utils::mpi::mpi_distributed_read_all_vector(rhs_path, vector, m, v_cols, rows_per_process, displacements);

    if(rank == 0)
        std::cout << n << std::endl;

    x = new double[n];
    //utils::print_matrix(matrix, n, n);
    //utils::print_matrix(vector, 1, n);

    solver.solve(matrix, vector, x, n, max_iter, res);

    //utils::print_matrix(x, 1, n);
    delete [] matrix;
    delete [] vector;
    delete [] x;

    if(rank == 0)
        solver.get_timer().print_last_formatted() ;
    
    MPI_Finalize();
    return 0;
}