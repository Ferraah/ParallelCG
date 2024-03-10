
        
#include "../challenge/include/cg/cgcore.hpp"


#include <mpi.h>
#include <time.h>

using namespace cgcore;

// benchmarking: measuring time
double wall_time(){
  struct timespec t;
  clock_gettime (CLOCK_MONOTONIC, &t);
  return 1.*t.tv_sec + 1.e-9*t.tv_nsec;
}


int main(int argc, char ** argv){

    MPI_Init(&argc, &argv);
    int rank, num_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double *matrix;
    double *vector;
    double *x;
    size_t n, m ; 
    int max_iter = 1000;
    double res = 1.e-6;

    CGSolver<MPI_OpenMP_CG> solver;

    const char *m_path =  "/project/home/p200301/tests/matrix20000.bin";
    const char *rhs_path =  "/project/home/p200301/tests/rhs20000.bin";

    double reading_time = - wall_time();
    utils::read_matrix_from_file(m_path , matrix, n, m);
    utils::read_vector_from_file(rhs_path, vector, n);
    reading_time += wall_time();

    if(rank == 0){
        std::cout << n << std::endl;
        std::cout << reading_time << std::endl;
    }
    x = new double[n];
    //utils::print_matrix(matrix, n, n);
    //utils::print_matrix(vector, 1, n);

   solver.solve(matrix, vector, x, n, max_iter, res);

    //utils::print_matrix(x, 1, n);
    delete [] matrix;
    delete [] vector;
    delete [] x;

    if(rank == 0){
        solver.get_timer().print_last_formatted() ;
    }
    MPI_Finalize();
    return 0;
}