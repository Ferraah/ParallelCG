#include "../challenge/include/cg/cgcore.hpp"

using namespace cgcore;

int main(int argc, char ** argv){

    double *matrix;
    double *vector;
    double *x;
    size_t n, m ; 
    int max_iter = 1000;
    double res = 1.e-6;

    CGSolver<Sequential> solver;

    const char *m_path =  "../test/assets/matrix_10000.bin";
    const char *rhs_path =  "../test/assets/rhs_10000.bin";

    utils::read_matrix_from_file(m_path , matrix, n, m);
    utils::read_vector_from_file(rhs_path, vector, n);
    std::cout << n << std::endl;
    x = new double[n];
    //utils::print_matrix(matrix, n, n);
    //utils::print_matrix(vector, 1, n);


    
    solver.solve(matrix, vector, x, n, max_iter, res);

    //utils::print_matrix(x, 1, n);
    delete [] matrix;
    delete [] vector;
    delete [] x;

    solver.get_timer().print_last_formatted() ;
    
    return 0;
}