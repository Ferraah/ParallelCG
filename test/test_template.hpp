#ifndef TEST_TEMPLATE_HPP
#define TEST_TEMPLATE_HPP

#include "../challenge/include/cg/cgcore.hpp"


using namespace cgcore;

/**
 * A template for testing non-mpi strategies.
*/
template <class STRATEGY>
void test_cg(int argc, char ** argv, const char *m_path, const char *rhs_path){

    double *matrix;
    double *vector;
    double *x;
    size_t n, m ; 
    int max_iter = 1000;
    double res = 1.e-6;

    CGSolver<STRATEGY> solver;

    utils::read_matrix_from_file(m_path , matrix, n, m);
    utils::read_vector_from_file(rhs_path, vector, n);
    std::cout << n << std::endl;
    x = new double[n];
        
    solver.solve(matrix, vector, x, n, max_iter, res);

    delete [] matrix;
    delete [] vector;
    delete [] x;

    solver.get_timer().print_last_formatted() ;
    
}

#endif