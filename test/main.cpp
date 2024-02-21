#include "../challenge/include/cg/cgcore.hpp"

using namespace cgcore;

int main(int argc, char ** argv){

    double *matrix;
    double *vector;
    double *x;
    size_t n = 10; 
    int max_iter = 100;
    double res = 1e-6;

    CGSolver<Sequential> solver;

    utils::create_matrix(matrix, n, n, 1);
    utils::create_vector(vector, n, 1);
    utils::print_matrix(matrix, n, n);

    solver.solve(matrix, vector, x, n, max_iter, res);
    
    
    return 0;
}