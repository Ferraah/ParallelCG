#include "../challenge/include/cg/cgcore.hpp"
#include <CL/cl.h>

using namespace cgcore;

int main(int argc, char ** argv){

    double *matrix;
    double *vector;
    double *x;
    size_t n, m ; 
    int max_iter = 100;
    double res = 1.e-6;

    CGSolver<OpenCL> solver;

    x = new double[n];
    const char *path =  "../test/assets/matrix.bin";
    utils::read_matrix_from_file(path , matrix, n, m);
    utils::create_vector(vector, n, 20);
    utils::print_matrix(matrix, n, n);
    utils::print_matrix(vector, 1, n);


    OpenCLUtils::CreateContext();
    
    solver.solve(matrix, vector, x, n, max_iter, res);

    delete [] matrix;
    delete [] vector;
    delete [] x;

    solver.get_timer().print_last_formatted() ;
    
    return 0;
}