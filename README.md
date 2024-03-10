# EuMaster4HPC CG Challenge
The following project has been structured in a way to offer a common interface for the final user, acting as a library to compute the Conjugate Gradient through different implementations.

## Tests and benchmarks compilation
The following instructions are intended to compile the code on Meluxina supercomputer.
```
cd ParallelCG
mkdir build && cd build
module load intel CMake CUDA OpenBLAS
```
### OpenMp, MPI, OpenCL tests
To compile non-cuda tests and benchmarks, run:
```
cmake ..
```
### Cuda tests
To compile only cuda tests, run:
```
cmake -DCUDA_TESTS=ON ..
```
> Note: remember to clean the `build` folder when switching between cuda/non cuda tests.

Then, compile in parallel using `make`:
```
make -j
```
The executables are now ready in the `build/test` folder, and some of the speedups benchmarks in `build/benchmark/SpeedupTemplates`.

Finally, to execute the compiled programs, use the usual ```srun``` commands.

## Usages examples
Without Mpi:
```cpp     
#include "../challenge/include/cg/cgcore.hpp"

using namespace cgcore;

int main(int argc, char ** argv){

    double *matrix;
    double *vector;
    double *x;
    size_t n, m ; 
    int max_iter = 1000;
    double res = 1.e-6;

    // CGSolver<OpenMP_BLAS_CG> solver;
    // CGSolver<OpenMP_CG> solver;
    // CGSolver<OpenCL_CG> solver;
    // CGSolver<...> solver;

    CGSolver<OpenMP_CG> solver;

    const char *m_path =  "../test/assets/matrix_10000.bin";
    const char *rhs_path =  "../test/assets/rhs_10000.bin";

    utils::read_matrix_from_file(m_path , matrix, n, m);
    utils::read_vector_from_file(rhs_path, vector, n);
    x = new double[n];

   solver.solve(matrix, vector, x, n, max_iter, res);

    delete [] matrix;
    delete [] vector;
    delete [] x;

    solver.get_timer().print_last_formatted() ;
    return 0;
}
```

With MPI and distribution of the matrix (if supported by the strategy): 

```cpp
#include "../challenge/include/cg/cgcore.hpp"

using namespace cgcore;

int main(int argc, char ** argv){

    MPI_Init(&argc, &argv);

    double *matrix;
    double *vector;
    double *x;
    size_t n, m ; 
    size_t v_cols;
    int max_iter = 1000;
    double res = 1.e-6;

    // CGSolver<MPI_DISTRIBUTED> solver;
    CGSolver<CUBLAS_CG> solver;

    const char *m_path =  "../test/assets/matrix_10000.bin";
    const char *rhs_path =  "../test/assets/rhs_10000.bin";

    int* rows_per_process;
    int* displacements;

    utils::mpi::mpi_distributed_read_matrix(m_path, matrix, m, n, rows_per_process, displacements);
    utils::mpi::mpi_distributed_read_all_vector(rhs_path, vector, m, v_cols, rows_per_process, displacements);
    x = new double[n];

    solver.solve(matrix, vector, x, n, max_iter, res);

    delete [] matrix;
    delete [] vector;
    delete [] x;

    solver.get_timer().print_last_formatted() ;
    
    MPI_Finalize();
    return 0;
}
```
## Running specific benchmarks
In the [Benchmark folder](./benchmark) you can find different folders that contain the code used for benchmarks. Such programs are intended for benchmarking purposes only, as 
the library is built via CMake.
To run a specific benchmark, make sure to allocate enough resources, load the intended modules, and then execute the ```.sh``` script. 

In the [Benchmark folder readme](./benchmark/README.md) you will find a comprehensive list describing the modules that must be loaded to run each benchmark.
