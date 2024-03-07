# EuMaster4HPC CG Challenge

## Tests compilation
The following instructions are intended to compile the code on Meluxina supercomputer.
```
cd ParallelCG
mkdir build && cd build
module load intel CMake CUDA OpenBLAS
```
### OpenMp, MPI, OpenCL tests
To compile non-cuda tests, run:
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
The executables are now ready in the `build/test` folder.
