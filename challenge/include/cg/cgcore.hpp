#ifndef CG_CORE_INCLUDE
#define CG_CORE_INCLUDE

#include "../../src/cgcore/CGSolver.hpp"
#include "../../src/cgcore/strategy/CGStrategy.hpp"
#include "../../src/cgcore/strategy/sequential/Sequential_CG.hpp"
#include "../../src/cgcore/strategy/openCL/OpenCL_CG.hpp"
#include "../../src/cgcore/strategy/openMP/OpenMP_CG.hpp"
#include "../../src/cgcore/strategy/openMP_BLAS/OpenMP_BLAS_CG.hpp"
#include "../../src/cgcore/strategy/MPI/MPI_CG.hpp"
#include "../../src/cgcore/strategy/openCL/OpenCLUtils.hpp"
#include "../../src/cgcore/strategy/CUDA/CUBLAS_CG.hpp"
#include "../../src/cgcore/strategy/CUDA/CUDA_MPI.hpp"

#include "../../src/cgcore/utils/utils.hpp"

#include "../../src/cgcore/timer/Timer.hpp"
#endif