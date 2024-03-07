# Define a macro for setting up an executable
macro(setup_executable TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})

    target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/challenge/include")
    target_include_directories(${TARGET_NAME} PUBLIC /home/users/u101373/OpenCL-CLHPP/include)

    if(CUDA_TESTS)
        #no opencl flag
        target_link_options(${TARGET_NAME} PRIVATE -lcublas -lOpenCL -lopenblas)
    else()
        target_compile_options(${TARGET_NAME} PRIVATE -fsycl -m64 -g)
        target_link_options(${TARGET_NAME} PRIVATE -lOpenCL -fopenmp -lopenblas -lcublas )
    endif(CUDA_TESTS)
    
    # Link with CG_CORE, 
    target_link_libraries(${TARGET_NAME} PUBLIC CG_CORE)
    # Additional settings for MPI
    target_include_directories(${TARGET_NAME} PUBLIC ${MPI_C_INCLUDE_PATH})

endmacro()

if(CUDA_TESTS)
   setup_executable(cuda cuda.cpp)
   setup_executable(cuda_mpi cuda_mpi.cpp)
else()
 # Use the macro for each executable
    setup_executable(sequential sequential.cpp)
    setup_executable(opencl opencl.cpp)
    setup_executable(openmp openmp.cpp)
    setup_executable(mpi mpi.cpp)
    setup_executable(mpi_openmp mpi_openmp.cpp)
    setup_executable(mpi_matrix_distr mpi_matrix_distr.cpp)
    setup_executable(openmp_blas openmp_blas.cpp)
    setup_executable(micro_benchmark_opencl micro_benchmark_opencl.cpp)
endif(CUDA_TESTS)
