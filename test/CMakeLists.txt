# Define a macro for setting up an executable
macro(setup_executable TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})

    target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/challenge/include")
    target_include_directories(${TARGET_NAME} PUBLIC /home/users/u101373/OpenCL-CLHPP/include)
    target_compile_options(${TARGET_NAME} PRIVATE -fsycl -m64 -g)

    # Link with CG_CORE, 
    target_link_libraries(${TARGET_NAME} PUBLIC CG_CORE)
    target_link_options(${TARGET_NAME} PRIVATE -lOpenCL -fopenmp -lopenblas)
    # Additional settings for MPI
    target_include_directories(${TARGET_NAME} PUBLIC ${MPI_C_INCLUDE_PATH})

endmacro()

# Use the macro for each executable
setup_executable(sequential sequential.cpp)
setup_executable(opencl opencl.cpp)
setup_executable(openmp openmp.cpp)
setup_executable(mpi mpi.cpp)
setup_executable(openmp_blas openmp_blas.cpp)
setup_executable(micro_benchmark_opencl micro_benchmark_opencl.cpp)
