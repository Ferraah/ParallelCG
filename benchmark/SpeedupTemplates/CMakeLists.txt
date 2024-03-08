# Define a macro for setting up an executable
macro(setup_executable TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})

    target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/challenge/include")

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
else()
    setup_executable(openmp_benchmark openmp_benchmark.cpp)
    setup_executable(opencl_benchmark opencl_benchmark.cpp)
    setup_executable(mpi_benchmark mpi_benchmark.cpp)
endif(CUDA_TESTS)
