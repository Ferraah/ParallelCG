
file(GLOB_RECURSE SOURCES "src/*.cpp")

if(CUDA_TESTS)
    # Add CUDA source files to the SOURCES variable
    file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")
    list(APPEND SOURCES ${CUDA_SOURCES})

endif(CUDA_TESTS)


add_library(CG_CORE SHARED ${SOURCES})

target_include_directories(CG_CORE PRIVATE "include/CL")
target_link_libraries(CG_CORE PUBLIC MPI::MPI_CXX )

if(CUDA_TESTS)
else()
    target_compile_options(CG_CORE PRIVATE -fsycl)
    target_link_libraries(CG_CORE PUBLIC OpenMP::OpenMP_CXX)
endif()

target_include_directories(CG_CORE PUBLIC ${MPI_C_INCLUDE_PATH})
# Additional settings for MPI
