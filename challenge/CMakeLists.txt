
file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(CG_CORE SHARED ${SOURCES})

# PROVVISORY @
target_include_directories(CG_CORE PRIVATE /home/users/u101373/OpenCL-CLHPP/include)
target_compile_options(CG_CORE PRIVATE -fsycl -m64 -g)
target_link_libraries(CG_CORE PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX)

# Additional settings for MPI
target_include_directories(CG_CORE PUBLIC ${MPI_C_INCLUDE_PATH})
