 
add_executable(sequential sequential.cpp)
add_executable(opencl opencl.cpp)

target_link_libraries(sequential PUBLIC CG_CORE)
target_include_directories(sequential PUBLIC "${CMAKE_SOURCE_DIR}/challenge/include")
target_include_directories(sequential PUBLIC /home/users/u101373/OpenCL-CLHPP/include)
target_include_directories(sequential PUBLIC /home/users/u101373/OpenCL-CLHPP/include)
target_compile_options(sequential PRIVATE -fsycl -m64)
target_link_options(sequential PRIVATE -lOpenCL)

target_link_libraries(opencl PUBLIC CG_CORE)
target_include_directories(opencl PUBLIC "${CMAKE_SOURCE_DIR}/challenge/include")
target_include_directories(opencl PUBLIC /home/users/u101373/OpenCL-CLHPP/include)
target_compile_options(opencl PRIVATE -fsycl -m64)
target_link_options(opencl PRIVATE -lOpenCL)