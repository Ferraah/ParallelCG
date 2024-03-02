 
add_executable(sequential sequential.cpp)
add_executable(opencl opencl.cpp)
add_executable(micro_benchmark_opencl micro_benchmark_opencl.cpp)

target_link_libraries(sequential PUBLIC CG_CORE)
target_include_directories(sequential PUBLIC "${CMAKE_SOURCE_DIR}/challenge/include")
target_include_directories(sequential PUBLIC /home/users/u101373/OpenCL-CLHPP/include)
target_include_directories(sequential PUBLIC /home/users/u101373/OpenCL-CLHPP/include)
target_compile_options(sequential PRIVATE -fsycl -m64 -g)
target_link_options(sequential PRIVATE -lOpenCL)

target_link_libraries(opencl PUBLIC CG_CORE)
target_include_directories(opencl PUBLIC "${CMAKE_SOURCE_DIR}/challenge/include")
target_include_directories(opencl PUBLIC /home/users/u101373/OpenCL-CLHPP/include)
target_compile_options(opencl PRIVATE -fsycl -m64 -g)
target_link_options(opencl PRIVATE -lOpenCL)

target_link_libraries(micro_benchmark_opencl PUBLIC CG_CORE)
target_include_directories(micro_benchmark_opencl PUBLIC "${CMAKE_SOURCE_DIR}/challenge/include")
target_include_directories(micro_benchmark_opencl PUBLIC /home/users/u101373/OpenCL-CLHPP/include)
target_include_directories(micro_benchmark_opencl PUBLIC /home/users/u101373/OpenCL-CLHPP/include)
target_compile_options(micro_benchmark_opencl PRIVATE -fsycl -m64 -g)
target_link_options(micro_benchmark_opencl PRIVATE -lOpenCL)

