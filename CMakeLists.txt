# establishes policy settings and ensures that the following CMake
# functions are run with a compatible version of CMake
  cmake_minimum_required(VERSION 3.13)

# Project name and version
project(CHALLENGE VERSION 1.0)

# Set C++ standard

option(CUDA_TESTS "Compile CUDA tests." OFF)


# Define default build type if not provided
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Set build type options
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
  STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")

# Set compiler flags for different build types
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# find required packages
# find MPI
find_package(MPI REQUIRED)

if(CUDA_TESTS)
  find_package(CUDA)
  enable_language(CUDA) #enable NVCC compiler

else()
  set(CMAKE_CXX_COMPILER icpx)
  # find OpenMP
  cmake_policy(SET CMP0074 NEW)
  find_package(OpenMP REQUIRED)
  # Find OpenBLAS package
  find_package(OpenBLAS REQUIRED)
endif(CUDA_TESTS)

add_subdirectory(challenge)
add_subdirectory(test)
add_subdirectory(benchmark/SpeedupTemplates)


