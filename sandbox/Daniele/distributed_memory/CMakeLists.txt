cmake_minimum_required(VERSION 3.1)

project(
  ParallelCG 
  VERSION 1.0
  LANGUAGES CXX)

find_package(MPI REQUIRED) 
find_package(OpenMP REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
 
file(GLOB_RECURSE SOURCES "utils.cpp") 
add_executable(test test.cpp)
add_library(UTILS SHARED ${SOURCES})

target_link_libraries(test MPI::MPI_CXX OpenMP::OpenMP_CXX UTILS)