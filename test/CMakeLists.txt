 
add_executable(main main.cpp)

target_link_libraries(main PUBLIC CG_CORE)
target_include_directories(main PUBLIC "${CMAKE_SOURCE_DIR}/challenge/include")