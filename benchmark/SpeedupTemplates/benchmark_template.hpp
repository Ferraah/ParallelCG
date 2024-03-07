#ifndef BENCHMARK_TEMPLATE_HPP
#define BENCHMARK_TEMPLATE_HPP

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <time.h>
#include "../../challenge/include/cg/cgcore.hpp"

// benchmarking: measuring time
double wall_time(){
  struct timespec t;
  clock_gettime (CLOCK_MONOTONIC, &t);
  return 1.*t.tv_sec + 1.e-9*t.tv_nsec;
}

using namespace cgcore;

template <class STRATEGY>
int benchmark_cg(int argc, char **argv, const char * input_file_matrix, const char * input_file_rhs, const char *file_name){

    // Usage instructions
    printf("Usage: ./random_matrix input_file_matrix.bin input_file_rhs.bin output_file_sol.bin max_iters rel_error\n");
    printf("All parameters are optional and have default values\n");
    printf("\n");
    //

    // Settings for the benchmark  -----------------------
    int max_iters = 1000;
    double rel_error = 1e-9;

    if(argc > 1) input_file_matrix = argv[1];
    if(argc > 2) input_file_rhs = argv[2];
    if(argc > 4) max_iters = atoi(argv[4]);
    if(argc > 5) rel_error = atof(argv[5]);

    printf("Provided settings :\n");
    printf("  input_file_matrix: %s\n", input_file_matrix);
    printf("  input_file_rhs:    %s\n", input_file_rhs);
    printf("  max_iters:         %d\n", max_iters);
    printf("  rel_error:         %e\n", rel_error);
    printf("\n");

    // -------------------------------------------------


    // Defining system structures
    double * matrix;
    double * rhs;
    size_t size;
    //

    // Data loading benchmarks 
    {

        // Reading matrix
        printf("Reading matrix from file ...\n");

        size_t matrix_rows;
        size_t matrix_cols;

        double t_read_m = -wall_time();
        bool success_read_matrix = utils::read_matrix_from_file(input_file_matrix, matrix, matrix_rows, matrix_cols);

        if(!success_read_matrix)
        {
            fprintf(stderr, "Failed to read matrix\n");
            return 1;
        }

        t_read_m += wall_time();
        printf("Done\n");
        printf("Reading the matrix took %8g s. \n", t_read_m);
        printf("\n");
        // 

        // Reading rhs
        printf("Reading right hand side from file ...\n");
        size_t rhs_rows;
        size_t rhs_cols;
        double t_read_right_side = - wall_time();
        bool success_read_rhs = utils::read_matrix_from_file(input_file_rhs, rhs, rhs_rows, rhs_cols);
        if(!success_read_rhs)
        {
            fprintf(stderr, "Failed to read right hand side\n");
            return 2;
        }
        t_read_right_side += wall_time();
        printf("Done\n");
        printf("Reading the right side took %8g s. \n", t_read_right_side);
        printf("\n");
        
        // 

        // System topology check 
        if(matrix_rows != matrix_cols)
        {
            fprintf(stderr, "Matrix has to be square\n");
            return 3;
        }
        if(rhs_rows != matrix_rows)
        {
            fprintf(stderr, "Size of right hand side does not match the matrix\n");
            return 4;
        }
        if(rhs_cols != 1)
        {
            fprintf(stderr, "Right hand side has to have just a single column\n");
            return 5;
        }
        // 

        size = matrix_rows;
    }

    // Strategy benchmark

    printf("Solving the system ...\n");

    double * sol = new double[size];

    CGSolver<STRATEGY> solver;
    solver.solve(matrix, rhs, sol, size, max_iters, rel_error);

    double solving_time = solver.get_timer().get_last(); 

    printf("Done\n");
    std::cout << "The c-g-algorithm took " << solving_time <<  " s." << std::endl;

    printf("\n");

    // 

    std::ofstream myfile;
    myfile.open(file_name, std::ios::app);
    // Append to file
    myfile << size << "\t" << solving_time << "\n";
    myfile.close();

    delete[] matrix;
    delete[] rhs;
    delete[] sol;

    printf("Finished successfully\n");

    return 0;
}

#endif