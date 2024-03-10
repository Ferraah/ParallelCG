#ifndef BENCHMARK_TEMPLATE_HPP
#define BENCHMARK_TEMPLATE_HPP

#define PRINTF(format, ...) \
    do { \
        int rank; \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
        if (rank == 0) { \
            printf(format, ##__VA_ARGS__); \
        } \
    } while (0)

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

template <class FASTER_STRATEGY, class SLOWER_STRATEGY>
int benchmark_cg(int argc, char **argv, const char * input_file_matrix, const char * input_file_rhs, const char *file_name, bool run_sequential, bool distribute_matrix){

    
    int rank, num_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    // Usage instructions
    PRINTF("Usage: ./random_matrix input_file_matrix.bin input_file_rhs.bin output_file_sol.bin max_iters rel_error\n");
    PRINTF("All parameters are optional and have default values\n");
    PRINTF("\n");
    //



    // Settings for the benchmark  -----------------------
    int max_iters = 1000;
    double rel_error = 1e-9;

    if(argc > 1) input_file_matrix = argv[1];
    if(argc > 2) input_file_rhs = argv[2];
    if(argc > 4) max_iters = atoi(argv[4]);
    if(argc > 5) rel_error = atof(argv[5]);

    PRINTF("Provided settings :\n");
    PRINTF("  input_file_matrix: %s\n", input_file_matrix);
    PRINTF("  input_file_rhs:    %s\n", input_file_rhs);
    PRINTF("  max_iters:         %d\n", max_iters);
    PRINTF("  rel_error:         %e\n", rel_error);
    PRINTF("\n");

    // -------------------------------------------------


    // Defining system structures
    double * matrix;
    double * rhs;
    size_t size;
    //

    // Data loading benchmarks 
    {

        int* rows_per_process;
        int* displacements;

        // Reading matrix
        PRINTF("Reading matrix from file ...\n");

        size_t matrix_rows;
        size_t matrix_cols;

        double t_read_m = -wall_time();
        
        bool success_read_matrix;

        if(!distribute_matrix) 
            success_read_matrix = utils::read_matrix_from_file(input_file_matrix, matrix, matrix_rows, matrix_cols);
        else
            success_read_matrix = utils::mpi::mpi_distributed_read_matrix(input_file_matrix, matrix, matrix_rows, matrix_cols, rows_per_process, displacements);

        if(!success_read_matrix)
        {
            fprintf(stderr, "Failed to read matrix\n");
            return 1;
        }

        t_read_m += wall_time();
        PRINTF("Done\n");
        PRINTF("Reading the matrix took %8g s. \n", t_read_m);
        PRINTF("\n");
        // 

        // Reading rhs
        PRINTF("Reading right hand side from file ...\n");
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
        PRINTF("Done\n");
        PRINTF("Reading the right side took %8g s. \n", t_read_right_side);
        PRINTF("\n");
        
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
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Strategy benchmark

    PRINTF("Solving the system ...\n");

    double * sol = new double[size];

    CGSolver<FASTER_STRATEGY> solver1;

    solver1.solve(matrix, rhs, sol, size, max_iters, rel_error);
    double solving_time1 = solver1.get_timer().get_last(); 
    if(rank == 0)
        std::cout << "Solver 1 took " << solving_time1 <<  " s." << std::endl;

    double solving_time2;
    if(rank == 0 && run_sequential){
        CGSolver<SLOWER_STRATEGY> solver2;
        solver2.solve(matrix, rhs, sol, size, max_iters, rel_error);
        solving_time2 = solver2.get_timer().get_last(); 
        std::cout << "Solver 2 took " << solving_time2 <<  " s." << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    PRINTF("Done\n");

    PRINTF("\n");

    // 

    if(rank == 0){
        std::ofstream myfile;
        myfile.open(file_name, std::ios::app);
        // Append to file
        if(run_sequential)
            myfile << size << "\t" << solving_time1 << "\t" << solving_time2 <<  "\t" << solving_time2/solving_time1 << "\n";
        else
            myfile << size << "\t" << solving_time1 << "\n";  
        myfile.close();
    }

    delete[] matrix;
    delete[] rhs;
    delete[] sol;

    PRINTF("Finished successfully\n");

    return 0;
}

#endif