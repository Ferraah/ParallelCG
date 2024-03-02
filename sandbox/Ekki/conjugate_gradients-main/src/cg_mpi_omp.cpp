//
// Execute with: module load OpenMPI
//               mpic++ -o cg_mpi_omp src/cg_mpi_omp.cpp -fopenmp
//               export OMP_NUM_THREADS=4
//               srun -n 4 ./cg_mpi_omp io/matrix.bin io/rhs.bin io/sol.bin        // up to 256 processes
//


#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <iostream>
#include <fstream>
#include <time.h>

#include <mpi.h>
#include <omp.h>

// benchmarking: measuring time
double wall_time(){
  struct timespec t;
  clock_gettime (CLOCK_MONOTONIC, &t);
  return 1.*t.tv_sec + 1.e-9*t.tv_nsec;
}

// declare output file
std::string file_name = "io/temp.txt";
std::ofstream myfile;


// start of programm
bool read_matrix_from_file(const char * filename, double ** matrix_out, size_t * num_rows_out, size_t * num_cols_out)
{
    double * matrix;
    size_t num_rows;
    size_t num_cols;

    FILE * file = fopen(filename, "rb");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fread(&num_rows, sizeof(size_t), 1, file);
    fread(&num_cols, sizeof(size_t), 1, file);
    matrix = new double[num_rows * num_cols];
    fread(matrix, sizeof(double), num_rows * num_cols, file);

    *matrix_out = matrix;
    *num_rows_out = num_rows;
    *num_cols_out = num_cols;

    fclose(file);

    return true;
}



bool write_matrix_to_file(const char * filename, const double * matrix, size_t num_rows, size_t num_cols)
{
    FILE * file = fopen(filename, "wb");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fwrite(&num_rows, sizeof(size_t), 1, file);
    fwrite(&num_cols, sizeof(size_t), 1, file);
    fwrite(matrix, sizeof(double), num_rows * num_cols, file);

    fclose(file);

    return true;
}



void print_matrix(const double * matrix, size_t num_rows, size_t num_cols, FILE * file = stdout)
{
    fprintf(file, "%zu %zu\n", num_rows, num_cols);
    for(size_t r = 0; r < num_rows; r++)
    {
        for(size_t c = 0; c < num_cols; c++)
        {
            double val = matrix[r * num_cols + c];
            printf("%+6.3f ", val);
        }
        printf("\n");
    }
}


// Parallelize dot product with MPI: pass whole vectors and take care of subsets within function
// double dot(const double * x, const double * y, size_t size)
// {
//     double result_tot;
//     double sub_result = 0.0;
//     for(size_t i = 0; i < sub_size; i++)
//     {
//         sub_result += sub_x[i] * sub_y[i];
//     }

//     MPI_Allreduce(&sub_result, &result_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

//     return result_tot;
// }

double dot(const double * x, const double * y, size_t size)
{
    double result = 0.0;
    // #pragma omp parallel for default(none) reduction(+:result) shared(x, y, size) // this might have no impact on the performance
    for(size_t i = 0; i < size; i++)
    {
        result += x[i] * y[i];
    }
    return result;
}



void axpby(double alpha, const double * x, double beta, double * y, size_t size)
{
    // y = alpha * x + beta * y
    // #pragma omp parallel for default(none) shared(alpha, x, beta, y, size)
    for(size_t i = 0; i < size; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}



void gemv(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols, int num_processes, int my_rank, int * displacements, int * counts)
{
    // y = alpha * A * x + beta * y;

    // Split computation along A and y: e.g.: p1 has rows 1 to num_roms/num_processes
    // Compute remainder for load balancing -> use modulo
    int rest = num_rows % num_processes;
    int my_num_rows = (int(num_rows / num_processes));
    int my_start = -1;
    int my_end = -1;
    if (my_rank < rest){ // distribute the rest of the rows across all processes evenly; my_rank starts at 0 & rest at 1 -> <
        my_num_rows += 1;
        my_start = my_rank * my_num_rows;
        my_end = my_start + my_num_rows;
    }
    else {
        my_start = my_rank * my_num_rows + rest;
        my_end = my_start + my_num_rows;
    }

    // local y
    double my_y[my_num_rows];

    // Determine correct range for each rank
    // p1: 0 - my_num_rows - 1, p2: my_num_rows - 
    #pragma omp parallel for simd shared(alpha, A, x, beta, y, my_start, my_end, num_cols, my_y)
    for(size_t r = my_start ; r < my_end; r++)
    {
        double y_val = 0.0;
        for(size_t c = 0; c < num_cols; c++)
        {
            y_val += alpha * A[r * num_cols + c] * x[c];
        }
        my_y[r - my_start] = beta * y[r] + y_val;
    }

    // #pragma omp parallel default(none) shared(alpha, A, x, beta, y, my_num_rows, my_start, my_end, num_cols, my_y)
    // {
    //     int num_threads = omp_get_num_threads();
    //     int my_tid = omp_get_thread_num();
    //     int rest_omp = my_num_rows % num_threads;
    //     int my_num_rows_omp = (int(my_num_rows / num_threads));
    //     int my_start_omp = -1;
    //     int my_end_omp = -1;
    //     if (my_tid < rest_omp){ // distribute the rest of the rows across all threads evenly; my_rank starts at 0 & rest at 1 -> <
    //         my_num_rows_omp += 1;
    //         my_start_omp = my_tid * my_num_rows_omp;
    //         my_end_omp = my_start_omp + my_num_rows_omp;
    //     }
    //     else {
    //         my_start_omp = my_tid * my_num_rows_omp + rest_omp;
    //         my_end_omp = my_start_omp + my_num_rows_omp;
    //     }

    //     for(size_t r = my_start_omp; r < my_end_omp; r++){
    //         double y_val = 0.0;
    //         for(size_t c = 0; c < num_cols; c++)
    //         {
    //             y_val += alpha * A[r * num_cols + c] * x[c];
    //         }
    //         y[r] = beta * y[r] + y_val;
    //     }
    // }

    // Stack all local y-vectors (my_y)
    MPI_Allgatherv(my_y, my_num_rows, MPI_DOUBLE, y, counts, displacements, MPI_DOUBLE, MPI_COMM_WORLD);
}



void conjugate_gradients(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error)
{
    int rank, num_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double alpha, beta, bb, rr, rr_new;
    double * r = new double[size];
    double * p = new double[size];
    double * Ap = new double[size];
    int num_iters;

    // Displacement and counts for MPI_Allgatherv

    int displacements[num_processes];
    int counts[num_processes];
    for (int i = 0; i < num_processes; i++){
        int rest = size % num_processes;
        int my_num_rows = (int(size / num_processes));
        if (i < rest){ 
            my_num_rows += 1;
            displacements[i] = my_num_rows * i;
            counts[i] = my_num_rows;
        }
        else {
            displacements[i] = my_num_rows * i + rest;
            counts[i] = my_num_rows;
        }
    }

    for(size_t i = 0; i < size; i++)
    {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
    }

    bb = dot(b, b, size);
    rr = bb;
    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        gemv(1.0, A, p, 0.0, Ap, size, size, num_processes, rank, displacements, counts); // Parallelized with Allgatherv
        alpha = rr / dot(p, Ap, size);
        axpby(alpha, p, 1.0, x, size);      // do not parallize with MPI due to overhead
        axpby(-alpha, Ap, 1.0, r, size);
        rr_new = dot(r, r, size);
        beta = rr_new / rr;
        rr = rr_new;
        if(std::sqrt(rr / bb) < rel_error) { break; }
        axpby(1.0, r, beta, p, size);
    }

    delete[] r;
    delete[] p;
    delete[] Ap;

    if(num_iters <= max_iters)
    {
        printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
    }
}





int main(int argc, char ** argv)
{
    printf("Usage: ./random_matrix input_file_matrix.bin input_file_rhs.bin output_file_sol.bin max_iters rel_error\n");
    printf("All parameters are optional and have default values\n");
    printf("\n");

    const char * input_file_matrix = "io/matrix.bin";
    const char * input_file_rhs = "io/rhs.bin";
    const char * output_file_sol = "io/sol.bin";
    int max_iters = 1000;
    double rel_error = 1e-9;

    if(argc > 1) input_file_matrix = argv[1];
    if(argc > 2) input_file_rhs = argv[2];
    if(argc > 3) output_file_sol = argv[3];
    if(argc > 4) max_iters = atoi(argv[4]);
    if(argc > 5) rel_error = atof(argv[5]);

    printf("Command line arguments:\n");
    printf("  input_file_matrix: %s\n", input_file_matrix);
    printf("  input_file_rhs:    %s\n", input_file_rhs);
    printf("  output_file_sol:   %s\n", output_file_sol);
    printf("  max_iters:         %d\n", max_iters);
    printf("  rel_error:         %e\n", rel_error);
    printf("\n");

    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, num_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    // TODO: Have to think about how to read in matrices and divide them upon ranks
    // Done: Do not divide them explicitly, but rather assign local domain within each function for each specific rank
    double * matrix;
    double * rhs;
    size_t size;

    {
        printf("Reading matrix from file ...\n");
        size_t matrix_rows;
        size_t matrix_cols;
        double t_read_m = -wall_time();
        bool success_read_matrix = read_matrix_from_file(input_file_matrix, &matrix, &matrix_rows, &matrix_cols);
        if(!success_read_matrix)
        {
            fprintf(stderr, "Failed to read matrix\n");
            return 1;
        }
        t_read_m += wall_time();
        printf("Done\n");
        printf("Reading the matrix took %8g s. \n", t_read_m);
        printf("\n");

        printf("Reading right hand side from file ...\n");
        size_t rhs_rows;
        size_t rhs_cols;
        double t_read_right_side = - wall_time();
        bool success_read_rhs = read_matrix_from_file(input_file_rhs, &rhs, &rhs_rows, &rhs_cols);
        if(!success_read_rhs)
        {
            fprintf(stderr, "Failed to read right hand side\n");
            return 2;
        }
        t_read_right_side += wall_time();
        printf("Done\n");
        printf("Reading the right side took %8g s. \n", t_read_right_side);
        printf("\n");

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

        size = matrix_rows;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0){
        printf("Solving the system ...\n");
    }
    double * sol = new double[size];

    double t_comp = - wall_time();
    conjugate_gradients(matrix, rhs, sol, size, max_iters, rel_error);
    t_comp += wall_time();
    printf("Done\n");
    printf("The c-g-algorithm took %8g s. \n", t_comp);
    printf("\n");

    

    if (rank == 0){
         // write header to file
        // myfile.open(file_name);
        // myfile << "matrix_size \t" << "time \n";
        // myfile.close(); 

        // Writing results to file
        myfile.open(file_name);
        myfile << size << "\t" << t_comp << "\n";
        myfile.close();

        printf("Writing solution to file ...\n");
        double t_write_m = - wall_time();
        bool success_write_sol = write_matrix_to_file(output_file_sol, sol, size, 1);
        if(!success_write_sol)
        {
            fprintf(stderr, "Failed to save solution\n");
            return 6;
        }
        t_write_m += wall_time();
        printf("Done\n");
        printf("Writing the matrix back took %8g s. \n", t_write_m);
        printf("\n");
    }

    delete[] matrix;
    delete[] rhs;
    delete[] sol;

    printf("%d: Finished successfully\n", rank);

    MPI_Finalize();
    return 0;
}
