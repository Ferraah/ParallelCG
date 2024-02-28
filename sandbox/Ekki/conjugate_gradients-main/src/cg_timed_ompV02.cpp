#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <iostream>
#include <fstream>
#include <time.h>

#include <omp.h>
#include <numeric>

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



double dot(const double * x, const double * y, size_t size)
{
    double result = 0.0;
    #pragma omp parallel for default(none) reduction(+:result) shared(x, y, size) // this might have no impact on the performance
    for(size_t i = 0; i < size; i++)
    {
        result += x[i] * y[i];
    }
    return result;
    
    // Also a possibility
    // return std::inner_product(x, x+size, y, 0.0);

    // Splitting it among the threads and every thread uses the inner product functionality
    // #pragma omp parallel default(none) reduction(+:result) shared(x, y, size)
    // {
    //     int num_threads = omp_get_num_threads();
    //     int my_tid = omp_get_thread_num();
    //     int num_items = size/num_threads;
    //     int my_start = my_tid * num_items;
    //     int my_end = my_start + num_items;
    //     if (my_tid == num_threads-1){
    //         my_end = size;
    //     }
    //     result += std::inner_product(x+my_start, x+my_end, y+my_start, 0.0);
    // }
    // return result;
}



void axpby(double alpha, const double * x, double beta, double * y, size_t size)
{
    // y = alpha * x + beta * y
    #pragma omp parallel for default(none) shared(alpha, x, beta, y, size)
    for(size_t i = 0; i < size; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}



void gemv(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols)
{
    // y = alpha * A * x + beta * y;
    // parallelizing this for-loop is the most important and has the most significant impact on performance
    // #pragma omp parallel for simd shared(alpha, A, x, beta, y, num_rows, num_cols) // or use collapse
    // for(size_t r = 0; r < num_rows; r++)
    // {
    //     double y_val = 0.0;
    //     for(size_t c = 0; c < num_cols; c++)
    //     {
    //         y_val += alpha * A[r * num_cols + c] * x[c];
    //     }
    //     y[r] = beta * y[r] + y_val;
    // }

    #pragma omp parallel default(none) shared(alpha, A, x, beta, y, num_rows, num_cols)
    {
        int num_threads = omp_get_num_threads();
        int my_tid = omp_get_thread_num();
        int num_items = num_rows/num_threads;
        int my_start = my_tid * num_items;
        int my_end = my_start + num_items;
        if (my_tid == num_threads-1){
            my_end = num_rows;
        }
        for(size_t r = my_start; r < my_end; r++){
            double y_val = 0.0;
            // for(size_t c = 0; c < num_cols; c++)
            // {
            //     y_val += alpha * A[r * num_cols + c] * x[c];
            // }
            y_val = alpha * std::inner_product(A+my_start*num_cols, A+my_start*num_cols+num_cols, x, 0.0);
            y[r] = beta * y[r] + y_val;
        }
    }
}



void conjugate_gradients(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error)
{
    double alpha, beta, bb, rr, rr_new; // bb = b*b
    double * r = new double[size];      // r = residual = b - A * x
    double * p = new double[size];      // commonly denoted as d
    double * Ap = new double[size];     // Ap = A * d
    int num_iters;

    // Initializing: x(0)=vec(0) -> r = b & start with d(0) = p(0) = r(0)
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
        gemv(1.0, A, p, 0.0, Ap, size, size);           // = A * d = Ap (we are working with square matrices) -> stored in Ap
        alpha = rr / dot(p, Ap, size);                  // = alpha
        axpby(alpha, p, 1.0, x, size);                  // calc. new x -> store in x
        axpby(-alpha, Ap, 1.0, r, size);                // calc. new r -> store in r
        rr_new = dot(r, r, size);                       // r_new * r_new
        beta = rr_new / rr;                             // = beta
        rr = rr_new;                                    // set for next iteration
        if(std::sqrt(rr / bb) < rel_error) { break; }   // -> break condition (rel. error small enough)
        axpby(1.0, r, beta, p, size);                   // next direction p
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

    printf("Solving the system ...\n");
    double * sol = new double[size];
    double t_comp = - wall_time();
    conjugate_gradients(matrix, rhs, sol, size, max_iters, rel_error);
    t_comp += wall_time();
    printf("Done\n");
    printf("The c-g-algorithm took %8g s. \n", t_comp);
    printf("\n");

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

    delete[] matrix;
    delete[] rhs;
    delete[] sol;

    printf("Finished successfully\n");

    return 0;
}
