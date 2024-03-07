
#include "MPI_DISTRIBUTED.hpp"

    void MPI_DISTRIBUTED::run(const double * A , const double * b, double * x, size_t size, int max_iter, double res_error) const{
        int rank;
        int mpi_size;

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

        int* rows_per_process = new int[mpi_size];
        int* displacements = new int[mpi_size];

        for (int i = 0; i < mpi_size; ++i)
        {
            rows_per_process[i] = size / mpi_size;
            if (i < size % mpi_size)
                rows_per_process[i]++;
        }

        displacements[0] = 0;
        for (int i = 1; i < mpi_size; ++i)
        {
            displacements[i] = displacements[i-1] + rows_per_process[i-1];   
        }

        conjugate_gradient(A, b, x, size, size, rows_per_process, displacements, max_iter, res_error);

        delete [] rows_per_process;
        delete [] displacements;
    }

    double MPI_DISTRIBUTED::dot(const double * x, const double * y, size_t size) 
    {
        double result = 0.0;
        for(size_t i = 0; i < size; i++)
        {
            result += x[i] * y[i];
        }
        return result;
    }



    void MPI_DISTRIBUTED::axpby(double alpha, const double * x, double beta, double * y, size_t size) 
    {
        // y = alpha * x + beta * y

        for(size_t i = 0; i < size; i++)
        {
            y[i] = alpha * x[i] + beta * y[i];
        }
    }



    void MPI_DISTRIBUTED::gemv(double alpha, const double * A, const double * x, double beta, double *& y, size_t num_rows, size_t num_cols, int displ) 
    {
        for(size_t r = 0; r < num_rows; r++)
        {
            double y_val = 0.0;
            for(size_t c = 0; c < num_cols; c++)
            {
                y_val += alpha * A[(r) * num_cols + c] * x[c];

            }
            y[r + displ] = y_val;
        }
    }


    /**
     * 
    */ 
    void MPI_DISTRIBUTED::conjugate_gradient(const double * A, const double * b, double * x, size_t rows, size_t cols, int* rows_per_process, int* displacements, int max_iters, double rel_error) 
    {
        double alpha, beta, bb, rr, rr_new;
        double * r = new double[rows];
        double * p = new double[rows];
        double * Ap = new double[rows];
        int num_iters;

        int size;
        int rank;

        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        memset(x, 0, rows * sizeof(double));
        memcpy(r, b, rows * sizeof(double));
        memcpy(p, b, rows * sizeof(double));

        
        MPI_Barrier(MPI_COMM_WORLD);

        bb = dot(b, b, rows);
        rr = bb;
        for(num_iters = 1; num_iters <= max_iters; num_iters++)
        {
            gemv(1.0, A, p, 0.0, Ap, rows_per_process[rank], rows, displacements[rank]);
            
            MPI_Barrier(MPI_COMM_WORLD);

            MPI_Allgatherv(Ap + displacements[rank], rows_per_process[rank], MPI_DOUBLE, Ap, rows_per_process, displacements, MPI_DOUBLE, MPI_COMM_WORLD);
            
            alpha = rr / dot(p, Ap, rows);
            axpby(alpha, p, 1.0, x, rows);
            axpby(-alpha, Ap, 1.0, r, rows);
            rr_new = dot(r, r, rows);

            beta = rr_new / rr;
            rr = rr_new;

            if(std::sqrt(rr / bb) < rel_error) { break; }
            axpby(1.0, r, beta, p, rows);
            MPI_Barrier(MPI_COMM_WORLD);
        }

        delete[] r;
        delete[] p;
        delete[] Ap;

        if(num_iters <= max_iters && rank == 0)  
        {
            printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
        }
        else if (num_iters > max_iters)
        {
            printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
        }
        
    }
    
