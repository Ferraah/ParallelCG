
#include "MPI_CG.hpp"

namespace cgcore{
    
    void MPI_CG::run(const double * A , const double * b, double * x, size_t size, int max_iter, double res_error) const{
        conjugate_gradient(A, b, x, size, max_iter, res_error);
    }

    double MPI_CG::dot(const double * x, const double * y, size_t size) const
    {
        double result = 0.0;
        for(size_t i = 0; i < size; i++)
        {
            result += x[i] * y[i];
        }
        return result;

    }



    void MPI_CG::axpby(double alpha, const double * x, double beta, double * y, size_t size) const
    {
        // y = alpha * x + beta * y

        for(size_t i = 0; i < size; i++)
        {
            y[i] = alpha * x[i] + beta * y[i];
        }
    }
   



    void MPI_CG::gemv(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols, int num_processes, int my_rank, int * displacements, int * counts) const
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
        for(size_t r = my_start ; r < my_end; r++)
        {
            double y_val = 0.0;
            for(size_t c = 0; c < num_cols; c++)
            {
                y_val += alpha * A[r * num_cols + c] * x[c];
            }
            my_y[r - my_start] = beta * y[r] + y_val;
        }
        // Stack all local y-vectors (my_y)
        MPI_Allgatherv(my_y, my_num_rows, MPI_DOUBLE, y, counts, displacements, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    /**
     * 
    */ 
    void MPI_CG::conjugate_gradient(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error) const
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
        
        if(rank == 0){
            if(num_iters <= max_iters)
            {
                printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
            }
            else
            {
                printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
            }  
        }
    }
        
}
