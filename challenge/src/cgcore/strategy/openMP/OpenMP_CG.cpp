
#include "OpenMP_CG.hpp"

namespace cgcore{
    
    void OpenMP_CG::run(const double * A , const double * b, double * x, size_t size, int max_iter, double res_error) const{
        conjugate_gradient(A, b, x, size, max_iter, res_error);
    }

    double OpenMP_CG::dot(const double * x, const double * y, size_t size) const
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



    void OpenMP_CG::axpby(double alpha, const double * x, double beta, double * y, size_t size) const
    {
        #pragma omp parallel for default(none) shared(alpha, x, beta, y, size)
        for(size_t i = 0; i < size; i++)
        {
            y[i] = alpha * x[i] + beta * y[i];
        }
    }



    void OpenMP_CG::gemv(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols) const
    {
        // y = alpha * A * x + beta * y;
        #pragma omp parallel for default(none) shared(alpha, A, x, beta, y, num_rows, num_cols)
        for(size_t r = 0; r < num_rows; r++)
        {
            double y_val = 0.0;
            for(size_t c = 0; c < num_cols; c++)
            {
                y_val += alpha * A[r * num_cols + c] * x[c];
            }
            y[r] = beta * y[r] + y_val;
        }
    }

    /**
     * 
    */ 
    void OpenMP_CG::conjugate_gradient(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error) const
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

        // Determine the number of threads
        // int num_threads = 0;
        // #pragma omp parallel reduction(+:num_threads){
        //     num_threads += 1;
        // }
        // int num_threads = 1;
        // #pragma omp parallel{
        //     num_threads = omp_get_max_num_threads();
        // }

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
        
}
