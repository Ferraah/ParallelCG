
#include "Sequential.hpp"

namespace cgcore{
    
    void Sequential::run(const double * A , const double * b, double * x, size_t size, int max_iter, double res_error) const{
        conjugate_gradient(A, b, x, size, max_iter, res_error);
    }

    double Sequential::dot(const double * x, const double * y, size_t size) const
    {
        double result = 0.0;
        for(size_t i = 0; i < size; i++)
        {
            result += x[i] * y[i];
        }
        return result;
    }



    void Sequential::axpby(double alpha, const double * x, double beta, double * y, size_t size) const
    {
        // y = alpha * x + beta * y

        for(size_t i = 0; i < size; i++)
        {
            y[i] = alpha * x[i] + beta * y[i];
        }
    }



    void Sequential::gemv(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols) const
    {
        // y = alpha * A * x + beta * y;

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
    void Sequential::conjugate_gradient(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error) const
    {
        double alpha, beta, bb, rr, rr_new;
        double * r = new double[size];
        double * p = new double[size];
        double * Ap = new double[size];
        int num_iters;

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
            gemv(1.0, A, p, 0.0, Ap, size, size);
            alpha = rr / dot(p, Ap, size);
            axpby(alpha, p, 1.0, x, size);
            axpby(-alpha, Ap, 1.0, r, size);
            rr_new = dot(r, r, size);
            std::cout << r[0] << std::endl;
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
    
}
