#ifndef CG_STRATEGY
#define CG_STRATEGY

#include <iostream> 

namespace cgcore{

    class CGStrategy{
        public: 

            /**
             * @brief Run the selected stategy.
             * @param A Pointer to matrix
             * @param b Pointer to rhs
             * @param x Pointer to the vector solution 
             * @param size size of the problem
             * @param max_iters Max iterations to perform 
             * @param rel_error Relative error to achieve 
             * 
            */
            virtual void run(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error) const = 0;
    };

}

#endif