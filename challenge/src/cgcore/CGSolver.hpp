#ifndef CG_SOLVER
#define CG_SOLVER

#include <memory>
#include "strategy/CGStrategy.hpp"
#include "timer/Timer.hpp"

namespace cgcore{

    /**
     * @brief Describes a solver for the conjugate gradient equipped with
     * a particular strategy.
    */
    template <typename Strategy>
    class CGSolver{

        public:

            CGSolver(){};

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
            void solve(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error){
                timer.start();
                strategy.run(A, b, x, size, max_iters, rel_error);
                timer.stop();
            }

            
            const Timer& get_timer() const
			{
				return timer;
			};


        private:
            Strategy strategy;
            Timer timer;

    };
}

#endif