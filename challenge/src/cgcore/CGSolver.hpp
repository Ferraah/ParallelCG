#ifndef CG_SOLVER
#define CG_SOLVER

#include <memory>

#include "strategy/CGStrategy.hpp"

namespace cgcore{

    /**
     * @brief Describes a solver for the conjugate gradient equipped with
     * a particular strategy.
    */
    class CGSolver{

        public:

            CGSolver(std::unique_pointer<CGStrategy> _strategy):
                strategy(std::move(_strategy))
            {};
        
            void solve(/* Parameters for CG */){
                strategy.run(/* Same params */);
            }

        private:
            CGStrategy strategy;

    };
}

#endif