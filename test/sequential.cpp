#include "../challenge/include/cg/cgcore.hpp"
#include "test_template.hpp"

using namespace cgcore;

int main(int argc, char ** argv){


    const char *m_path =  "../test/assets/matrix_10000.bin";
    const char *rhs_path =  "../test/assets/rhs_10000.bin";

    test_cg<Sequential>(argc, argv, m_path, rhs_path);

    return 0;
}