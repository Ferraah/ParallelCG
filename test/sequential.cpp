#include "../challenge/include/cg/cgcore.hpp"
#include "test_template.hpp"

using namespace cgcore;

int main(int argc, char ** argv){


    const char *m_path =  "/project/home/p200301/tests/matrix70000.bin";
    const char *rhs_path =  "/project/home/p200301/tests/rhs70000.bin";

    test_cg<Sequential>(argc, argv, m_path, rhs_path);

    return 0;
}