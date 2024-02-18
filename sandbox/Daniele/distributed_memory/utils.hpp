
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <cassert>

namespace utils{
    bool read_matrix_from_file(const char *, double **, size_t &, size_t &);
    bool read_matrix_rows(const char *, double **, size_t , size_t , size_t &);
    bool read_matrix_dims(const char * , size_t * , size_t * );
    void print_matrix(const double * , size_t , size_t , FILE * = stdout);
}