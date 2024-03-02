#include <iostream>
#include "mpi.h"

#define DIM 1<<30

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int size;
    int rank;

    MPI_Comm_size(&size);
    MPI_Comm_rank(&rank);

    double* global_vec;

    if (rank == 0)
    {
        // Building the global vector
        global_vec = double[DIM];
        for (unsigned int i = 0; i < DIM; ++i)
        {
            global_vec[i] = 1.0;
        }
        // Building
    }

    // Computing the size of the vector associated to each process (for the scatter)
    int* sizes = new int[size];
    for (unsigned int i = 0; i < size; ++i)
    {
        sizes[i] = DIM/size;
        if (rank < DIM % size)
            sizes[i]++;
    }

    int personal_size = sizes[rank];

    delete [] global_vec;
    delete [] sizes;

}