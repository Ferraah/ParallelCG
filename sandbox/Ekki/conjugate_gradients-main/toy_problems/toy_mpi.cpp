// From https://docs.lxp.lu/hpc/compiling/

//
// Execute with: module load foss/2021a
//               mpic++ -o toy_mpi toy_mpi.cpp
//               mpirun -np 4 ./toy_mpi
//

#include <iostream>
#include <mpi.h>

int main(int argc, char **argv) 
{
  MPI_Init(&argc, &argv);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  std::cout << "hello world from process " << my_rank << " of " << world_size << std::endl;

  MPI_Finalize();

  return 0;
}
