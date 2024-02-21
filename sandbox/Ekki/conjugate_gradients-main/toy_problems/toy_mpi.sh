#!/bin/bash

#SBATCH --ntasks=4  # number of MPI processes
#SBATCH --time=0-00:05

module load env/release/2021.3
module load env/release/2021.5
module load env/release/latest
module load env/staging/2021.5

module load foss/2021a
mpic++ -o toy_mpi toy_mpi.cpp

mpirun -np $SLURM_NTASKS ./toy_mpi # or 'srun ./toy_mpi'