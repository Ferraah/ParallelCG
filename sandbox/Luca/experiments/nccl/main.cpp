#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"
#include "nccl.h"

int main(int argc, char** argv)
{
    // NCCL WITHOUT MPI.
    int device_count;

    cudaGetDeviceCount(&device_count);
    int devices[device_count];

    ncclComm_t comms[device_count];

    for (int i = 0; i < device_count; ++i)
    {
        devices[i] = i;
    }

    // Then, initialize the nccl communicator
    ncclCommInitAll(comms, device_count, devices);

    for (int i = 0; i < device_count; ++i)
    {
        ncclCommDestroy(comms[i]);
    }
}