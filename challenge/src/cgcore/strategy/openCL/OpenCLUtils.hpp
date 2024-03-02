#ifndef OPENCLUTILS_HPP
#define OPENCLUTILS_HPP


#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

#include <CL/cl.h>

namespace cgcore
{
   class OpenCLUtils {
        public:
            static cl_context CreateContext();
            static cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device);
            static cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName);
            static cl_program CreateProgramFromBinary(cl_context context, cl_device_id device, const char* fileName);
            static bool SaveProgramBinary(cl_program program, cl_device_id device, const char* fileName);
            static void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel, cl_mem memObjects[3]);
            static bool CreateMemObjects(cl_context context, cl_mem memObjects[3], float *a, float *b, size_t size);
   }; 
} // namespace cgcore

#endif