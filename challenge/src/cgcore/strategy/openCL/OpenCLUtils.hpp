#ifndef OPENCLUTILS_HPP
#define OPENCLUTILS_HPP


#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cassert>

#include <CL/cl.h>

namespace cgcore
{
   class OpenCLUtils {
        public:
            static void InitializePlatforms(cl_device_id &device, cl_context &context, cl_command_queue &command_queue);
            static cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName);
            static cl_program CreateProgramFromBinary(cl_context context, cl_device_id device, const char* fileName);
            static bool SaveProgramBinary(cl_program program, cl_device_id device, const char* fileName);
            static void Wait( cl_command_queue queue );
            static void InitializeProgram(const char * kernel_source_path, const char * kernel_binary_path, cl_program &program, cl_device_id &device, cl_context &context);
   }; 
} // namespace cgcore

#endif