
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include "OpenCLUtils.hpp"


namespace cgcore{

    /**
     * Initialize the platfor and the context on which the kernels will be executed. 
    */
    void OpenCLUtils::InitializePlatforms(cl_device_id &device, cl_context &context, cl_command_queue &command_queue){

        cl_int err_num;
        cl_uint numPlatforms;
        cl_platform_id platformIds[2];
        cl_platform_id selected_platform;
        context = NULL;

        // First, select an OpenCL platform to run on.  For this example, we
        // simply choose the first available platform.  Normally, you would
        // query for all available platforms and select the most appropriate one.
        std::cout << "Fetching platforms..";
        err_num = clGetPlatformIDs(2, platformIds, &numPlatforms);

        std::cerr << "OpenCL platforms found" << std::endl;
        if (err_num != CL_SUCCESS || numPlatforms <= 0)
        {
            std::cerr << "Failed to find any OpenCL platforms." << std::endl;
          
        }
        
        std::cout << "Number of available platforms: " << numPlatforms << std::endl; 
        selected_platform = platformIds[1];

        // Next, create an OpenCL context on the platform.  Attempt to
        // create a GPU-based context, and if that fails, try to create
        // a CPU-based context.
        cl_context_properties contextProperties[] =
        {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)selected_platform,
            0
        };

        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                        NULL, NULL, &err_num);
        if (err_num != CL_SUCCESS)
        {
            std::cout << "Could not create GPU context, trying CPU..." << std::endl;
            context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                            NULL, NULL, &err_num);

            err_num = clGetDeviceIDs(selected_platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);

            if (err_num!= CL_SUCCESS)
            {
                std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
                
            }
        }else{
            std::cout << "GPU context created. " << std::endl;
            err_num = clGetDeviceIDs(selected_platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        }

        // Create a command queue 
        command_queue = clCreateCommandQueue(context, device, 0, &err_num);
        if(err_num != CL_SUCCESS){
            std::cerr << "Failed to create command queue.";
        }
    }

    /**
     * Create a program not yet saved on a binary file. 
    */
    cl_program OpenCLUtils::CreateProgram(cl_context context, cl_device_id device, const char* fileName)
    {
        cl_int errNum;
        cl_program program;

        std::ifstream kernelFile(fileName, std::ios::in);
        if (!kernelFile.is_open())
        {
            std::cerr << "Failed to open file for reading: " << fileName << std::endl;
            return NULL;
        }

        std::ostringstream oss;
        oss << kernelFile.rdbuf();

        std::string srcStdStr = oss.str();
        const char *srcStr = srcStdStr.c_str();
        program = clCreateProgramWithSource(context, 1,
                                            (const char**)&srcStr,
                                            NULL, NULL);
        if (program == NULL)
        {
            std::cerr << "Failed to create CL program from source." << std::endl;
            return NULL;
        }

        errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        if (errNum != CL_SUCCESS)
        {
            // Determine the reason for the error
            char buildLog[16384];
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                sizeof(buildLog), buildLog, NULL);

            std::cerr << "Error in kernel: " << std::endl;
            std::cerr << buildLog;
            clReleaseProgram(program);
            return NULL;
        }

        return program;
    }

    /**
     * Create a program, loading the binary from file. 
    */
    cl_program OpenCLUtils::CreateProgramFromBinary(cl_context context, cl_device_id device, const char* fileName)
    {
        FILE *fp = fopen(fileName, "rb");
        if (fp == NULL)
        {
            return NULL;
        }

        // Determine the size of the binary
        size_t binarySize;
        fseek(fp, 0, SEEK_END);
        binarySize = ftell(fp);
        rewind(fp);

        unsigned char *programBinary = new unsigned char[binarySize];
        fread(programBinary, 1, binarySize, fp);
        fclose(fp);

        cl_int errNum = 0;
        cl_program program;
        cl_int binaryStatus;

        program = clCreateProgramWithBinary(context,
                                            1,
                                            &device,
                                            &binarySize,
                                            (const unsigned char**)&programBinary,
                                            &binaryStatus,
                                            &errNum);
        delete [] programBinary;
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Error loading program binary." << std::endl;
            return NULL;
        }

        if (binaryStatus != CL_SUCCESS)
        {
            std::cerr << "Invalid binary for device" << std::endl;
            return NULL;
        }

        errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        if (errNum != CL_SUCCESS)
        {
            // Determine the reason for the error
            char buildLog[16384];
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                sizeof(buildLog), buildLog, NULL);

            std::cerr << "Error in program: " << std::endl;
            std::cerr << buildLog << std::endl;
            clReleaseProgram(program);
            return NULL;
        }

        return program;
    }

    /**
     * Save the created binary of the kernel file for future runs.
    */
    bool OpenCLUtils::SaveProgramBinary(cl_program program, cl_device_id device, const char* fileName)
    {
        cl_uint numDevices = 0;
        cl_int errNum;

        // 1 - Query for number of devices attached to program
        errNum = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint),
                                &numDevices, NULL);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Error querying for number of devices." << std::endl;
            return false;
        }

        // 2 - Get all of the Device IDs
        cl_device_id *devices = new cl_device_id[numDevices];
        errNum = clGetProgramInfo(program, CL_PROGRAM_DEVICES,
                                sizeof(cl_device_id) * numDevices,
                                devices, NULL);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Error querying for devices." << std::endl;
            delete [] devices;
            return false;
        }

        // 3 - Determine the size of each program binary
        size_t *programBinarySizes = new size_t [numDevices];
        errNum = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                                sizeof(size_t) * numDevices,
                                programBinarySizes, NULL);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Error querying for program binary sizes." << std::endl;
            delete [] devices;
            delete [] programBinarySizes;
            return false;
        }

        unsigned char **programBinaries = new unsigned char*[numDevices];
        for (cl_uint i = 0; i < numDevices; i++)
        {
            programBinaries[i] = new unsigned char[programBinarySizes[i]];
        }

        // 4 - Get all of the program binaries
        errNum = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*) * numDevices,
                                programBinaries, NULL);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Error querying for program binaries" << std::endl;

            delete [] devices;
            delete [] programBinarySizes;
            for (cl_uint i = 0; i < numDevices; i++)
            {
                delete [] programBinaries[i];
            }
            delete [] programBinaries;
            return false;
        }

        // 5 - Finally store the binaries for the device requested out to disk for future reading.
        for (cl_uint i = 0; i < numDevices; i++)
        {
            // Store the binary just for the device requested.  In a scenario where
            // multiple devices were being used you would save all of the binaries out here.
            if (devices[i] == device)
            {
                FILE *fp = fopen(fileName, "wb");
                fwrite(programBinaries[i], 1, programBinarySizes[i], fp);
                fclose(fp);
                break;
            }
        }

        // Cleanup
        delete [] devices;
        delete [] programBinarySizes;
        for (cl_uint i = 0; i < numDevices; i++)
        {
            delete [] programBinaries[i];
        }
        delete [] programBinaries;
        return true;
    }


    // wait until all queued tasks have taken place:
    void OpenCLUtils::Wait( cl_command_queue queue )
    {
        cl_event wait;
        cl_int status;

        status = clEnqueueMarker( queue, &wait );

        if( status != CL_SUCCESS )
            fprintf( stderr, "Wait: clEnqueueMarker failed\n" );

        status = clWaitForEvents( 1, &wait ); // blocks until everything is done
        
        if( status != CL_SUCCESS )
            fprintf( stderr, "Wait: clWaitForEvents failed\n" );
    }

    void OpenCLUtils::InitializeProgram(const char * kernel_source_path, const char * kernel_binary_path, cl_program &program, cl_device_id &device, cl_context &context){

        // Create OpenCL program - first attempt to load cached binary.
        //  If that is not available, then create the program from source
        //  and store the binary for future use.
        std::cout << "Attempting to create program from binary..." << std::endl;
        program = CreateProgramFromBinary(context, device, kernel_binary_path) ;
        if (program == NULL)
        {
            std::cout << "Binary not loaded, create from source..." << std::endl;
            program = CreateProgram(context, device, kernel_source_path);

            assert(program != NULL);

            std::cout << "Save program binary for future run..." << std::endl;
            if (SaveProgramBinary(program, device, kernel_binary_path) == false)
            {
                std::cerr << "Failed to write program binary" << std::endl;
            }
        }
        else
        {
            std::cout << "Read program from binary." << std::endl;
        }
    }
}