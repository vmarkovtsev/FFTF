/*

  Copyright 2013 Samsung R&D Institute Russia
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met: 

  1. Redistributions of source code must retain the above copyright notice, this
     list of conditions and the following disclaimer. 
  2. Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution. 

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  
 */

/*! @file opencl.c
 *  @brief OpenCL service functions.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifdef OPENCL

#include "src/opencl/opencl.h"
#include <assert.h>
#include <malloc.h>

cl_device_type FFTFOptionsToCLDeviceType(FFTFOptions opt) {
  switch (opt & FFTF_OPTION_OPENCL_DEVICE_TYPE_ALL) {
    default:
    case FFTF_OPTION_OPENCL_DEVICE_TYPE_DEFAULT:
      return CL_DEVICE_TYPE_DEFAULT;
    case FFTF_OPTION_OPENCL_DEVICE_TYPE_CPU:
      return CL_DEVICE_TYPE_CPU;
    case FFTF_OPTION_OPENCL_DEVICE_TYPE_GPU:
      return CL_DEVICE_TYPE_GPU;
    case FFTF_OPTION_OPENCL_DEVICE_TYPE_ACCELERATOR:
      return CL_DEVICE_TYPE_ACCELERATOR;
    case FFTF_OPTION_OPENCL_DEVICE_TYPE_ALL:
      return CL_DEVICE_TYPE_ALL;
  }
}

static const char *opencl_status_str(cl_int status) {
  switch (status) {
    case CL_INVALID_GLOBAL_WORK_SIZE:
      return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_MIP_LEVEL:
      return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_BUFFER_SIZE:
      return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_GL_OBJECT:
      return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_OPERATION:
      return "CL_INVALID_OPERATION";
    case CL_INVALID_EVENT:
      return "CL_INVALID_EVENT";
    case CL_INVALID_EVENT_WAIT_LIST:
      return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_GLOBAL_OFFSET:
      return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_WORK_ITEM_SIZE:
      return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_WORK_GROUP_SIZE:
      return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_DIMENSION:
      return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_KERNEL_ARGS:
      return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_ARG_SIZE:
      return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_ARG_VALUE:
      return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_INDEX:
      return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_KERNEL:
      return "CL_INVALID_KERNEL";
    case CL_INVALID_KERNEL_DEFINITION:
      return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL_NAME:
      return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_PROGRAM:
      return "CL_INVALID_PROGRAM";
    case CL_INVALID_BUILD_OPTIONS:
      return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_BINARY:
      return "CL_INVALID_BINARY";
    case CL_INVALID_SAMPLER:
      return "CL_INVALID_SAMPLER";
    case CL_INVALID_IMAGE_SIZE:
      return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_MEM_OBJECT:
      return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_HOST_PTR:
      return "CL_INVALID_HOST_PTR";
    case CL_INVALID_COMMAND_QUEUE:
      return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_QUEUE_PROPERTIES:
      return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_CONTEXT:
      return "CL_INVALID_CONTEXT";
    case CL_INVALID_DEVICE:
      return "CL_INVALID_DEVICE";
    case CL_INVALID_PLATFORM:
      return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE_TYPE:
      return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_VALUE:
      return "CL_INVALID_VALUE";
    case CL_MAP_FAILURE:
      return "CL_MAP_FAILURE";
    case CL_BUILD_PROGRAM_FAILURE:
      return "CL_BUILD_PROGRAM_FAILURE";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_IMAGE_FORMAT_MISMATCH:
      return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_MEM_COPY_OVERLAP:
      return "CL_MEM_COPY_OVERLAP";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_OUT_OF_HOST_MEMORY:
      return "CL_OUT_OF_HOST_MEMORY";
    case CL_OUT_OF_RESOURCES:
      return "CL_OUT_OF_RESOURCES";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_COMPILER_NOT_AVAILABLE:
      return "CL_COMPILER_NOT_AVAILABLE";
    case CL_DEVICE_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_AVAILABLE";
    case CL_DEVICE_NOT_FOUND:
      return "CL_DEVICE_NOT_FOUND";
    case CL_SUCCESS:
      return "CL_SUCCESS";
    case -1001:
      return "-1001 (OpenCL is not configured or unavailable)";
    default: {
      static char str[256];
      snprintf(str, sizeof(str), "a not recognized error code (%i)", status);
      return str;
    }
  }
}

cl_int opencl_checked(cl_int res, const char *msg, const char *file,
                      int line) {
  switch (res) {
    case CL_SUCCESS:
      break;
    default:
      fprintf(stderr, "%s:%i\t%s returned %s\n", file, line,
              msg, opencl_status_str(res));
      break;
  }
  return res;
}

cl_context initializeOpenCL(cl_device_type deviceType,
                            cl_device_id **devicesPtr,
                            cl_uint *numDevicesPtr) {
  cl_uint numPlatforms;
  OPENCL_CHECKED_RETNULL(clGetPlatformIDs(0, NULL, &numPlatforms));
  if (numPlatforms == 0) {
    fprintf(stderr, "clGetPlatformIDs() returned zero number of platforms\n");
    return NULL;
  }
  cl_platform_id platforms[numPlatforms];
  OPENCL_CHECKED_RETNULL(clGetPlatformIDs(numPlatforms, platforms, NULL));

  // Setting the last one, as it is done in clAmdFft example
  cl_platform_id platform = platforms[numPlatforms - 1];

  cl_uint numEarlyDevices;
  OPENCL_CHECKED_RETNULL(clGetDeviceIDs(platform, deviceType,
                                       0, NULL, &numEarlyDevices));
  if (numEarlyDevices == 0) {
    fprintf(stderr, "clGetearlyDevices() returned zero number of devices\n");
    return NULL;
  }

  cl_device_id earlyDevices[numEarlyDevices];
  OPENCL_CHECKED_RETNULL(clGetDeviceIDs(platform, deviceType,
                                       numEarlyDevices, &earlyDevices[0], NULL));

  cl_context_properties cps[3] = {
      CL_CONTEXT_PLATFORM,
      (cl_context_properties)platform,
      0
  };

  /////////////////////////////////////////////////////////////////
  // Create an OpenCL context
  /////////////////////////////////////////////////////////////////
  cl_int clCreateContext_status;
  cl_context context = clCreateContext(cps, numEarlyDevices, &earlyDevices[0],
                                       NULL, NULL,
                                       &clCreateContext_status);
  OPENCL_CHECKED_RETNULL(clCreateContext_status);

  /////////////////////////////////////////////////////////////////
  // Get OpenCL devices
  /////////////////////////////////////////////////////////////////
  if (numDevicesPtr != NULL) {
    /* First, get the size of device list data */
    size_t sizeDevices;
    OPENCL_CHECKED_RETNULL(clGetContextInfo(context, CL_CONTEXT_DEVICES,
                                           0, NULL, &sizeDevices));
    cl_uint numDevices = sizeDevices / sizeof(cl_device_id);
    *numDevicesPtr = numDevices;

    if (devicesPtr != NULL) {
      cl_device_id *devices = malloc(numDevices * sizeof(cl_device_id));
      /* Now, get the device list data */
      OPENCL_CHECKED_RETNULL(clGetContextInfo(context, CL_CONTEXT_DEVICES,
                                              sizeDevices, devices, NULL));
      *devicesPtr = devices;
    }
  }
  return context;
}

#endif
