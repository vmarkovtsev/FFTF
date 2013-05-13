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
 *  @brief APPML service functions.
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

#include "src/opencl/clamdfft.h"
#include <stdio.h>

static const char *clamdfft_status_str(cl_int status) {
  switch (status) {
    case CLFFT_INVALID_GLOBAL_WORK_SIZE:
      return "CLFFT_INVALID_GLOBAL_WORK_SIZE";
    case CLFFT_INVALID_MIP_LEVEL:
      return "CLFFT_INVALID_MIP_LEVEL";
    case CLFFT_INVALID_BUFFER_SIZE:
      return "CLFFT_INVALID_BUFFER_SIZE";
    case CLFFT_INVALID_GL_OBJECT:
      return "CLFFT_INVALID_GL_OBJECT";
    case CLFFT_INVALID_OPERATION:
      return "CLFFT_INVALID_OPERATION";
    case CLFFT_INVALID_EVENT:
      return "CLFFT_INVALID_EVENT";
    case CLFFT_INVALID_EVENT_WAIT_LIST:
      return "CLFFT_INVALID_EVENT_WAIT_LIST";
    case CLFFT_INVALID_GLOBAL_OFFSET:
      return "CLFFT_INVALID_GLOBAL_OFFSET";
    case CLFFT_INVALID_WORK_ITEM_SIZE:
      return "CLFFT_INVALID_WORK_ITEM_SIZE";
    case CLFFT_INVALID_WORK_GROUP_SIZE:
      return "CLFFT_INVALID_WORK_GROUP_SIZE";
    case CLFFT_INVALID_WORK_DIMENSION:
      return "CLFFT_INVALID_WORK_DIMENSION";
    case CLFFT_INVALID_KERNEL_ARGS:
      return "CLFFT_INVALID_KERNEL_ARGS";
    case CLFFT_INVALID_ARG_SIZE:
      return "CLFFT_INVALID_ARG_SIZE";
    case CLFFT_INVALID_ARG_VALUE:
      return "CLFFT_INVALID_ARG_VALUE";
    case CLFFT_INVALID_ARG_INDEX:
      return "CLFFT_INVALID_ARG_INDEX";
    case CLFFT_INVALID_KERNEL:
      return "CLFFT_INVALID_KERNEL";
    case CLFFT_INVALID_KERNEL_DEFINITION:
      return "CLFFT_INVALID_KERNEL_DEFINITION";
    case CLFFT_INVALID_KERNEL_NAME:
      return "CLFFT_INVALID_KERNEL_NAME";
    case CLFFT_INVALID_PROGRAM_EXECUTABLE:
      return "CLFFT_INVALID_PROGRAM_EXECUTABLE";
    case CLFFT_INVALID_PROGRAM:
      return "CLFFT_INVALID_PROGRAM";
    case CLFFT_INVALID_BUILD_OPTIONS:
      return "CLFFT_INVALID_BUILD_OPTIONS";
    case CLFFT_INVALID_BINARY:
      return "CLFFT_INVALID_BINARY";
    case CLFFT_INVALID_SAMPLER:
      return "CLFFT_INVALID_SAMPLER";
    case CLFFT_INVALID_IMAGE_SIZE:
      return "CLFFT_INVALID_IMAGE_SIZE";
    case CLFFT_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return "CLFFT_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CLFFT_INVALID_MEM_OBJECT:
      return "CLFFT_INVALID_MEM_OBJECT";
    case CLFFT_INVALID_HOST_PTR:
      return "CLFFT_INVALID_HOST_PTR";
    case CLFFT_INVALID_COMMAND_QUEUE:
      return "CLFFT_INVALID_COMMAND_QUEUE";
    case CLFFT_INVALID_QUEUE_PROPERTIES:
      return "CLFFT_INVALID_QUEUE_PROPERTIES";
    case CLFFT_INVALID_CONTEXT:
      return "CLFFT_INVALID_CONTEXT";
    case CLFFT_INVALID_DEVICE:
      return "CLFFT_INVALID_DEVICE";
    case CLFFT_INVALID_PLATFORM:
      return "CLFFT_INVALID_PLATFORM";
    case CLFFT_INVALID_DEVICE_TYPE:
      return "CLFFT_INVALID_DEVICE_TYPE";
    case CLFFT_INVALID_VALUE:
      return "CLFFT_INVALID_VALUE";
    case CLFFT_MAP_FAILURE:
      return "CLFFT_MAP_FAILURE";
    case CLFFT_BUILD_PROGRAM_FAILURE:
      return "CLFFT_BUILD_PROGRAM_FAILURE";
    case CLFFT_IMAGE_FORMAT_NOT_SUPPORTED:
      return "CLFFT_IMAGE_FORMAT_NOT_SUPPORTED";
    case CLFFT_IMAGE_FORMAT_MISMATCH:
      return "CLFFT_IMAGE_FORMAT_MISMATCH";
    case CLFFT_MEM_COPY_OVERLAP:
      return "CLFFT_MEM_COPY_OVERLAP";
    case CLFFT_PROFILING_INFO_NOT_AVAILABLE:
      return "CLFFT_PROFILING_INFO_NOT_AVAILABLE";
    case CLFFT_OUT_OF_HOST_MEMORY:
      return "CLFFT_OUT_OF_HOST_MEMORY";
    case CLFFT_OUT_OF_RESOURCES:
      return "CLFFT_OUT_OF_RESOURCES";
    case CLFFT_MEM_OBJECT_ALLOCATION_FAILURE:
      return "CLFFT_MEM_OBJECT_ALLOCATION_FAILURE";
    case CLFFT_COMPILER_NOT_AVAILABLE:
      return "CLFFT_COMPILER_NOT_AVAILABLE";
    case CLFFT_DEVICE_NOT_AVAILABLE:
      return "CLFFT_DEVICE_NOT_AVAILABLE";
    case CLFFT_DEVICE_NOT_FOUND:
      return "CLFFT_DEVICE_NOT_FOUND";
    case CLFFT_SUCCESS:
      return "CLFFT_SUCCESS";
    case CLFFT_NOTIMPLEMENTED:
      return "CLFFT_NOTIMPLEMENTED";
    case CLFFT_FILE_NOT_FOUND:
      return "CLFFT_FILE_NOT_FOUND";
    case CLFFT_FILE_CREATE_FAILURE:
      return "CLFFT_FILE_CREATE_FAILURE";
    case CLFFT_VERSION_MISMATCH:
      return "CLFFT_VERSION_MISMATCH";
    case CLFFT_INVALID_PLAN:
      return "CLFFT_INVALID_PLAN";
    default: {
      static char str[256];
      snprintf(str, sizeof(str), "a not recognized error code (%i)", status);
      return str;
    }
  }
}

cl_int clamdfft_checked(cl_int res, const char *msg, const char *file,
                        int line) {
  switch (res) {
    case CL_SUCCESS:
      break;
    default:
      fprintf(stderr, "%s:%i\t%s returned %s\n", file, line,
              msg, clamdfft_status_str(res));
      break;
  }
  return res;
}

#endif
