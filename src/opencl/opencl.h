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

/*! @file opencl.h
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

#ifndef SRC_OPENCL_OPENCL_H_
#define SRC_OPENCL_OPENCL_H_

#include <fftf/api.h>

#if defined(__APPLE__) || defined(__MACOSX)
  #include <CLAMDFFT/cl.h>
#else
  #include <CL/cl.h>
#endif

cl_device_type FFTFOptionsToCLDeviceType(FFTFOptions opt);

cl_int opencl_checked(cl_int res, const char *msg, const char *file,
                      int line);

#define STR_VALUE(arg)      #arg
#define STR(name) STR_VALUE(name)

#define OPENCL_CHECKED(x) opencl_checked(x, STR(x), __FILE__, __LINE__)

#define OPENCL_CHECKED_RET(x, val) do { \
  if (OPENCL_CHECKED(x) != CL_SUCCESS) { \
    return val; \
  } \
} while (0)

#define OPENCL_CHECKED_RET0(x) OPENCL_CHECKED_RET(x, 0)
#define OPENCL_CHECKED_RETNULL(x) OPENCL_CHECKED_RET(x, NULL)
#define OPENCL_CHECKED_RETURN(x) OPENCL_CHECKED_RET(x, )

cl_context initializeOpenCL(cl_device_type deviceType,
                            cl_device_id **devicesPtr, cl_uint *numDevicesPtr);

#endif  // SRC_OPENCL_OPENCL_H_
