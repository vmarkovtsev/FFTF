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

/*! @file cufft.h
 *  @brief New file description.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef SRC_CUDA_CUFFT_H_
#define SRC_CUDA_CUFFT_H_

#include <cuComplex.h>
#include "src/config.h"
#include "src/cuda/cuda.h"
#include "src/backend.h"

// CUFFT API function return values
typedef enum cufftResult_t {
  CUFFT_SUCCESS        = 0x0,
  CUFFT_INVALID_PLAN   = 0x1,
  CUFFT_ALLOC_FAILED   = 0x2,
  CUFFT_INVALID_TYPE   = 0x3,
  CUFFT_INVALID_VALUE  = 0x4,
  CUFFT_INTERNAL_ERROR = 0x5,
  CUFFT_EXEC_FAILED    = 0x6,
  CUFFT_SETUP_FAILED   = 0x7,
  CUFFT_INVALID_SIZE   = 0x8,
  CUFFT_UNALIGNED_DATA = 0x9
} cufftResult;

// CUFFT supports the following transform types
typedef enum cufftType_t {
  CUFFT_R2C = 0x2a,     // Real to Complex (interleaved)
  CUFFT_C2R = 0x2c,     // Complex (interleaved) to Real
  CUFFT_C2C = 0x29,     // Complex to Complex, interleaved
  CUFFT_D2Z = 0x6a,     // Double to Double-Complex
  CUFFT_Z2D = 0x6c,     // Double-Complex to Double
  CUFFT_Z2Z = 0x69      // Double-Complex to Double-Complex
} cufftType;

cufftResult cufft_checked(cufftResult res, const char *msg, const char *file,
                          int line);

#define CUFFT_CHECKED(x) cufft_checked(x, STR(x), __FILE__, __LINE__)

#define CUFFT_CHECKED_RET(x, val) do { \
  if (CUFFT_CHECKED(x) != CUFFT_SUCCESS) { \
    return val; \
  } \
} while (0)

#define CUFFT_CHECKED_RET0(x) CUFFT_CHECKED_RET(x, 0)
#define CUFFT_CHECKED_RETNULL(x) CUFFT_CHECKED_RET(x, NULL)
#define CUFFT_CHECKED_RETURN(x) CUFFT_CHECKED_RET(x, )

cufftType fftf_instance_to_cufft_type(const FFTFInstance *instance);

typedef unsigned int cufftHandle;
typedef float cufftReal;
typedef cuComplex cufftComplex;

typedef cufftResult (*cufftPlanMany_func)(
    cufftHandle *plan,
    int rank,
    int *n,
    int *inembed, int istride, int idist,
    int *onembed, int ostride, int odist,
    cufftType type,
    int batch);

typedef cufftResult (*cufftExecC2C_func)(
    cufftHandle plan,
    cufftComplex *idata,
    cufftComplex *odata,
    int direction);

typedef cufftResult (*cufftExecR2C_func)(
    cufftHandle plan,
    cufftReal *idata,
    cufftComplex *odata);

typedef cufftResult (*cufftExecC2R_func)(
    cufftHandle plan,
    cufftComplex *idata,
    cufftReal *odata);

typedef cufftResult (*cufftDestroy_func)(cufftHandle plan);

#endif  // SRC_CUDA_CUFFT_H_
