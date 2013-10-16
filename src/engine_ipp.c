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

/*! @file engine_ipp.c
 *  @brief Intel Integrated Performance Primitives FFT wrapper functions implementation.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */
#ifndef __arm__

#include "src/engine_ipp.h"
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* /////////////////////////////////////////////////////////////////////////////
//        The following enumerator defines a status of IPP operations
//                     negative value means error
*/
typedef enum {
    /* errors */
    ippStsNotSupportedModeErr    = -9999,/* The requested mode is currently not supported.  */
    ippStsCpuNotSupportedErr     = -9998,/* The target CPU is not supported. */

    ippStsMisalignedOffsetErr    = -227, /* The offset is not aligned on an element */

    ippStsQuadraticNonResidueErr = -226, /* SQRT operation on quadratic non-residue value. */

    ippStsBorderErr              = -225, /* Illegal value for border type.*/

    ippStsDitherTypeErr          = -224, /* Dithering type is not supported. */
    ippStsH264BufferFullErr      = -223, /* Buffer for the output bitstream is full. */
    ippStsWrongAffinitySettingErr= -222, /* An affinity setting does not correspond to the affinity setting that was set by f.ippSetAffinity(). */
    ippStsLoadDynErr             = -221, /* Error when loading the dynamic library. */

    ippStsPointAtInfinity        = -220, /* Point at infinity is detected.  */

    ippStsI18nUnsupportedErr     = -219, /* Internationalization (i18n) is not supported.                                                                 */
    ippStsI18nMsgCatalogOpenErr  = -218, /* Message catalog cannot be opened, for more information use errno on Linux* OS and GetLastError on Windows* OS. */
    ippStsI18nMsgCatalogCloseErr = -217, /* Message catalog cannot be closed, for more information use errno on Linux* OS and GetLastError on Windows* OS. */
    ippStsUnknownStatusCodeErr   = -216, /* Unknown status code. */

    ippStsContextMatchErr       = -17,   /* Context parameter does not match the operation. */
    ippStsFftFlagErr            = -16,   /* Invalid value for the FFT flag parameter. */
    ippStsFftOrderErr           = -15,   /* Invalid value for the FFT order parameter. */
    ippStsStepErr               = -14,   /* Step value is not valid. */
    ippStsScaleRangeErr         = -13,   /* Scale bounds are out of range. */
    ippStsDataTypeErr           = -12,   /* Data type is incorrect or not supported. */
    ippStsOutOfRangeErr         = -11,   /* Argument is out of range, or point is outside the image. */
    ippStsDivByZeroErr          = -10,   /* An attempt to divide by zero. */
    ippStsMemAllocErr           = -9,    /* Memory allocated for the operation is not enough.*/
    ippStsNullPtrErr            = -8,    /* Null pointer error. */
    ippStsRangeErr              = -7,    /* Incorrect values for bounds: the lower bound is greater than the upper bound. */
    ippStsSizeErr               = -6,    /* Incorrect value for data size. */
    ippStsBadArgErr             = -5,    /* Incorrect arg/param of the function.  */
    ippStsNoMemErr              = -4,    /* Not enough memory for the operation. */
    ippStsSAReservedErr3        = -3,    /* Unknown/unspecified error, -3. */
    ippStsErr                   = -2,    /* Unknown/unspecified error, -2. */
    ippStsSAReservedErr1        = -1,    /* Unknown/unspecified error, -1. */

     /* no errors */
    ippStsNoErr                 =   0,   /* No errors. */

     /* warnings  */
    ippStsNoOperation       =   1,       /* No operation has been executed */
    ippStsMisalignedBuf     =   2,       /* Misaligned pointer in operation in which it must be aligned. */
    ippStsSqrtNegArg        =   3,       /* Negative value(s) for the argument in the function Sqrt. */
    ippStsInvZero           =   4,       /* INF result. Zero value was met by InvThresh with zero level. */
    ippStsEvenMedianMaskSize=   5,       /* Even size of the Median Filter mask was replaced by the odd one. */
    ippStsDivByZero         =   6,       /* Zero value(s) for the divisor in the function Div. */
    ippStsLnZeroArg         =   7,       /* Zero value(s) for the argument in the function Ln.     */
    ippStsLnNegArg          =   8,       /* Negative value(s) for the argument in the function Ln. */
    ippStsNanArg            =   9,       /* Argument value is not a number.                  */
    ippStsNotSupportedCpu   =   36,      /* The CPU is not supported. */
    ippStsUnknownCacheSize  =   37,      /* The CPU is supported, but the size of the cache is unknown. */
    ippStsI18nMsgCatalogInvalid=41,      /* Message Catalog is invalid, English message returned.                                                    */
    ippStsI18nGetMessageFail  = 42,      /* Failed to fetch a localized message, English message returned. For more information use errno on Linux* OS and GetLastError on Windows* OS. */
    ippStsWaterfall           = 43,      /* Cannot load required library, waterfall is used. */
    ippStsPrevLibraryUsed     = 44,      /* Cannot load required library, previous dynamic library is used. */
    ippStsLLADisabled         = 45,      /* OpenMP* Low Level Affinity is disabled. */
} IppStatus;

typedef void* (*ippMalloc_func)(int length);
typedef void (*ippFree_func)(void* ptr);

typedef unsigned char   Ipp8u;
typedef float   Ipp32f;

typedef struct {
    Ipp32f  re;
    Ipp32f  im;
} Ipp32fc;

enum {
    IPP_FFT_DIV_FWD_BY_N = 1,
    IPP_FFT_DIV_INV_BY_N = 2,
    IPP_FFT_DIV_BY_SQRTN = 4,
    IPP_FFT_NODIV_BY_ANY = 8
};

typedef enum {
    ippAlgHintNone,
    ippAlgHintFast,
    ippAlgHintAccurate
} IppHintAlgorithm;

typedef struct DFTSpec_C_32fc       IppsDFTSpec_C_32fc;
typedef struct DFTSpec_R_32f        IppsDFTSpec_R_32f;
typedef struct DCTFwdSpec_32f IppsDCTFwdSpec_32f;
typedef struct DCTInvSpec_32f IppsDCTInvSpec_32f;

typedef IppStatus (*ippsDFTInitAlloc_C_32fc_func)(
    IppsDFTSpec_C_32fc** pDFTSpec,
    int length, int flag, IppHintAlgorithm hint);
typedef IppStatus (*ippsDFTInitAlloc_R_32f_func)(
    IppsDFTSpec_R_32f** pDFTSpec,
    int length, int flag, IppHintAlgorithm hint);

typedef IppStatus (*ippsDFTFree_C_32fc_func)(IppsDFTSpec_C_32fc* pDFTSpec);
typedef IppStatus (*ippsDFTFree_R_32f_func)(IppsDFTSpec_R_32f* pDFTSpec);

typedef IppStatus (*ippsDFTGetBufSize_C_32fc_func)(
    const IppsDFTSpec_C_32fc* pDFTSpec, int* pSize);
typedef IppStatus (*ippsDFTGetBufSize_R_32f_func)(
    const IppsDFTSpec_R_32f*  pDFTSpec, int* pSize);

typedef IppStatus (*ippsDFTFwd_CToC_32fc_func)(
    const Ipp32fc* pSrc, Ipp32fc* pDst,
    const IppsDFTSpec_C_32fc* pDFTSpec, Ipp8u* pBuffer);
typedef IppStatus (*ippsDFTInv_CToC_32fc_func)(
    const Ipp32fc* pSrc, Ipp32fc* pDst,
    const IppsDFTSpec_C_32fc* pDFTSpec, Ipp8u* pBuffer);
typedef IppStatus (*ippsDFTFwd_RToCCS_32f_func)(
    const Ipp32f* pSrc, Ipp32f* pDst,
    const IppsDFTSpec_R_32f* pDFTSpec, Ipp8u* pBuffer);
typedef IppStatus (*ippsDFTInv_CCSToR_32f_func)(
    const Ipp32f* pSrc, Ipp32f* pDst,
    const IppsDFTSpec_R_32f* pDFTSpec, Ipp8u* pBuffer);

typedef IppStatus (*ippsDCTFwdInitAlloc_32f_func)(
    IppsDCTFwdSpec_32f** ppDCTSpec,
    int len, IppHintAlgorithm hint);
typedef IppStatus (*ippsDCTInvInitAlloc_32f_func)(
    IppsDCTInvSpec_32f** ppDCTSpec,
    int len, IppHintAlgorithm hint);

typedef IppStatus (*ippsDCTFwdGetBufSize_32f_func)(
    const IppsDCTFwdSpec_32f* pDCTSpec, int* pBufferSize);
typedef IppStatus (*ippsDCTInvGetBufSize_32f_func)(
    const IppsDCTInvSpec_32f* pDCTSpec, int* pBufferSize);

typedef IppStatus (*ippsDCTFwdFree_32f_func)(IppsDCTFwdSpec_32f* pDCTSpec);
typedef IppStatus (*ippsDCTInvFree_32f_func)(IppsDCTInvSpec_32f* pDCTSpec);
typedef IppStatus (*ippsDCTFwd_32f_func)(
    const Ipp32f* pSrc, Ipp32f* pDst,
    const IppsDCTFwdSpec_32f* pDCTSpec, Ipp8u* pBuffer);
typedef IppStatus (*ippsDCTInv_32f_func)(
    const Ipp32f* pSrc, Ipp32f* pDst,
    const IppsDCTInvSpec_32f* pDCTSpec, Ipp8u* pBuffer);

typedef struct {
  void *libHandleOpenMP;
  void *libHandleCore;
  void *libHandleDSP;

  // Below are the functions which are dynamically loaded
  ippMalloc_func ippMalloc;
  ippFree_func ippFree;

  ippsDFTInitAlloc_C_32fc_func ippsDFTInitAlloc_C_32fc;
  ippsDFTInitAlloc_R_32f_func ippsDFTInitAlloc_R_32f;
  ippsDFTGetBufSize_C_32fc_func ippsDFTGetBufSize_C_32fc;
  ippsDFTGetBufSize_R_32f_func ippsDFTGetBufSize_R_32f;
  ippsDFTFree_C_32fc_func ippsDFTFree_C_32fc;
  ippsDFTFree_R_32f_func ippsDFTFree_R_32f;

  ippsDFTFwd_CToC_32fc_func ippsDFTFwd_CToC_32fc;
  ippsDFTInv_CToC_32fc_func ippsDFTInv_CToC_32fc;
  ippsDFTFwd_RToCCS_32f_func ippsDFTFwd_RToCCS_32f;
  ippsDFTInv_CCSToR_32f_func ippsDFTInv_CCSToR_32f;

  ippsDCTFwdInitAlloc_32f_func ippsDCTFwdInitAlloc_32f;
  ippsDCTInvInitAlloc_32f_func ippsDCTInvInitAlloc_32f;
  ippsDCTFwdGetBufSize_32f_func ippsDCTFwdGetBufSize_32f;
  ippsDCTInvGetBufSize_32f_func ippsDCTInvGetBufSize_32f;
  ippsDCTFwdFree_32f_func ippsDCTFwdFree_32f;
  ippsDCTInvFree_32f_func ippsDCTInvFree_32f;

  ippsDCTFwd_32f_func ippsDCTFwd_32f;
  ippsDCTInv_32f_func ippsDCTInv_32f;
} LibIPP;

#define dlsym_ipp_core(lib, symbol) dlsym_checked(&lib, libHandleCore, symbol)
#define dlsym_ipp_dsp(lib, symbol) dlsym_checked(&lib, libHandleDSP, symbol)

typedef struct {
  void *spec;
  Ipp8u *buffer;
} IppInternalData;

int load_ipp(const char *libPath, void **engineInternalData) {
  LibIPP ipp;
  int libPathLength = strlen(libPath);
  char openMP_path[libPathLength + 100];
  char core_path[libPathLength + 20];

  int i;
  for (i = libPathLength - 1; i > 0 && libPath[i] != '/'; i--);
  if (i > 0) {
    char basePath[libPathLength];
    strncpy(basePath, libPath, i);
    basePath[i] = 0;
    snprintf(core_path, sizeof(core_path), "%s/libippcore.so",
             basePath);
    int baseLength = i;
    for (--i; i > 0 && libPath[i] != '/'; i--);
    if (i > 0) {
      char arch[20] = {0};
      strncpy(arch, libPath + i + 1, baseLength - i - 1);
      snprintf(openMP_path, sizeof(openMP_path),
               "%s/../../../compiler/lib/%s/libiomp5.so",
               basePath, arch);
    } else {
      snprintf(openMP_path, sizeof(openMP_path), "%s/libiomp5.so",
               basePath);
    }
  } else {
    snprintf(openMP_path, sizeof(openMP_path), "libiomp5.so");
    snprintf(core_path, sizeof(core_path), "libippcore.so");
  }

  dlopen_checked(&ipp, libHandleOpenMP, openMP_path);
  dlopen_checked(&ipp, libHandleCore, core_path);
  dlopen_checked(&ipp, libHandleDSP, libPath);

  dlsym_ipp_core(ipp, ippMalloc);
  dlsym_ipp_core(ipp, ippFree);

  dlsym_ipp_dsp(ipp, ippsDFTInitAlloc_C_32fc);
  dlsym_ipp_dsp(ipp, ippsDFTInitAlloc_R_32f);
  dlsym_ipp_dsp(ipp, ippsDFTGetBufSize_C_32fc);
  dlsym_ipp_dsp(ipp, ippsDFTGetBufSize_R_32f);
  dlsym_ipp_dsp(ipp, ippsDFTFree_C_32fc);
  dlsym_ipp_dsp(ipp, ippsDFTFree_R_32f);
  dlsym_ipp_dsp(ipp, ippsDFTFwd_CToC_32fc);
  dlsym_ipp_dsp(ipp, ippsDFTInv_CToC_32fc);
  dlsym_ipp_dsp(ipp, ippsDFTFwd_RToCCS_32f);
  dlsym_ipp_dsp(ipp, ippsDFTInv_CCSToR_32f);
  dlsym_ipp_dsp(ipp, ippsDCTFwdInitAlloc_32f);
  dlsym_ipp_dsp(ipp, ippsDCTInvInitAlloc_32f);
  dlsym_ipp_dsp(ipp, ippsDCTFwdGetBufSize_32f);
  dlsym_ipp_dsp(ipp, ippsDCTInvGetBufSize_32f);
  dlsym_ipp_dsp(ipp, ippsDCTFwdFree_32f);
  dlsym_ipp_dsp(ipp, ippsDCTInvFree_32f);
  dlsym_ipp_dsp(ipp, ippsDCTFwd_32f);
  dlsym_ipp_dsp(ipp, ippsDCTInv_32f);

  *engineInternalData = malloc(sizeof(LibIPP));
  memcpy(*engineInternalData, &ipp, sizeof(LibIPP));
  return 1;
}

void unload_ipp(void *engineInternalData) {
  LibIPP *ipp = (LibIPP *)engineInternalData;
  dlclose(ipp->libHandleDSP);
  dlclose(ipp->libHandleCore);
  dlclose(ipp->libHandleOpenMP);
  free(ipp);
}

void init_ipp(void *engineInternalData, FFTFSingleInstance *instance) {
  assert((uintptr_t)(instance->output) % 32 == 0 &&
         "output buffer should be aligned to 32 bytes (use fftf_malloc())");

  LibIPP *ipp = (LibIPP *)engineInternalData;
  IppInternalData *data = malloc(sizeof(IppInternalData));
  instance->internalData = data;
  int bufferSize;
  switch (instance->type) {
    case FFTF_TYPE_COMPLEX:
      assert(
          ipp->ippsDFTInitAlloc_C_32fc(
              (IppsDFTSpec_C_32fc **)&data->spec,
              instance->length, IPP_FFT_NODIV_BY_ANY,
              (IppHintAlgorithm)instance->options)
          == ippStsNoErr);
      ipp->ippsDFTGetBufSize_C_32fc((const IppsDFTSpec_C_32fc *)data->spec,
                                    &bufferSize);
      break;
    case FFTF_TYPE_REAL:
      assert(
          ipp->ippsDFTInitAlloc_R_32f(
              (IppsDFTSpec_R_32f **)&data->spec,
              instance->length, IPP_FFT_NODIV_BY_ANY,
              (IppHintAlgorithm)instance->options)
          == ippStsNoErr);
      ipp->ippsDFTGetBufSize_R_32f((const IppsDFTSpec_R_32f *)data->spec,
                                   &bufferSize);
      break;
    case FFTF_TYPE_DCT:
      if (instance->direction == FFTF_DIRECTION_FORWARD) {
        assert(
          ipp->ippsDCTFwdInitAlloc_32f(
              (IppsDCTFwdSpec_32f **)&data->spec,
              instance->length, (IppHintAlgorithm)instance->options)
          == ippStsNoErr);
        ipp->ippsDCTFwdGetBufSize_32f((const IppsDCTFwdSpec_32f *)data->spec,
                                      &bufferSize);
      } else {
        assert(
          ipp->ippsDCTInvInitAlloc_32f(
              (IppsDCTInvSpec_32f **)&data->spec,
              instance->length, (IppHintAlgorithm)instance->options)
          == ippStsNoErr);
        ipp->ippsDCTInvGetBufSize_32f((const IppsDCTInvSpec_32f *)data->spec,
                                      &bufferSize);
      }
      break;
  }
  data->buffer = ipp->ippMalloc(bufferSize);
}

void calc_ipp(void *engineInternalData,
              const FFTFSingleInstance *instance) {
  LibIPP *ipp = (LibIPP *)engineInternalData;
  IppInternalData *data = (IppInternalData *)instance->internalData;
  switch (instance->type) {
    case FFTF_TYPE_COMPLEX:
      if (instance->direction == FFTF_DIRECTION_FORWARD) {
        ipp->ippsDFTFwd_CToC_32fc(
            (const Ipp32fc *)instance->input,
            (Ipp32fc *)instance->output,
            (const IppsDFTSpec_C_32fc*)data->spec,
            data->buffer);
      } else {
        ipp->ippsDFTInv_CToC_32fc(
            (const Ipp32fc *)instance->input,
            (Ipp32fc *)instance->output,
            (const IppsDFTSpec_C_32fc*)data->spec,
            data->buffer);
      }
      break;
    case FFTF_TYPE_REAL:
      if (instance->direction == FFTF_DIRECTION_FORWARD) {
        ipp->ippsDFTFwd_RToCCS_32f(
            instance->input, instance->output,
            (const IppsDFTSpec_R_32f*)data->spec,
            data->buffer);
      } else {
        ipp->ippsDFTInv_CCSToR_32f(
            instance->input, instance->output,
            (const IppsDFTSpec_R_32f*)data->spec,
            data->buffer);
      }
      break;
    case FFTF_TYPE_DCT:
      if (instance->direction == FFTF_DIRECTION_FORWARD) {
        ipp->ippsDCTFwd_32f(
            instance->input, instance->output,
            (const IppsDCTFwdSpec_32f *)data->spec,
            data->buffer);
        // Cancel normalization
        int N = instance->length;
        for (int i = 0; i < N; i++) {
          instance->output[i] *= 2 * N;
        }
      } else {
        ipp->ippsDCTInv_32f(
            instance->input, instance->output,
            (const IppsDCTInvSpec_32f *)data->spec,
            data->buffer);
      }
      break;
  }
}

void destroy_ipp(void *engineInternalData, FFTFSingleInstance *instance) {
  LibIPP *ipp = (LibIPP *)engineInternalData;
  IppInternalData *data = (IppInternalData *)instance->internalData;
  switch (instance->type) {
    case FFTF_TYPE_COMPLEX:
      ipp->ippsDFTFree_C_32fc((IppsDFTSpec_C_32fc *)data->spec);
      break;
    case FFTF_TYPE_REAL:
      ipp->ippsDFTFree_R_32f((IppsDFTSpec_R_32f *)data->spec);
      break;
    case FFTF_TYPE_DCT:
      if (instance->direction == FFTF_DIRECTION_FORWARD) {
        ipp->ippsDCTFwdFree_32f((IppsDCTFwdSpec_32f *)data->spec);
      } else {
        ipp->ippsDCTInvFree_32f((IppsDCTInvSpec_32f *)data->spec);
      }
      break;
  }
  ipp->ippFree(data->buffer);
  free(data);
}

void *malloc_ipp(void *engineInternalData, size_t size) {
  LibIPP *ipp = (LibIPP *)engineInternalData;
  return ipp->ippMalloc(size);
}

void free_ipp(void *engineInternalData, void *ptr) {
  LibIPP *ipp = (LibIPP *)engineInternalData;
  ipp->ippFree(ptr);
}

#endif  // #ifndef __arm__
