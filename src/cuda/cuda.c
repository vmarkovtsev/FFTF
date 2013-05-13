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

/*! @file cuda.c
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

#include "src/cuda/cuda.h"
#include <assert.h>
#include <stdio.h>

static const char *cuda_status_str(cudaError_t status) {
  switch (status) {
    case cudaSuccess:
      return "cudaSuccess";
    case cudaErrorMissingConfiguration:
      return "cudaErrorMissingConfiguration";
    case cudaErrorMemoryAllocation:
      return "cudaErrorMemoryAllocation";
    case cudaErrorInitializationError:
      return "cudaErrorInitializationError";
    case cudaErrorLaunchFailure:
      return "cudaErrorLaunchFailure";
    case cudaErrorPriorLaunchFailure:
      return "cudaErrorPriorLaunchFailure";
    case cudaErrorLaunchTimeout:
      return "cudaErrorLaunchTimeout";
    case cudaErrorLaunchOutOfResources:
      return "cudaErrorLaunchOutOfResources";
    case cudaErrorInvalidDeviceFunction:
      return "cudaErrorInvalidDeviceFunction";
    case cudaErrorInvalidConfiguration:
      return "cudaErrorInvalidConfiguration";
    case cudaErrorInvalidDevice:
      return "cudaErrorInvalidDevice";
    case cudaErrorInvalidValue:
      return "cudaErrorInvalidValue";
    case cudaErrorInvalidPitchValue:
      return "cudaErrorInvalidPitchValue";
    case cudaErrorInvalidSymbol:
      return "cudaErrorInvalidSymbol";
    case cudaErrorMapBufferObjectFailed:
      return "cudaErrorMapBufferObjectFailed";
    case cudaErrorUnmapBufferObjectFailed:
      return "cudaErrorUnmapBufferObjectFailed";
    case cudaErrorInvalidHostPointer:
      return "cudaErrorInvalidHostPointer";
    case cudaErrorInvalidDevicePointer:
      return "cudaErrorInvalidDevicePointer";
    case cudaErrorInvalidTexture:
      return "cudaErrorInvalidTexture";
    case cudaErrorInvalidTextureBinding:
      return "cudaErrorInvalidTextureBinding";
    case cudaErrorInvalidChannelDescriptor:
      return "cudaErrorInvalidChannelDescriptor";
    case cudaErrorInvalidMemcpyDirection:
      return "cudaErrorInvalidMemcpyDirection";
    case cudaErrorAddressOfConstant:
      return "cudaErrorAddressOfConstant";
    case cudaErrorTextureFetchFailed:
      return "cudaErrorTextureFetchFailed";
    case cudaErrorTextureNotBound:
      return "cudaErrorTextureNotBound";
    case cudaErrorSynchronizationError:
      return "cudaErrorSynchronizationError";
    case cudaErrorInvalidFilterSetting:
      return "cudaErrorInvalidFilterSetting";
    case cudaErrorInvalidNormSetting:
      return "cudaErrorInvalidNormSetting";
    case cudaErrorMixedDeviceExecution:
      return "cudaErrorMixedDeviceExecution";
    case cudaErrorCudartUnloading:
      return "cudaErrorCudartUnloading";
    case cudaErrorUnknown:
      return "cudaErrorUnknown";
    case cudaErrorNotYetImplemented:
      return "cudaErrorNotYetImplemented";
    case cudaErrorMemoryValueTooLarge:
      return "cudaErrorMemoryValueTooLarge";
    case cudaErrorInvalidResourceHandle:
      return "cudaErrorInvalidResourceHandle";
    case cudaErrorNotReady:
      return "cudaErrorNotReady";
    case cudaErrorInsufficientDriver:
      return "cudaErrorInsufficientDriver";
    case cudaErrorSetOnActiveProcess:
      return "cudaErrorSetOnActiveProcess";
    case cudaErrorInvalidSurface:
      return "cudaErrorInvalidSurface";
    case cudaErrorNoDevice:
      return "cudaErrorNoDevice";
    case cudaErrorECCUncorrectable:
      return "cudaErrorECCUncorrectable";
    case cudaErrorSharedObjectSymbolNotFound:
      return "cudaErrorSharedObjectSymbolNotFound";
    case cudaErrorSharedObjectInitFailed:
      return "cudaErrorSharedObjectInitFailed";
    case cudaErrorUnsupportedLimit:
      return "cudaErrorUnsupportedLimit";
    case cudaErrorDuplicateVariableName:
      return "cudaErrorDuplicateVariableName";
    case cudaErrorDuplicateTextureName:
      return "cudaErrorDuplicateTextureName";
    case cudaErrorDuplicateSurfaceName:
      return "cudaErrorDuplicateSurfaceName";
    case cudaErrorDevicesUnavailable:
      return "cudaErrorDevicesUnavailable";
    case cudaErrorInvalidKernelImage:
      return "cudaErrorInvalidKernelImage";
    case cudaErrorNoKernelImageForDevice:
      return "cudaErrorNoKernelImageForDevice";
    case cudaErrorIncompatibleDriverContext:
      return "cudaErrorIncompatibleDriverContext";
    case cudaErrorPeerAccessAlreadyEnabled:
      return "cudaErrorPeerAccessAlreadyEnabled";
    case cudaErrorPeerAccessNotEnabled:
      return "cudaErrorPeerAccessNotEnabled";
    case cudaErrorDeviceAlreadyInUse:
      return "cudaErrorDeviceAlreadyInUse";
    case cudaErrorProfilerDisabled:
      return "cudaErrorProfilerDisabled";
    case cudaErrorProfilerNotInitialized:
      return "cudaErrorProfilerNotInitialized";
    case cudaErrorProfilerAlreadyStarted:
      return "cudaErrorProfilerAlreadyStarted";
    case cudaErrorProfilerAlreadyStopped:
      return "cudaErrorProfilerAlreadyStopped";
#if __CUDA_API_VERSION >= 0x4000
    case cudaErrorAssert:
      return "cudaErrorAssert";
    case cudaErrorTooManyPeers:
      return "cudaErrorTooManyPeers";
    case cudaErrorHostMemoryAlreadyRegistered:
      return "cudaErrorHostMemoryAlreadyRegistered";
    case cudaErrorHostMemoryNotRegistered:
      return "cudaErrorHostMemoryNotRegistered";
#endif
    case cudaErrorStartupFailure:
      return "cudaErrorStartupFailure";
    case cudaErrorApiFailureBase:
      return "cudaErrorApiFailureBase";
    default: {
      static char str[256];
      snprintf(str, sizeof(str), "a not recognized error code (%i)", status);
      return str;
    }
  }
}

cudaError_t cuda_checked(cudaError_t res, const char *msg, const char *file,
                         int line) {
  switch (res) {
    case cudaSuccess:
      break;
    default:
      fprintf(stderr, "%s:%i\t%s returned %s (%s)\n", file, line,
              msg, cuda_status_str(res), cudaGetErrorString(res));
      break;
  }
  return res;
}

int initializeCUDA(int deviceId) {
  assert(deviceId >= 0);

  int deviceCount;
  CUDA_CHECKED_RET(cudaGetDeviceCount(&deviceCount), -1);
  if (deviceCount == 0) {
    fprintf(stderr, "cudaGetDeviceCount() returned 0: no devices "
                    "supporting CUDA was found.\n");
    return -1;
  }
  if (deviceId >= deviceCount) {
    fprintf(stderr, "cudaGetDeviceCount() returned %i devices, however, "
                    "#%i was requested (index starts with 0).",
            deviceCount, deviceId);
    return -1;
  }

  struct cudaDeviceProp deviceProp;
  CUDA_CHECKED_RET(cudaGetDeviceProperties(&deviceProp, deviceId), -1);
  if (deviceProp.computeMode == cudaComputeModeProhibited) {
    fprintf(stderr, "Device #%i is running in <Compute Mode Prohibited>.\n",
            deviceId);
    return -1;
  }
  if (deviceProp.major < 1) {
    fprintf(stderr, "Device #%i does not support CUDA.\n", deviceId);
    return -1;
  }

  int prevDeviceId;
  CUDA_CHECKED_RET(cudaGetDevice(&prevDeviceId), -1);
  if (prevDeviceId != deviceId) {
    CUDA_CHECKED_RET(cudaSetDevice(deviceId), -1);
  }
  struct cudaDeviceProp props;
  CUDA_CHECKED_RET(cudaGetDeviceProperties(&props, deviceId), -1);
  if (!props.canMapHostMemory) {
    cudaSetDeviceFlags(cudaDeviceMapHost);
  }
  return deviceId;
}
