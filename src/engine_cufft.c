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

/*! @file engine_cufft.c
 *  @brief Nvidia cuFFT wrapper functions implementation.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifdef CUDA

#include "src/engine_cufft.h"
#include <assert.h>
#include <string.h>
#include "src/cuda/cufft.h"

typedef struct {
  void *libHandle;

  cufftPlanMany_func cufftPlanMany;
  cufftExecC2C_func cufftExecC2C;
  cufftExecR2C_func cufftExecR2C;
  cufftExecC2R_func cufftExecC2R;
  cufftDestroy_func cufftDestroy;
} LibCuFFT;

#define dlopen_cufft(lib, path) dlopen_checked(&lib, libHandle, path)
#define dlsym_cufft(lib, symbol) dlsym_checked(&lib, libHandle, symbol)

int load_cufft(const char *libPath, void **engineInternalData) {
  LibCuFFT libcufft;
  dlopen_cufft(libcufft, libPath);

  dlsym_cufft(libcufft, cufftPlanMany);
  dlsym_cufft(libcufft, cufftExecC2C);
  dlsym_cufft(libcufft, cufftExecR2C);
  dlsym_cufft(libcufft, cufftExecC2R);
  dlsym_cufft(libcufft, cufftDestroy);

  *engineInternalData = malloc(sizeof(LibCuFFT));
  memcpy(*engineInternalData, &libcufft, sizeof(LibCuFFT));
  return 1;
}

void unload_cufft(void *engineInternalData) {
  LibCuFFT *libcufft = (LibCuFFT *)engineInternalData;
  dlclose(libcufft->libHandle);
  free(libcufft);
}

typedef struct {
  cufftHandle handle;
  void *inputs;
  void *outputs;
  int zerocopy;
  int inplace;
  int init_successful;
} cuFftInstance;

static size_t calc_buffer_size(const FFTFInstance *instance, int input) {
  int length = 0;
  for (int i = 0; i < (int)instance->dimension; i++) {
    length += instance->lengths[i];
  }
  size_t res  = length * sizeof(float);
  if (instance->type == FFTF_TYPE_REAL) {
    if ((instance->direction == FFTF_DIRECTION_BACKWARD && input) ||
        (instance->direction == FFTF_DIRECTION_FORWARD && !input)) {
      res += 2 * sizeof(float);
    }
  } else {
    res *= 2;
  }
  return res;
}

static int init_data_buffers(const FFTFInstance *instance) {
  cuFftInstance *data = (cuFftInstance *)instance->internalData;
  int inplace = instance->inputs == (const float **)instance->outputs ||
      *instance->inputs == *(const float **)instance->outputs;
  data->inplace = inplace;
  int zerocopy = (instance->options & FFTF_OPTION_CUDA_ZEROCOPY);
  data->zerocopy = zerocopy;

  size_t inSize;
  if (instance->batchSize == 1) {
    inSize = calc_buffer_size(instance, 1);
  } else {
    inSize = instance->inputs[1] - instance->inputs[0];
    inSize *= sizeof(float);
  }
  size_t outSize;
  if (instance->batchSize == 1) {
    outSize = calc_buffer_size(instance, 0);
  } else {
    outSize = instance->outputs[1] - instance->outputs[0];
    outSize *= sizeof(float);
  }
  if (zerocopy && inplace && inSize < outSize) {
    inSize = outSize;
  }

  if (data->zerocopy) {
    CUDA_CHECKED_RET0(cudaHostRegister(
        (void *)*instance->inputs,
        inSize * instance->batchSize,
        cudaHostRegisterMapped));
    CUDA_CHECKED_RET0(cudaHostGetDevicePointer(
        &data->inputs, (void *)*instance->inputs, 0));
    if (!inplace) {
      CUDA_CHECKED_RET0(cudaHostRegister(
          *instance->outputs,
          outSize * instance->batchSize,
          cudaHostRegisterMapped));
      CUDA_CHECKED_RET0(cudaHostGetDevicePointer(
          &data->outputs, (void *)*instance->outputs, 0));
    }
  } else {
    CUDA_CHECKED_RET0(cudaMalloc(
        &data->inputs, inSize * instance->batchSize));
    if (!inplace) {
      CUDA_CHECKED_RET0(cudaMalloc(
          &data->outputs, outSize * instance->batchSize));
    }
  }

  if (data->inplace) {
    data->outputs = data->inputs;
  }
  return 1;
}

void init_many_cufft(void *engineInternalData, FFTFInstance *instance) {
  assert(instance->type != FFTF_TYPE_DCT &&
         "DCT is not supported by this backend");

  LibCuFFT *libcufft = (LibCuFFT *)engineInternalData;
  cuFftInstance *data = malloc(sizeof(cuFftInstance));
  instance->internalData = (void **)data;
  data->init_successful = 0;
  if (initializeCUDA(instance->options & 0xF) < 0) {
    return;
  }
  if (!init_data_buffers(instance)) {
    return;
  }

  int *lengths = (int *)instance->lengths;
  cufftType cufft_type = fftf_instance_to_cufft_type(instance);
  if (instance->batchSize > 1 && data->zerocopy) {
    int inDist = instance->inputs[1] - instance->inputs[0];
    assert(inDist * sizeof(float) >= calc_buffer_size(instance, 1));
    int outDist = instance->outputs[1] - instance->outputs[0];
    assert(outDist * sizeof(float) >= calc_buffer_size(instance, 0));
    if (instance->type == FFTF_TYPE_COMPLEX) {
      inDist /= 2;
      outDist /= 2;
    }
    if (instance->type == FFTF_TYPE_REAL) {
      if (instance->direction == FFTF_DIRECTION_FORWARD) {
        outDist /= 2;
      } else {
        inDist /= 2;
      }
    }
    CUFFT_CHECKED_RETURN(libcufft->cufftPlanMany(
        &data->handle, instance->dimension, lengths,
        lengths, 1, inDist,
        lengths, 1, outDist,
        cufft_type, instance->batchSize));
  } else {
    CUFFT_CHECKED_RETURN(libcufft->cufftPlanMany(
        &data->handle, instance->dimension, lengths,
        NULL, 0, 0, NULL, 0, 0,
        cufft_type, instance->batchSize));
  }
  data->init_successful = 1;
}

static void data_in(const FFTFInstance *instance) {
  cuFftInstance *data = (cuFftInstance *)instance->internalData;

  if (!data->zerocopy) {
    size_t eachBufferSize = calc_buffer_size(instance, 1);
    for (int i = 0; i < instance->batchSize; i++) {
      CUDA_CHECKED_RETURN(cudaMemcpy(
          (char *)data->inputs + i * eachBufferSize,
          instance->inputs[i], eachBufferSize,
          cudaMemcpyHostToDevice));
    }
  }
}

static void data_out(const FFTFInstance *instance) {
  cuFftInstance *data = (cuFftInstance *)instance->internalData;

  if (!data->zerocopy) {
    size_t eachBufferSize = calc_buffer_size(instance, 0);
    for (int i = 0; i < instance->batchSize; i++) {
      CUDA_CHECKED_RETURN(cudaMemcpy(
          instance->outputs[i],
          (char *)data->outputs + i * eachBufferSize,
          eachBufferSize,
          cudaMemcpyDeviceToHost));
    }
  }
}

void calc_many_cufft(void *engineInternalData, const FFTFInstance *instance) {
  LibCuFFT *libcufft = (LibCuFFT *)engineInternalData;
  cuFftInstance *data = (cuFftInstance *)instance->internalData;
  if (!data->init_successful) return;
  data_in(instance);
  if (instance->type == FFTF_TYPE_COMPLEX) {
    libcufft->cufftExecC2C(data->handle,
                           (cufftComplex *)data->inputs,
                           (cufftComplex *)data->outputs,
                           instance->direction);
  } else {
    if (instance->direction == FFTF_DIRECTION_FORWARD) {
      libcufft->cufftExecR2C(data->handle,
                             (cufftReal *)data->inputs,
                             (cufftComplex *)data->outputs);
    } else {
      libcufft->cufftExecC2R(data->handle,
                             (cufftComplex *)data->inputs,
                             (cufftReal *)data->outputs);
    }
  }
  CUDA_CHECKED_RETURN(cudaThreadSynchronize());
  data_out(instance);
}

void destroy_many_cufft(void *engineInternalData, FFTFInstance *instance) {
  LibCuFFT *libcufft = (LibCuFFT *)engineInternalData;
  cuFftInstance *data = (cuFftInstance *)instance->internalData;
  libcufft->cufftDestroy(data->handle);
  if (data->zerocopy) {
    CUDA_CHECKED_RETURN(cudaHostUnregister((void *)*instance->inputs));
  } else {
    CUDA_CHECKED_RETURN(cudaFree(data->inputs));
  }

  if (!data->inplace) {
    if (data->zerocopy) {
      CUDA_CHECKED_RETURN(cudaHostUnregister(*instance->outputs));
    } else {
      CUDA_CHECKED_RETURN(cudaFree(data->outputs));
    }
  }
  free(data);
}

#endif

