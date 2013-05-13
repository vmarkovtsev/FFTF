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

/*! @file api.c
 *  @brief Fast Fourier Transform Frontend API implementation.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */


#include <fftf/api.h>
#include <assert.h>
#include <stdlib.h>
#include "src/backend.h"

#define FFTF_BACKEND_ID_CHECK(id) do { \
  assert(id > FFTF_BACKEND_NONE && id < FFTF_COUNT_BACKENDS); \
} while(0)

int FFTFBackendPriorities[FFTF_COUNT_BACKENDS] = {
     0,   // FFTF_BACKEND_KISS
    10,   // FFTF_BACKEND_OOURA
#ifdef GPL
    20,   // FFTF_BACKEND_FFTW3
#endif
    30,   // FFTF_BACKEND_LIBAV
    40,   // FFTF_BACKEND_IMKL
    41,   // FFTF_BACKEND_IIPP
#ifdef CUDA
    50,  // FFTF_BACKEND_CUFFT
#endif
#ifdef OPENCL
    45,  // FFTF_BACKEND_APPML
    15   // FFTF_BACKEND_VIENNACL
#endif
};

FFTFBackend FFTFBackends[FFTF_COUNT_BACKENDS] = {
    { FFTF_BACKEND_KISS,    NULL},
    { FFTF_BACKEND_OOURA,   NULL},
#ifdef GPL
    { FFTF_BACKEND_FFTW3,   NULL},
#endif
    { FFTF_BACKEND_LIBAV,   NULL},
    { FFTF_BACKEND_IMKL,    NULL},
    { FFTF_BACKEND_IIPP,    NULL},
#ifdef CUDA
    { FFTF_BACKEND_CUFFT,    NULL},
#endif
#ifdef OPENCL
    { FFTF_BACKEND_APPML,   NULL}
#endif
};

FFTFBackendId FFTFCurrentBackendId = FFTF_BACKEND_NONE;

void fftf_cleanup() {
  free_backends(FFTFBackends);
}

const FFTFBackend *fftf_available_backends(const char *const *additionalPaths,
                                           const FFTFBackend *additionalLibs) {
  scan_backends(FFTFBackends, additionalPaths, additionalLibs);
  return FFTFBackends;
}

FFTFBackendId fftf_current_backend(void) {
  if (FFTFCurrentBackendId == FFTF_BACKEND_NONE) {
    int highestPriority = FFTFBackendPriorities[FFTF_BACKEND_NONE + 1];
    for (int i = FFTF_BACKEND_NONE + 1; i < FFTF_COUNT_BACKENDS; i++) {
       if (FFTFBackends[i].path != NULL &&
           FFTFBackendPriorities[i] >= highestPriority) {
         FFTFCurrentBackendId = i;
         highestPriority = FFTFBackendPriorities[i];
       }
    }
  }
  return FFTFCurrentBackendId;
}

FFTF_SET_BACKEND_RESULT fftf_set_backend(FFTFBackendId id) {
  FFTF_BACKEND_ID_CHECK(id);
  if (FFTFBackends[id].path != NULL) {
    FFTFCurrentBackendId = id;
    return FFTF_SET_BACKEND_SUCCESS;
  }
  int result = load_backend(&FFTFBackends[id]);
  if (result == FFTF_SET_BACKEND_SUCCESS) {
    FFTFCurrentBackendId = id;
  }
  return result;
}

int fftf_get_backend_priority(FFTFBackendId id) {
  FFTF_BACKEND_ID_CHECK(id);
  return FFTFBackendPriorities[id];
}

void fftf_set_backend_priority(FFTFBackendId id, int value) {
  FFTF_BACKEND_ID_CHECK(id);
  FFTFBackendPriorities[id] = value;
  if (value >= FFTFBackendPriorities[FFTFCurrentBackendId]) {
    FFTFCurrentBackendId = id;
  }
}

static void fftf_check_backend() {
  if (FFTFCurrentBackendId == FFTF_BACKEND_NONE) {
    fftf_available_backends(NULL, NULL);
  }
  assert(fftf_current_backend() != FFTF_BACKEND_NONE &&
         "No available backends were found");
  assert(fftf_set_backend(FFTFCurrentBackendId) ==
         FFTF_SET_BACKEND_SUCCESS && "Backend is not available now");
}

FFTFInstance *fftf_init(FFTFType type, FFTFDirection direction,
                        FFTFDimension dimension, const int *lengths,
                        FFTFOptions options, const float *input,
                        float *output) {
  assert(input != NULL);
  assert(output != NULL);
  assert(lengths != NULL);
  for (int i = 0; i < (int)dimension; i++) {
    assert(lengths[i] > 1);
  }
  fftf_check_backend();
  const float *inputs[1] = { input };
  float *outputs[1] = { output };
  return backend_init(FFTFCurrentBackendId, type, direction, dimension,
                      lengths, options, 1, inputs, outputs);
}

FFTFInstance *fftf_init_batch(FFTFType type, FFTFDirection direction,
                              FFTFDimension dimension, const int *lengths,
                              FFTFOptions options, int batchSize,
                              const float *const *inputs,
                              float *const *outputs) {
  assert(inputs != NULL);
  assert(outputs != NULL);
  assert(lengths != NULL);
  for (int i = 0; i < (int)dimension; i++) {
    assert(lengths[i] > 1);
  }
  assert(batchSize > 0);
  fftf_check_backend();
  return backend_init(FFTFCurrentBackendId, type, direction, dimension,
                      lengths, options, batchSize, inputs, outputs);
}

static void check_instance(const FFTFInstance *instance) {
  assert(instance != NULL);
  assert(instance->id != FFTF_BACKEND_NONE &&
         "Invalid pointer to FFTFInstance");
}

void fftf_calc(const FFTFInstance *instance) {
  check_instance(instance);
  backend_calc(instance);
}

void fftf_destroy(FFTFInstance *instance) {
  check_instance(instance);
  backend_destroy(instance);
}

void *fftf_malloc(size_t size) {
  fftf_check_backend();
  return backend_malloc(FFTFCurrentBackendId, size);
}

void fftf_free(void *ptr) {
  fftf_check_backend();
  backend_free(FFTFCurrentBackendId, ptr);
}
