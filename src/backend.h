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

/*! @file backend.h
 *  @brief FFTF backends engine declaration.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef SRC_BACKEND_H_
#define SRC_BACKEND_H_

#include <dlfcn.h>
#include <fftf/api.h>
#include <pthread.h>
#include <stdio.h>
#include "src/config.h"

#ifdef __cplusplus
extern "C" {
#endif

struct FFTFInstance {
  FFTFBackendId id;
  void **internalData;
  FFTFType type;
  FFTFDirection direction;
  FFTFDimension dimension;
  FFTFOptions options;
  int batchSize;
  const int *lengths;
  const float *const *inputs;
  float *const *outputs;
  pthread_mutex_t lock;
};

typedef struct {
  FFTFBackendId id;
  void *internalData;
  FFTFType type;
  FFTFDirection direction;
  FFTFDimension dimension;
  FFTFOptions options;
  const int *lengths;
  int length;
  const float *input;
  float *output;
} FFTFSingleInstance;

struct Backend;

typedef int (*load_backend_func)(const char *libPath,
                                 void **engineInternalData,
                                 int trial);

typedef void (*unload_backend_func)(void *engineInternalData);

typedef void (*init_func)(void *engineInternalData,
                          FFTFSingleInstance *instance);

typedef void (*init_many_func)(void *engineInternalData,
                               FFTFInstance *instance);

typedef void (*calc_func)(void *engineInternalData,
                          const FFTFSingleInstance *instance);

typedef void (*calc_many_func)(void *engineInternalData,
                               const FFTFInstance *instance);

typedef void (*destroy_func)(void *engineInternalData,
                             FFTFSingleInstance *instance);

typedef void (*destroy_many_func)(void *engineInternalData,
                                  FFTFInstance *instance);

typedef void *(*malloc_func)(void *engineInternalData, size_t size);

typedef void (*free_func)(void *engineInternalData, void *ptr);

/// @brief Backend internal representation.
/// @details Each backend may implement only one of two API sets:
/// { init, calc, destroy } (the usual one) and
/// { init_many, calc_many, destroy_many } (for custom batch processing).
typedef struct Backend {
  const FFTFBackendId id;
  const char *libraryDefaultPath;
  int only1d;

  // API table
  load_backend_func load;
  unload_backend_func unload;
  init_func init;
  init_many_func init_many;
  calc_func calc;
  calc_many_func calc_many;
  destroy_func destroy;
  destroy_many_func destroy_many;
  malloc_func malloc;
  free_func free;

  // The fields below can be modified
  const char *libraryCurrentPath;
  void *internalData;
} Backend;

FFTF_SET_BACKEND_RESULT load_backend(FFTFBackend *lib, int trial);

void scan_backends(FFTFBackend *libs, const char *const *additionalPaths,
                   const FFTFBackend *additionalLibs);

void free_backends(FFTFBackend *libs);

FFTFInstance *backend_init(FFTFBackendId id, FFTFType type,
                           FFTFDirection direction, FFTFDimension dimension,
                           const int *lengths, FFTFOptions options,
                           int batchSize, const float *const *inputs,
                           float *const *outputs);

void backend_calc(const FFTFInstance *instance);

void backend_destroy(FFTFInstance *instance);

void *backend_malloc(FFTFBackendId id, size_t size);

void backend_free(FFTFBackendId id, void *ptr);

///////////////////////////////////////////////////////////////////////////////

#define dlopen_checked(lib, handle, path, trial) do { \
  (lib)->handle = dlopen(path, RTLD_LAZY); \
  if ((lib)->handle == NULL) { \
    if (!trial) { \
      fprintf(stderr, "dlopen(%s) failed: %s.\n", path, dlerror()); \
    } \
    return 0; \
  } \
} while (0)

#define STR_VALUE(arg)      #arg
#define FUNCTION_NAME(name) STR_VALUE(name)

#define dlsym_checked(lib, handle, symbol) do { \
  (lib)->symbol = dlsym((lib)->handle, FUNCTION_NAME(symbol)); \
  if ((lib)->symbol == NULL) { \
    fprintf(stderr, "dlsym(%s) failed: %s.\n", #symbol, dlerror()); \
    dlclose((lib)->handle); \
    return 0; \
  } \
} while (0)

INLINE int is_power_of_two(int x) {
    return (x & (x - 1)) == 0;
}

INLINE int log2int(int n) {
  int res = 0;
  while (n >>= 1) {
    res++;
  }
  return res;
}

void copy_input_to_output(const FFTFSingleInstance *instance);

#ifdef __cplusplus
}
#endif

#endif  // SRC_BACKEND_H_
