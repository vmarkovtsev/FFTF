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

/*! @file backend.c
 *  @brief FFTF backends engine implementation.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */


#include "src/backend.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "src/engine_kiss.h"
#include "src/engine_ooura.h"
#include "src/engine_libav.h"
#include "src/engine_ipp.h"
#include "src/engine_mkl.h"
#ifdef GPL
#include "src/engine_fftw3.h"
#endif
#ifdef OPENCL
#include "src/engine_appml.h"
#include "src/engine_viennacl.h"
#endif
#ifdef CUDA
#include "src/engine_cufft.h"
#endif

#ifndef strdup
char *strdup(const char *str) {
    int n = strlen(str) + 1;
    char *dup = malloc(n);
    if (dup) {
        strcpy(dup, str);
    }
    return dup;
}
#endif

#define BACKEND_INIT(id, lib) \
  { id, lib, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, \
    NULL, NULL, NULL }

Backend Backends[FFTF_COUNT_BACKENDS] = {
    { FFTF_BACKEND_KISS, NULL, 0,
      NULL, NULL,
      init_kiss, NULL, calc_kiss, NULL, destroy_kiss, NULL,
      malloc_kiss, free_kiss,
      NULL, NULL },

    { FFTF_BACKEND_OOURA, NULL, 1,
      NULL, NULL,
      init_ooura, NULL, calc_ooura, NULL, destroy_ooura, NULL,
      NULL, NULL, NULL, NULL },

#ifdef GPL
    { FFTF_BACKEND_FFTW3, "libfftw3f.so.3", 0,
      load_fftw3, unload_fftw3,
      init_fftw3, NULL, calc_fftw3, NULL, destroy_fftw3, NULL,
      malloc_fftw3, free_fftw3,
      NULL, NULL },
#endif

    { FFTF_BACKEND_LIBAV, "libavcodec.so.53", 1,
      load_libav, unload_libav,
      init_libav, NULL, calc_libav, NULL, destroy_libav, NULL,
      malloc_libav, free_libav,
      NULL, NULL },

    { FFTF_BACKEND_IMKL, "libmkl_rt.so", 1,
      load_mkl, unload_mkl,
      init_mkl, NULL, calc_mkl, NULL, destroy_mkl, NULL,
      malloc_mkl, free_mkl,
      NULL, NULL },

    { FFTF_BACKEND_IIPP, "libipps.so", 1,
      load_ipp, unload_ipp,
      init_ipp, NULL, calc_ipp, NULL, destroy_ipp, NULL,
      malloc_ipp, free_ipp,
      NULL, NULL },
#ifdef CUDA
    { FFTF_BACKEND_CUFFT, "libcufft.so", 0,
      load_cufft, unload_cufft,
      NULL, init_many_cufft, NULL, calc_many_cufft, NULL, destroy_many_cufft,
      NULL, NULL, NULL, NULL },
#endif
#ifdef OPENCL
    { FFTF_BACKEND_APPML, "libclAmdFft.Runtime.so", 0,
      load_appml, unload_appml,
      NULL, init_many_appml, NULL, calc_many_appml, NULL, destroy_many_appml,
      NULL, NULL, NULL, NULL },

    { FFTF_BACKEND_VIENNACL, NULL, 1,
      NULL, NULL,
      NULL, init_many_viennacl, NULL, calc_many_viennacl,
      NULL, destroy_many_viennacl,
      NULL, NULL, NULL, NULL },
#endif
};

const char **BackendAdditionalPaths = NULL;
FFTFBackend BackendAdditionalLibs[FFTF_COUNT_BACKENDS] = {
    { FFTF_BACKEND_KISS,    NULL},
    { FFTF_BACKEND_OOURA,   NULL},
#ifdef GPL
    { FFTF_BACKEND_FFTW3,   NULL},
#endif
    { FFTF_BACKEND_LIBAV,   NULL},
    { FFTF_BACKEND_IMKL,    NULL},
    { FFTF_BACKEND_IIPP,    NULL},
#ifdef CUDA
    { FFTF_BACKEND_CUFFT,   NULL},
#endif
#ifdef OPENCL
    { FFTF_BACKEND_APPML,   NULL},
    { FFTF_BACKEND_VIENNACL,NULL}
#endif
};

int InstancesCount = 0;
pthread_mutex_t InstancesMutex = PTHREAD_MUTEX_INITIALIZER;

static void unload_backend(FFTFBackend *lib) {
  pthread_mutex_lock(&InstancesMutex);
  assert(Backends[lib->id].unload != NULL);
  Backends[lib->id].unload(Backends[lib->id].internalData);
  bzero(&Backends[lib->id].libraryCurrentPath,
        offsetof(Backend, internalData)
          - offsetof(Backend, libraryCurrentPath));
  pthread_mutex_unlock(&InstancesMutex);
}

static int load_backend_internal(FFTFBackendId id, const char *path) {
  assert(Backends[id].load != NULL);
  if (path != NULL && path[0] == '/') {
    struct stat check;
    if (stat(path, &check)) {
      return 0;
    }
  }
  pthread_mutex_lock(&InstancesMutex);
  int res = Backends[id].load(path, &Backends[id].internalData);
  if (res) {
    Backends[id].libraryCurrentPath = path;
  }
  pthread_mutex_unlock(&InstancesMutex);
  return res;
}

FFTF_SET_BACKEND_RESULT load_backend(FFTFBackend *lib) {
  assert(lib != NULL);

  // These are built-in
  if (lib->id == FFTF_BACKEND_KISS || lib->id == FFTF_BACKEND_OOURA) {
    return FFTF_SET_BACKEND_SUCCESS;
  }

  assert(Backends[lib->id].load != NULL);

  // Unload the previous library
  if (Backends[lib->id].libraryCurrentPath != NULL) {
    unload_backend(lib);
  }

  int loaded = 0;
  lib->path = NULL;
  if (BackendAdditionalLibs[lib->id].path != NULL) {
    if (load_backend_internal(lib->id, BackendAdditionalLibs[lib->id].path)) {
      loaded = 1;
    }
  }
  if (!loaded && BackendAdditionalPaths != NULL) {
    for (int i = 0; BackendAdditionalPaths[i] != NULL; i++) {
      char libpath[strlen(BackendAdditionalPaths[i]) +
                   strlen(Backends[lib->id].libraryDefaultPath) + 2];
      snprintf(libpath, sizeof(libpath), "%s/%s", BackendAdditionalPaths[i],
               Backends[lib->id].libraryDefaultPath);
      if (load_backend_internal(lib->id, libpath)) {
        loaded = 1;
        break;
      }
    }
  }
  if (!loaded) {
    if (!load_backend_internal(lib->id, Backends[lib->id].libraryDefaultPath)) {
      if (BackendAdditionalPaths == NULL &&
          BackendAdditionalLibs[lib->id].path == NULL) {
        void *handle = dlopen(Backends[lib->id].libraryDefaultPath, RTLD_NOW);
        if (handle == NULL) {
          return FFTF_SET_BACKEND_NO_LIBS_FOUND;
        }
        dlclose(handle);
      }
      return FFTF_SET_BACKEND_FAILED_TO_LOAD;
    }
  }
  lib->path = Backends[lib->id].libraryCurrentPath;
  return FFTF_SET_BACKEND_SUCCESS;
}

static void copy_paths_and_libs(const char *const *additionalPaths,
                                const FFTFBackend *additionalLibs) {
  // free() old paths
  if (BackendAdditionalPaths != NULL) {
    for (int i = 0; BackendAdditionalPaths[i] != NULL; i++) {
      free((char *)BackendAdditionalPaths[i]);
    }
    free(BackendAdditionalPaths);
  }
  // copy new paths
  if (additionalPaths != NULL) {
    int i;
    for (i = 1; additionalPaths[i] != NULL; i++);
    BackendAdditionalPaths = malloc(sizeof(const char *) * (i + 1));
    for (i = 0; additionalPaths[i] != NULL; i++) {
      BackendAdditionalPaths[i] = strdup(additionalPaths[i]);
    }
    BackendAdditionalPaths[i] = NULL;
  } else {
    BackendAdditionalPaths = NULL;
  }
  // copy new libs
  if (additionalLibs != NULL) {
    int i;
    for (i = 0; additionalLibs[i].id != FFTF_BACKEND_NONE; i++) {
      assert(additionalLibs[i].path != NULL);
      if (BackendAdditionalLibs[additionalLibs[i].id].path != NULL) {
        free((char *)BackendAdditionalLibs[additionalLibs[i].id].path);
      }
      BackendAdditionalLibs[additionalLibs[i].id].path =
          strdup(additionalLibs[i].path);
    }
  }
}

void scan_backends(FFTFBackend *libs,
                   const char *const *additionalPaths,
                   const FFTFBackend *additionalLibs) {
  pthread_mutex_lock(&InstancesMutex);
  // Do not let the backends reloading invalidate any existing FFTF instances
  assert(InstancesCount == 0);
  pthread_mutex_unlock(&InstancesMutex);
  copy_paths_and_libs(additionalPaths, additionalLibs);
  for (int i = FFTF_BACKEND_NONE + 1; i < FFTF_COUNT_BACKENDS; i++) {
    // TODO: implement all backends and remove this check
    if (Backends[i].load == NULL) continue;
    load_backend(&libs[i]);
  }
}

void free_backends(FFTFBackend *libs) {
  for (int i = FFTF_BACKEND_NONE + 1; i < FFTF_COUNT_BACKENDS; i++) {
    // TODO: implement all backends and remove this check
    if (Backends[i].load == NULL) continue;
    if (Backends[i].libraryCurrentPath != NULL) {
      unload_backend(&libs[i]);
    }
  }
}

#define FFTF_SINGLE_INSTANCE(instance, i) { \
  instance->id, \
  instance->internalData[i], \
  instance->type, \
  instance->direction, \
  instance->dimension, \
  instance->options, \
  instance->lengths, \
  instance->lengths[0], \
  instance->inputs[i], \
  instance->outputs[i] }

FFTFInstance *backend_init(FFTFBackendId id, FFTFType type,
                           FFTFDirection direction, FFTFDimension dimension,
                           const int *lengths, FFTFOptions options,
                           int batchSize, const float *const *inputs,
                           float *const *outputs) {
  assert((!Backends[id].only1d || dimension == FFTF_DIMENSION_1D) &&
         "Not implemented");
  pthread_mutex_lock(&InstancesMutex);

  FFTFInstance *instance = malloc(sizeof(FFTFInstance));
  instance->id = id;
  size_t ptr_table_size = sizeof(void *) * batchSize;
  instance->internalData = NULL;
  instance->batchSize = batchSize;
  instance->inputs = malloc(ptr_table_size);
  memcpy((const float **)instance->inputs, inputs, ptr_table_size);
  instance->lengths = lengths;
  instance->direction = direction;
  instance->options = options;
  instance->outputs = malloc(ptr_table_size);
  memcpy((const float **)instance->outputs, outputs, ptr_table_size);
  instance->type = type;
  instance->dimension = dimension;
  assert(pthread_mutex_init(&instance->lock, NULL) == 0);

  if (Backends[id].init == NULL) {
    assert(Backends[id].calc == NULL);
    assert(Backends[id].init_many != NULL);
    assert(Backends[id].calc_many != NULL);
    Backends[id].init_many(Backends[id].internalData, instance);
  } else {
    instance->internalData = malloc(ptr_table_size);
    bzero(instance->internalData, ptr_table_size);
    if (batchSize == 1) {
      FFTFSingleInstance si = FFTF_SINGLE_INSTANCE(instance, 0);
      Backends[id].init(Backends[id].internalData, &si);
      instance->internalData[0] = si.internalData;
    } else {
      for (int i = 0; i < batchSize; i++) {
        FFTFSingleInstance si = FFTF_SINGLE_INSTANCE(instance, i);
        Backends[id].init(Backends[id].internalData, &si);
        instance->internalData[i] = si.internalData;
      }
    }
  }

  InstancesCount++;
  pthread_mutex_unlock(&InstancesMutex);
  return instance;
}

void backend_calc(const FFTFInstance *instance) {
  pthread_mutex_lock((pthread_mutex_t *)&instance->lock);
  assert(instance->id != FFTF_BACKEND_NONE);
  if (Backends[instance->id].calc == NULL) {
    assert(Backends[instance->id].calc_many != NULL);
    Backends[instance->id].calc_many(Backends[instance->id].internalData,
                                     instance);
  } else if (instance->batchSize == 1) {
    FFTFSingleInstance si = FFTF_SINGLE_INSTANCE(instance, 0);
    Backends[instance->id].calc(Backends[instance->id].internalData, &si);
  } else {
    #pragma omp parallel for
    for (int i = 0; i < instance->batchSize; i++) {
      FFTFSingleInstance si = FFTF_SINGLE_INSTANCE(instance, i);
      Backends[instance->id].calc(Backends[instance->id].internalData, &si);
    }
  }
  pthread_mutex_unlock((pthread_mutex_t *)&instance->lock);
}

void backend_destroy(FFTFInstance *instance) {
  pthread_mutex_lock(&InstancesMutex);
  pthread_mutex_lock(&instance->lock);
  assert(instance->id != FFTF_BACKEND_NONE);
  if (Backends[instance->id].destroy != NULL) {
    assert(Backends[instance->id].destroy_many == NULL);
    for (int i = 0; i < instance->batchSize; i++) {
      FFTFSingleInstance si = FFTF_SINGLE_INSTANCE(instance, i);
      Backends[instance->id].destroy(Backends[instance->id].internalData,
                                     &si);
    }
  } else {
    assert(Backends[instance->id].destroy_many != NULL);
    Backends[instance->id].destroy_many(Backends[instance->id].internalData,
                                        instance);
  }
  if (Backends[instance->id].init != NULL) {
    free(instance->internalData);
  }
  free((const float **)instance->inputs);
  free((const float **)instance->outputs);
  instance->id = FFTF_BACKEND_NONE;
  pthread_mutex_unlock(&instance->lock);
  // Give a chance for any pending fftf_calc() to assert.
  sched_yield();
  pthread_mutex_destroy(&instance->lock);
  free(instance);
  InstancesCount--;
  pthread_mutex_unlock(&InstancesMutex);
}

void *backend_malloc(FFTFBackendId id, size_t size) {
  if (Backends[id].malloc != NULL) {
    return Backends[id].malloc(Backends[id].internalData, size);
  }
  return malloc(size);
}

void backend_free(FFTFBackendId id, void *ptr) {
  if (Backends[id].free != NULL) {
    Backends[id].free(Backends[id].internalData, ptr);
  } else {
    free(ptr);
  }
}

void copy_input_to_output(const FFTFSingleInstance *instance) {
  if (instance->output != instance->input) {
    int length = 0;
    for (int i = 0; i < (int)instance->dimension; i++) {
      length += instance->lengths[i];
    }
    size_t size = length * sizeof(float) *
        (instance->type == FFTF_TYPE_COMPLEX? 2 : 1);
    memcpy(instance->output, instance->input, size);
  }
}
