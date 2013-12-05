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

/*! @file engine_fftw3.c
 *  @brief FFTW3 wrapper functions declaration.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#include "src/engine_fftw3.h"
#include <stdlib.h>
#include <string.h>

typedef float fftwf_complex[2];
typedef void *fftwf_plan;

typedef enum {
     FFTW_R2HC=0, FFTW_HC2R=1, FFTW_DHT=2,
     FFTW_REDFT00=3, FFTW_REDFT01=4, FFTW_REDFT10=5, FFTW_REDFT11=6,
     FFTW_RODFT00=7, FFTW_RODFT01=8, FFTW_RODFT10=9, FFTW_RODFT11=10
} fftwf_r2r_kind;

typedef fftwf_plan (*fftwf_plan_dft_func)(int rank, const int *n,
    fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags);
typedef fftwf_plan (*fftwf_plan_dft_r2c_func)(int rank, const int *n,
    float *in, fftwf_complex *out, unsigned flags);
typedef fftwf_plan (*fftwf_plan_dft_c2r_func)(int rank, const int *n,
    fftwf_complex *in, float *out, unsigned flags);
typedef fftwf_plan (*fftwf_plan_r2r_func)(int rank, const int *n,
    float *in, float *out, const fftwf_r2r_kind *kind, unsigned flags);
typedef void (*fftwf_execute_func)(const fftwf_plan p);
typedef void (*fftwf_destroy_plan_func)(fftwf_plan p);
typedef void *(*fftwf_malloc_func)(size_t n);
typedef void (*fftwf_free_func)(void *p);

typedef struct {
  void *libHandle;

  // Below are the functions which are dynamically loaded
  fftwf_plan_dft_func fftwf_plan_dft;
  fftwf_plan_dft_r2c_func fftwf_plan_dft_r2c;
  fftwf_plan_dft_c2r_func fftwf_plan_dft_c2r;
  fftwf_plan_r2r_func fftwf_plan_r2r;
  fftwf_execute_func fftwf_execute;
  fftwf_destroy_plan_func fftwf_destroy_plan;
  fftwf_malloc_func fftwf_malloc;
  fftwf_free_func fftwf_free;
} LibFFTW3;

#define dlopen_fftw3(lib, path, trial) dlopen_checked(&lib, libHandle, path, \
                                                      trial)
#define dlsym_fftw3(lib, symbol) dlsym_checked(&lib, libHandle, symbol)

int load_fftw3(const char *libPath, void **engineInternalData, int trial) {
  LibFFTW3 libfftw3;
  dlopen_fftw3(libfftw3, libPath, trial);

  dlsym_fftw3(libfftw3, fftwf_plan_dft);
  dlsym_fftw3(libfftw3, fftwf_plan_dft_r2c);
  dlsym_fftw3(libfftw3, fftwf_plan_dft_c2r);
  dlsym_fftw3(libfftw3, fftwf_plan_r2r);
  dlsym_fftw3(libfftw3, fftwf_execute);
  dlsym_fftw3(libfftw3, fftwf_destroy_plan);
  dlsym_fftw3(libfftw3, fftwf_malloc);
  dlsym_fftw3(libfftw3, fftwf_free);

  *engineInternalData = malloc(sizeof(LibFFTW3));
  memcpy(*engineInternalData, &libfftw3, sizeof(LibFFTW3));
  return 1;
}

void unload_fftw3(void *engineInternalData) {
  LibFFTW3 *libfftw3 = (LibFFTW3 *)engineInternalData;
  dlclose(libfftw3->libHandle);
  free(libfftw3);
}

#define FFTW_MEASURE (0U)
#define FFTW_DESTROY_INPUT (1U << 0)
#define FFTW_UNALIGNED (1U << 1)
#define FFTW_CONSERVE_MEMORY (1U << 2)
#define FFTW_EXHAUSTIVE (1U << 3) /* NO_EXHAUSTIVE is default */
#define FFTW_PRESERVE_INPUT (1U << 4) /* cancels FFTW_DESTROY_INPUT */
#define FFTW_PATIENT (1U << 5) /* IMPATIENT is default */
#define FFTW_ESTIMATE (1U << 6)
#define FFTW_WISDOM_ONLY (1U << 21)

void init_fftw3(void *engineInternalData, FFTFSingleInstance *instance) {
  LibFFTW3 *libfftw3 = (LibFFTW3 *)engineInternalData;

  unsigned flags = instance->options | FFTW_ESTIMATE;
  if (instance->options & FFTF_OPTION_FFTW_MEASURE) {
    flags |= FFTW_MEASURE;
    flags &= ~FFTW_ESTIMATE;
  }

  switch (instance->type) {
    case FFTF_TYPE_COMPLEX:
      instance->internalData = libfftw3->fftwf_plan_dft(
            instance->dimension, instance->lengths,
            (fftwf_complex *)instance->input,
            (fftwf_complex *)instance->output,
            instance->direction, flags);
      break;
    case FFTF_TYPE_REAL:
      if (instance->direction == FFTF_DIRECTION_FORWARD) {
        instance->internalData = libfftw3->fftwf_plan_dft_r2c(
            instance->dimension, instance->lengths,
            (float *)instance->input,
            (fftwf_complex *)instance->output, flags);
      } else {
        if (!(flags & FFTW_DESTROY_INPUT) &&
            instance->input != instance->output) {
          // Conform to const declaration, unless explicitly requested
          flags |= FFTW_PRESERVE_INPUT;
        }
        instance->internalData = libfftw3->fftwf_plan_dft_c2r(
            instance->dimension, instance->lengths,
            (fftwf_complex *)instance->input,
            instance->output, flags);
      }
      break;
    case FFTF_TYPE_DCT: {
      fftwf_r2r_kind kinds[instance->dimension];
      for (int i = 0; i < (int)instance->dimension; i++) {
        kinds[i] = instance->direction == FFTF_DIRECTION_FORWARD?
            FFTW_REDFT10 : FFTW_REDFT01;
      }
      instance->internalData = libfftw3->fftwf_plan_r2r(
          instance->dimension, instance->lengths,
          (float *)instance->input, instance->output,
          kinds, flags);
      break;
    }
  }
}

void calc_fftw3(void *engineInternalData,
                const FFTFSingleInstance *instance) {
  LibFFTW3 *libfftw3 = (LibFFTW3 *)engineInternalData;
  libfftw3->fftwf_execute((fftwf_plan)instance->internalData);
}

void destroy_fftw3(void *engineInternalData, FFTFSingleInstance *instance) {
  LibFFTW3 *libfftw3 = (LibFFTW3 *)engineInternalData;
  libfftw3->fftwf_destroy_plan((fftwf_plan)instance->internalData);
}

void *malloc_fftw3(void *engineInternalData, size_t size) {
  LibFFTW3 *libfftw3 = (LibFFTW3 *)engineInternalData;
  return libfftw3->fftwf_malloc(size);
}

void free_fftw3(void *engineInternalData, void *ptr) {
  LibFFTW3 *libfftw3 = (LibFFTW3 *)engineInternalData;
  libfftw3->fftwf_free(ptr);
}
