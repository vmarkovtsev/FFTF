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

/*! @file engine_libav.c
 *  @brief libav FFT wrapper functions implementation.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#include "src/engine_libav.h"
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef float FFTSample;

typedef struct FFTComplex {
    FFTSample re, im;
} FFTComplex;

typedef struct FFTContext FFTContext;
typedef struct RDFTContext RDFTContext;
typedef struct DCTContext DCTContext;

enum RDFTransformType {
    DFT_R2C,
    IDFT_C2R,
    IDFT_R2C,
    DFT_C2R,
};

enum DCTTransformType {
    DCT_II = 0,
    DCT_III,
    DCT_I,
    DST_I,
};

typedef FFTContext *(*av_fft_init_func)(int nbits, int inverse);
typedef void (*av_fft_permute_func)(FFTContext *s, FFTComplex *z);
typedef void (*av_fft_calc_func)(FFTContext *s, FFTComplex *z);
typedef void (*av_fft_end_func)(FFTContext *s);

typedef RDFTContext *(*av_rdft_init_func)(int nbits, enum RDFTransformType trans);
typedef void (*av_rdft_calc_func)(RDFTContext *s, FFTSample *data);
typedef void (*av_rdft_end_func)(RDFTContext *s);

typedef DCTContext *(*av_dct_init_func)(int nbits, enum DCTTransformType type);
typedef void (*av_dct_calc_func)(DCTContext *s, FFTSample *data);
typedef void (*av_dct_end_func)(DCTContext *s);

typedef void *(*av_malloc_func)(size_t size);
typedef void (*av_free_func)(void *ptr);

typedef struct {
  void *libHandle;

  // Below are the functions which are dynamically loaded
  av_fft_init_func av_fft_init;
  av_fft_permute_func av_fft_permute;
  av_fft_calc_func av_fft_calc;
  av_fft_end_func av_fft_end;

  av_rdft_init_func av_rdft_init;
  av_rdft_calc_func av_rdft_calc;
  av_rdft_end_func av_rdft_end;

  av_dct_init_func av_dct_init;
  av_dct_calc_func av_dct_calc;
  av_dct_end_func av_dct_end;

  av_malloc_func av_malloc;
  av_free_func av_free;
} Libav;

#define dlopen_libav(lib, path) dlopen_checked(&lib, libHandle, path)
#define dlsym_libav(lib, symbol) dlsym_checked(&lib, libHandle, symbol)

int load_libav(const char *libPath, void **engineInternalData) {
  Libav libav;
  dlopen_libav(libav, libPath);

  dlsym_libav(libav, av_fft_init);
  dlsym_libav(libav, av_fft_permute);
  dlsym_libav(libav, av_fft_calc);
  dlsym_libav(libav, av_fft_end);

  dlsym_libav(libav, av_rdft_init);
  dlsym_libav(libav, av_rdft_calc);
  dlsym_libav(libav, av_rdft_end);

  dlsym_libav(libav, av_dct_init);
  dlsym_libav(libav, av_dct_calc);
  dlsym_libav(libav, av_dct_end);

  dlsym_libav(libav, av_malloc);
  dlsym_libav(libav, av_free);

  *engineInternalData = malloc(sizeof(Libav));
  memcpy(*engineInternalData, &libav, sizeof(Libav));
  return 1;
}

void unload_libav(void *engineInternalData) {
  Libav *libav = (Libav *)engineInternalData;
  dlclose(libav->libHandle);
  free(libav);
}

void init_libav(void *engineInternalData, FFTFSingleInstance *instance) {
  assert(is_power_of_two(instance->length) &&
         "length must be a power of 2 for libav FFT");
  assert((uintptr_t)(instance->output) % 32 == 0 &&
         "output buffer should be aligned to 32 bytes (use fftf_malloc())");
  int n = log2int(instance->length);

  Libav *libav = (Libav *)engineInternalData;
  switch (instance->type) {
    case FFTF_TYPE_COMPLEX:
      instance->internalData = libav->av_fft_init(
          n, instance->direction == FFTF_DIRECTION_BACKWARD);
      break;
    case FFTF_TYPE_REAL:
      instance->internalData = libav->av_rdft_init(
          n, instance->direction == FFTF_DIRECTION_FORWARD?
             DFT_R2C : IDFT_C2R);
      break;
    case FFTF_TYPE_DCT:
      if (instance->direction == FFTF_DIRECTION_FORWARD) {
        instance->internalData = libav->av_dct_init(
            n, DCT_II);
      } else {
        instance->internalData = libav->av_dct_init(
            n, DCT_III);
      }
      break;
  }
}

void calc_libav(void *engineInternalData,
                const FFTFSingleInstance *instance) {
  Libav *libav = (Libav *)engineInternalData;
  copy_input_to_output(instance);
  switch (instance->type) {
    case FFTF_TYPE_COMPLEX:
      libav->av_fft_permute((FFTContext *)instance->internalData,
                            (FFTComplex *)instance->output);
      libav->av_fft_calc((FFTContext *)instance->internalData,
                         (FFTComplex *)instance->output);
      break;
    case FFTF_TYPE_REAL:
      if (instance->direction == FFTF_DIRECTION_BACKWARD) {
        instance->output[1] = instance->output[instance->length];
      }
      libav->av_rdft_calc((RDFTContext *)instance->internalData,
                          instance->output);
      if (instance->direction == FFTF_DIRECTION_FORWARD) {
        instance->output[instance->length] = instance->output[1];
        instance->output[instance->length + 1] = 0.0f;
        instance->output[1] = 0.0f;
      } else {
        // Multiply each number by 2 so that normalization is compatible with
        // FFTW3-like engines
        for (int i = 0; i < instance->length; i++) {
          instance->output[i] *= 2.0f;
        }
      }
      break;
    case FFTF_TYPE_DCT:
      libav->av_dct_calc((DCTContext *)instance->internalData,
                         instance->output);
      if (instance->direction == FFTF_DIRECTION_BACKWARD) {
        // Multiply each number by (2 * N) so that normalization is compatible with
        // FFTW3-like engines
        for (int i = 0; i < instance->length; i++) {
          instance->output[i] *= 2.0f * instance->length;
        }
      }
      break;
  }
}

void destroy_libav(void *engineInternalData, FFTFSingleInstance *instance) {
  Libav *libav = (Libav *)engineInternalData;
  switch (instance->type) {
    case FFTF_TYPE_COMPLEX:
      libav->av_fft_end((FFTContext *)instance->internalData);
      break;
    case FFTF_TYPE_REAL:
      libav->av_rdft_end((RDFTContext *)instance->internalData);
      break;
    case FFTF_TYPE_DCT:
      libav->av_dct_end((DCTContext *)instance->internalData);
      break;
  }
}

void *malloc_libav(void *engineInternalData, size_t size) {
  Libav *libav = (Libav *)engineInternalData;
  return libav->av_malloc(size);
}

void free_libav(void *engineInternalData, void *ptr) {
  Libav *libav = (Libav *)engineInternalData;
  libav->av_free(ptr);
}
