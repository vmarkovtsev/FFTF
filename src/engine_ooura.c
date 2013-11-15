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

/*! @file engine_ooura.c
 *  @brief Ooura split-radix FFT wrapper functions implementation.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#include "src/engine_ooura.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "src/ooura/fftsg.h"

typedef struct {
  int *ip;
  float *w;
} Aux;

void init_ooura(void *engineInternalData UNUSED,
                FFTFSingleInstance *instance) {
  assert(is_power_of_two(instance->length) &&
         "length must be a power of 2 for Ooura FFT");
  instance->internalData = malloc(sizeof(Aux));
  Aux *aux = (Aux *)instance->internalData;
  switch (instance->type) {
    case FFTF_TYPE_COMPLEX:
      aux->ip = malloc((2 + sqrtf(instance->length * 2)) * sizeof(int));
      aux->w = malloc(instance->length * sizeof(float));
      break;
    case FFTF_TYPE_REAL:
      aux->ip = malloc((2 + sqrtf(instance->length / 2)) * sizeof(int));
      aux->w = malloc((instance->length / 2) * sizeof(float));
      break;
    case FFTF_TYPE_DCT:
      aux->ip = malloc((2 + sqrtf(instance->length / 2)) * sizeof(int));
      aux->w = malloc((5 * instance->length / 4) * sizeof(float));
      break;
  }
  aux->ip[0] = 0;
}

void calc_ooura(void *engineInternalData UNUSED,
                const FFTFSingleInstance *instance) {
  assert(is_power_of_two(instance->length) &&
         "Ooura FFT requires length to be a power of 2");
  Aux *aux = (Aux *)instance->internalData;
  copy_input_to_output(instance);
  switch (instance->type) {
    case FFTF_TYPE_COMPLEX:
      cdft(instance->length * 2, -instance->direction,
           instance->output, aux->ip, aux->w);
      break;
    case FFTF_TYPE_REAL:
      if (instance->direction == FFTF_DIRECTION_BACKWARD) {
        instance->output[1] = instance->input[instance->length];
      }
      rdft(instance->length, -instance->direction,
           instance->output, aux->ip, aux->w);
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
      if (instance->direction == FFTF_DIRECTION_BACKWARD) {
        instance->output[0] *= 0.5f;
      }
      ddct(instance->length, instance->direction,
           instance->output, aux->ip, aux->w);
      if (instance->direction == FFTF_DIRECTION_BACKWARD) {
        // Multiply each number by 4 so that normalization is compatible with
        // FFTW3-like engines
        for (int i = 0; i < instance->length; i++) {
          instance->output[i] *= 4.0f;
        }
      }
      break;
  }
}

void destroy_ooura(void *engineInternalData UNUSED,
                   FFTFSingleInstance *instance) {
  Aux *aux = (Aux *)instance->internalData;
  free(aux->ip);
  free(aux->w);
  free(aux);
}
