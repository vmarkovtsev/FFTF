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

/*! @file testenv.cc
 *  @brief FFTF testing environment class.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#include "tests/testenv.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

#define UNINITIALIZED_OUTPUT_VALUE 77.0f

TestEnv::TestEnv(FFTFDirection direction, FFTFType type, bool inplace,
                 int count, int N)
: inputs_(NULL)
, outputs_(NULL)
, type_(type)
, direction_(direction)
, count_(count)
, N_(N)
, N_2_(N / 2) {
  inputs_ = new float*[count_];
  outputs_ = inplace? inputs_ : new float*[count_];
  int alignedN = N + 8;  // sizeof(float) * 8 = 32
  auto allocatorIn = reinterpret_cast<float*>(fftf_malloc(
      sizeof(**inputs_) * alignedN * count));
  float *allocatorOut = NULL;
  if (!inplace) {
    allocatorOut = reinterpret_cast<float*>(fftf_malloc(
        sizeof(**outputs_) * alignedN * count));
  }
  for (int i = 0; i < count_; i++) {
    inputs_[i] = allocatorIn + i * alignedN;
    if (!inplace) {
      outputs_[i] = allocatorOut + i * alignedN;
    }
    InitializeInput(inputs_[i]);
  }

  if (!inplace) {
    for (int idx = 0; idx < count_; idx++) {
      for (int i = 0; i < N; i++) {
        outputs_[idx][i] = UNINITIALIZED_OUTPUT_VALUE;
      }
    }
  }
}

TestEnv::~TestEnv() {
  fftf_free(inputs_[0]);
  delete[] inputs_;
  if (outputs_ != inputs_) {
    fftf_free(outputs_[0]);
    delete[] outputs_;
  }
}

#if 0
  /*                                max
   *       /\      /\      /\
   *      /  \    /  \    /  \
   *     /    \  /    \  /    \
   *    /      \/      \/           min
   *
   *    |------|
   *     2*freq
   */
  const float max = 100.0f;
  const float min = 0.0f;
  int inv = -1;
  for (int i = 0; i < length; complex? i+=2 : i++) {
    if ((i % freq) == 0) {
      inv = -inv;
    }
    array[i] = (inv? min : max) + inv * (max - min) * (i % freq) / (freq - 1);
    if (complex) {
      array[i + 1] = 0.0f;
    }
  }
#endif

void TestEnv::InitializeInput(float* in) {
  if (direction_ == FFTF_DIRECTION_FORWARD) {
    const int freq = N_ / 64;
    bool complex = (type_ == FFTF_TYPE_COMPLEX);
    for (int i = 0; i < N_; complex? i+=2 : i++) {
      in[i] = sinf(i * 2 * M_PI / freq);
      if (complex) {
        in[i + 1] = 0.0f;
      }
    }
  } else {
    for (int i = 0; i < N_; i+=2) {
      in[i] = 0.0f;
      if (i == ((N_ * 7) / 8)) {
        in[i + 1] = 256.0f;
      } else {
        in[i + 1] = 0.0f;
      }
    }
    in[N_] = 0.0f;
    in[N_ + 1] = 0.0f;
  }
}

VerificationResult TestEnv::Verify() {
  if (inputs_ != outputs_) {
    float orig[N_];
    InitializeInput(orig);
    for (int i = 0; i < count_; i++) {
      if (memcmp(orig, inputs_[i], N_ * sizeof(**inputs_))) {
        return kVerificationResultChangedInput;
      }
    }
  }

  for (int idx = 0; idx < count_; idx++) {
    if (direction_ == FFTF_DIRECTION_FORWARD) {
      switch (type_) {
        case FFTF_TYPE_COMPLEX: {
          float max1 = outputs_[idx][0];
          for (int i = 1; i < N_; i++) {
            if (outputs_[idx][i] > max1) {
              max1 = outputs_[idx][i];
            }
          }
          float max2 = outputs_[idx][0]? outputs_[idx][2] : outputs_[idx][0];
          for (int i = 1; i < N_; i++) {
            if (outputs_[idx][i] > max2 && outputs_[idx][i] < max1) {
              max2 = outputs_[idx][i];
            }
          }
          float ratio = max1 / max2;
          if (ratio < 500000) {
            return kVerificationResultWrongOutput;
          }
          break;
        }
        case FFTF_TYPE_REAL:
          if (outputs_[idx][1] != 0.0f || outputs_[idx][N_ + 1] != 0.0f ||
              outputs_[idx][N_] == UNINITIALIZED_OUTPUT_VALUE ||
              outputs_[idx][20] < 0.001f) {
            return kVerificationResultWrongOutput;
          }
          break;
        case FFTF_TYPE_DCT:
          // Hard-code
          if (outputs_[idx][127] < 600 || outputs_[idx][128] > -100 ||
              outputs_[idx][129] > -600) {
            return kVerificationResultWrongOutput;
          }
          break;
      }
    } else {
      switch (type_) {
        case FFTF_TYPE_REAL:
        case FFTF_TYPE_COMPLEX: {
          int zeros = 0;
          bool complex = (type_ == FFTF_TYPE_COMPLEX);
          for (int i = 2; i < N_; complex? i+= 2: i++) {
            if (outputs_[idx][i] * outputs_[idx][i - 2] <= 0) {
              zeros++;
            }
          }
          if (zeros < 510) {
            return kVerificationResultWrongOutput;
          }
          break;
        }
        case FFTF_TYPE_DCT:
          int zeros = 0;
          for (int i = 1; i < N_; i++) {
            if (outputs_[idx][i] * outputs_[idx][i - 1] <= 0) {
              zeros++;
            }
          }
          if (zeros < 1700) {
            return kVerificationResultWrongOutput;
          }
          break;
      }
    }
  }
  return kVerificationResultOk;
}

void TestEnv::swap_input_output() {
  float* tmp = outputs_[0];
  outputs_[0] = inputs_[0];
  inputs_[0] = tmp;
}

const float* TestEnv::input() {
  return inputs_[0];
}

float* TestEnv::output() {
  return outputs_[0];
}

const float** TestEnv::inputs() {
  return const_cast<const float**>(inputs_);
}

float** TestEnv::outputs() {
  return outputs_;
}

const int* TestEnv::N() {
  if (type_ == FFTF_TYPE_COMPLEX) {
    return &N_2_;
  } else {
    return &N_;
  }
}
