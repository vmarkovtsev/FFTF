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

/*! @file cufft.cc
 *  @brief Tests for cuFFT backend.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#define NAME CUFFT
#define BACKEND FFTF_BACKEND_CUFFT
#define INPLACE
#if __SIZEOF_POINTER__ == 4
#define LD_LIBRARY_PATH "/usr/lib/i386-linux-gnu/"
#else
#define LD_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu/"
#endif

#include "tests/template.inc"

TEST(NAME, ComplexZerocopy) {
  ASSERT_FALSE(fatalError);
  fftf_set_backend(BACKEND);
  ASSERT_EQ(BACKEND, fftf_current_backend());
  TestEnv te(FFTF_DIRECTION_FORWARD, FFTF_TYPE_COMPLEX);
  auto handle = fftf_init(FFTF_TYPE_COMPLEX, FFTF_DIRECTION_FORWARD,
                          FFTF_DIMENSION_1D, te.N(),
                          FFTF_OPTION_CUDA_ZEROCOPY,
                          te.input(), te.output());
  fftf_calc(handle);
  fftf_destroy(handle);
  ASSERT_EQ(kVerificationResultOk, te.Verify());
}

TEST(NAME, ComplexZerocopyInplace) {
  ASSERT_FALSE(fatalError);
  fftf_set_backend(BACKEND);
  ASSERT_EQ(BACKEND, fftf_current_backend());
  TestEnv te(FFTF_DIRECTION_FORWARD, FFTF_TYPE_COMPLEX, true);
  auto handle = fftf_init(FFTF_TYPE_COMPLEX, FFTF_DIRECTION_BACKWARD,
                          FFTF_DIMENSION_1D, te.N(),
                          FFTF_OPTION_CUDA_ZEROCOPY,
                          te.input(), te.output());
  fftf_calc(handle);
  fftf_destroy(handle);
  ASSERT_EQ(kVerificationResultOk, te.Verify());
}

TEST(NAME, ComplexForwardBatchZerocopy) {
  ASSERT_FALSE(fatalError);
  fftf_set_backend(BACKEND);
  ASSERT_EQ(BACKEND, fftf_current_backend());
  TestEnv te(FFTF_DIRECTION_FORWARD, FFTF_TYPE_COMPLEX, false, 100);
  auto handle = fftf_init_batch(FFTF_TYPE_COMPLEX, FFTF_DIRECTION_FORWARD,
                                FFTF_DIMENSION_1D, te.N(),
                                FFTF_OPTION_CUDA_ZEROCOPY,
                                100, te.inputs(), te.outputs());
  fftf_calc(handle);
  fftf_destroy(handle);
  ASSERT_EQ(kVerificationResultOk, te.Verify());
}

TEST(NAME, ComplexForwardBatchZerocopyInplace) {
  ASSERT_FALSE(fatalError);
  fftf_set_backend(BACKEND);
  ASSERT_EQ(BACKEND, fftf_current_backend());
  TestEnv te(FFTF_DIRECTION_FORWARD, FFTF_TYPE_COMPLEX, true, 100);
  auto handle = fftf_init_batch(FFTF_TYPE_COMPLEX, FFTF_DIRECTION_FORWARD,
                                FFTF_DIMENSION_1D, te.N(),
                                FFTF_OPTION_CUDA_ZEROCOPY,
                                100, te.inputs(), te.outputs());
  fftf_calc(handle);
  fftf_destroy(handle);
  ASSERT_EQ(kVerificationResultOk, te.Verify());
}
