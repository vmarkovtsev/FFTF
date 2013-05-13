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

/*! @file engine_viennacl.cc
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

#ifdef OPENCL

#include "src/engine_viennacl.h"
#include <viennacl/fft.hpp>

int test_main()
{
  // Change this type definition to double if your gpu supports that
  typedef float       ScalarType;

  // Create vectors of eight complex values (represented as pairs of floating point values: [real_0, imag_0, real_1, imag_1, etc.])
  viennacl::vector<ScalarType> input_vec(16);
  viennacl::vector<ScalarType> output_vec(16);

  // Fill with values (use viennacl::copy() for larger data!)
  for (std::size_t i=0; i<input_vec.size(); ++i)
  {
    if (i%2 == 0)
      input_vec(i) = ScalarType(i/2);  // even indices represent real part
    else
      input_vec(i) = 0;                // odd indices represent imaginary part
  }

  // Print the vector
  std::cout << "input_vec: " << input_vec << std::endl;

  // Compute FFT and store result in 'output_vec'
  std::cout << "Computing FFT..." << std::endl;
  viennacl::fft(input_vec, output_vec);

  // Compute FFT and store result directly in 'input_vec'
  viennacl::inplace_fft(input_vec);

  // Print result
  std::cout << "input_vec: " << input_vec << std::endl;
  std::cout << "output_vec: " << output_vec << std::endl;

  std::cout << "Computing inverse FFT..." << std::endl;
  viennacl::ifft(input_vec, output_vec); // either store result into output_vec
  viennacl::inplace_ifft(input_vec);     // or compute in-place

  std::cout << "input_vec: " << input_vec << std::endl;
  std::cout << "output_vec: " << output_vec << std::endl;

  //
  //  That's it.
  //
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

extern "C" {

typedef struct {
  viennacl::vector<float> input;
  viennacl::vector<float> output;
} ViennaCLVectors;

void init_many_viennacl(void *engineInternalData UNUSED,
                        FFTFInstance *instance) {
  assert(instance->type != FFTF_TYPE_REAL &&
         "Real transform is not supported by this backend.");
  assert(instance->type != FFTF_TYPE_DCT &&
         "DCT is not supported by this backend.");
  assert(instance->dimension == FFTF_DIMENSION_1D &&
         "2D FFT is not implemented");

  auto vectors = new ViennaCLVectors();
  instance->internalData = reinterpret_cast<void **>(vectors);
  vectors->input.resize(instance->lengths[0] * 2 * instance->batchSize);
  if (instance->inputs != instance->outputs &&
      instance->inputs[0] != instance->outputs[0]) {
    vectors->output.resize(instance->lengths[0] * 2 * instance->batchSize);
  }
}

void calc_many_viennacl(void *engineInternalData UNUSED,
                        const FFTFInstance *instance) {
  auto vectors = reinterpret_cast<ViennaCLVectors *>(instance->internalData);

  for (int i = 0; i < instance->batchSize; i++) {
    viennacl::fast_copy(
        instance->inputs[i],
        instance->inputs[i] + instance->lengths[0] * 2,
        vectors->input.begin() + instance->lengths[0] * 2 * i);
  }

  viennacl::vector<float> *result = &vectors->output;
  if (instance->inputs == instance->outputs ||
      instance->inputs[0] == instance->outputs[0]) {
    result = &vectors->input;
    viennacl::inplace_fft(vectors->input, instance->batchSize,
                          static_cast<float>(instance->direction));
  } else {
    viennacl::fft(vectors->input, vectors->output, instance->batchSize,
                  static_cast<float>(instance->direction));
  }
  for (int i = 0; i < instance->batchSize; i++) {
    viennacl::fast_copy(result->begin() + instance->lengths[0] * 2 * i,
                        result->begin() + instance->lengths[0] * 2 * (i + 1),
                        instance->outputs[i]);
  }
}

void destroy_many_viennacl(void *engineInternalData UNUSED,
                           FFTFInstance *instance) {
  delete reinterpret_cast<ViennaCLVectors *>(instance->internalData);
}

}

#endif

