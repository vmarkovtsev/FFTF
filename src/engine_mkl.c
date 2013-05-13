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

/*! @file engine_mkl.c
 *  @brief Intel Math Kernel Library FFT wrapper functions implementation.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#include "src/engine_mkl.h"
#include <stdlib.h>
#include <string.h>
#include "src/engine_fftw3.h"

// I wish all backends were as easy to implement as this one!

int load_mkl(const char *libPath, void **engineInternalData) {
  return load_fftw3(libPath,engineInternalData);
}

void unload_mkl(void *engineInternalData) {
  unload_fftw3(engineInternalData);
}

void init_mkl(void *engineInternalData, FFTFSingleInstance *instance) {
  init_fftw3(engineInternalData, instance);
}

void calc_mkl(void *engineInternalData,
              const FFTFSingleInstance *instance) {
  calc_fftw3(engineInternalData, instance);
}

void destroy_mkl(void *engineInternalData, FFTFSingleInstance *instance) {
  destroy_fftw3(engineInternalData, instance);
}

void *malloc_mkl(void *engineInternalData, size_t size) {
  return malloc_fftw3(engineInternalData, size);
}

void free_mkl(void *engineInternalData, void *ptr) {
  free_fftw3(engineInternalData, ptr);
}
