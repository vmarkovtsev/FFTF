Fast Fourier Transform Frontend
===============================

This library puts various Fast Fourier Transform implementations (built-in and third party) under a single interface. Supported backends are:
* KissFFT
* Ooura FFT
* libavcodec
* FFTW3
* Intel Integrated Performance Primitives
* Intel Math Kernel Library
* Nvidia cuFFT
* AMD OpenCL
* ViennaCL

Refer to Doxygen documentation for details. FFTF uses [SEAPT](https://github.com/vmarkovtsev/SEAPT) build system, so is compiled on UNIX systems only.

Installation
------------
### Getting the source ###
~~~~{.sh}
git clone https://github.com/vmarkovtsev/FFTF.git
~~~~

### Compiling ###

The following commands will install FFTF to /opt:

~~~~{.sh}
./autogen.sh build --enable-gpl --enable-opencl --enable-cuda --prefix=/opt
cd build && make -j$(getconf _NPROCESSORS_ONLN)
sudo make install
~~~~

Example
-------
~~~~{.c}
#include <assert.h>  // for assert()
#include <fftf/api.h>
#include <stdlib.h>  // for atexit()

// Scan for libraries
FFTFBackendbackends = fftf_available_backends(NULL, NULL);

// Check if FFTW3 was successfully loaded
assert(backends[FFTF_BACKEND_FFTW3].path != NULL);

// Give it maximal priority
fftf_set_backend_priority(FFTF_BACKEND_FFTW3,
                          BACKEND_MAX_INITIAL_PRIORITY + 1);

// Note: the alternative way to use that backend is just
assert(fftf_set_backend(FFTF_BACKEND_FFTW3) == FFTF_SET_BACKEND_SUCCESS);
// without even scanning for available backends

// Schedule the FFTF resources cleanup
atexit(fftf_cleanup);

// Transform N complex numbers
const int N = 2048;
floatin = fftf_malloc(N2sizeof(float));
fill_with_numbers(in, N);

// Prepare the space for the result
floatout = fftf_malloc(N2sizeof(float));

// Obtain the calculation plan (instance)
FFTFInstanceinstance = fftf_init(FFTF_TYPE_COMPLEX, FFTF_DIRECTION_FORWARD,
                                 FFTF_DIMENSION_1D, N, FFTF_NO_OPTIONS,
                                 in, out);

// Calculate the forward 1-dimensional complex FFT
fftf_calc(instance);

// Free the plan
fftf_destroy(instance);

// Free in and out arrays
fftf_free(in);
fftf_free(out);
~~~~

FFTF is released under the Simplified BSD License. If FFTW3 backend is configured, FFTF automatically becomes GPL.
Copyright 2013 Samsung R&D Institute Russia.
