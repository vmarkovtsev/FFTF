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

/*! @file api.h
 *  @brief Fast Fourier Transform Frontend API declaration.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

/*! @mainpage Fast Fourier Transform Frontend
 *
 * Installation
 * ------------
 * ### Getting the source ###
 * ~~~~{.sh}
 * git clone https://github.com/vmarkovtsev/FFTF.git
 * ~~~~
 *
 * ### Compiling ###
 *
 * The following commands will install FFTF to /opt:
 *
 * ~~~~{.sh}
 * ./autogen.sh build --enable-gpl --enable-opencl --enable-cuda --prefix=/opt
 * cd build && make -j$(getconf _NPROCESSORS_ONLN)
 * sudo make install
 * ~~~~
 *
 * Example
 * -------
 * ~~~~{.c}
 * #include <assert.h>  // for assert()
 * #include <fftf/api.h>
 * #include <stdlib.h>  // for atexit()
 *
 * // Scan for libraries
 * FFTFBackend *backends = fftf_available_backends(NULL, NULL);
 *
 * // Check if FFTW3 was successfully loaded
 * assert(backends[FFTF_BACKEND_FFTW3].path != NULL);
 *
 * // Give it maximal priority
 * fftf_set_backend_priority(FFTF_BACKEND_FFTW3,
 *                           BACKEND_MAX_INITIAL_PRIORITY + 1);
 *
 * // Note: the alternative way to use that backend is just
 * assert(fftf_set_backend(FFTF_BACKEND_FFTW3) == FFTF_SET_BACKEND_SUCCESS);
 * // without even scanning for available backends
 *
 * // Schedule the FFTF resources cleanup
 * atexit(fftf_cleanup);
 *
 * // Transform N complex numbers
 * const int N = 2048;
 * float *in = fftf_malloc(N * 2 * sizeof(float));
 * fill_with_numbers(in, N);
 *
 * // Prepare the space for the result
 * float *out = fftf_malloc(N * 2 * sizeof(float));
 *
 * // Obtain the calculation plan (instance)
 * FFTFInstance *instance = fftf_init(FFTF_TYPE_COMPLEX, FFTF_DIRECTION_FORWARD,
 *                                    FFTF_DIMENSION_1D, N, FFTF_NO_OPTIONS,
 *                                    in, out);
 *
 * // Calculate the forward 1-dimensional complex FFT
 * fftf_calc(instance);
 *
 * // Free the plan
 * fftf_destroy(instance);
 *
 * // Free in and out arrays
 * fftf_free(in);
 * fftf_free(out);
 * ~~~~
 *
 * Detailed documentation is in api.h.
 */

#ifndef INC_FFTF_API_H_
#define INC_FFTF_API_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

#ifdef __GNUC__

/* malloc() function attribute */
#define MALLOC __attribute__ ((__malloc__))

/* Mark pointer parameters which must not be NULL */
#ifndef NOTNULL
#define NOTNULL(...) __attribute__ ((__nonnull__ (__VA_ARGS__)))
#endif

/* warn about unused result function attribute */
#define WARN_UNUSED_RESULT __attribute__ ((__warn_unused_result__))

#endif

#if __GNUC__ >= 4
#pragma GCC visibility push(default)
#endif

typedef enum {
  /// @brief No backend.
  FFTF_BACKEND_NONE = -1,  // This should always be the first
  /// @brief KissFFT (http://sourceforge.net/projects/kissfft) backend.
  /// It is built-in.
  FFTF_BACKEND_KISS,
  /// @brief Ooura FFT (http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html)
  /// backend. It is built-in.
  FFTF_BACKEND_OOURA,
#ifdef GPL
  /// @brief FFTW3 (http://www.fftw.org/) backend.
  FFTF_BACKEND_FFTW3,
#endif
  /// @brief Libavcodec FFT (http://libav.org) backend.
  FFTF_BACKEND_LIBAV,
#ifndef __arm__
  /// @brief Intel Math Kernel Libraries
  /// (http://software.intel.com/en-us/intel-mkl) backend.
  FFTF_BACKEND_IMKL,
  /// @brief Intel Integrated Performance Primitives
  /// (http://software.intel.com/en-us/intel-ipp) backend.
  FFTF_BACKEND_IIPP,
#endif
#ifdef CUDA
  /// @brief Nvidia cuFFT (https://developer.nvidia.com/cufft) backend.
  FFTF_BACKEND_CUFFT,
#endif
#ifdef OPENCL
  /// @brief AMD OpenCL FFT
  /// (http://developer.amd.com/tools/heterogeneous-computing/amd-accelerated-parallel-processing-math-libraries/)
  /// backend.
  FFTF_BACKEND_APPML,

  /// @brief ViennaCL FFT (http://viennacl.sourceforge.net/) backend.
  /// It is built-in.
  FFTF_BACKEND_VIENNACL,
#endif
  /// @brief The overall number of supported backends.
  FFTF_COUNT_BACKENDS     // This should always be the last
} FFTFBackendId;

/// @brief FFTF backend state object.
typedef struct {
  /// @brief The backend's identifier.
  FFTFBackendId id;
  /// @brief The path to the used library which backend
  /// delegates the real work to.
  const char *path;
} FFTFBackend;

/// @brief Searches for backends which are usable in the system.
/// Any backends which are already loaded are reloaded.
/// @param additionalPaths NULL-terminated array of additional paths
/// to search for libraries in. For example, { "/usr/local/lib", NULL }.
/// It can be NULL.
/// @param additionalLibs FFTF_BACKEND_NONE-terminated array of additional
/// libraries to try to load. For example,
/// { { FFTF_BACKEND_LIBAV, "/usr/local/lib/libavcodec.so" },
/// { FFTF_BACKEND_NONE, NULL} }. It can be NULL.
/// @note fftf_available_backends() would invalidate any existing FFTF
/// instances, so it explicitly throws an assertion if any not built-in
/// FFTF instances exist.
/// @return The statically allocated list of all supported FFTF backends
/// (FFTF_BACKEND_NONE < id < FFTF_COUNT_BACKENDS). In each record, path
/// is set to the currently active library path, or NULL if backend
/// could not be loaded.
const FFTFBackend *fftf_available_backends(
    const char *const *__restrict additionalPaths,
    const FFTFBackend *__restrict additionalLibs);

/// @brief Returns the currently active backend.
/// If no backend is currently active, it will be selected according to
/// its priority from loaded backends.
/// @return The active backend's identifier.
/// @note Initially, only built-in backends are loaded. To scan for other
/// backends, call fftf_available_backends().
FFTFBackendId fftf_current_backend(void);

/// @brief Indicates the status of backend loading procedure, returned by
/// fftf_set_backend().
typedef enum {
  /// @brief Backend was set successfully.
  FFTF_SET_BACKEND_SUCCESS = 0,
  /// @brief No backend libraries were found.
  FFTF_SET_BACKEND_NO_LIBS_FOUND,
  /// @brief An error occured during loading of the backend.
  FFTF_SET_BACKEND_FAILED_TO_LOAD
} FFTF_SET_BACKEND_RESULT;

/// @brief Tries to set the current backend. All previously initialized FFTF
/// instances remain bound to their own backends.
/// @return The result of the attempt.
FFTF_SET_BACKEND_RESULT fftf_set_backend(FFTFBackendId id);

/// @brief This is the maximal initial value of priority of any of
/// the supported backends.
#define BACKEND_MAX_INITIAL_PRIORITY 100

/// @brief Get the backend's priority value.
int fftf_get_backend_priority(FFTFBackendId id);

/// @brief Sets the backend's priority value.
void fftf_set_backend_priority(FFTFBackendId id, int value);

/// @brief Frees any resources allocated by FFTF runtime. Particularly,
/// unloads any loaded backends. This function is expected to be invoked
/// with atexit().
void fftf_cleanup();

/// @brief FFTF instance (that is, a calculation plan) forward declaration.
typedef struct FFTFInstance FFTFInstance;

/// @brief FFTF transform direction descriptor.
typedef enum {
  /// @brief Calculate the direct transform.
  FFTF_DIRECTION_FORWARD = -1,
  /// @brief Calculate the inverse transform.
  FFTF_DIRECTION_BACKWARD = 1
} FFTFDirection;

/// @brief FFTF transform kind.
typedef enum {
  /// @brief FFT from complex numbers space to complex numbers space.
  /// @note The layout of complex numbers in FFTF arrays is conventional
  /// ([real part][imaginary part]) and is compatible with stored structs
  /// like typedef struct { float reall float imag; } Complex;
  FFTF_TYPE_COMPLEX,
  /// @brief FFT from real numbers space to complex numbers space
  /// (and vice versa).
  FFTF_TYPE_REAL,
  /// @brief Discrete Cosine Transform (real to real).
  FFTF_TYPE_DCT
} FFTFType;

/// @brief The dimension of FFTF transform.
/// @note Some backends do not support higher dimensions (2D and 3D).
typedef enum {
  /// @brief One dimension (e.g., audio data).
  FFTF_DIMENSION_1D = 1,
  /// @brief Two dimensions (e.g., picture data).
  FFTF_DIMENSION_2D,
  /// @brief Three dimensions.
  FFTF_DIMENSION_3D
} FFTFDimension;

/// @brief FFTF backend additional options.
typedef int FFTFOptions;

/// @brief Empty backend options (defaults).
#define FFTF_NO_OPTIONS 0

#ifdef OPENCL
typedef enum {
  FFTF_OPTION_OPENCL_DEVICE_TYPE_DEFAULT = 0,
  FFTF_OPTION_OPENCL_DEVICE_TYPE_CPU = 1,
  FFTF_OPTION_OPENCL_DEVICE_TYPE_GPU = 2,
  FFTF_OPTION_OPENCL_DEVICE_TYPE_ACCELERATOR = 3,
  FFTF_OPTION_OPENCL_DEVICE_TYPE_ALL = 0xF
} FFTFOptionsOpenCLDeviceType;

#define FFTF_OPTION_OPENCL_ZEROCOPY 0x100
#endif

#ifdef CUDA
#define FFTF_OPTION_CUDA_ZEROCOPY 0x100
#endif

// Note the different order
#define FFTF_OPTION_FFTW_ESTIMATE (0U)
#define FFTF_OPTION_FFTW_MEASURE (1U << 6)

/// @brief Initializes a new FFTF instance. After initialization,
/// perform calculations with fftf_calc(). When you are done, free
/// the allocated resources with fftf_destroy().
/// You may write new data to input and calculate again and again.
/// @param type The type of the transform.
/// @param direction The direction of the transform.
/// @param dimension The dimension of the transform. If the active backend's
/// implementation does not support the requested number of dimensions, an
/// assertion will be thrown.
/// @param lengths The length of each dimension in units of FFTFType.
/// It may not be NULL.
/// @param options Additional options to be passed to the backend.
/// @param input The source buffer to read numbers from.
/// If type is FFTF_TYPE_COMPLEX, it must
/// have size of sizeof(float) * (sum of lengths) * 2 bytes. If type is
/// FFTF_TYPE_REAL and direction is FFTF_DIRECTION_FORWARD, it must have
/// size of sizeof(float) * (sum of lengths). If type is FFTF_TYPE_REAL
/// and direction is FFTF_DIRECTION_BACKWARD, it must have size of
/// sizeof(float) * ((sum of lengths) + 2 * dimensions), that is, the last
/// complex number is Nyquist frequency.
/// If type is FFTF_TYPE_DCT, it must have size of
/// sizeof(float) * (sum of lengths).
/// It may not be NULL.
/// @param output The destination buffer to place the results in.
/// If type is FFTF_TYPE_COMPLEX, it must
/// have size of sizeof(float) * (sum of lengths) * 2 bytes. If type is
/// FFTF_TYPE_REAL and direction is FFTF_DIRECTION_FORWARD, it must have
/// size of sizeof(float) * ((sum of lengths) + 2 * dimensions), that is,
/// the last complex number is for Nyquist frequency. If type is
/// FFTF_TYPE_REAL and direction is FFTF_DIRECTION_BACKWARD, it must have
/// size of sizeof(float) * (sum of lengths).
/// It may not be NULL.
/// If output is the same as input, FFTF engine tries to do an in-place
/// transform. Please note that in such a case, if type is FFTF_TYPE_REAL,
/// input size is always sizeof(float) * ((sum of lengths) + 2 * dimensions).
/// @return A newly allocated FFTF instance.
/// @note It is recommended to allocate input and output arrays with
/// fftf_malloc() and free them with fftf_free(). Some backends require such
/// allocation, since fftf_malloc() guarantees the proper data alignment.
/// For example, libavcodec's FFT x86 AVX implementation throws a segmentation
/// fault in case of invalid input alignment.
FFTFInstance *fftf_init(FFTFType type, FFTFDirection direction,
                        FFTFDimension dimension, const int *lengths,
                        FFTFOptions options, const float *input,
                        float *output) WARN_UNUSED_RESULT NOTNULL(4, 6, 7);

/// @brief Initializes a new FFTF instance for batch calculations.
/// After initialization, perform calculations with fftf_calc().
/// When you are done, free the allocated resources with fftf_destroy().
/// Unlike fftf_init(), this function plans a set of identical FFT-s,
/// using the active backend's special features when possible.
/// You may write to inputs and calculate again and again.
/// @param type The type of the transform.
/// @param direction The direction of the transform.
/// @param dimension The dimension of the transform. If the active backend's
/// implementation does not support the requested number of dimensions, an
/// assertion will be thrown.
/// @param lengths The length of each dimension in units of FFTFType.
/// It may not be NULL.
/// @param options Additional options to be passed to the backend.
/// @param batchCount The number of buffers in inputs and outputs.
/// @param inputs The array of identical source buffers to read numbers from.
/// If type is FFTF_TYPE_COMPLEX, each buffer must
/// have size of sizeof(float) * (sum of lengths) * 2 bytes. If type is
/// FFTF_TYPE_REAL and direction is FFTF_DIRECTION_FORWARD, it must have
/// size of sizeof(float) * (sum of lengths). If type is FFTF_TYPE_REAL
/// and direction is FFTF_DIRECTION_BACKWARD, it must have size of
/// sizeof(float) * ((sum of lengths) + 2 * dimensions), that is, the last
/// complex number is Nyquist frequency.
/// If type is FFTF_TYPE_DCT, it must have size of
/// sizeof(float) * (sum of lengths).
/// inputs may not be NULL.
/// @param outputs The array of destination buffers to place the results in.
/// If type is FFTF_TYPE_COMPLEX, each buffer must
/// have size of sizeof(float) * (sum of lengths) * 2 bytes. If type is
/// FFTF_TYPE_REAL and direction is FFTF_DIRECTION_FORWARD, it must have
/// size of sizeof(float) * ((sum of lengths) + 2 * dimensions), that is,
/// the last complex number is for Nyquist frequency. If type is
/// FFTF_TYPE_REAL and direction is FFTF_DIRECTION_BACKWARD, it must have
/// size of sizeof(float) * (sum of lengths).
/// outputs may not be NULL.
/// If outputs is the same as inputs, FFTF engine tries to do in-place
/// transforms. Please note that in such a case, if type is FFTF_TYPE_REAL,
/// each input buffer size is always
/// sizeof(float) * ((sum of lengths) + 2 * dimensions).
/// @return A newly allocated FFTF instance.
/// @note It is recommended to allocate each buffer in inputs and outputs with
/// fftf_malloc() and free them with fftf_free() (but no matter how inputs and
/// outputs were allocated themselves). Some backends require such
/// allocation, since fftf_malloc() guarantees the proper data alignment.
/// For example, libavcodec's FFT x86 AVX implementation throws a segmentation
/// fault in case of invalid input alignment.
FFTFInstance *fftf_init_batch(FFTFType type, FFTFDirection direction,
                              FFTFDimension dimension, const int *lengths,
                              FFTFOptions options, int batchCount,
                              const float *const *inputs,
                              float *const *outputs)
                              WARN_UNUSED_RESULT NOTNULL(4, 7, 8);

/// @brief Executes the operation(s) set for the specified FFTF instance.
/// @param instance A FFTF instance. It may not be null.
/// @note This function is thread-safe.
void fftf_calc(const FFTFInstance *instance) NOTNULL(1);

/// @brief Releases any resources allocated by the specified FFTF
/// instance. After the destruction, any subsequent calls to
/// fftf_calc() are invalid.
/// @param instance A FFTF instance. It may not be null.
void fftf_destroy(FFTFInstance *instance) NOTNULL(1);

/// @brief Like malloc(), allocates the requested amount of memory.
/// @param size The number of bytes to allocate.
/// @return The newly allocated memory pointer.
/// @note It is recommended to allocate memory with fftf_malloc() for input
/// and output buffers used in fftf_init() and fftf_init_batch().
/// fftf_malloc() is not compatible with free()! Use fftf_free() instead.
void *fftf_malloc(size_t size) MALLOC;

/// @brief Like free(), frees the memory previously allocated with
/// fftf_malloc().
/// @param ptr The pointer to be freed.
/// @note fftf_free() is not compatible with malloc()! Use
/// fftf_malloc() instead.
void fftf_free(void *ptr);

/// @brief Gets the maximal number of OpenMP threads in batch mode for backends
/// without the explicit batch support.
/// @return value A positive number.
int fftf_get_openmp_num_threads();

/// @brief Sets the maximal number of OpenMP threads in batch mode for backends
/// without the explicit batch support.
/// @param value A positive number.
void fftf_set_openmp_num_threads(int value);

#if __GNUC__ >= 4
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif

#endif  // INC_FFTF_API_H_
