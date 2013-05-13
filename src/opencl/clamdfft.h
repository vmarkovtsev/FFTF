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

/*! @file clamdfft.h
 *  @brief APPML service functions.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef SRC_OPENCL_CLAMDFFT_H_
#define SRC_OPENCL_CLAMDFFT_H_

#include "src/config.h"
#include "src/opencl/opencl.h"

cl_int clamdfft_checked(cl_int res, const char *msg, const char *file,
                        int line);

#define CLAMDFFT_CHECKED(x) clamdfft_checked(x, STR(x), __FILE__, __LINE__)

#define CLAMDFFT_CHECKED_RET(x, val) do { \
  if (CLAMDFFT_CHECKED(x) != CL_SUCCESS) { \
    return val; \
  } \
} while (0)

#define CLAMDFFT_CHECKED_RET0(x) CLAMDFFT_CHECKED_RET(x, 0)
#define CLAMDFFT_CHECKED_RETNULL(x) CLAMDFFT_CHECKED_RET(x, NULL)
#define CLAMDFFT_CHECKED_RETURN(x) CLAMDFFT_CHECKED_RET(x, )

typedef enum {
  CLFFT_INVALID_GLOBAL_WORK_SIZE      = CL_INVALID_GLOBAL_WORK_SIZE,
  CLFFT_INVALID_MIP_LEVEL         = CL_INVALID_MIP_LEVEL,
  CLFFT_INVALID_BUFFER_SIZE       = CL_INVALID_BUFFER_SIZE,
  CLFFT_INVALID_GL_OBJECT         = CL_INVALID_GL_OBJECT,
  CLFFT_INVALID_OPERATION         = CL_INVALID_OPERATION,
  CLFFT_INVALID_EVENT           = CL_INVALID_EVENT,
  CLFFT_INVALID_EVENT_WAIT_LIST     = CL_INVALID_EVENT_WAIT_LIST,
  CLFFT_INVALID_GLOBAL_OFFSET       = CL_INVALID_GLOBAL_OFFSET,
  CLFFT_INVALID_WORK_ITEM_SIZE      = CL_INVALID_WORK_ITEM_SIZE,
  CLFFT_INVALID_WORK_GROUP_SIZE     = CL_INVALID_WORK_GROUP_SIZE,
  CLFFT_INVALID_WORK_DIMENSION      = CL_INVALID_WORK_DIMENSION,
  CLFFT_INVALID_KERNEL_ARGS       = CL_INVALID_KERNEL_ARGS,
  CLFFT_INVALID_ARG_SIZE          = CL_INVALID_ARG_SIZE,
  CLFFT_INVALID_ARG_VALUE         = CL_INVALID_ARG_VALUE,
  CLFFT_INVALID_ARG_INDEX         = CL_INVALID_ARG_INDEX,
  CLFFT_INVALID_KERNEL          = CL_INVALID_KERNEL,
  CLFFT_INVALID_KERNEL_DEFINITION     = CL_INVALID_KERNEL_DEFINITION,
  CLFFT_INVALID_KERNEL_NAME       = CL_INVALID_KERNEL_NAME,
  CLFFT_INVALID_PROGRAM_EXECUTABLE    = CL_INVALID_PROGRAM_EXECUTABLE,
  CLFFT_INVALID_PROGRAM         = CL_INVALID_PROGRAM,
  CLFFT_INVALID_BUILD_OPTIONS       = CL_INVALID_BUILD_OPTIONS,
  CLFFT_INVALID_BINARY          = CL_INVALID_BINARY,
  CLFFT_INVALID_SAMPLER         = CL_INVALID_SAMPLER,
  CLFFT_INVALID_IMAGE_SIZE        = CL_INVALID_IMAGE_SIZE,
  CLFFT_INVALID_IMAGE_FORMAT_DESCRIPTOR = CL_INVALID_IMAGE_FORMAT_DESCRIPTOR,
  CLFFT_INVALID_MEM_OBJECT        = CL_INVALID_MEM_OBJECT,
  CLFFT_INVALID_HOST_PTR          = CL_INVALID_HOST_PTR,
  CLFFT_INVALID_COMMAND_QUEUE       = CL_INVALID_COMMAND_QUEUE,
  CLFFT_INVALID_QUEUE_PROPERTIES      = CL_INVALID_QUEUE_PROPERTIES,
  CLFFT_INVALID_CONTEXT         = CL_INVALID_CONTEXT,
  CLFFT_INVALID_DEVICE          = CL_INVALID_DEVICE,
  CLFFT_INVALID_PLATFORM          = CL_INVALID_PLATFORM,
  CLFFT_INVALID_DEVICE_TYPE       = CL_INVALID_DEVICE_TYPE,
  CLFFT_INVALID_VALUE           = CL_INVALID_VALUE,
  CLFFT_MAP_FAILURE           = CL_MAP_FAILURE,
  CLFFT_BUILD_PROGRAM_FAILURE       = CL_BUILD_PROGRAM_FAILURE,
  CLFFT_IMAGE_FORMAT_NOT_SUPPORTED    = CL_IMAGE_FORMAT_NOT_SUPPORTED,
  CLFFT_IMAGE_FORMAT_MISMATCH       = CL_IMAGE_FORMAT_MISMATCH,
  CLFFT_MEM_COPY_OVERLAP          = CL_MEM_COPY_OVERLAP,
  CLFFT_PROFILING_INFO_NOT_AVAILABLE    = CL_PROFILING_INFO_NOT_AVAILABLE,
  CLFFT_OUT_OF_HOST_MEMORY        = CL_OUT_OF_HOST_MEMORY,
  CLFFT_OUT_OF_RESOURCES          = CL_OUT_OF_RESOURCES,
  CLFFT_MEM_OBJECT_ALLOCATION_FAILURE   = CL_MEM_OBJECT_ALLOCATION_FAILURE,
  CLFFT_COMPILER_NOT_AVAILABLE      = CL_COMPILER_NOT_AVAILABLE,
  CLFFT_DEVICE_NOT_AVAILABLE        = CL_DEVICE_NOT_AVAILABLE,
  CLFFT_DEVICE_NOT_FOUND          = CL_DEVICE_NOT_FOUND,
  CLFFT_SUCCESS             = CL_SUCCESS,
  //-------------------------- Extended status codes for clAmdFft -------------
  CLFFT_BUGCHECK =  4*1024, /*!< Bugcheck. */
  CLFFT_NOTIMPLEMENTED,   /*!< Functionality is not implemented yet. */
  CLFFT_TRANSPOSED_NOTIMPLEMENTED,  /// < Transposed functionality is not
                                    ///   implemented for this transformation.
  CLFFT_FILE_NOT_FOUND,   /// < Tried to open an existing file on the host
                          ///   system, but failed.
  CLFFT_FILE_CREATE_FAILURE,  /// < Tried to create a file on the host system,
                              ///   but failed.
  CLFFT_VERSION_MISMATCH,   /// < Version conflict between client and
                            ///   library.
  CLFFT_INVALID_PLAN,     /*!< Requested plan could not be found. */
  CLFFT_DEVICE_NO_DOUBLE,   /// < Double precision not supported on
                            ///   this device.
  CLFFT_ENDSTATUS       /// < This value will always be last, and marks
                        ///  the length of clAmdFftStatus.
} clAmdFftStatus;

/*! @brief Data structure that can be passed to clAmdFftSetup() to control
 *  the behavior of the FFT runtime
 *  @details This structure contains values that can be initialized before
 *  instantiation of the FFT runtime
 *  with ::clAmdFftSetup().  To initialize this structure, pass a pointer
 *  to a user struct to ::clAmdFftInitSetupData( ),
 *  which will clear the structure and set the version member variables
 *  to the current values.
 */
typedef struct {
  cl_uint major;    /// < Major version number of the project; signifies
                    ///   major API changes.
  cl_uint minor;    /// < Minor version number of the project; minor API
                    ///   changes that could break backwards compatibility.
  cl_uint patch;    /// < Patch version number of the project;
                    ///   Always incrementing number, signifies
                    ///   change over time.

  /*!   Bitwise flags that control the behavior of library debug logic. */
  cl_ulong debugFlags;  /// < This should be set to zero, except when debugging
                        ///  the clAmdFft library.
                        ///  <p> debugFlags can be set to CLFFT_DUMP_PROGRAMS,
                        ///  in which case the dynamically generated OpenCL
                        ///  kernels will be written to text files in the
                        ///  current working directory.
                        ///  These files will have a *.cl suffix.
} clAmdFftSetupData;

/*!  @brief The dimension of the input and output buffers that will be fed
 *          into all FFT transforms */
typedef enum {
  CLFFT_1D    = 1,    /*!< 1 Dimensional FFT transform (default). */
  CLFFT_2D,         /*!< 2 Dimensional FFT transform. */
  CLFFT_3D,         /*!< 3 Dimensional FFT transform. */
  ENDDIMENSION      /// < This value will always be last, and marks the
                    ///  length of clAmdFftDim.
} clAmdFftDim;

/*!  @brief An abstract handle to the object that represents the state of
 *          the FFT(s) */
typedef size_t clAmdFftPlanHandle;

/*!  @brief Are the input buffers overwritten with the results */
typedef enum {
  CLFFT_INPLACE   = 1,    /// < The input and output buffers are the same
                          ///   (default).
  CLFFT_OUTOFPLACE,       /*!< Seperate input and output buffers. */
  ENDPLACE        /// < This value will always be last, and marks the length of
                  ///   clAmdFftPlaceness.
} clAmdFftResultLocation;

///   @brief What are the expected layout's of the complex numbers <p>
///   <b> For Release 1.0,</b> only the CLFFT_COMPLEX_INTERLEAVED and
///   CLFFT_COMPLEX_PLANAR formats are supported.
///   The real and hermitian formats should be supported in a future release.
typedef enum {
  CLFFT_COMPLEX_INTERLEAVED = 1,  /// < An array of complex numbers, with real
                                  ///   and imaginary components together
                                  ///   (default).
  CLFFT_COMPLEX_PLANAR,       /// < Arrays of real componets and arrays of
                              ///   imaginary components that have been
                              ///   separated out.
  CLFFT_HERMITIAN_INTERLEAVED,    /// < Compressed form of complex numbers;
                                  ///   complex-conjugates not stored, real and
                                  ///   imaginary components in same array.
                                  ///   TODO: Document layout
  CLFFT_HERMITIAN_PLANAR,       /// < Compressed form of complex numbers;
                                ///   complex-conjugates not stored, real and
                                ///   imaginary components in separate arrays.
                                ///   TODO: Document layout
  CLFFT_REAL,             /// < An array of real numbers, with no corresponding
                          ///   imaginary components.
  ENDLAYOUT     /// < This value will always be last, and marks the length of
                ///   clAmdFftLayout.
} clAmdFftLayout;

/// @brief What is the expected direction of each FFT, time or the
/// frequency domains
typedef enum {
  CLFFT_FORWARD = -1,   /// < FFT transform from the time to the frequency
                        ///   domain.
  CLFFT_BACKWARD  = 1,    /// < FFT transform from the frequency to the time
                          ///   domain.
  CLFFT_MINUS   = -1,   /*!< Alias for the forward transform. */
  CLFFT_PLUS    = 1,    /*!< Alias for the backward transform. */
  ENDDIRECTION      /// < This value will always be last, and marks the length
                    ///   of clAmdFftDirection.
} clAmdFftDirection;

// My God, what AMD developers were thinking about when they made
// this function inline?.. My dear friends, inline functions are NOT got
// included into compiled shared libraries as separate symbols. Consequently,
// it is impossible to check version at runtime now. Well done!
// P.S. If you pretend to not understand me, run
// readelf -s libclAmdFft.Runtime.so | grep clAmdFftInitSetupData
INLINE clAmdFftStatus clAmdFftInitSetupData(clAmdFftSetupData *setupData) {
  setupData->major  = 1;
  setupData->minor  = 8;
  setupData->patch  = 291;
  setupData->debugFlags = 0;

  return  CLFFT_SUCCESS;
}

typedef clAmdFftStatus (*clAmdFftSetupFunc)(
    const clAmdFftSetupData *setupData);

typedef clAmdFftStatus (*clAmdFftTeardownFunc)(void);

typedef clAmdFftStatus (*clAmdFftCreateDefaultPlanFunc)(
    clAmdFftPlanHandle* plHandle, cl_context context, const clAmdFftDim dim,
    const size_t* clLengths);

typedef clAmdFftStatus (*clAmdFftSetResultLocationFunc)(
    clAmdFftPlanHandle plHandle, clAmdFftResultLocation placeness);

typedef clAmdFftStatus (*clAmdFftSetLayoutFunc)(clAmdFftPlanHandle plHandle,
    clAmdFftLayout iLayout, clAmdFftLayout oLayout);

typedef clAmdFftStatus (*clAmdFftBakePlanFunc)(clAmdFftPlanHandle plHandle,
    cl_uint numQueues, cl_command_queue* commQueueFFT,
    void (CL_CALLBACK *pfn_notify)(clAmdFftPlanHandle plHandle,
                                   void *user_data),
    void* user_data);

typedef clAmdFftStatus (*clAmdFftDestroyPlanFunc)(
    clAmdFftPlanHandle* plHandle);

typedef clAmdFftStatus (*clAmdFftEnqueueTransformFunc)(
    clAmdFftPlanHandle plHandle,
    clAmdFftDirection dir,
    cl_uint numQueuesAndEvents,
    cl_command_queue* commQueues,
    cl_uint numWaitEvents,
    const cl_event* waitEvents,
    cl_event* outEvents,
    cl_mem* inputBuffers,
    cl_mem* outputBuffers,
    cl_mem tmpBuffer);

#endif  // SRC_OPENCL_CLAMDFFT_H_
