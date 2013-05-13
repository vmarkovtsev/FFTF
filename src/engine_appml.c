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

/*! @file engine_appml.c
 *  @brief AMD OpenCL FFT wrapper functions implementation.
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

#include "src/engine_appml.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "src/opencl/clamdfft.h"

typedef struct {
  void *libHandle;

  clAmdFftSetupFunc clAmdFftSetup;
  clAmdFftTeardownFunc clAmdFftTeardown;
  clAmdFftCreateDefaultPlanFunc clAmdFftCreateDefaultPlan;
  clAmdFftSetResultLocationFunc clAmdFftSetResultLocation;
  clAmdFftSetLayoutFunc clAmdFftSetLayout;
  clAmdFftBakePlanFunc clAmdFftBakePlan;
  clAmdFftDestroyPlanFunc clAmdFftDestroyPlan;
  clAmdFftEnqueueTransformFunc clAmdFftEnqueueTransform;
} LibAPPML;

#define dlopen_appml(lib, path) dlopen_checked(&lib, libHandle, path)
#define dlsym_appml(lib, symbol) dlsym_checked(&lib, libHandle, symbol)

int load_appml(const char *libPath, void **engineInternalData) {
  LibAPPML libappml;
  dlopen_appml(libappml, libPath);

  dlsym_appml(libappml, clAmdFftSetup);
  dlsym_appml(libappml, clAmdFftTeardown);
  dlsym_appml(libappml, clAmdFftCreateDefaultPlan);
  dlsym_appml(libappml, clAmdFftSetResultLocation);
  dlsym_appml(libappml, clAmdFftSetLayout);
  dlsym_appml(libappml, clAmdFftBakePlan);
  dlsym_appml(libappml, clAmdFftDestroyPlan);
  dlsym_appml(libappml, clAmdFftEnqueueTransform);

  clAmdFftSetupData sd;
  CLAMDFFT_CHECKED_RET0(clAmdFftInitSetupData(&sd));
  if (sd.major < 1 || sd.minor < 8) {
    fprintf(stderr, "%s is of too old version %u.%u, skipped\n",
            libPath, sd.major, sd.minor);
    return 0;
  }
  CLAMDFFT_CHECKED_RET0(libappml.clAmdFftSetup(&sd));

  *engineInternalData = malloc(sizeof(LibAPPML));
  memcpy(*engineInternalData, &libappml, sizeof(LibAPPML));
  return 1;
}

void unload_appml(void *engineInternalData) {
  LibAPPML *libappml = (LibAPPML *)engineInternalData;
  CLAMDFFT_CHECKED(libappml->clAmdFftTeardown());
  dlclose(libappml->libHandle);
  free(libappml);
}

typedef struct {
  cl_context context;
  cl_command_queue queue;
  cl_mem *inputs;
  cl_mem *outputs;
  clAmdFftPlanHandle handle;
  int zerocopy;
  int inplace;
  int init_successful;
} clAmdFftInstance;

static size_t calc_buffer_size(const FFTFInstance *instance, int input) {
  int length = 0;
  for (int i = 0; i < (int)instance->dimension; i++) {
    length += instance->lengths[i];
  }
  size_t res  = length * sizeof(float);
  if (instance->type == FFTF_TYPE_REAL) {
    if ((instance->direction == FFTF_DIRECTION_BACKWARD && input) ||
        (instance->direction == FFTF_DIRECTION_FORWARD && !input)) {
      res += 2 * sizeof(float);
    }
  } else {
    res *= 2;
  }
  return res;
}

static void init_data_buffers(const FFTFInstance *instance) {
  clAmdFftInstance *data = (clAmdFftInstance *)instance->internalData;
  int inplace = instance->inputs == (const float **)instance->outputs;
  data->inplace = inplace;
  int zerocopy = (instance->options & FFTF_OPTION_OPENCL_ZEROCOPY);
  data->zerocopy = zerocopy;

  data->inputs = malloc(sizeof(void *) * instance->batchSize);
  if (!inplace) {
    data->outputs = malloc(sizeof(void *) * instance->batchSize);
  }

  for (int i = 0; i < instance->batchSize; i++) {
    cl_int clCreateBuffer_status;
    if (data->zerocopy) {
      data->inputs[i] = clCreateBuffer(
          data->context,
          CL_MEM_USE_HOST_PTR | (inplace? CL_MEM_READ_WRITE : CL_MEM_READ_ONLY),
          calc_buffer_size(instance, 1),
          (void *)instance->inputs[i],
          &clCreateBuffer_status);
    } else {
      data->inputs[i] = clCreateBuffer(
          data->context,
          inplace? CL_MEM_READ_WRITE : CL_MEM_READ_ONLY,
          calc_buffer_size(instance, 1),
          NULL,
          &clCreateBuffer_status);
    }
    OPENCL_CHECKED_RETURN(clCreateBuffer_status);

    if (!inplace) {
      if (data->zerocopy) {
        // CL_MEM_USE_HOST_PTR raises Segmentation fault
        data->outputs[i] = clCreateBuffer(
            data->context,
            CL_MEM_ALLOC_HOST_PTR | CL_MEM_WRITE_ONLY,
            calc_buffer_size(instance, 0),
            NULL,
            &clCreateBuffer_status);
      } else {
        data->outputs[i] = clCreateBuffer(
            data->context,
            CL_MEM_WRITE_ONLY,
            calc_buffer_size(instance, 0),
            NULL,
            &clCreateBuffer_status);
      }
      OPENCL_CHECKED_RETURN(clCreateBuffer_status);
    }
  }
}

void init_many_appml(void *engineInternalData, FFTFInstance *instance) {
  assert(instance->type != FFTF_TYPE_DCT &&
         "DCT is not supported by this backend");

  LibAPPML *libappml = (LibAPPML *)engineInternalData;
  clAmdFftInstance *data = malloc(sizeof(clAmdFftInstance));
  instance->internalData = (void **)data;
  data->init_successful = 0;
  cl_device_id *devices;
  cl_uint numDevices;
  data->context = initializeOpenCL(
      FFTFOptionsToCLDeviceType(instance->options),
      &devices, &numDevices);
  if (data->context == NULL) {
    fprintf(stderr, "Failed to initialize an OpenCL context\n");
    return;
  }
  int deviceIndex = instance->options & 0xF0;
  cl_int clCreateCommandQueue_status;
  data->queue = clCreateCommandQueue(data->context, devices[deviceIndex],
                                     0, &clCreateCommandQueue_status);
  OPENCL_CHECKED_RETURN(clCreateCommandQueue_status);
  free(devices);
  init_data_buffers(instance);

  // Done with OpenCL stuff, pass over to clAmdFft
  size_t lengths[3] = { 0, 0, 0 };
  for (int i = 0; i < (int)instance->dimension; i++) {
    lengths[i] = instance->lengths[i];
  }
  CLAMDFFT_CHECKED_RETURN(libappml->clAmdFftCreateDefaultPlan(
      &data->handle, data->context, instance->dimension, lengths));
  CLAMDFFT_CHECKED_RETURN(libappml->clAmdFftSetResultLocation(
      data->handle, data->inplace? CLFFT_INPLACE : CLFFT_OUTOFPLACE));
  clAmdFftLayout inLayout = CLFFT_COMPLEX_INTERLEAVED;
  clAmdFftLayout outLayout = CLFFT_COMPLEX_INTERLEAVED;
  if (instance->type == FFTF_TYPE_REAL) {
    if (instance->direction == FFTF_DIRECTION_FORWARD) {
     inLayout = CLFFT_REAL;
    } else {
      outLayout = CLFFT_REAL;
    }
  }
  CLAMDFFT_CHECKED_RETURN(libappml->clAmdFftSetLayout(
      data->handle, inLayout, outLayout));
  CLAMDFFT_CHECKED_RETURN(libappml->clAmdFftBakePlan(
      data->handle, 1, &data->queue, NULL, NULL));
  data->init_successful = 1;
}

static void data_in(const FFTFInstance *instance, cl_event *events) {
  clAmdFftInstance *data = (clAmdFftInstance *)instance->internalData;
  size_t eachBufferSize = calc_buffer_size(instance, 1);

  for (int i = 0; i < instance->batchSize; i++) {
    if (data->zerocopy) {
      cl_int clEnqueueMapBuffer_status;
      void *p = clEnqueueMapBuffer(
          data->queue, data->inputs[i], CL_TRUE,
          CL_MAP_WRITE, 0,
          eachBufferSize,
          0, NULL, NULL,
          &clEnqueueMapBuffer_status);
      OPENCL_CHECKED(clEnqueueMapBuffer_status);
      assert(p == instance->inputs[i] && "Wrong zero-copy");
      clEnqueueUnmapMemObject(data->queue, data->inputs[i],
                              p, 0, NULL, &events[i]);
    } else {
      OPENCL_CHECKED_RETURN(clEnqueueWriteBuffer(
          data->queue, data->inputs[i], CL_TRUE, 0,
          eachBufferSize, instance->inputs[i],
          0, NULL, &events[i]));
    }
  }
}

static void data_out(const FFTFInstance *instance, const cl_event *events) {
  clAmdFftInstance *data = (clAmdFftInstance *)instance->internalData;

  size_t eachBufferSize = calc_buffer_size(instance, 0);

  for (int i = 0; i < instance->batchSize; i++) {
    if (data->zerocopy) {
      cl_int clEnqueueMapBuffer_status;
      void *p = clEnqueueMapBuffer(
          data->queue, data->outputs[i], CL_TRUE,
          CL_MAP_READ, 0,
          eachBufferSize, 1, &events[i], NULL,
          &clEnqueueMapBuffer_status);
      OPENCL_CHECKED(clEnqueueMapBuffer_status);
      if (!data->inplace) {
        // We still need to copy data if not in-place,
        // since CL_MEM_USE_HOST_PTR on data->output results in
        // Segmentation fault
        memcpy(instance->outputs[i], p, eachBufferSize);
      } else {
        assert(p == instance->inputs[0] && "Wrong zero-copy");
      }
      clEnqueueUnmapMemObject(data->queue, data->outputs[i],
                              p, 0, NULL, NULL);
    } else {
      OPENCL_CHECKED(clEnqueueReadBuffer(
          data->queue, data->outputs[i], CL_TRUE, 0,
          eachBufferSize, instance->outputs[i], 1, &events[i], NULL));
    }
  }
}

void calc_many_appml(void *engineInternalData, const FFTFInstance *instance) {
  LibAPPML *libappml = (LibAPPML *)engineInternalData;
  clAmdFftInstance *data = (clAmdFftInstance *)instance->internalData;
  if (!data->init_successful) return;
  assert(data->context != NULL && "OpenCL context is not set");

  cl_event eventIn[instance->batchSize], eventOut[instance->batchSize];
  data_in(instance, eventIn);

  for (int i = 0; i < instance->batchSize; i++) {
    CLAMDFFT_CHECKED_RETURN(libappml->clAmdFftEnqueueTransform(
        data->handle, instance->direction, 1, &data->queue,
        data->zerocopy? 0 : 1,
        data->zerocopy? NULL : &eventIn[i],
        &eventOut[i], &data->inputs[i], &data->outputs[i], NULL));
  }

  data_out(instance, eventOut);

  if (instance->direction == FFTF_DIRECTION_BACKWARD) {
    if (instance->dimension == FFTF_DIMENSION_1D) {
      for (int i = 0; i < instance->batchSize; i++) {
        for (int j = 0; j < instance->lengths[0] * 2; j++) {
          instance->outputs[i][j] *= instance->lengths[0];
        }
      }
    } else {
      fprintf(stderr, "Warning: multidimensional normalization unification "
              "is not implemented. APPML is likely to produce an already "
              "normalized result.");
    }
  }
}

void destroy_many_appml(void *engineInternalData, FFTFInstance *instance) {
  LibAPPML *libappml = (LibAPPML *)engineInternalData;
  clAmdFftInstance *data = (clAmdFftInstance *)instance->internalData;
  CLAMDFFT_CHECKED(libappml->clAmdFftDestroyPlan(&data->handle));
  CLAMDFFT_CHECKED(libappml->clAmdFftTeardown());
  for (int i = 0; i < instance->batchSize; i++) {
    OPENCL_CHECKED(clReleaseMemObject(data->inputs[i]));
    if (!data->inplace) {
      OPENCL_CHECKED(clReleaseMemObject(data->outputs[i]));
    }
  }
  free(data->inputs);
  if (!data->inplace) {
    free(data->outputs);
  }
  OPENCL_CHECKED(clReleaseCommandQueue(data->queue));
  OPENCL_CHECKED(clReleaseContext(data->context));
  free(data);
}

#endif  // #ifdef OPENCL
