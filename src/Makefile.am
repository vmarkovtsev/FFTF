include $(top_srcdir)/Makefile.common

lib_LTLIBRARIES = libfftf.la

# Source files
libfftf_la_SOURCES = api.c backend.c engine_kiss.c engine_ooura.c engine_libav.c \
engine_fftw3.c engine_appml.c engine_ipp.c engine_cufft.c engine_mkl.c engine_viennacl.cc

# C++ additional flags
libfftf_la_CXXFLAGS = -I$(top_srcdir)/src/opencl -DVIENNACL_WITH_OPENCL

# Used libraries
libfftf_la_LIBADD = kiss/lib_kissfft.la ooura/lib_ooura.la $(OPENCL_LT) $(CUDA_LT) -ldl \
@OPENCL_LIBS@

PARALLEL_SUBDIRS = kiss ooura $(OPENCL_DIR) $(CUDA_DIR)
