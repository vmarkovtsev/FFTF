dnl Process this file with autoconf to produce a configure script.

AC_INIT(fftf, 1.0)
m4_include([common.ac])
AC_SUBST(INTERFACE_VERSION, [0])
AC_SUBST(REVISION_NUMBER, [$(cd $srcdir && git rev-list HEAD --count)])
AC_SUBST(AGE_NUMBER, [0])

dnl Check whether to build GPL backends
AC_ARG_WITH([gpl],
	AS_HELP_STRING([--with-gpl], [Build GPL backends, including FFTW3 (FFTF becomes GPL)]),
	[CPPFLAGS="$CPPFLAGS -DGPL"]
)

dnl Check whether to build OpenCL backends
AC_ARG_WITH([opencl],
	AS_HELP_STRING([--with-opencl], [Build OpenCL backends, including APPML]), [
	CPPFLAGS="$CPPFLAGS -DOPENCL"
	AS_IF([test "$(uname -s)" != "Darwin"], [
		AC_CHECK_HEADERS(CL/cl.h)
	], [
		AC_CHECK_HEADERS(OpenCL/cl.h)
	])	
	OPENCL_DIR=opencl
	OPENCL_LT=opencl/lib_opencl.la
	PKG_CHECK_MODULES([OPENCL], [OpenCL >= 1.1])]	
)
AC_SUBST([OPENCL_DIR])
AC_SUBST([OPENCL_LT])

dnl Check whether to build CUDA backends
AC_ARG_WITH([cuda],
	AS_HELP_STRING([--with-cuda], [Build CUDA backends, including cuFFT]), [
	CPPFLAGS="$CPPFLAGS -DCUDA"
	AC_CHECK_HEADERS(cufft.h)	
	CUDA_DIR=cuda
	CUDA_LT="cuda/lib_cuda.la -lcudart"]	
)
AC_SUBST([CUDA_DIR])
AC_SUBST([CUDA_LT])

AM_COND_IF([TESTS], [
	AC_CONFIG_FILES(tests/gpl/Makefile
                	tests/opencl/Makefile
                	tests/cuda/Makefile
                        tests/intel/Makefile
                	)
    AS_IF([test "x$with_gpl" == "xyes"], [
    	EXTRA_TESTS="$EXTRA_TESTS gpl"
    ])
    AS_IF([test "x$with_opencl" == "xyes"], [    	
    	EXTRA_TESTS="$EXTRA_TESTS opencl"
    ])

    AS_IF([test "x$with_cuda" == "xyes"], [
    	EXTRA_TESTS="$EXTRA_TESTS cuda"
    ])
    AS_IF([test ${host%%-*} != arm], [
	    EXTRA_TESTS="$EXTRA_TESTS intel"
	])
])
AC_SUBST([EXTRA_TESTS])

dnl Link with Makefile.am in additional source directories
AC_CONFIG_FILES(src/kiss/Makefile
                src/ooura/Makefile
                src/opencl/Makefile
                src/cuda/Makefile
                )

AC_OUTPUT
