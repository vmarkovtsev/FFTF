/*! @file template.inc
 *  @brief Tests template.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Common::Notes
 *  This code conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#include <gtest/gtest.h>
#include <fftf/api.h>
#include "tests/testenv.h"
#include "tests/testenv.cc"

bool fatalError = false;

TEST(NAME, SetBackend) {
#ifdef LD_LIBRARY_PATH
  const char* paths[2] = { LD_LIBRARY_PATH, NULL };
  auto backends = fftf_available_backends(paths, NULL);
  EXPECT_STRNE(NULL, backends[BACKEND].path);
  if (backends[BACKEND].path == NULL) {
    fatalError = true;
  }
#else
  fftf_set_backend(BACKEND);
  EXPECT_EQ(BACKEND, fftf_current_backend());
  if (fftf_current_backend() != BACKEND) {
    fatalError = true;
  }
#endif
  atexit(fftf_cleanup);
}

TEST(NAME, ComplexForward) {
  ASSERT_FALSE(fatalError);
  fftf_set_backend(BACKEND);
  ASSERT_EQ(BACKEND, fftf_current_backend());
  TestEnv te(FFTF_DIRECTION_FORWARD, FFTF_TYPE_COMPLEX);
  auto handle = fftf_init(FFTF_TYPE_COMPLEX, FFTF_DIRECTION_FORWARD,
                          FFTF_DIMENSION_1D, te.N(), FFTF_NO_OPTIONS,
                          te.input(), te.output());
  fftf_calc(handle);
  fftf_destroy(handle);
  ASSERT_EQ(kVerificationResultOk, te.Verify());
}

TEST(NAME, ComplexBackward) {
  ASSERT_FALSE(fatalError);
  fftf_set_backend(BACKEND);
  ASSERT_EQ(BACKEND, fftf_current_backend());
  TestEnv te(FFTF_DIRECTION_BACKWARD, FFTF_TYPE_COMPLEX);
  auto handle = fftf_init(FFTF_TYPE_COMPLEX, FFTF_DIRECTION_BACKWARD,
                          FFTF_DIMENSION_1D, te.N(), FFTF_NO_OPTIONS,
                          te.input(), te.output());
  fftf_calc(handle);
  fftf_destroy(handle);
  ASSERT_EQ(kVerificationResultOk, te.Verify());
}

TEST(NAME, ComplexForwardBatch) {
  ASSERT_FALSE(fatalError);
  fftf_set_backend(BACKEND);
  ASSERT_EQ(BACKEND, fftf_current_backend());
  TestEnv te(FFTF_DIRECTION_FORWARD, FFTF_TYPE_COMPLEX, false, 100);
  auto handle = fftf_init_batch(FFTF_TYPE_COMPLEX, FFTF_DIRECTION_FORWARD,
                                FFTF_DIMENSION_1D, te.N(), FFTF_NO_OPTIONS,
                                100, te.inputs(), te.outputs());
  fftf_calc(handle);
  fftf_destroy(handle);
  ASSERT_EQ(kVerificationResultOk, te.Verify());
}

TEST(NAME, ComplexForwardBackwardIdentity) {
  ASSERT_FALSE(fatalError);
  fftf_set_backend(BACKEND);
  ASSERT_EQ(BACKEND, fftf_current_backend());

  TestEnv te(FFTF_DIRECTION_FORWARD, FFTF_TYPE_COMPLEX);
  auto handle = fftf_init(FFTF_TYPE_COMPLEX, FFTF_DIRECTION_FORWARD,
                          FFTF_DIMENSION_1D, te.N(), FFTF_NO_OPTIONS,
                          te.input(), te.output());
  fftf_calc(handle);
  fftf_destroy(handle);
  te.swap_input_output();
  handle = fftf_init(FFTF_TYPE_COMPLEX, FFTF_DIRECTION_BACKWARD,
                     FFTF_DIMENSION_1D, te.N(), FFTF_NO_OPTIONS,
                     te.input(), te.output());
  fftf_calc(handle);
  fftf_destroy(handle);

  TestEnv te_src(FFTF_DIRECTION_FORWARD, FFTF_TYPE_COMPLEX);

  float error = 0.0f;
  int N = te.N()[0];
  for (int i = 0; i < N * 2; i++) {
     float delta = te_src.input()[i] - te.output()[i] / N;
     error += delta * delta;
  }
  ASSERT_GT(1E-8, error);
}

#ifndef NO_REAL
TEST(NAME, RealForward) {
  ASSERT_FALSE(fatalError);
  fftf_set_backend(BACKEND);
  ASSERT_EQ(BACKEND, fftf_current_backend());
  TestEnv te(FFTF_DIRECTION_FORWARD, FFTF_TYPE_REAL);
  auto handle = fftf_init(FFTF_TYPE_REAL, FFTF_DIRECTION_FORWARD,
                          FFTF_DIMENSION_1D, te.N(), FFTF_NO_OPTIONS,
                          te.input(), te.output());
  fftf_calc(handle);
  fftf_destroy(handle);
  ASSERT_EQ(kVerificationResultOk, te.Verify());
}

TEST(NAME, RealBackward) {
  ASSERT_FALSE(fatalError);
  fftf_set_backend(BACKEND);
  ASSERT_EQ(BACKEND, fftf_current_backend());
  TestEnv te(FFTF_DIRECTION_BACKWARD, FFTF_TYPE_REAL);
  auto handle = fftf_init(FFTF_TYPE_REAL, FFTF_DIRECTION_BACKWARD,
                          FFTF_DIMENSION_1D, te.N(), FFTF_NO_OPTIONS,
                          te.input(), te.output());
  fftf_calc(handle);
  fftf_destroy(handle);
  ASSERT_EQ(kVerificationResultOk, te.Verify());
}

TEST(NAME, RealForwardBackwardIdentity) {
  ASSERT_FALSE(fatalError);
  fftf_set_backend(BACKEND);
  ASSERT_EQ(BACKEND, fftf_current_backend());

  TestEnv te(FFTF_DIRECTION_FORWARD, FFTF_TYPE_REAL);
  auto handle = fftf_init(FFTF_TYPE_REAL, FFTF_DIRECTION_FORWARD,
                          FFTF_DIMENSION_1D, te.N(), FFTF_NO_OPTIONS,
                          te.input(), te.output());
  fftf_calc(handle);
  fftf_destroy(handle);
  handle = fftf_init(FFTF_TYPE_REAL, FFTF_DIRECTION_BACKWARD,
                     FFTF_DIMENSION_1D, te.N(), FFTF_NO_OPTIONS,
                     te.output(), te.output());
  fftf_calc(handle);
  fftf_destroy(handle);

  float error = 0.0f;
  int N = te.N()[0];
  for (int i = 0; i < N; i++) {
     float delta = te.input()[i] - te.output()[i] / N;
     error += delta * delta;
  }
  ASSERT_GT(1E-8, error);
}
#endif

#ifndef DCT_TESTS
TEST(NAME, DCTAssertion) {
  ASSERT_FALSE(fatalError);
  fftf_set_backend(BACKEND);
  ASSERT_EQ(BACKEND, fftf_current_backend());
  TestEnv te(FFTF_DIRECTION_FORWARD, FFTF_TYPE_DCT);
  // There are problems with stderr redirection otherwise
  // ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_DEATH({
    auto handle = fftf_init(FFTF_TYPE_DCT, FFTF_DIRECTION_FORWARD,
                            FFTF_DIMENSION_1D, te.N(), FFTF_NO_OPTIONS,
                            te.input(), te.output());
    assert(handle != nullptr);
  }, ".*Assertion.*DCT is not supported by this backend.*");
  EXPECT_DEATH({
    auto handle = fftf_init(FFTF_TYPE_DCT, FFTF_DIRECTION_BACKWARD,
                            FFTF_DIMENSION_1D, te.N(), FFTF_NO_OPTIONS,
                            te.input(), te.output());
    assert(handle != nullptr);
  }, ".*Assertion.*DCT is not supported by this backend.*");
}
#else
TEST(NAME, DCTForward) {
  ASSERT_FALSE(fatalError);
  fftf_set_backend(BACKEND);
  ASSERT_EQ(BACKEND, fftf_current_backend());
  TestEnv te(FFTF_DIRECTION_FORWARD, FFTF_TYPE_DCT);
  auto handle = fftf_init(FFTF_TYPE_DCT, FFTF_DIRECTION_FORWARD,
                          FFTF_DIMENSION_1D, te.N(), FFTF_NO_OPTIONS,
                          te.input(), te.output());
  fftf_calc(handle);
  fftf_destroy(handle);
  ASSERT_EQ(kVerificationResultOk, te.Verify());
}

TEST(NAME, DCTBackward) {
  ASSERT_FALSE(fatalError);
  fftf_set_backend(BACKEND);
  ASSERT_EQ(BACKEND, fftf_current_backend());
  TestEnv te(FFTF_DIRECTION_BACKWARD, FFTF_TYPE_DCT);
  auto handle = fftf_init(FFTF_TYPE_DCT, FFTF_DIRECTION_BACKWARD,
                          FFTF_DIMENSION_1D, te.N(), FFTF_NO_OPTIONS,
                          te.input(), te.output());
  fftf_calc(handle);
  fftf_destroy(handle);
  ASSERT_EQ(kVerificationResultOk, te.Verify());
}

TEST(NAME, DCTForwardBackwardIdentity) {
  ASSERT_FALSE(fatalError);
  fftf_set_backend(BACKEND);
  ASSERT_EQ(BACKEND, fftf_current_backend());

  TestEnv te(FFTF_DIRECTION_FORWARD, FFTF_TYPE_DCT);
  auto handle = fftf_init(FFTF_TYPE_DCT, FFTF_DIRECTION_FORWARD,
                          FFTF_DIMENSION_1D, te.N(), FFTF_NO_OPTIONS,
                          te.input(), te.output());
  fftf_calc(handle);
  fftf_destroy(handle);
  handle = fftf_init(FFTF_TYPE_DCT, FFTF_DIRECTION_BACKWARD,
                     FFTF_DIMENSION_1D, te.N(), FFTF_NO_OPTIONS,
                     te.output(), te.output());
  fftf_calc(handle);
  fftf_destroy(handle);

  float error = 0.0f;
  int N = te.N()[0];
  for (int i = 0; i < N; i++) {
     float delta = te.input()[i] - te.output()[i] / (2 * N);
     error += delta * delta;
  }
  ASSERT_GT(1E-8, error);
}

#ifdef INPLACE
TEST(NAME, DCTForwardInplace) {
  ASSERT_FALSE(fatalError);
  fftf_set_backend(BACKEND);
  ASSERT_EQ(BACKEND, fftf_current_backend());
  TestEnv te(FFTF_DIRECTION_FORWARD, FFTF_TYPE_DCT, true);
  auto handle = fftf_init(FFTF_TYPE_DCT, FFTF_DIRECTION_FORWARD,
                          FFTF_DIMENSION_1D, te.N(), FFTF_NO_OPTIONS,
                          te.input(), te.output());
  fftf_calc(handle);
  fftf_destroy(handle);
  ASSERT_EQ(kVerificationResultOk, te.Verify());
}
#endif
#endif

#ifdef INPLACE
TEST(NAME, InplaceComplexForward) {
  ASSERT_FALSE(fatalError);
  fftf_set_backend(BACKEND);
  ASSERT_EQ(BACKEND, fftf_current_backend());
  TestEnv te(FFTF_DIRECTION_FORWARD, FFTF_TYPE_COMPLEX, true);
  auto handle = fftf_init(FFTF_TYPE_COMPLEX, FFTF_DIRECTION_FORWARD,
                          FFTF_DIMENSION_1D, te.N(), FFTF_NO_OPTIONS,
                          te.input(), te.output());
  fftf_calc(handle);
  fftf_destroy(handle);
  ASSERT_EQ(kVerificationResultOk, te.Verify());
}

TEST(NAME, InplaceComplexBackward) {
  ASSERT_FALSE(fatalError);
  fftf_set_backend(BACKEND);
  ASSERT_EQ(BACKEND, fftf_current_backend());
  TestEnv te(FFTF_DIRECTION_BACKWARD, FFTF_TYPE_COMPLEX, true);
  auto handle = fftf_init(FFTF_TYPE_COMPLEX, FFTF_DIRECTION_BACKWARD,
                          FFTF_DIMENSION_1D, te.N(), FFTF_NO_OPTIONS,
                          te.input(), te.output());
  fftf_calc(handle);
  fftf_destroy(handle);
  ASSERT_EQ(kVerificationResultOk, te.Verify());
}

TEST(NAME, ComplexForwardBatchInplace) {
  ASSERT_FALSE(fatalError);
  fftf_set_backend(BACKEND);
  ASSERT_EQ(BACKEND, fftf_current_backend());
  TestEnv te(FFTF_DIRECTION_FORWARD, FFTF_TYPE_COMPLEX, true, 100);
  auto handle = fftf_init_batch(FFTF_TYPE_COMPLEX, FFTF_DIRECTION_FORWARD,
                                FFTF_DIMENSION_1D, te.N(), FFTF_NO_OPTIONS,
                                100, te.inputs(), te.outputs());
  fftf_calc(handle);
  fftf_destroy(handle);
  ASSERT_EQ(kVerificationResultOk, te.Verify());
}
#endif

#include "tests/google/src/gtest_main.cc"
