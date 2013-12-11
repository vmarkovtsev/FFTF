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

#ifndef NO_SET_BACKEND

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

#endif

bool set_backend(FFTFBackendId backend
#ifdef NO_SET_BACKEND
                 __attribute__((unused))
#endif
                 ) {
#ifndef NO_SET_BACKEND
  fftf_set_backend(backend);
  return fftf_current_backend() == backend;
#else
  return true;
#endif
}

TEST(NAME, ComplexForward) {
  ASSERT_FALSE(fatalError);
  ASSERT_TRUE(set_backend(BACKEND));
  TestEnv te(FFTF_DIRECTION_FORWARD, FFTF_TYPE_COMPLEX);
  auto handle = fftf_init(FFTF_TYPE_COMPLEX, FFTF_DIRECTION_FORWARD,
                          FFTF_DIMENSION_1D, te.N(), FFTF_NO_OPTIONS,
                          te.input(), te.output());
  fftf_calc(handle);
  fftf_destroy(handle);
  ASSERT_EQ(kVerificationResultOk, te.Verify());
}

TEST(NAME, ComplexForwardLooong) {
  ASSERT_FALSE(fatalError);
  ASSERT_TRUE(set_backend(BACKEND));
  TestEnv te(FFTF_DIRECTION_FORWARD, FFTF_TYPE_COMPLEX, false, 1, 65536);
  auto handle = fftf_init(FFTF_TYPE_COMPLEX, FFTF_DIRECTION_FORWARD,
                          FFTF_DIMENSION_1D, te.N(), FFTF_NO_OPTIONS,
                          te.input(), te.output());
  fftf_calc(handle);
  fftf_destroy(handle);
  ASSERT_EQ(kVerificationResultOk, te.Verify());
}

TEST(NAME, ComplexBackward) {
  ASSERT_FALSE(fatalError);
  ASSERT_TRUE(set_backend(BACKEND));
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
  ASSERT_TRUE(set_backend(BACKEND));
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
  ASSERT_TRUE(set_backend(BACKEND));

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
  ASSERT_TRUE(set_backend(BACKEND));
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
  ASSERT_TRUE(set_backend(BACKEND));
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
  ASSERT_TRUE(set_backend(BACKEND));

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
/*
TEST(NAME, RealForwardBatchInplace) {
  ASSERT_FALSE(fatalError);
  ASSERT_TRUE(set_backend(BACKEND));
  TestEnv te(FFTF_DIRECTION_FORWARD, FFTF_TYPE_REAL, true, 100);
  auto handle = fftf_init_batch(FFTF_TYPE_COMPLEX, FFTF_DIRECTION_FORWARD,
                                FFTF_DIMENSION_1D, te.N(), FFTF_NO_OPTIONS,
                                100, te.inputs(), te.outputs());
  fftf_calc(handle);
  fftf_destroy(handle);
  ASSERT_EQ(kVerificationResultOk, te.Verify());
  for (int i = 0; i < 99; i++) {
    ASSERT_EQ(0, memcmp(te.outputs()[i], te.outputs()[i + 1],
                        (*te.N() + 2) * sizeof(float)));
  }
  exit(0);
}
*/
#endif

#ifndef DCT_TESTS
#ifndef NO_SET_BACKEND
TEST(NAME, DCTAssertion) {
  ASSERT_FALSE(fatalError);
  ASSERT_TRUE(set_backend(BACKEND));
  TestEnv te(FFTF_DIRECTION_FORWARD, FFTF_TYPE_DCT);
  // There are problems with stderr redirection otherwise
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
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
#endif
#else
TEST(NAME, DCTForward) {
  ASSERT_FALSE(fatalError);
  ASSERT_TRUE(set_backend(BACKEND));
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
  ASSERT_TRUE(set_backend(BACKEND));
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
  ASSERT_TRUE(set_backend(BACKEND));

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
  ASSERT_TRUE(set_backend(BACKEND));
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
  ASSERT_TRUE(set_backend(BACKEND));
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
  ASSERT_TRUE(set_backend(BACKEND));
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
  ASSERT_TRUE(set_backend(BACKEND));
  TestEnv te(FFTF_DIRECTION_FORWARD, FFTF_TYPE_COMPLEX, true, 100);
  auto handle = fftf_init_batch(FFTF_TYPE_COMPLEX, FFTF_DIRECTION_FORWARD,
                                FFTF_DIMENSION_1D, te.N(), FFTF_NO_OPTIONS,
                                100, te.inputs(), te.outputs());
  fftf_calc(handle);
  fftf_destroy(handle);
  ASSERT_EQ(kVerificationResultOk, te.Verify());
}

TEST(NAME, ComplexForwardMultipleHandles) {
  ASSERT_FALSE(fatalError);
  ASSERT_TRUE(set_backend(BACKEND));
  TestEnv te(FFTF_DIRECTION_FORWARD, FFTF_TYPE_COMPLEX);
  auto handle1 = fftf_init(FFTF_TYPE_COMPLEX, FFTF_DIRECTION_FORWARD,
                           FFTF_DIMENSION_1D, te.N(), FFTF_NO_OPTIONS,
                           te.input(), te.output());
  auto handle2 = fftf_init(FFTF_TYPE_COMPLEX, FFTF_DIRECTION_FORWARD,
                           FFTF_DIMENSION_1D, te.N(), FFTF_NO_OPTIONS,
                           te.input(), te.output());
  TestEnv te2(FFTF_DIRECTION_FORWARD, FFTF_TYPE_COMPLEX, true, 100);
  auto handle3 = fftf_init_batch(FFTF_TYPE_COMPLEX, FFTF_DIRECTION_FORWARD,
                                 FFTF_DIMENSION_1D, te2.N(), FFTF_NO_OPTIONS,
                                 100, te2.inputs(), te2.outputs());
#ifndef NO_REAL
  TestEnv te3(FFTF_DIRECTION_BACKWARD, FFTF_TYPE_REAL);
  auto handle4 = fftf_init(FFTF_TYPE_REAL, FFTF_DIRECTION_BACKWARD,
                           FFTF_DIMENSION_1D, te3.N(), FFTF_NO_OPTIONS,
                           te3.input(), te3.output());
#endif
  fftf_calc(handle3);
  ASSERT_EQ(kVerificationResultOk, te2.Verify());
  fftf_calc(handle1);
  ASSERT_EQ(kVerificationResultOk, te.Verify());
  fftf_calc(handle2);
  ASSERT_EQ(kVerificationResultOk, te.Verify());
#ifndef NO_REAL
  fftf_calc(handle4);
  ASSERT_EQ(kVerificationResultOk, te3.Verify());
#endif
  fftf_destroy(handle1);
  fftf_destroy(handle2);
  fftf_destroy(handle3);
#ifndef NO_REAL
  fftf_destroy(handle4);
#endif
}
#endif

#include "tests/google/src/gtest_main.cc"
