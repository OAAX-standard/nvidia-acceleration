// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ORT_MINIMAL_BUILD
#include <gsl/narrow>

#include "core/common/span_utils.h"
#include "core/framework/tensor.h"
#include "core/mlas/inc/mlas_qnbit.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/mlas/inc/mlas.h"
#include "core/session/inference_session.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/framework/test_utils.h"
#include "test/optimizer/graph_transform_test_builder.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/ort_env.h"
#include "core/util/qmath.h"

#include <chrono>
#include <random>

#include "gtest/gtest.h"
#include "gmock/gmock.h"
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {

namespace test {

static constexpr int QBits = 4;
void QuantizeDequantize(std::vector<float>& raw_vals,
                        std::vector<uint8_t>& quant_vals,
                        std::vector<float>& scales,
                        std::vector<uint8_t>* zp,
                        int32_t N,
                        int32_t K,
                        int32_t block_size) {
  auto& ortenv = **ort_env.get();
  onnxruntime::concurrency::ThreadPool* tp = ortenv.GetEnvironment().GetIntraOpThreadPool();

  MlasQuantizeBlockwise<float, 4>(
      quant_vals.data(),
      scales.data(),
      zp != nullptr ? zp->data() : nullptr,
      raw_vals.data(),
      block_size,
      true,
      K,
      N,
      N,
      tp);

  // Note that input1_f_vals is NxK after dequant
  MlasDequantizeBlockwise<float, 4>(
      raw_vals.data(),                       // dequantized output
      quant_vals.data(),                     // quantized input
      scales.data(),                         // quantization scales
      zp != nullptr ? zp->data() : nullptr,  // quantization zero points
      block_size,                            // quantization block size
      true,                                  // columnwise quantization
      K,                                     // number of rows
      N,                                     // number of columns
      tp);
}

void RunTest(int64_t M, int64_t N, int64_t K, int64_t block_size, int64_t accuracy_level,
             bool has_zeropoint, bool use_float16, bool has_g_idx = false,
             bool zp_is_4bit = true, float fp16_abs_error = 0.02f) {
  zp_is_4bit = zp_is_4bit | has_g_idx;
  RandomValueGenerator random{1234};
  std::vector<float> input0_vals(random.Gaussian<float>(std::vector<int64_t>({M, K}), 0.0f, 0.25f));
  std::vector<float> input1_f_vals(random.Gaussian<float>(std::vector<int64_t>({K, N}), 0.0f, 0.25f));

#if 0  // for Debugging
  std::vector<float> input1_f_vals_trans(N * K);
  MlasTranspose(input1_f_vals.data(), input1_f_vals_trans.data(), K, N);
#endif

  int q_rows, q_cols;
  MlasBlockwiseQuantizedShape<float, 4>((int)block_size, true, (int)K, (int)N, q_rows, q_cols);

  size_t q_data_size_in_bytes, q_scale_size, q_zp_size_in_bytes;
  MlasBlockwiseQuantizedBufferSizes(4, static_cast<int>(block_size), /* columnwise */ true,
                                    static_cast<int>(K), static_cast<int>(N),
                                    q_data_size_in_bytes, q_scale_size, &q_zp_size_in_bytes);

  std::vector<uint8_t> input1_vals(q_data_size_in_bytes);
  std::vector<float> scales(q_scale_size);
  std::vector<uint8_t> zp(q_zp_size_in_bytes);

  QuantizeDequantize(input1_f_vals,
                     input1_vals,
                     scales,
                     has_zeropoint ? &zp : nullptr,
                     static_cast<int32_t>(N),
                     static_cast<int32_t>(K),
                     static_cast<int32_t>(block_size));

#if 0
  for (int i = 0; i < input1_vals.size(); i++)
  {
    uint8_t byte = input1_vals[i];
    uint8_t val_lo = byte & 0x0f;
    uint8_t val_hi = byte >> 4;
    std::cout << (int)val_lo << ", " << (int)val_hi << ", ";
  }
#endif
  std::vector<float> expected_vals(M * N);
  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        sum += input0_vals[m * K + k] * input1_f_vals[n * K + k];
      }
      expected_vals[m * N + n] = sum;
    }
  }

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddAttribute<int64_t>("bits", QBits);
  test.AddAttribute<int64_t>("accuracy_level", accuracy_level);
  auto ceildiv = [](int64_t a, int64_t b) { return (a + b - 1) / b; };

  if (use_float16) {
    test.AddInput<MLFloat16>("A", {M, K}, ToFloat16(input0_vals), false);
    test.AddInput<uint8_t>("B", {q_cols, q_rows}, input1_vals, true);
    test.AddInput<MLFloat16>("scales", {static_cast<int64_t>(q_scale_size)}, ToFloat16(scales), true);
    if (has_zeropoint) {
      if (zp_is_4bit) {
        test.AddInput<uint8_t>("zero_points", {static_cast<int64_t>(q_zp_size_in_bytes)}, zp, true);
      } else {
        std::vector<float> zp_f;
        zp_f.reserve(q_zp_size_in_bytes * 2);
        for (size_t i = 0; i < zp.size(); i++) {
          zp_f.push_back(static_cast<float>(zp[i] & 0xf));
          zp_f.push_back(static_cast<float>((zp[i] >> 4) & 0xf));
        }
        size_t ind = zp_f.size() - 1;
        while (zp_f.size() != q_scale_size) {
          zp_f.erase(zp_f.begin() + ind);
          ind -= q_scale_size / N + 1;
        }

        test.AddInput<MLFloat16>("zero_points", {static_cast<int64_t>(q_scale_size)}, ToFloat16(zp_f), true);
      }
    } else {
      test.AddInput<uint8_t>("", {0}, {});
    }
    if (has_g_idx) {
      int K_pad = gsl::narrow<int32_t>(ceildiv(K, block_size) * block_size);
      std::vector<int32_t> g_idx(K_pad);
      for (int64_t i = 0; i < K_pad; i++) {
        g_idx[i] = gsl::narrow<int32_t>(i / block_size);
      }
      test.AddInput<int32_t>("g_idx", {static_cast<int64_t>(K_pad)}, g_idx, true);
    }

    test.AddOutput<MLFloat16>("Y", {M, N}, ToFloat16(expected_vals));
    test.SetOutputAbsErr("Y", fp16_abs_error);

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;

#ifdef USE_CUDA
    execution_providers.push_back(DefaultCudaExecutionProvider());
#endif

#ifdef USE_DML
    execution_providers.push_back(DefaultDmlExecutionProvider());
#endif

    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  } else {
    test.AddInput<float>("A", {M, K}, input0_vals, false);
    test.AddInput<uint8_t>("B", {q_cols, q_rows}, input1_vals, true);
    test.AddInput<float>("scales", {static_cast<int64_t>(q_scale_size)}, scales, true);
    if (has_zeropoint) {
      if (zp_is_4bit) {
        test.AddInput<uint8_t>("zero_points", {static_cast<int64_t>(q_zp_size_in_bytes)}, zp, true);
      } else {
        std::vector<float> zp_f;
        zp_f.reserve(q_zp_size_in_bytes * 2);
        for (size_t i = 0; i < zp.size(); i++) {
          zp_f.push_back(static_cast<float>(zp[i] & 0xf));
          zp_f.push_back(static_cast<float>((zp[i] >> 4) & 0xf));
        }
        size_t ind = zp_f.size() - 1;
        while (zp_f.size() != q_scale_size) {
          zp_f.erase(zp_f.begin() + ind);
          ind -= q_scale_size / N + 1;
        }

        test.AddInput<float>("zero_points", {static_cast<int64_t>(q_scale_size)}, zp_f, true);
      }
    } else {
      test.AddInput<uint8_t>("", {0}, {});
    }
    if (has_g_idx) {
      int K_pad = gsl::narrow<int32_t>(ceildiv(K, block_size) * block_size);
      std::vector<int32_t> g_idx(K_pad);
      for (int64_t i = 0; i < K_pad; i++) {
        g_idx[i] = gsl::narrow<int32_t>(i / block_size);
      }
      test.AddInput<int32_t>("g_idx", {static_cast<int64_t>(K_pad)}, g_idx, true);
    }
    test.AddOutput<float>("Y", {M, N}, expected_vals);
    if (accuracy_level == 4) {
      test.SetOutputAbsErr("Y", 0.1f);
    }

    test.Run();
  }
}

TEST(MatMulNBits, Float32) {
  for (auto M : {1, 2, 100}) {
    for (auto N : {1, 2, 32, 288}) {
      for (auto K : {16, 32, 64, 128, 256, 1024, 93, 1234}) {
        for (auto block_size : {16, 32, 64, 128}) {
#if defined(ORT_NEURAL_SPEED) || defined(USE_DML)
          for (auto accuracy_level : {0, 1, 4}) {
            RunTest(M, N, K, block_size, accuracy_level, false, false);
            RunTest(M, N, K, block_size, accuracy_level, true, false);
          }
#else
          for (auto accuracy_level : {0, 1, 4}) {
            RunTest(M, N, K, block_size, accuracy_level, false, false);
            RunTest(M, N, K, block_size, accuracy_level, true, false);
            RunTest(M, N, K, block_size, accuracy_level, false, false, true);
            RunTest(M, N, K, block_size, accuracy_level, true, false, false, false);
          }
#endif
        }
      }
    }
  }
}

#if defined(USE_CUDA) || defined(USE_DML)
TEST(MatMulNBits, Float16) {
#ifdef USE_CUDA
  auto has_gidx_options = {true, false};
#else
  auto has_gidx_options = {false};
#endif

  for (auto M : {1, 2, 100}) {
    for (auto N : {1, 2, 32, 288}) {
      for (auto K : {16, 32, 64, 128, 256, 1024, 93, 1234}) {
        for (auto block_size : {16, 32, 64, 128}) {
          for (auto has_gidx : has_gidx_options) {
#ifdef USE_DML
            RunTest(M, N, K, block_size, 0, false, true, has_gidx, true, 0.04f);
#else
            RunTest(M, N, K, block_size, 0, false, true, has_gidx);
            RunTest(M, N, K, block_size, 0, true, true, has_gidx, false);
#endif
          }
        }
      }
    }
  }
}

TEST(MatMulNBits, Float16Large) {
#ifdef USE_DML
  // For some reason, the A10 machine that runs these tests during CI has a much bigger error than all retail
  // machines we tested on. All consumer-grade machines from Nvidia/AMD/Intel seem to pass these tests with an
  // absolute error of 0.08, but the A10 has errors going as high as 0.22. Ultimately, given the large number
  // of elements in this test, ULPs should probably be used instead of absolute/relative tolerances.
  float abs_error = 0.3f;
#else
  float abs_error = 0.05f;
#endif

  for (auto block_size : {16, 32, 64, 128}) {
    for (auto symmetric : {false, true}) {
      RunTest(1, 4096, 4096, block_size, 0, symmetric, true, false, true, abs_error);
      RunTest(1, 4096, 11008, block_size, 0, symmetric, true, false, true, abs_error);
      RunTest(1, 11008, 4096, block_size, 0, symmetric, true, false, true, abs_error);
    }
  }
}

#endif

void RunSharedPrepackedWeightsTest(int64_t M, int64_t N, int64_t K, int block_size, bool is_asym,
                                   int64_t acc_lvl) {
  // (M x K) X (K x N)

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("accuracy_level", acc_lvl);
  test.AddAttribute<int64_t>("block_size", int64_t(block_size));
  test.AddAttribute<int64_t>("bits", QBits);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("K", K);

  std::vector<float> input0_vals(M * K);
  float fv = -135.f;
  for (auto& f : input0_vals) {
    f = fv / 127;
    fv++;
    if (fv > 135.f) {
      fv = -135.f;
    }
  }

  size_t kblks = K / block_size;
  std::vector<uint8_t> input1_vals(N * K / 2);
  for (size_t i = 0; i < input1_vals.size(); i++) {
    input1_vals[i] = uint8_t(i);
  }
  std::vector<float> input2_vals(N * kblks, 0.002f);
  for (size_t i = 0; i < N * kblks; i++) {
    input2_vals[i] += (i % 100) * 0.00003f;
  }
  std::vector<uint8_t> input3_vals(N * kblks / 2, static_cast<uint8_t>(0x88));

  std::vector<float> input1_f_vals(N * K);
  if (is_asym) {
    for (size_t i = 0; i < N * kblks; i += 2) {
      input3_vals[i / 2] = static_cast<uint8_t>(i + 1);
    }
    for (int64_t i = 0; i < K; i += 2) {
      for (int64_t j = 0; j < N; j++) {
        auto srcv = input1_vals[j * K / 2 + i / 2];
        auto koff = i % (block_size * 2);
        auto zpv = input3_vals[j * kblks / 2 + i / block_size / 2];
        auto zp0 = koff < block_size ? (zpv & 0xf) - 8 : ((zpv & 0xf0) >> 4) - 8;
        auto src0 = (srcv & 0xf) - 8;
        auto src1 = ((srcv & 0xf0) >> 4) - 8;
        auto scale0 = input2_vals[j * kblks + i / block_size];
        auto scale1 = input2_vals[j * kblks + (i + 1) / block_size];
        input1_f_vals[i * N + j] = (static_cast<float>(src0) - zp0) * scale0;
        input1_f_vals[(i + 1) * N + j] = (static_cast<float>(src1) - zp0) * scale1;
      }
    }
  } else {
    for (int64_t i = 0; i < K; i += 2) {
      for (int64_t j = 0; j < N; j++) {
        auto srcv = input1_vals[j * K / 2 + i / 2];
        auto src0 = (srcv & 0xf) - 8;
        auto src1 = ((srcv & 0xf0) >> 4) - 8;
        auto scale0 = input2_vals[j * kblks + i / block_size];
        auto scale1 = input2_vals[j * kblks + (i + 1) / block_size];
        input1_f_vals[i * N + j] = static_cast<float>(src0) * scale0;
        input1_f_vals[(i + 1) * N + j] = static_cast<float>(src1) * scale1;
      }
    }
  }

  std::vector<float> expected_vals(M * N);
  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        sum += input0_vals[m * K + k] * input1_f_vals[k * N + n];
      }
      expected_vals[m * N + n] = sum;
    }
  }

  test.AddInput<float>("A", {M, K}, input0_vals, false);

  test.AddInput<uint8_t>("B", {N, static_cast<int64_t>(kblks), static_cast<int64_t>(block_size / 2)}, input1_vals,
                         true);
  test.AddInput<float>("scales", {N, static_cast<int64_t>(kblks)}, input2_vals, true);
  if (is_asym) {
    test.AddInput<uint8_t>("zero_points", {N, static_cast<int64_t>(kblks / 2)}, input3_vals, true);
  }
  test.AddOutput<float>("Y", {M, N}, expected_vals, false);
  if (acc_lvl == 4) {
    test.SetOutputAbsErr("Y", 0.1f);
  }

  OrtValue b, scale, zp;
  Tensor::InitOrtValue(DataTypeImpl::GetType<uint8_t>(),
                       TensorShape({N, static_cast<int64_t>(kblks), static_cast<int64_t>(block_size / 2)}),
                       input1_vals.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), b);

  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape({N, static_cast<int64_t>(kblks)}),
                       input2_vals.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), scale);
  if (is_asym) {
    Tensor::InitOrtValue(DataTypeImpl::GetType<uint8_t>(), TensorShape({N, static_cast<int64_t>(kblks / 2)}),
                         input3_vals.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), zp);
  }
  SessionOptions so;
  // Set up B as a shared initializer to be shared between sessions
  ASSERT_EQ(so.AddInitializer("B", &b), Status::OK());
  ASSERT_EQ(so.AddInitializer("scales", &scale), Status::OK());
  if (is_asym) {
    ASSERT_EQ(so.AddInitializer("zero_points", &zp), Status::OK());
  }

  // We want all sessions running using this OpTester to be able to share pre-packed weights if applicable
  test.EnableSharingOfPrePackedWeightsAcrossSessions();

  // Pre-packing is limited just to the CPU EP for now and we will only test the CPU EP
  // and we want to ensure that it is available in this build
  auto cpu_ep = []() -> std::vector<std::unique_ptr<IExecutionProvider>> {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    return execution_providers;
  };

  size_t number_of_pre_packed_weights_counter_session_1 = 0;
  size_t number_of_shared_pre_packed_weights_counter = 0;

  // Session 1
  {
    auto ep_vec = cpu_ep();
    test.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &ep_vec, {},
             &number_of_pre_packed_weights_counter_session_1, &number_of_shared_pre_packed_weights_counter);
    // Assert that no pre-packed weights have been shared thus far
    ASSERT_EQ(number_of_shared_pre_packed_weights_counter, static_cast<size_t>(0));
  }

  auto number_of_elements_in_shared_prepacked_buffers_container = test.GetNumPrePackedWeightsShared();
  // Assert that the number of elements in the shared container
  // is the same as the number of weights that have been pre-packed
  ASSERT_EQ(number_of_pre_packed_weights_counter_session_1, number_of_elements_in_shared_prepacked_buffers_container);

  // On some platforms/architectures MLAS may choose to not do any pre-packing and the number of elements
  // that have been pre-packed will be zero in which case we do not continue with the testing
  // of "sharing" of pre-packed weights as there are no pre-packed weights to be shared at all.
  if (number_of_pre_packed_weights_counter_session_1 == 0) return;

  // Session 2
  {
    size_t number_of_pre_packed_weights_counter_session_2 = 0;
    auto ep_vec = cpu_ep();
    test.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &ep_vec, {},
             &number_of_pre_packed_weights_counter_session_2, &number_of_shared_pre_packed_weights_counter);

    // Assert that the same number of weights were pre-packed in both sessions
    ASSERT_EQ(number_of_pre_packed_weights_counter_session_1, number_of_pre_packed_weights_counter_session_2);

    // Assert that the number of pre-packed weights that were shared equals
    // the number of pre-packed weights in the second session
    ASSERT_EQ(number_of_pre_packed_weights_counter_session_2,
              static_cast<size_t>(number_of_shared_pre_packed_weights_counter));
  }
}

#ifdef ORT_NEURAL_SPEED
TEST(MatMulNBits, SharedPrepackedWeights) {
  RunSharedPrepackedWeightsTest(2, 4096, 4096, 32, true, 1);
  RunSharedPrepackedWeightsTest(2, 4096, 4096, 32, false, 1);
  RunSharedPrepackedWeightsTest(2, 4096, 4096, 128, false, 1);
  RunSharedPrepackedWeightsTest(2, 4096, 4096, 128, false, 4);
  RunSharedPrepackedWeightsTest(2, 4096, 4096, 1024, false, 4);
  RunSharedPrepackedWeightsTest(2, 4096, 4096, 4096, false, 4);
}
#endif
}  // namespace test
}  // namespace onnxruntime

#endif  // ORT_MINIMAL_BUILD
