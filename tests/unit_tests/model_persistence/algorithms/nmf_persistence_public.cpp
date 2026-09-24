/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "../../utest_utils.hpp"
#include "../persistence_test_utils.hpp"
#include "aoclda.h"
#include "aoclda_cpp_overloads.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

// Test parameters for NMF serialization tests
struct nmf_serial_params {
    std::string test_name;
    std::string init;
    std::string storage_order;
    da_int n_components;
};

void PrintTo(const nmf_serial_params &param, ::std::ostream *os) {
    *os << param.test_name;
}

// Configurations covering different init methods and storage orders.
const std::vector<nmf_serial_params> nmf_serialization_params = {
    {"nndsvda_colmaj_2comp", "nndsvda", "column-major", 2},
    {"nndsvd_rowmaj_2comp", "nndsvd", "row-major", 2},
    {"random_colmaj_3comp", "random", "column-major", 3},
    {"nndsvdar_rowmaj_2comp", "nndsvdar", "row-major", 2},
};

class NMFSerializationTest : public testing::TestWithParam<nmf_serial_params> {
  protected:
    std::string model_file;

    da_int n_samples = 5;
    da_int n_features = 4;
    da_int lda = 5;
    std::vector<double> A_colmaj = {1, 2, 3, 4, 5, 2, 4, 6,  1, 3,
                                    3, 6, 9, 5, 8, 4, 8, 12, 2, 4};

    // Row-major layout of the same matrix (lda = n_features = 4)
    std::vector<double> A_rowmaj = {1, 2,  3, 4, 2, 4, 6, 8, 3, 6,
                                    9, 12, 4, 1, 5, 2, 5, 3, 8, 4};

    da_int m_samples = 2;
    da_int ldx = 2;
    std::vector<double> X_new_colmaj = {2.0, 1.0, 3.0, 1.0, 5.0, 2.0, 4.0, 1.0};

    std::vector<double> X_new_rowmaj = {2.0, 3.0, 5.0, 4.0, 1.0, 1.0, 2.0, 1.0};

    void SetUp() override {
        const nmf_serial_params &pr = GetParam();
        const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string test_case = test_info->name();
        std::replace(test_case.begin(), test_case.end(), '/', '_');
        model_file = model_persistence_test_utils::get_test_file_dir() + "/nmf_" +
                     pr.test_name + "_" + test_case + ".bin";
    }

    void TearDown() override { std::remove(model_file.c_str()); }
};

template <typename T>
void nmf_serialization_test(const nmf_serial_params &pr, da_int n_samples,
                            da_int n_features, da_int lda_colmaj,
                            const std::vector<double> &A_colmaj_d,
                            const std::vector<double> &A_rowmaj_d, da_int m_samples,
                            da_int ldx_colmaj, const std::vector<double> &X_new_colmaj_d,
                            const std::vector<double> &X_new_rowmaj_d,
                            const std::string &model_file) {

    const bool row_major = (pr.storage_order == "row-major");

    // Select the appropriate layout for training and transform input
    const std::vector<double> &A_d = row_major ? A_rowmaj_d : A_colmaj_d;
    da_int lda = row_major ? n_features : lda_colmaj;

    const std::vector<double> &X_d = row_major ? X_new_rowmaj_d : X_new_colmaj_d;
    da_int ldx = row_major ? n_features : ldx_colmaj;

    // Convert to type T
    std::vector<T> A(A_d.begin(), A_d.end());
    std::vector<T> X_new(X_d.begin(), X_d.end());

    const da_int k = pr.n_components;

    // Leading dimensions for transform/inverse_transform outputs respect storage order:
    //   column-major: ld = n_rows;  row-major: ld = n_cols
    const da_int lxt = row_major ? k : m_samples; // W_new (m_samples x k)
    const da_int lxi =
        row_major ? n_features : m_samples; // X_inv (m_samples x n_features)

    // Result arrays for the original model
    std::vector<T> W_orig(n_samples * k);
    std::vector<T> H_orig(k * n_features);
    std::vector<T> transform_orig(m_samples * k);
    std::vector<T> inv_transform_orig(m_samples * n_features);
    T recon_err_orig = T{0};
    da_int n_iter_orig = 0;

    // ==================== ORIGINAL MODEL BLOCK ====================
    {
        da_handle handle = nullptr;
        EXPECT_EQ(da_handle_init<T>(&handle, da_handle_nmf), da_status_success);

        EXPECT_EQ(
            da_options_set_string(handle, "storage order", pr.storage_order.c_str()),
            da_status_success);
        EXPECT_EQ(da_nmf_init(handle, n_samples, n_features, A.data(), lda),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "n_components", k), da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "init", pr.init.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "random_state", 42), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "max_iter", 200), da_status_success);
        EXPECT_EQ(da_options_set(handle, "tol", (T)1.0e-6), da_status_success);

        EXPECT_EQ(da_nmf_compute<T>(handle), da_status_success);

        da_int w_dim = n_samples * k;
        EXPECT_EQ(da_handle_get_result(handle, da_nmf_w, &w_dim, W_orig.data()),
                  da_status_success);

        da_int h_dim = k * n_features;
        EXPECT_EQ(da_handle_get_result(handle, da_nmf_h, &h_dim, H_orig.data()),
                  da_status_success);

        da_int err_dim = 1;
        EXPECT_EQ(da_handle_get_result(handle, da_nmf_reconstruction_err, &err_dim,
                                       &recon_err_orig),
                  da_status_success);

        da_int niter_dim = 1;
        EXPECT_EQ(
            da_handle_get_result_int(handle, da_nmf_n_iter, &niter_dim, &n_iter_orig),
            da_status_success);

        // transform: X_new -> W_new (m_samples x k)
        EXPECT_EQ(da_nmf_transform(handle, m_samples, n_features, X_new.data(), ldx,
                                   transform_orig.data(), lxt),
                  da_status_success);

        // inverse_transform: W_new -> X_inv (m_samples x n_features)
        EXPECT_EQ(da_nmf_inverse_transform(handle, m_samples, k, transform_orig.data(),
                                           lxt, inv_transform_orig.data(), lxi),
                  da_status_success);

        EXPECT_EQ(da_handle_save_model(handle, model_file.c_str()), da_status_success);
        da_handle_destroy(&handle);
    }

    // ==================== LOADED MODEL BLOCK ====================
    {
        da_handle handle_loaded = nullptr;
        EXPECT_EQ(da_handle_load_model(&handle_loaded, model_file.c_str()),
                  da_status_success);

        std::vector<T> W_loaded(n_samples * k);
        std::vector<T> H_loaded(k * n_features);
        std::vector<T> transform_loaded(m_samples * k);
        std::vector<T> inv_transform_loaded(m_samples * n_features);
        T recon_err_loaded = T{0};
        da_int n_iter_loaded = 0;

        da_int w_dim = n_samples * k;
        EXPECT_EQ(da_handle_get_result(handle_loaded, da_nmf_w, &w_dim, W_loaded.data()),
                  da_status_success);

        da_int h_dim = k * n_features;
        EXPECT_EQ(da_handle_get_result(handle_loaded, da_nmf_h, &h_dim, H_loaded.data()),
                  da_status_success);

        da_int err_dim = 1;
        EXPECT_EQ(da_handle_get_result(handle_loaded, da_nmf_reconstruction_err, &err_dim,
                                       &recon_err_loaded),
                  da_status_success);

        da_int niter_dim = 1;
        EXPECT_EQ(da_handle_get_result_int(handle_loaded, da_nmf_n_iter, &niter_dim,
                                           &n_iter_loaded),
                  da_status_success);

        EXPECT_EQ(da_nmf_transform(handle_loaded, m_samples, n_features, X_new.data(),
                                   ldx, transform_loaded.data(), lxt),
                  da_status_success);

        EXPECT_EQ(da_nmf_inverse_transform(handle_loaded, m_samples, k,
                                           transform_loaded.data(), lxt,
                                           inv_transform_loaded.data(), lxi),
                  da_status_success);

        // ==================== COMPARE RESULTS ====================

        EXPECT_ARR_EQ(n_samples * k, W_orig.data(), W_loaded.data(), 1, 1, 0, 0);
        EXPECT_ARR_EQ(k * n_features, H_orig.data(), H_loaded.data(), 1, 1, 0, 0);
        EXPECT_EQ(recon_err_orig, recon_err_loaded);
        EXPECT_EQ(n_iter_orig, n_iter_loaded);
        EXPECT_ARR_EQ(m_samples * k, transform_orig.data(), transform_loaded.data(), 1, 1,
                      0, 0);
        EXPECT_ARR_EQ(m_samples * n_features, inv_transform_orig.data(),
                      inv_transform_loaded.data(), 1, 1, 0, 0);

        // Repeat transform on loaded model to check stability
        std::vector<T> transform_again(m_samples * k);
        EXPECT_EQ(da_nmf_transform(handle_loaded, m_samples, n_features, X_new.data(),
                                   ldx, transform_again.data(), lxt),
                  da_status_success);
        EXPECT_ARR_EQ(m_samples * k, transform_orig.data(), transform_again.data(), 1, 1,
                      0, 0);

        // ==================== VERIFY OPTIONS ====================

        char init_loaded[64];
        da_int init_len = 64;
        EXPECT_EQ(da_options_get_string(handle_loaded, "init", init_loaded, &init_len),
                  da_status_success);
        EXPECT_STREQ(init_loaded, pr.init.c_str());

        da_int n_components_loaded = 0;
        EXPECT_EQ(da_options_get_int(handle_loaded, "n_components", &n_components_loaded),
                  da_status_success);
        EXPECT_EQ(n_components_loaded, k);

        char order_loaded[64];
        da_int order_len = 64;
        EXPECT_EQ(da_options_get_string(handle_loaded, "storage order", order_loaded,
                                        &order_len),
                  da_status_success);
        EXPECT_STREQ(order_loaded, pr.storage_order.c_str());

        model_persistence_test_utils::test_print_model_versions(handle_loaded);

        da_handle_destroy(&handle_loaded);
    }
}

TEST_P(NMFSerializationTest, double) {
    const nmf_serial_params &pr = GetParam();
    nmf_serialization_test<double>(pr, n_samples, n_features, lda, A_colmaj, A_rowmaj,
                                   m_samples, ldx, X_new_colmaj, X_new_rowmaj,
                                   model_file);
}

TEST_P(NMFSerializationTest, float) {
    const nmf_serial_params &pr = GetParam();
    nmf_serialization_test<float>(pr, n_samples, n_features, lda, A_colmaj, A_rowmaj,
                                  m_samples, ldx, X_new_colmaj, X_new_rowmaj, model_file);
}

INSTANTIATE_TEST_SUITE_P(NMFSerializationSuite, NMFSerializationTest,
                         testing::ValuesIn(nmf_serialization_params));

// ==================== ERROR HANDLING TESTS ====================

class NMFSerializationErrorTest : public testing::Test {
  protected:
    std::string model_file;
    void SetUp() override {
        const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string test_name = test_info->name();
        model_file = model_persistence_test_utils::get_test_file_dir() + "/nmf_error_" +
                     test_name + ".bin";
    }
    void TearDown() override { std::remove(model_file.c_str()); }
};

TEST_F(NMFSerializationErrorTest, SaveBeforeComputeFails) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_nmf), da_status_success);
    std::vector<double> A = {1, 2, 3, 4, 2, 4, 6, 8};
    EXPECT_EQ(da_nmf_init_d(handle, 4, 2, A.data(), 4), da_status_success);
    EXPECT_EQ(da_handle_save_model(handle, model_file.c_str()), da_status_no_data);
    da_handle_destroy(&handle);
}

TEST_F(NMFSerializationErrorTest, ComputeAfterLoadWithoutInitFails) {
    // Train and save
    da_handle handle_train = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle_train, da_handle_nmf), da_status_success);
    std::vector<double> A = {1, 2, 3, 4, 5, 2, 4, 6, 1, 3, 3, 6, 9, 5, 8, 4, 8, 12, 2, 4};
    EXPECT_EQ(da_nmf_init_d(handle_train, 5, 4, A.data(), 5), da_status_success);
    EXPECT_EQ(da_options_set_int(handle_train, "n_components", 2), da_status_success);
    EXPECT_EQ(da_nmf_compute_d(handle_train), da_status_success);
    EXPECT_EQ(da_handle_save_model(handle_train, model_file.c_str()), da_status_success);
    da_handle_destroy(&handle_train);

    // Load and try to compute without calling da_nmf_init
    da_handle handle_load = nullptr;
    EXPECT_EQ(da_handle_load_model(&handle_load, model_file.c_str()), da_status_success);
    EXPECT_EQ(da_nmf_compute_d(handle_load), da_status_no_data);
    da_handle_destroy(&handle_load);
}

TEST_F(NMFSerializationErrorTest, PrintModelMetadataFromFile) {
    da_handle handle = nullptr;
    ASSERT_EQ(da_handle_init_d(&handle, da_handle_nmf), da_status_success);
    std::vector<double> A = {1, 2, 3, 4, 5, 2, 4, 6, 1, 3, 3, 6, 9, 5, 8, 4, 8, 12, 2, 4};
    ASSERT_EQ(da_nmf_init_d(handle, 5, 4, A.data(), 5), da_status_success);
    ASSERT_EQ(da_options_set_int(handle, "n_components", 2), da_status_success);
    ASSERT_EQ(da_nmf_compute_d(handle), da_status_success);
    ASSERT_EQ(da_handle_save_model(handle, model_file.c_str()), da_status_success);
    da_handle_destroy(&handle);

    testing::internal::CaptureStdout();
    EXPECT_EQ(da_print_model_metadata(model_file.c_str()), da_status_success);
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_NE(output.find("Header keyword:"), std::string::npos);
    EXPECT_NE(output.find("Serialization version:"), std::string::npos);
    EXPECT_NE(output.find("Saved AOCL-DA build version:"), std::string::npos);
}

TEST_F(NMFSerializationErrorTest, PrintModelMetadataFromBuffer) {
    // Save to an in-memory buffer (float variant to exercise the single-precision path)
    da_handle handle = nullptr;
    ASSERT_EQ(da_handle_init_s(&handle, da_handle_nmf), da_status_success);
    std::vector<float> A = {1, 2, 3, 4, 5, 2, 4, 6, 1, 3, 3, 6, 9, 5, 8, 4, 8, 12, 2, 4};
    ASSERT_EQ(da_nmf_init_s(handle, 5, 4, A.data(), 5), da_status_success);
    ASSERT_EQ(da_options_set_int(handle, "n_components", 2), da_status_success);
    ASSERT_EQ(da_nmf_compute_s(handle), da_status_success);

    std::vector<char> buffer;
    ASSERT_EQ(da_handle_save_model(handle, buffer), da_status_success);
    da_handle_destroy(&handle);

    testing::internal::CaptureStdout();
    EXPECT_EQ(da_print_model_metadata(buffer), da_status_success);
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_NE(output.find("Header keyword:"), std::string::npos);
    EXPECT_NE(output.find("Serialization version:"), std::string::npos);
    EXPECT_NE(output.find("Saved AOCL-DA build version:"), std::string::npos);
}

TEST_F(NMFSerializationErrorTest, SetDataAfterLoadFails) {
    // After loading, A is not available so da_nmf_set_data must fail
    da_handle handle_train = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle_train, da_handle_nmf), da_status_success);
    std::vector<double> A = {1, 2, 3, 4, 5, 2, 4, 6, 1, 3, 3, 6, 9, 5, 8, 4, 8, 12, 2, 4};
    EXPECT_EQ(da_nmf_init_d(handle_train, 5, 4, A.data(), 5), da_status_success);
    EXPECT_EQ(da_options_set_int(handle_train, "n_components", 2), da_status_success);
    EXPECT_EQ(da_nmf_compute_d(handle_train), da_status_success);
    EXPECT_EQ(da_handle_save_model(handle_train, model_file.c_str()), da_status_success);
    da_handle_destroy(&handle_train);

    da_handle handle_load = nullptr;
    EXPECT_EQ(da_handle_load_model(&handle_load, model_file.c_str()), da_status_success);

    // W0 and H0 for custom init (values don't matter since we expect an error)
    std::vector<double> W0(5 * 2, 1.0);
    std::vector<double> H0(2 * 4, 1.0);
    EXPECT_EQ(da_nmf_set_data_d(handle_load, 5, 2, W0.data(), 5, 4, H0.data(), 2),
              da_status_no_data);
    da_handle_destroy(&handle_load);
}
