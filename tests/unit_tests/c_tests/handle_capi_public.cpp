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

#include "aoclda.h"
#include "gtest/gtest.h"
#include <cstdio>
#include <string>
#ifdef _WIN32
#include <direct.h>
#define DA_MKDIR(dir) _mkdir(dir)
#else
#include <sys/stat.h>
#define DA_MKDIR(dir) mkdir(dir, 0755)
#endif

static std::string get_test_output_dir() {
#ifdef TEST_OUTPUT_DIR
    struct stat st;
    // Try the absolute compile-time path first
    if (stat(TEST_OUTPUT_DIR, &st) == 0)
        return TEST_OUTPUT_DIR;
#endif
    // Fall back to a relative directory
    const char *fallback = "tmp_test_files";
    struct stat st2;
    if (stat(fallback, &st2) != 0)
        DA_MKDIR(fallback);
    return fallback;
}

/*
 * Tests for handle management APIs (aoclda_handle.h)
 */
TEST(HandleCAPI, InitDestroyDouble) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_pca), da_status_success);
    EXPECT_NE(handle, nullptr);
    da_handle_destroy(&handle);
    EXPECT_EQ(handle, nullptr);
}

TEST(HandleCAPI, InitDestroyFloat) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_s(&handle, da_handle_kmeans), da_status_success);
    EXPECT_NE(handle, nullptr);
    da_handle_destroy(&handle);
    EXPECT_EQ(handle, nullptr);
}

TEST(HandleCAPI, PrintErrorMessage) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_linmod), da_status_success);

    EXPECT_EQ(da_handle_print_error_message(handle), da_status_success);

    da_handle_destroy(&handle);
}

TEST(HandleCAPI, GetErrorMessage) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_linmod), da_status_success);

    // Trigger an error by calling compute without setting data
    EXPECT_NE(da_linmod_fit_d(handle), da_status_success);

    char *message = nullptr;
    EXPECT_EQ(da_handle_get_error_message(handle, &message), da_status_success);
    EXPECT_NE(message, nullptr);
    free(message);

    da_handle_destroy(&handle);
}

TEST(HandleCAPI, GetErrorSeverity) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_pca), da_status_success);

    da_severity severity;
    EXPECT_EQ(da_handle_get_error_severity(handle, &severity), da_status_success);

    da_handle_destroy(&handle);
}

TEST(HandleCAPI, Refresh) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_pca), da_status_success);
    da_handle_refresh(handle);
    da_handle_destroy(&handle);
}

TEST(HandleCAPI, SaveModel) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_kmeans), da_status_success);

    double A[8] = {1.0, 2.0, -1.0, -2.0, 1.0, 2.0, -1.0, -2.0};
    EXPECT_EQ(da_options_set_int(handle, "n_clusters", 2), da_status_success);
    EXPECT_EQ(da_kmeans_set_data_d(handle, 4, 2, A, 4), da_status_success);
    EXPECT_EQ(da_kmeans_compute_d(handle), da_status_success);

    std::string model_file = get_test_output_dir() + "/test_capi_model.bin";
    da_status status = da_handle_save_model(handle, model_file.c_str());
    EXPECT_EQ(status, da_status_success);

    da_handle_destroy(&handle);
    std::remove(model_file.c_str());
}

TEST(HandleCAPI, LoadModel) {
    // First save a model
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_kmeans), da_status_success);

    double A[8] = {1.0, 2.0, -1.0, -2.0, 1.0, 2.0, -1.0, -2.0};
    EXPECT_EQ(da_options_set_int(handle, "n_clusters", 2), da_status_success);
    EXPECT_EQ(da_kmeans_set_data_d(handle, 4, 2, A, 4), da_status_success);
    EXPECT_EQ(da_kmeans_compute_d(handle), da_status_success);

    std::string model_file = get_test_output_dir() + "/test_capi_load_model.bin";
    EXPECT_EQ(da_handle_save_model(handle, model_file.c_str()), da_status_success);
    da_handle_destroy(&handle);

    // Load it back
    da_handle loaded_handle = nullptr;
    da_status status = da_handle_load_model(&loaded_handle, model_file.c_str());
    EXPECT_EQ(status, da_status_success);

    if (loaded_handle) {
        da_handle_destroy(&loaded_handle);
    }
    std::remove(model_file.c_str());
}
