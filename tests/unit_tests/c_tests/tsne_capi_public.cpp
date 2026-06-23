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

/*
 * Test the t-SNE C API (double precision).
 * Based on tests/examples/tsne.cpp
 */
TEST(TsneCAPI, BasicDouble) {
    da_handle handle = nullptr;

    // Input data: 6 samples, 3 features (column-major)
    double X[18] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.1, 1.9, 3.1,
                    4.1, 5.2, 5.8, 0.9, 2.2, 2.9, 4.2, 4.9, 6.2};
    da_int n_samples = 6, n_features = 3, ldx = 6;

    EXPECT_EQ(da_handle_init_d(&handle, da_handle_tsne), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_components", 2), da_status_success);
    EXPECT_EQ(da_options_set_real_d(handle, "perplexity", 2.0), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "max_iter", 300), da_status_success);
    EXPECT_EQ(da_options_set_real_d(handle, "theta", 0.0), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "seed", 42), da_status_success);

    EXPECT_EQ(da_tsne_set_data_d(handle, n_samples, n_features, X, ldx),
              da_status_success);

    // Set initial embedding
    double Y_init[12] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1};
    EXPECT_EQ(da_tsne_set_init_embedding_d(handle, Y_init, n_samples), da_status_success);

    EXPECT_EQ(da_tsne_compute_d(handle), da_status_success);

    // Get embedding result
    da_int n_components = 2;
    da_int emb_dim = n_samples * n_components;
    double embedding[12];
    EXPECT_EQ(da_handle_get_result_d(handle, da_tsne_embedding, &emb_dim, embedding),
              da_status_success);

    da_handle_destroy(&handle);
}

/*
 * Test the t-SNE C API (single precision).
 */
TEST(TsneCAPI, BasicFloat) {
    da_handle handle = nullptr;

    float X[18] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 1.1f, 1.9f, 3.1f,
                   4.1f, 5.2f, 5.8f, 0.9f, 2.2f, 2.9f, 4.2f, 4.9f, 6.2f};
    da_int n_samples = 6, n_features = 3, ldx = 6;

    EXPECT_EQ(da_handle_init_s(&handle, da_handle_tsne), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_components", 2), da_status_success);
    EXPECT_EQ(da_options_set_real_s(handle, "perplexity", 2.0f), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "max_iter", 300), da_status_success);
    EXPECT_EQ(da_options_set_real_s(handle, "theta", 0.0f), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "seed", 42), da_status_success);

    EXPECT_EQ(da_tsne_set_data_s(handle, n_samples, n_features, X, ldx),
              da_status_success);

    float Y_init[12] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                        0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f};
    EXPECT_EQ(da_tsne_set_init_embedding_s(handle, Y_init, n_samples), da_status_success);

    EXPECT_EQ(da_tsne_compute_s(handle), da_status_success);

    da_int n_components = 2;
    da_int emb_dim = n_samples * n_components;
    float embedding[12];
    EXPECT_EQ(da_handle_get_result_s(handle, da_tsne_embedding, &emb_dim, embedding),
              da_status_success);

    da_handle_destroy(&handle);
}
