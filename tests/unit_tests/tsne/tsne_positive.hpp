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
 */

#ifndef TSNE_POSITIVE_HPP
#define TSNE_POSITIVE_HPP

#include "../utest_utils.hpp"
#include "aoclda.h"
#include "tsne_utils.hpp"
#include "gtest/gtest.h"
#include <cmath>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

/**
 * Run t-SNE quality test with target solution comparison.
 *
 * This function:
 * 1. Loads input data from CSV
 * 2. Runs AOCL-DA t-SNE
 * 3. Computes quality metrics (trustworthiness, KL divergence)
 * 4. Compares against target solution metrics (passed as parameters)
 *
 * Note: target_kl_div <= 0 skips KL divergence comparison (useful for tiny datasets)
 */
template <typename T>
void test_tsne_quality(const std::string &data_name, da_int n_components, T perplexity,
                       da_int max_iter, T theta, const std::string &init, da_int seed,
                       da_int k_neighbors, T target_trustworthiness, T target_kl_div,
                       T check_tol_scale) {

    std::string data_file =
        std::string(DATA_DIR) + "/tsne_data/" + data_name + "_data.csv";
    std::vector<T> X;
    da_int n_samples, n_features;
    ASSERT_TRUE(da_test::read_csv_data(data_file, X, n_samples, n_features, row_major))
        << "Failed to read data file: " << data_file;
    ASSERT_GT(n_samples, 0) << "No samples in data file";
    ASSERT_GT(n_features, 0) << "No features in data file";

    // Create t-SNE handle
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_tsne), da_status_success);

    // Set options
    EXPECT_EQ(da_options_set_int(handle, "n_components", n_components),
              da_status_success);
    EXPECT_EQ(da_options_set(handle, "perplexity", perplexity), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "max_iter", max_iter), da_status_success);
    EXPECT_EQ(da_options_set(handle, "theta", theta), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "init", init.c_str()), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "seed", seed), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "storage order", "row-major"),
              da_status_success);
    EXPECT_EQ(da_tsne_set_data(handle, n_samples, n_features, X.data(), n_features),
              da_status_success);

    // Compute t-SNE
    EXPECT_EQ(da_tsne_compute<T>(handle), da_status_success);

    // Get embedding
    da_int emb_dim = n_samples * n_components;
    std::vector<T> embedding(emb_dim);
    EXPECT_EQ(da_handle_get_result(handle, da_tsne_embedding, &emb_dim, embedding.data()),
              da_status_success);

    // Get info (KL divergence)
    da_int info_dim = 6;
    T rinfo[6];
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &info_dim, rinfo),
              da_status_success);
    T kl_divergence = rinfo[4];

    // Check embedding validity
    EXPECT_TRUE(
        tsne_metrics::check_embedding_finite(embedding.data(), n_samples, n_components))
        << "Embedding contains non-finite values";

    // Compute AOCL-DA quality metrics
    T trustworthiness = tsne_metrics::compute_trustworthiness(
        X.data(), embedding.data(), n_samples, n_features, n_components, k_neighbors);

    // Use a common base absolute tolerance across precisions.
    const T base_abs_tol = T(0.01);
    T effective_abs_tol = base_abs_tol * check_tol_scale;

    T min_trustworthiness = std::max(T(0), target_trustworthiness - effective_abs_tol);
    T max_kl_div = target_kl_div + effective_abs_tol;

    // Print results
    std::cout << "\n  Dataset: " << data_name
              << ", Precision: " << (std::is_same_v<T, float> ? "float" : "double")
              << "\n  AOCL-DA Trustworthiness: " << trustworthiness
              << " (target: " << target_trustworthiness
              << ", min: " << min_trustworthiness << ")"
              << "\n  AOCL-DA KL divergence: " << kl_divergence;
    if (target_kl_div > T(0)) {
        std::cout << " (target: " << target_kl_div << ", max: " << max_kl_div << ")";
    }
    std::cout << std::endl;

    // Assert compliance with target solution
    EXPECT_GE(trustworthiness, min_trustworthiness)
        << "AOCL-DA trustworthiness " << trustworthiness << " is below target "
        << target_trustworthiness << " by more than absolute tolerance "
        << effective_abs_tol;

    // Compare KL divergence against target solution (skip if target_kl_div <= 0)
    if (target_kl_div > T(0)) {
        EXPECT_LE(kl_divergence, max_kl_div) << "AOCL-DA KL divergence " << kl_divergence
                                             << " exceeds max allowed " << max_kl_div;
    }

    EXPECT_GT(kl_divergence, T(0)) << "KL divergence should be positive";
    EXPECT_TRUE(std::isfinite(kl_divergence)) << "KL divergence should be finite";

    // Cleanup
    da_handle_destroy(&handle);
}

#endif // TSNE_POSITIVE_HPP
