/*
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

// #include "../../source/core/decision_forest/decision_forest.hpp"

#include <random>

#include "gtest/gtest.h"

template <typename T>
da_status convert_2d_array_r_major_to_c_major(da_int n_row, da_int n_col, T *a_in,
                                              da_int lda, T *a_out) {
    da_status status = da_status_success;

    for (da_int i = 0; i < n_row; i++) {
        for (da_int j = 0; j < n_col; j++) {
            // a_in is row major (contiguous over j for fixed i)
            // a_out is column major (contiguous over i for fixed j)
            // outer loop is over rows so wthis code should read contiguous block from a_in
            // and do scattered write to a_out
            a_out[i + (j * n_row)] = a_in[(i * lda) + j];
        }
    }

    return status;
}

// TEST(decision_forest, cpp_api_sample_features) {
//     // Initialize the decision forest class and fit model
//     da_handle df_handle = nullptr;
//     da_status status;
//     status = da_handle_init_s(&df_handle, da_handle_decision_forest);

//     status = da_options_set_int(df_handle, "n_features", 20);
//     // status = da_options_set_int(df_handle, "seed", 1201);

//     da_int n = 10;

//     da_df_sample_features_s(df_handle, n);

// }

// TEST(decision_forest, cpp_api_generate_trees)
// {
//     da_int n_features          = 20;
//     da_int n_features_per_tree = 5;

//     da_int n_trees = 4;

//     da_int n_samples;

//     da_handle df_handle = nullptr;

//     da_status status;
//     status = da_handle_init_s(&df_handle, da_handle_decision_forest);

//     da_df_bootstrap_obs_s(df_handle, n_trees, n_features_per_tree);

// }

TEST(decision_forest, cpp_api_sample_features) {

    da_datastore csv_handle;
    da_status status;

    // Read in training data
    csv_handle = nullptr;
    status = da_datastore_init(&csv_handle);

    char features_fp[256] = DATA_DIR;
    strcat(features_fp, "df_data/");
    strcat(features_fp, "training_features");
    strcat(features_fp, ".csv");

    char labels_fp[256] = DATA_DIR;
    strcat(labels_fp, "df_data/");
    strcat(labels_fp, "training_labels");
    strcat(labels_fp, ".csv");

    float *x_r_major = nullptr, *x = nullptr;
    uint8_t *y = nullptr;
    da_int n_obs = 0, d = 0, nrows_y = 0, ncols_y = 0;
    // read in x (row major)
    status = da_read_csv_s(csv_handle, features_fp, &x_r_major, &n_obs, &d, nullptr);
    // read in y
    status = da_read_csv_uint8(csv_handle, labels_fp, &y, &nrows_y, &ncols_y, nullptr);

    // convert x from row major to column major
    x = new float[n_obs * d];
    status = convert_2d_array_r_major_to_c_major(n_obs, d, x_r_major, d, x);

    // Initialize the decision forest class and fit model
    da_handle df_handle = nullptr;
    status = da_handle_init_s(&df_handle, da_handle_decision_forest);

    status = da_options_set_int(df_handle, "seed", 988);
    status = da_options_set_int(df_handle, "n_obs_per_tree", 100);
    // status = da_options_set_int(df_handle, "n_obs_per_tree", n_obs);
    status = da_options_set_int(df_handle, "n_features_to_select", 3);
    //status = da_options_set_int(df_handle, "n_features_to_select", d);
    status = da_options_set_int(df_handle, "n_trees", 20);

    // copy x and y into df class members and convert x, y from row major to column major
    // ldx is leading domension of column-major input
    // (i.e. stride between columns in 2-d column-major array)
    status = da_df_set_training_data_s(df_handle, n_obs, d, x, n_obs, y);

    std::cout << "----------------------------------------" << std::endl;
    if (status == da_status_success) {
        std::cout << "Setup complete." << std::endl;
    } else {
        std::cout << "Something wrong happened during training setup." << std::endl;
    }

    // status = da_options_set_int(df_handle, "seed", 1201);

    da_df_fit_s(df_handle);

    std::cout << "----------------------------------------" << std::endl;
    if (status == da_status_success) {
        std::cout << "Fitting complete." << std::endl;
    } else {
        std::cout << "Something wrong happened during fitting." << std::endl;
    }

    // Read in data for making predictions
    char test_features_fp[256] = DATA_DIR;
    strcat(test_features_fp, "/df_data/");
    strcat(test_features_fp, "test_features");
    strcat(test_features_fp, ".csv");

    char test_labels_fp[256] = DATA_DIR;
    strcat(test_labels_fp, "/df_data/");
    strcat(test_labels_fp, "test_labels");
    strcat(test_labels_fp, ".csv");

    float *x_test = nullptr;
    uint8_t *y_test = nullptr;
    n_obs = 0;
    d = 0;
    nrows_y = 0;
    ncols_y = 0;

    status = da_read_csv_s(csv_handle, test_features_fp, &x_test, &n_obs, &d, nullptr);
    status = da_read_csv_uint8(csv_handle, test_labels_fp, &y_test, &nrows_y, &ncols_y,
                               nullptr);

    // Make predictions with model and evaluate score
    std::vector<uint8_t> y_pred(n_obs);
    status = da_df_predict_s(df_handle, n_obs, x_test, y_pred.data());

    float score = 0.0;
    status = da_df_score_s(df_handle, n_obs, x_test, y_test, &score);

    if (status == da_status_success) {
        std::cout << "Scoring complete." << std::endl;
        std::cout << "Score    = " << score << std::endl;
    } else {
        std::cout << "Something wrong happened during prediction setup or scoring."
                  << std::endl;
    }

    if (x_test)
        free(x_test);

    if (y_test)
        free(y_test);

    if (x)
        delete[] x;

    if (x_r_major)
        free(x_r_major);

    if (y)
        free(y);

    da_datastore_destroy(&csv_handle);
    da_handle_destroy(&df_handle);
}
