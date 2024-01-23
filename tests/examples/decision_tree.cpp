/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <iostream>
#include <vector>

#ifndef DATA_DIR
#define DATA_DIR "data"
#endif

void decision_tree_ex_d() {
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Decision tree model (double precision)" << std::endl;
    std::cout << std::fixed;
    std::cout.precision(5);

    da_handle df_handle;
    da_datastore csv_handle;
    da_status status;
    bool pass = true;

    // Read in training data
    csv_handle = nullptr;
    status = da_datastore_init(&csv_handle);
    pass = pass && (status == da_status_success);

    char features_fp[256] = DATA_DIR;
    strcat(features_fp, "/df_data/");
    strcat(features_fp, "training_features");
    strcat(features_fp, ".csv");

    char labels_fp[256] = DATA_DIR;
    strcat(labels_fp, "/df_data/");
    strcat(labels_fp, "training_labels");
    strcat(labels_fp, ".csv");

    double *x = nullptr;
    uint8_t *y = nullptr;
    da_int n_obs = 0, d = 0, nrows_y = 0, ncols_y = 0;
    status = da_read_csv_d(csv_handle, features_fp, &x, &n_obs, &d, nullptr);
    pass = pass && (status == da_status_success);
    status = da_read_csv_uint8(csv_handle, labels_fp, &y, &nrows_y, &ncols_y, nullptr);
    pass = pass && (status == da_status_success);

    // Initialize the decision tree class and fit model
    df_handle = nullptr;
    status = da_handle_init_d(&df_handle, da_handle_decision_tree);
    pass = pass && (status == da_status_success);
    status = da_df_set_training_data_d(df_handle, n_obs, d, x, n_obs, y);
    pass = pass && (status == da_status_success);

    status = da_options_set_int(df_handle, "depth", 5);
    pass = pass && (status == da_status_success);
    status = da_options_set_int(df_handle, "seed", 77);
    pass = pass && (status == da_status_success);
    status = da_options_set_int(df_handle, "n_features_to_select", d);
    pass = pass && (status == da_status_success);
    status = da_options_set_string(df_handle, "scoring function", "gini");
    pass = pass && (status == da_status_success);
    status = da_df_fit_d(df_handle);
    pass = pass && (status == da_status_success);

    std::cout << "----------------------------------------" << std::endl;
    if (pass) {
        std::cout << "Fitting complete." << std::endl;
    } else {
        std::cout << "Something wrong happened during training setup or fitting."
                  << std::endl;
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

    double *x_test = nullptr;
    uint8_t *y_test = nullptr;
    n_obs = 0;
    d = 0;
    nrows_y = 0;
    ncols_y = 0;

    status = da_read_csv_d(csv_handle, test_features_fp, &x_test, &n_obs, &d, nullptr);
    pass = pass && (status == da_status_success);
    status = da_read_csv_uint8(csv_handle, test_labels_fp, &y_test, &nrows_y, &ncols_y,
                               nullptr);
    pass = pass && (status == da_status_success);

    // Make predictions with model and evaluate score
    std::vector<uint8_t> y_pred(n_obs);
    status = da_df_predict_d(df_handle, n_obs, d, x_test, n_obs, y_pred.data());
    pass = pass && (status == da_status_success);
    double score = 0.0;
    status = da_df_score_d(df_handle, n_obs, d, x_test, n_obs, y_test, &score);
    pass = pass && (status == da_status_success);

    std::cout << "----------------------------------------" << std::endl;
    if (pass) {
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
        free(x);

    if (y)
        free(y);

    da_datastore_destroy(&csv_handle);
    da_handle_destroy(&df_handle);
}

int main() {
    decision_tree_ex_d();
    return 0;
}
