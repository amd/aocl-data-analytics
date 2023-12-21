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

int decision_forest_ex_s(std::string score_criteria) {
    int exit_code = 0;
    double tol = 1.0e-6;

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Decision forest model (single precision)" << std::endl;
    std::cout << "Scoring Criteria: " << score_criteria << std::endl;
    std::cout << std::fixed;
    std::cout.precision(5);

    da_datastore csv_handle;
    da_status status;

    // Read in training data
    csv_handle = nullptr;
    status = da_datastore_init(&csv_handle);
    status = da_datastore_options_set_string(csv_handle, "CSV data storage", "row major");

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
    status = da_options_set_string(df_handle, "scoring function", score_criteria.data());

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

    float *x_test = nullptr, *x_test_r_major = nullptr;
    uint8_t *y_test = nullptr;
    n_obs = 0;
    d = 0;
    nrows_y = 0;
    ncols_y = 0;

    status =
        da_read_csv_s(csv_handle, test_features_fp, &x_test_r_major, &n_obs, &d, nullptr);
    status = da_read_csv_uint8(csv_handle, test_labels_fp, &y_test, &nrows_y, &ncols_y,
                               nullptr);

    x_test = new float[n_obs * d];
    status = convert_2d_array_r_major_to_c_major(n_obs, d, x_test_r_major, d, x_test);

    // Make predictions with model and evaluate score
    std::vector<uint8_t> y_pred(n_obs);
    status = da_df_predict_s(df_handle, n_obs, x_test, n_obs, y_pred.data());

    float score = 0.0;
    status = da_df_score_s(df_handle, n_obs, x_test, n_obs, y_test, &score);

    if (status == da_status_success) {
        std::cout << "Scoring complete." << std::endl;
        std::cout << "Score    = " << score << std::endl;
        double score_exp = 0.0;
        if (score_criteria == "gini") {
            score_exp = 0.93250;
            std::cout << "Expected = " << score_exp << std::endl;
        } else if (score_criteria == "cross-entropy") {
            score_exp = 0.94250;
            std::cout << "Expected = " << score_exp << std::endl;
        } else if (score_criteria == "misclassification-error") {
            score_exp = 0.93750;
            std::cout << "Expected = " << score_exp << std::endl;
        }
        // Check result
        double err = std::abs(score - score_exp);
        if (err > tol) {
            std::cout << "Solution is not within the expected tolerance: " << err
                      << std::endl;
            exit_code = 1;
        }
    } else {
        std::cout << "Something wrong happened during prediction setup or scoring."
                  << std::endl;
    }

    if (x_test_r_major)
        free(x_test_r_major);

    if (x_test)
        delete[] x_test;

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

    return exit_code;
}

int decision_forest_ex_d(std::string score_criteria) {
    int exit_code = 0;
    double tol = 1.0e-6;

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Decision forest model (double precision)" << std::endl;
    std::cout << "Scoring Criteria: " << score_criteria << std::endl;
    std::cout << std::fixed;
    std::cout.precision(5);

    da_datastore csv_handle;
    da_status status;

    // Read in training data
    csv_handle = nullptr;
    status = da_datastore_init(&csv_handle);
    status = da_datastore_options_set_string(csv_handle, "CSV data storage", "row major");

    char features_fp[256] = DATA_DIR;
    strcat(features_fp, "df_data/");
    strcat(features_fp, "training_features");
    strcat(features_fp, ".csv");

    char labels_fp[256] = DATA_DIR;
    strcat(labels_fp, "df_data/");
    strcat(labels_fp, "training_labels");
    strcat(labels_fp, ".csv");

    double *x_r_major = nullptr, *x = nullptr;
    uint8_t *y = nullptr;
    da_int n_obs = 0, d = 0, nrows_y = 0, ncols_y = 0;
    // read in x (row major)
    status = da_read_csv_d(csv_handle, features_fp, &x_r_major, &n_obs, &d, nullptr);
    // read in y
    status = da_read_csv_uint8(csv_handle, labels_fp, &y, &nrows_y, &ncols_y, nullptr);

    // convert x from row major to column major
    x = new double[n_obs * d];
    status = convert_2d_array_r_major_to_c_major(n_obs, d, x_r_major, d, x);

    // Initialize the decision forest class and fit model
    da_handle df_handle = nullptr;
    status = da_handle_init_d(&df_handle, da_handle_decision_forest);

    status = da_options_set_int(df_handle, "seed", 988);
    status = da_options_set_int(df_handle, "n_obs_per_tree", 100);
    // status = da_options_set_int(df_handle, "n_obs_per_tree", n_obs);
    status = da_options_set_int(df_handle, "n_features_to_select", 3);
    //status = da_options_set_int(df_handle, "n_features_to_select", d);
    status = da_options_set_int(df_handle, "n_trees", 20);
    status = da_options_set_string(df_handle, "scoring function", score_criteria.data());

    // copy x and y into df class members and convert x, y from row major to column major
    // ldx is leading domension of column-major input
    // (i.e. stride between columns in 2-d column-major array)
    status = da_df_set_training_data_d(df_handle, n_obs, d, x, n_obs, y);

    std::cout << "----------------------------------------" << std::endl;
    if (status == da_status_success) {
        std::cout << "Setup complete." << std::endl;
    } else {
        std::cout << "Something wrong happened during training setup." << std::endl;
    }

    // status = da_options_set_int(df_handle, "seed", 1201);

    da_df_fit_d(df_handle);

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

    double *x_test = nullptr, *x_test_r_major = nullptr;
    uint8_t *y_test = nullptr;
    n_obs = 0;
    d = 0;
    nrows_y = 0;
    ncols_y = 0;

    status =
        da_read_csv_d(csv_handle, test_features_fp, &x_test_r_major, &n_obs, &d, nullptr);
    status = da_read_csv_uint8(csv_handle, test_labels_fp, &y_test, &nrows_y, &ncols_y,
                               nullptr);

    x_test = new double[n_obs * d];
    status = convert_2d_array_r_major_to_c_major(n_obs, d, x_test_r_major, d, x_test);

    // Make predictions with model and evaluate score
    std::vector<uint8_t> y_pred(n_obs);
    status = da_df_predict_d(df_handle, n_obs, x_test, n_obs, y_pred.data());

    double score = 0.0;
    status = da_df_score_d(df_handle, n_obs, x_test, n_obs, y_test, &score);

    if (status == da_status_success) {
        std::cout << "Scoring complete." << std::endl;
        std::cout << "Score    = " << score << std::endl;
        double score_exp = 0.0;
        if (score_criteria == "gini") {
            score_exp = 0.93250;
            std::cout << "Expected = " << score_exp << std::endl;
        } else if (score_criteria == "cross-entropy") {
            score_exp = 0.94250;
            std::cout << "Expected = " << score_exp << std::endl;
        } else if (score_criteria == "misclassification-error") {
            score_exp = 0.93750;
            std::cout << "Expected = " << score_exp << std::endl;
        }
        // Check result
        double err = std::abs(score - score_exp);
        if (err > tol) {
            std::cout << "Solution is not within the expected tolerance: " << err
                      << std::endl;
            exit_code = 1;
        }
    } else {
        std::cout << "Something wrong happened during prediction setup or scoring."
                  << std::endl;
    }

    if (x_test_r_major)
        free(x_test_r_major);

    if (x_test)
        delete[] x_test;

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

    return exit_code;
}

int main() {
    int exit_code = 0;

    std::vector<std::string> param_vec = {"gini", "cross-entropy",
                                          "misclassification-error"};

    for (std::string score_criteria : param_vec) {
        exit_code = decision_forest_ex_d(score_criteria);
        if (exit_code != 0) {
            return exit_code;
        }

        exit_code = decision_forest_ex_s(score_criteria);
        if (exit_code != 0) {
            return exit_code;
        }
    }

    return exit_code;
}
