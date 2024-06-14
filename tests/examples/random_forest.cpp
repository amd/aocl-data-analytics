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
#include <algorithm>
#include <iostream>
#include <vector>

#ifndef DATA_DIR
#define DATA_DIR "data"
#endif

int main() {
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Random forest model (double precision)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << std::fixed;
    std::cout.precision(5);

    da_status status;
    bool pass;

    // Read in training data
    da_datastore csv_store = nullptr;
    std::string filename = DATA_DIR;
    filename += "/decision_train.csv";
    da_int n_cols = 0, n_rows = 0;
    pass = da_datastore_init(&csv_store) == da_status_success;
    pass &= da_datastore_options_set_string(csv_store, "CSV datastore precision",
                                            "single") == da_status_success;
    pass &= da_data_load_from_csv(csv_store, filename.c_str()) == da_status_success;
    pass &= da_data_get_n_cols(csv_store, &n_cols) == da_status_success;
    pass &= da_data_get_n_rows(csv_store, &n_rows) == da_status_success;
    pass &=
        da_data_select_columns(csv_store, "features", 0, n_cols - 2) == da_status_success;
    pass &= da_data_select_columns(csv_store, "labels", n_cols - 1, n_cols - 1) ==
            da_status_success;
    da_int n_features = n_cols - 1;
    da_int n_samples = n_rows;
    std::vector<float> X(n_features * n_samples);
    std::vector<da_int> y(n_rows);
    pass &= da_data_extract_selection_real_s(csv_store, "features", X.data(), n_rows) ==
            da_status_success;
    pass &= da_data_extract_selection_int(csv_store, "labels", y.data(), n_rows) ==
            da_status_success;
    da_int n_class = *std::max_element(y.begin(), y.end()) + 1;
    da_datastore_destroy(&csv_store);
    if (!pass) {
        std::cout << "Something went wrong while loading data.\n";
        return 1;
    }

    // Initialize the decision forest class and fit model
    da_handle forest_handle = nullptr;
    pass =
        da_handle_init_s(&forest_handle, da_handle_decision_forest) == da_status_success;
    pass &=
        da_forest_set_training_data_s(forest_handle, n_samples, n_features, n_class,
                                      X.data(), n_samples, y.data()) == da_status_success;
    pass &=
        da_options_set_int(forest_handle, "number of trees", 100) == da_status_success;
    pass &= da_options_set_int(forest_handle, "seed", 42) == da_status_success;
    pass &= da_options_set_int(forest_handle, "maximum features", 5) == da_status_success;
    pass &= da_options_set_string(forest_handle, "scoring function", "gini") ==
            da_status_success;
    pass &= da_options_set_string(forest_handle, "bootstrap", "yes") == da_status_success;
    if (!pass) {
        std::cout << "Something went wrong setting up the decision tree data and "
                     "optional parameters.\n";
        return 1;
    }

    status = da_forest_fit_s(forest_handle);
    if (status != da_status_success) {
        std::cout << "Failure while fitting the trees.\n";
        return 1;
    }

    // Read in data for making predictions
    filename = DATA_DIR;
    filename += "/decision_test.csv";
    pass = da_datastore_init(&csv_store) == da_status_success;
    pass &= da_datastore_options_set_string(csv_store, "CSV datastore precision",
                                            "single") == da_status_success;
    pass &= da_data_load_from_csv(csv_store, filename.c_str()) == da_status_success;
    pass &= da_data_get_n_rows(csv_store, &n_rows) == da_status_success;
    pass &=
        da_data_select_columns(csv_store, "features", 0, n_cols - 2) == da_status_success;
    pass &= da_data_select_columns(csv_store, "labels", n_cols - 1, n_cols - 1) ==
            da_status_success;
    n_samples = n_rows;
    std::vector<float> X_test(n_features * n_samples);
    std::vector<da_int> y_test(n_samples);
    pass &= da_data_extract_selection_real_s(csv_store, "features", X_test.data(),
                                             n_samples) == da_status_success;
    pass &= da_data_extract_selection_int(csv_store, "labels", y_test.data(),
                                          n_samples) == da_status_success;
    da_datastore_destroy(&csv_store);
    if (!pass) {
        std::cout << "Something went wrong while loading test data.\n";
        return 1;
    }

    // Make predictions with model and evaluate score
    std::vector<da_int> y_pred(n_samples);
    float mean_accuracy;
    status = da_forest_predict_s(forest_handle, n_samples, n_features, X_test.data(),
                                 n_samples, y_pred.data());
    status = da_forest_score_s(forest_handle, n_samples, n_features, X_test.data(),
                               n_samples, y_test.data(), &mean_accuracy);
    std::cout << "Mean accuracy on the test data: " << mean_accuracy << std::endl;

    da_handle_destroy(&forest_handle);
    return 0;
}
