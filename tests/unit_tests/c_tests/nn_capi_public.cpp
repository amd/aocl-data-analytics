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
 * Test the k-nearest neighbors C API (double precision).
 * Based on tests/examples/knn_classification.cpp
 * Covers: set_data, set_labels, set_targets, knn_search_mode, radius_neighbors,
 *         classes, classifier_predict, classifier_predict_proba, regressor_predict
 */
TEST(NnCAPI, ClassificationDouble) {
    da_handle handle = nullptr;

    // Training data: 6 samples, 3 features (column-major)
    double X_train[18] = {-1.0, -2.0, -3.0, 1.0, 2.0, 3.0,  -1.0, -1.0, -2.0,
                          3.0,  5.0,  -1.0, 2.0, 3.0, -1.0, 1.0,  1.0,  2.0};
    da_int y_train[6] = {1, 2, 0, 1, 2, 2};

    da_int n_samples = 6, n_features = 3, n_queries = 3, n_neigh = 3;

    EXPECT_EQ(da_handle_init_d(&handle, da_handle_nn), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "number of neighbors", n_neigh),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "metric", "euclidean"), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "weights", "uniform"), da_status_success);

    EXPECT_EQ(da_nn_set_data_d(handle, n_samples, n_features, X_train, n_samples),
              da_status_success);

    // Query k-nearest neighbors
    double X_test[9] = {-2.0, -1.0, 2.0, 2.0, -2.0, 1.0, 3.0, -1.0, -3.0};
    double k_dist[9];
    da_int k_ind[9];
    EXPECT_EQ(da_nn_kneighbors_d(handle, n_queries, n_features, X_test, n_queries, k_ind,
                                 k_dist, n_neigh, 1),
              da_status_success);

    // Set labels for classification
    EXPECT_EQ(da_nn_set_labels_d(handle, n_samples, y_train), da_status_success);

    // Get classes
    da_int n_classes = 0;
    EXPECT_EQ(da_nn_classes_d(handle, &n_classes, nullptr), da_status_success);
    EXPECT_GT(n_classes, 0);
    da_int classes[3];
    EXPECT_EQ(da_nn_classes_d(handle, &n_classes, classes), da_status_success);

    // Classifier predict
    da_int y_pred[3];
    EXPECT_EQ(da_nn_classifier_predict_d(handle, n_queries, n_features, X_test, n_queries,
                                         y_pred, knn_search_mode),
              da_status_success);

    // Classifier predict proba
    double proba[9]; // n_queries * n_classes
    EXPECT_EQ(da_nn_classifier_predict_proba_d(handle, n_queries, n_features, X_test,
                                               n_queries, proba, knn_search_mode),
              da_status_success);

    da_handle_destroy(&handle);
}

/*
 * Test kNN radius neighbors (double precision).
 */
TEST(NnCAPI, RadiusDouble) {
    da_handle handle = nullptr;

    double X_train[18] = {-1.0, -2.0, -3.0, 1.0, 2.0, 3.0,  -1.0, -1.0, -2.0,
                          3.0,  5.0,  -1.0, 2.0, 3.0, -1.0, 1.0,  1.0,  2.0};
    da_int n_samples = 6, n_features = 3;

    EXPECT_EQ(da_handle_init_d(&handle, da_handle_nn), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "metric", "euclidean"), da_status_success);
    EXPECT_EQ(da_nn_set_data_d(handle, n_samples, n_features, X_train, n_samples),
              da_status_success);

    // Radius neighbors query
    double X_test[3] = {0.0, 0.0, 0.0};
    da_int n_queries = 1;
    double radius = 5.0;
    EXPECT_EQ(da_nn_radius_neighbors_d(handle, n_queries, n_features, X_test, n_queries,
                                       radius, 1, 1),
              da_status_success);

    da_handle_destroy(&handle);
}

/*
 * Test kNN regression (double precision).
 */
TEST(NnCAPI, RegressionDouble) {
    da_handle handle = nullptr;

    double X_train[18] = {-1.0, -2.0, -3.0, 1.0, 2.0, 3.0,  -1.0, -1.0, -2.0,
                          3.0,  5.0,  -1.0, 2.0, 3.0, -1.0, 1.0,  1.0,  2.0};
    double y_targets[6] = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5};

    da_int n_samples = 6, n_features = 3, n_queries = 2, n_neigh = 3;

    EXPECT_EQ(da_handle_init_d(&handle, da_handle_nn), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "number of neighbors", n_neigh),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "metric", "euclidean"), da_status_success);
    EXPECT_EQ(da_nn_set_data_d(handle, n_samples, n_features, X_train, n_samples),
              da_status_success);

    // Set targets for regression
    EXPECT_EQ(da_nn_set_targets_d(handle, n_samples, y_targets), da_status_success);

    // Regressor predict
    double X_test[6] = {-2.0, 2.0, 1.0, -1.0, 3.0, -3.0};
    double y_pred[2];
    EXPECT_EQ(da_nn_regressor_predict_d(handle, n_queries, n_features, X_test, n_queries,
                                        y_pred, knn_search_mode),
              da_status_success);

    da_handle_destroy(&handle);
}

/*
 * Test the k-nearest neighbors C API (single precision).
 */
TEST(NnCAPI, ClassificationFloat) {
    da_handle handle = nullptr;

    float X_train[18] = {-1.0f, -2.0f, -3.0f, 1.0f, 2.0f, 3.0f,  -1.0f, -1.0f, -2.0f,
                         3.0f,  5.0f,  -1.0f, 2.0f, 3.0f, -1.0f, 1.0f,  1.0f,  2.0f};
    da_int y_train[6] = {1, 2, 0, 1, 2, 2};

    da_int n_samples = 6, n_features = 3, n_queries = 3, n_neigh = 3;

    EXPECT_EQ(da_handle_init_s(&handle, da_handle_nn), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "number of neighbors", n_neigh),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "metric", "euclidean"), da_status_success);
    EXPECT_EQ(da_nn_set_data_s(handle, n_samples, n_features, X_train, n_samples),
              da_status_success);

    float X_test[9] = {-2.0f, -1.0f, 2.0f, 2.0f, -2.0f, 1.0f, 3.0f, -1.0f, -3.0f};
    float k_dist[9];
    da_int k_ind[9];
    EXPECT_EQ(da_nn_kneighbors_s(handle, n_queries, n_features, X_test, n_queries, k_ind,
                                 k_dist, n_neigh, 1),
              da_status_success);

    EXPECT_EQ(da_nn_set_labels_s(handle, n_samples, y_train), da_status_success);

    // Get classes
    da_int n_classes = 0;
    EXPECT_EQ(da_nn_classes_s(handle, &n_classes, nullptr), da_status_success);
    EXPECT_GT(n_classes, 0);
    da_int classes[3];
    EXPECT_EQ(da_nn_classes_s(handle, &n_classes, classes), da_status_success);

    da_int y_pred[3];
    EXPECT_EQ(da_nn_classifier_predict_s(handle, n_queries, n_features, X_test, n_queries,
                                         y_pred, knn_search_mode),
              da_status_success);

    float proba[9];
    EXPECT_EQ(da_nn_classifier_predict_proba_s(handle, n_queries, n_features, X_test,
                                               n_queries, proba, knn_search_mode),
              da_status_success);

    da_handle_destroy(&handle);
}

/*
 * Test kNN regression (single precision).
 */
TEST(NnCAPI, RegressionFloat) {
    da_handle handle = nullptr;

    float X_train[18] = {-1.0f, -2.0f, -3.0f, 1.0f, 2.0f, 3.0f,  -1.0f, -1.0f, -2.0f,
                         3.0f,  5.0f,  -1.0f, 2.0f, 3.0f, -1.0f, 1.0f,  1.0f,  2.0f};
    float y_targets[6] = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f};

    da_int n_samples = 6, n_features = 3, n_queries = 2, n_neigh = 3;

    EXPECT_EQ(da_handle_init_s(&handle, da_handle_nn), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "number of neighbors", n_neigh),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "metric", "euclidean"), da_status_success);
    EXPECT_EQ(da_nn_set_data_s(handle, n_samples, n_features, X_train, n_samples),
              da_status_success);
    EXPECT_EQ(da_nn_set_targets_s(handle, n_samples, y_targets), da_status_success);

    float X_test[6] = {-2.0f, 2.0f, 1.0f, -1.0f, 3.0f, -3.0f};
    float y_pred[2];
    EXPECT_EQ(da_nn_regressor_predict_s(handle, n_queries, n_features, X_test, n_queries,
                                        y_pred, knn_search_mode),
              da_status_success);

    da_handle_destroy(&handle);
}

/*
 * Test radius neighbors (single precision).
 */
TEST(NnCAPI, RadiusFloat) {
    da_handle handle = nullptr;

    float X_train[18] = {-1.0f, -2.0f, -3.0f, 1.0f, 2.0f, 3.0f,  -1.0f, -1.0f, -2.0f,
                         3.0f,  5.0f,  -1.0f, 2.0f, 3.0f, -1.0f, 1.0f,  1.0f,  2.0f};
    da_int n_samples = 6, n_features = 3;

    EXPECT_EQ(da_handle_init_s(&handle, da_handle_nn), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "metric", "euclidean"), da_status_success);
    EXPECT_EQ(da_nn_set_data_s(handle, n_samples, n_features, X_train, n_samples),
              da_status_success);

    float X_test[3] = {0.0f, 0.0f, 0.0f};
    EXPECT_EQ(da_nn_radius_neighbors_s(handle, 1, n_features, X_test, 1, 5.0f, 1, 1),
              da_status_success);

    da_handle_destroy(&handle);
}
