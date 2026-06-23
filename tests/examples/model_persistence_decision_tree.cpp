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
#include <cstdio>
#include <iostream>

/*
 * Decision Tree Model Serialization Example
 *
 * This example demonstrates saving and loading a trained decision tree model.
 */

int main() {

    std::cout << "-----------------------------------------------------------------------"
              << std::endl;
    std::cout << "Decision Tree Model Serialization Example" << std::endl;
    std::cout << "Training, saving, and loading a decision tree model" << std::endl
              << std::endl;

    bool pass = true;

    // Training data: 8 samples, 2 features, 2 classes
    // Simple dataset for binary classification
    da_int n_samples = 8, n_features = 2, n_classes = 2;
    double X_train[16] = {1.0, 1.0, 1.5, 1.8, 2.0, 2.5, 5.0, 5.5,
                          6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5};
    da_int y_train[8] = {0, 0, 0, 0, 1, 1, 1, 1};

    // Test data: 4 samples
    da_int n_test = 4;
    double X_test[8] = {1.2, 1.6, 6.2, 8.8, 2.2, 7.3, 3.0, 8.0};

    da_int predictions_before[4];

    // Train and save original model
    {
        std::cout << "Training decision tree classifier..." << std::endl;
        da_handle handle_orig = nullptr;
        pass = pass && (da_handle_init_d(&handle_orig, da_handle_decision_tree) ==
                        da_status_success);
        pass = pass && (da_tree_set_training_data_d(
                            handle_orig, n_samples, n_features, n_classes, X_train,
                            n_samples, y_train, nullptr) == da_status_success);
        pass = pass &&
               (da_options_set_int(handle_orig, "maximum depth", 3) == da_status_success);
        pass = pass && (da_tree_fit_d(handle_orig) == da_status_success);

        std::cout << "Model trained successfully" << std::endl;

        // Make predictions before saving
        pass = pass && (da_tree_predict_d(handle_orig, n_test, n_features, X_test, n_test,
                                          predictions_before) == da_status_success);

        std::cout << "Predictions before save: ";
        for (da_int i = 0; i < n_test; i++) {
            std::cout << predictions_before[i] << " ";
        }
        std::cout << std::endl;

        // Save the trained model
        std::cout << "\nSaving model to 'decision_tree_model.bin'..." << std::endl;
        pass = pass && (da_handle_save_model(handle_orig, "decision_tree_model.bin") ==
                        da_status_success);

        if (!pass) {
            std::cerr << "Error: Failed to train, predict, or save model" << std::endl;
            da_handle_destroy(&handle_orig);
            return 1;
        }

        da_handle_destroy(&handle_orig);
    }

    // Load and use the saved model
    {
        std::cout << "Loading model from 'decision_tree_model.bin'..." << std::endl;
        da_handle handle_loaded = nullptr;
        pass = pass && (da_handle_load_model(&handle_loaded, "decision_tree_model.bin") ==
                        da_status_success);

        if (!pass) {
            std::cerr << "Error: Failed to load model" << std::endl;
            return 1;
        }

        // Make predictions with loaded model
        da_int predictions_after[4];
        pass =
            pass && (da_tree_predict_d(handle_loaded, n_test, n_features, X_test, n_test,
                                       predictions_after) == da_status_success);

        std::cout << "Predictions after load:  ";
        for (da_int i = 0; i < n_test; i++) {
            std::cout << predictions_after[i] << " ";
        }
        std::cout << std::endl;

        // Verify predictions match
        bool predictions_match = true;
        for (da_int i = 0; i < n_test; i++) {
            if (predictions_before[i] != predictions_after[i]) {
                predictions_match = false;
                break;
            }
        }

        if (!pass || !predictions_match) {
            std::cerr << "\nError: ";
            if (!pass)
                std::cerr << "Failed to make predictions with loaded model";
            else
                std::cerr << "Predictions do not match";
            std::cerr << std::endl;
            da_handle_destroy(&handle_loaded);
            return 1;
        }

        std::cout << "\nModel persistence verified - predictions match!" << std::endl;

        da_handle_destroy(&handle_loaded);
    }

    // Clean up created files
    std::remove("decision_tree_model.bin");

    std::cout << "\nDecision tree model serialization example completed successfully"
              << std::endl;

    std::cout << "-----------------------------------------------------------------------"
              << std::endl;

    return 0;
}
