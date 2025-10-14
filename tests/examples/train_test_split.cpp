/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

/*
Train test split example

This example demonstrates how to use train_test_split to perform K-fold cross-validation.
*/

#include "aoclda_cpp_overloads.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

int main() {
    da_int pass = true;
    da_int exit_code = 0;

    // Define matrix
    std::vector<double> X = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3,
                             3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6,
                             6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9};
    da_order order = row_major;
    da_int m = 10;
    da_int n = 5;
    da_int ldx = 5;
    da_int ldx_train = 5;
    da_int ldx_test = 5;

    // Create and shuffle indices of the data
    std::vector<da_int> indices(m);
    std::iota(indices.begin(), indices.end(), 0);
    da_int seed = 1;
    da_status status_shuffle =
        da_get_shuffled_indices_int(m, seed, 0, 0, 10, nullptr, indices.data());

    pass = pass && (status_shuffle == da_status_success);
    if (!pass) {
        std::cout << "Error while performing the shuffling of indices !" << '\n';
    }

    // Define fold sizes
    // Leave remainder to the test fold so all the training folds are of equal size
    da_int n_folds = 3;
    da_int train_fold_size = m / n_folds;
    da_int test_fold_size = (m / n_folds);

    // Expected folds
    std::vector<std::vector<double>> folds_exp_train = {
        {1, 1, 1, 1, 1, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0, 6, 6, 6,
         6, 6, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4},
        {3, 3, 3, 3, 3, 9, 9, 9, 9, 9, 2, 2, 2, 2, 2, 6, 6, 6,
         6, 6, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4},
        {3, 3, 3, 3, 3, 9, 9, 9, 9, 9, 2, 2, 2, 2, 2,
         1, 1, 1, 1, 1, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0}};

    std::vector<std::vector<double>> folds_exp_test = {
        {3, 3, 3, 3, 3, 9, 9, 9, 9, 9, 2, 2, 2, 2, 2},
        {1, 1, 1, 1, 1, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0},
        {6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4}};

    // Implementation of the cross-validation
    for (da_int i = 0; i < n_folds; ++i) {
        std::vector<da_int> indices_kfold;
        indices_kfold.reserve(m);

        if ((i == n_folds - 1) && ((m % n_folds) > 0)) {
            test_fold_size += m % n_folds;
        }
        da_int test_start = i * train_fold_size;
        da_int test_end = test_start + test_fold_size;

        for (da_int j = 0; j < m; ++j) {
            if (j < test_start || j >= test_end)
                indices_kfold.push_back(indices[j]);
        }
        for (da_int j = test_start; j < test_end; ++j) {
            indices_kfold.push_back(indices[j]);
        }

        // Split the data into train and test
        std::vector<double> X_train((m - test_fold_size) * n);
        std::vector<double> X_test((test_fold_size)*n, 0.0);

        da_status status_split = da_train_test_split(
            order, m, n, X.data(), ldx, m - test_fold_size, test_fold_size,
            indices_kfold.data(), X_train.data(), ldx_train, X_test.data(), ldx_test);

        pass = pass && (status_split == da_status_success);
        if (!pass) {
            std::cout << "Error while performing the split on fold " << i + 1 << "!"
                      << '\n';
        }

        std::cout << '\n' << "Fold " << i + 1 << ":" << '\n';
        // Print train and test matrices.
        std::cout << "X_train" << '\n';
        for (da_int j = 0; j < (m - test_fold_size); ++j) {
            for (da_int k = 0; k < n; ++k) {
                std::cout << X_train[j * n + k];
                if (X_train[j * n + k] != folds_exp_train[i][j * n + k]) {
                    pass = false;
                }
            }
            std::cout << '\n';
        }
        std::cout << '\n';
        std::cout << "X_test" << '\n';
        for (da_int j = 0; j < test_fold_size; ++j) {
            for (da_int k = 0; k < n; ++k) {
                std::cout << X_test[j * n + k];
                if (X_test[j * n + k] != folds_exp_test[i][j * n + k]) {
                    pass = false;
                }
            }
            std::cout << '\n';
        }
        if (!pass) {
            std::cout << '\n'
                      << "Incorrect results for X_train and X_test for Fold " << i + 1
                      << "!" << '\n';
        }
    }

    if (pass) {
        std::cout << '\n'
                  << "Cross Validation Train-Test splitting was successful!" << '\n';
    } else {
        exit_code = 1;
    }

    return exit_code;
}