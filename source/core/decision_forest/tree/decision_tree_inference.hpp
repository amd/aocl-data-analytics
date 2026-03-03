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

#ifndef TREE_INFERENCE_HPP
#define TREE_INFERENCE_HPP

#include "aoclda.h"
#include "decision_tree.hpp"
#include "macros.h"

namespace ARCH {
namespace da_decision_forest {

template <typename T>
da_status decision_tree<T>::predict(da_int nsamp, da_int nfeat, const T *X_test,
                                    da_int ldx_test, da_int *y_pred, da_int mode) {
    if (y_pred == nullptr) {
        return da_error_bypass(this->err, da_status_invalid_pointer,
                               "y_pred is not a valid pointer.");
    }

    const T *X_test_temp;
    T *utility_ptr1 = nullptr;
    da_int ldx_test_temp;

    if (nfeat != n_features) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "n_features = " + std::to_string(nfeat) +
                                   " doesn't match the expected value " +
                                   std::to_string(n_features) + ".");
    }

    if (!model_trained) {
        return da_error_bypass(this->err, da_status_out_of_date,
                               "The model has not yet been trained or the data it is "
                               "associated with is out of date.");
    }

    da_status status = this->store_2D_array(nsamp, nfeat, X_test, ldx_test, &utility_ptr1,
                                            &X_test_temp, ldx_test_temp, "n_samples",
                                            "n_features", "X_test", "ldx_test", mode);
    if (status != da_status_success)
        return status;

    // Fill y_pred with the values of all the requested samples
    node<T> *current_node;
    for (da_int i = 0; i < nsamp; i++) {
        current_node = &tree[0];
        while (!current_node->is_leaf) {
            if (current_node->prop == continuous ||
                current_node->prop == categorical_ordered) {
                T feat_val = X_test_temp[ldx_test_temp * current_node->feature + i];
                if (feat_val < current_node->x_threshold)
                    current_node = &tree[current_node->left_child_idx];
                else
                    current_node = &tree[current_node->right_child_idx];
            } else {
                da_int cat_val =
                    std::round(X_test_temp[ldx_test_temp * current_node->feature + i]);
                if (cat_val == current_node->category)
                    current_node = &tree[current_node->left_child_idx];
                else
                    current_node = &tree[current_node->right_child_idx];
            }
        }
        y_pred[i] = current_node->y_pred;
    }
    if (utility_ptr1)
        delete[] (utility_ptr1);
    return da_status_success;
}

template <typename T>
da_status decision_tree<T>::predict_proba(da_int nsamp, da_int nfeat, const T *X_test,
                                          da_int ldx_test, T *y_proba_pred, da_int nclass,
                                          da_int ldy, da_int mode) {

    const T *X_test_temp;
    T *utility_ptr1;
    T *utility_ptr2;
    da_int ldx_test_temp;
    T *y_proba_pred_temp;
    da_int ldy_proba_pred_temp;

    if (!predict_proba_opt) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "predict_proba must be set to 1");
    }

    if (nfeat != n_features) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "n_features = " + std::to_string(nfeat) +
                                   " doesn't match the expected value " +
                                   std::to_string(n_features) + ".");
    }

    if (nclass != n_class) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "n_features = " + std::to_string(nclass) +
                                   " doesn't match the expected value " +
                                   std::to_string(nclass) + ".");
    }

    if (!model_trained) {
        return da_error_bypass(this->err, da_status_out_of_date,
                               "The model has not yet been trained or the data it is "
                               "associated with is out of date.");
    }

    da_status status = this->store_2D_array(nsamp, nfeat, X_test, ldx_test, &utility_ptr1,
                                            &X_test_temp, ldx_test_temp, "n_samples",
                                            "n_features", "X_test", "ldx_test", mode);
    if (status != da_status_success)
        return status;

    da_int mode_output = (mode == 0) ? 1 : mode;
    status = this->store_2D_array(nsamp, nclass, y_proba_pred, ldy, &utility_ptr2,
                                  const_cast<const T **>(&y_proba_pred_temp),
                                  ldy_proba_pred_temp, "n_samples", "n_class", "y_proba",
                                  "ldy", mode_output);
    if (status != da_status_success)
        return status;

    // Fill y_proba_pred with the values of all the requested samples
    node<T> *current_node;
    for (da_int i = 0; i < nsamp; i++) {
        current_node = &tree[0];
        da_int current_node_idx = 0;
        while (!current_node->is_leaf) {
            T feat_val = X_test_temp[ldx_test_temp * current_node->feature + i];
            if (feat_val < current_node->x_threshold) {
                current_node_idx = current_node->left_child_idx;
                current_node = &tree[current_node_idx];
            } else {
                current_node_idx = current_node->right_child_idx;
                current_node = &tree[current_node_idx];
            }
        }
        for (da_int j = 0; j < n_class; j++)
            y_proba_pred_temp[ldy_proba_pred_temp * j + i] =
                class_props[n_class * current_node_idx + j];
    }

    if (this->order == row_major) {

        da_utils::copy_transpose_2D_array_column_to_row_major(
            nsamp, n_class, y_proba_pred_temp, ldy_proba_pred_temp, y_proba_pred, ldy);
        if (utility_ptr1)
            delete[] (utility_ptr1);
        if (utility_ptr2)
            delete[] (utility_ptr2);
    }
    return da_status_success;
}

template <typename T>
da_status decision_tree<T>::predict_log_proba(da_int nsamp, da_int nfeat, const T *X_test,
                                              da_int ldx_test, T *y_log_proba,
                                              da_int nclass, da_int ldy) {
    da_status status = da_status_success;

    status = predict_proba(nsamp, nfeat, X_test, ldx_test, y_log_proba, n_class, ldy);
    if (status != da_status_success)
        return status;

    if (this->order == column_major) {
        for (da_int j = 0; j < nclass; j++) {
            for (da_int i = 0; i < nsamp; i++) {
                y_log_proba[ldy * j + i] = log(y_log_proba[ldy * j + i]);
            }
        }
    } else {
        for (da_int j = 0; j < nsamp; j++) {
            for (da_int i = 0; i < nclass; i++) {
                y_log_proba[j * ldy + i] = log(y_log_proba[j * ldy + i]);
            }
        }
    }
    return status;
}

template <typename T>
da_status decision_tree<T>::score(da_int nsamp, da_int nfeat, const T *X_test,
                                  da_int ldx_test, const da_int *y_test, T *accuracy) {

    const T *X_test_temp;
    T *utility_ptr1 = nullptr;
    da_int ldx_test_temp;

    if (accuracy == nullptr) {
        return da_error_bypass(this->err, da_status_invalid_pointer,
                               "mean_accuracy is not valid pointers.");
    }

    if (nfeat != n_features) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "nfeat = " + std::to_string(nfeat) +
                                   " doesn't match the expected value " +
                                   std::to_string(n_features) + ".");
    }

    if (!model_trained) {
        return da_error_bypass(this->err, da_status_out_of_date,
                               "The model has not yet been trained or the data it is "
                               "associated with is out of date.");
    }

    da_status status = this->store_2D_array(nsamp, nfeat, X_test, ldx_test, &utility_ptr1,
                                            &X_test_temp, ldx_test_temp, "n_samples",
                                            "n_features", "X_test", "ldx_test");
    if (status != da_status_success)
        return status;

    status = this->check_1D_array(nsamp, y_test, "n_samples", "y_test", 1);
    if (status != da_status_success)
        return status;

    node<T> *current_node;
    *accuracy = 0.;
    for (da_int i = 0; i < nsamp; i++) {
        current_node = &tree[0];
        while (!current_node->is_leaf) {
            if (current_node->prop == continuous ||
                current_node->prop == categorical_ordered) {
                T feat_val = X_test_temp[ldx_test_temp * current_node->feature + i];
                if (feat_val < current_node->x_threshold)
                    current_node = &tree[current_node->left_child_idx];
                else
                    current_node = &tree[current_node->right_child_idx];
            } else {
                da_int cat_val =
                    std::round(X_test_temp[ldx_test_temp * current_node->feature + i]);
                if (cat_val == current_node->category)
                    current_node = &tree[current_node->left_child_idx];
                else
                    current_node = &tree[current_node->right_child_idx];
            }
        }

        if (current_node->y_pred == y_test[i])
            *accuracy += (T)1.0;
    }
    *accuracy = *accuracy / (T)nsamp;
    if (utility_ptr1)
        delete[] (utility_ptr1);

    return da_status_success;
}

} // namespace da_decision_forest
} // namespace ARCH

#endif