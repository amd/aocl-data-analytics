/*
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef SVM_HPP
#define SVM_HPP

#include "aoclda.h"
#include "basic_handle.hpp"
#include "c_svm.hpp"
#include "da_error.hpp"
#include "moment_statistics.hpp"
#include "nu_svm.hpp"
#include "options.hpp"
#include "svm_options.hpp"
#include "svm_types.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

/*
 * SVM handle class that contains definitions to all the user 
 * facing functionalities like set_data(), 
 */
namespace da_svm {

template <typename T> class svm : public basic_handle<T> {
  private:
    // Pointers to SVM problem class that will be specialised
    std::vector<std::unique_ptr<base_svm<T>>> classifiers;

    da_int n_class, n_classifiers;
    // Only used in multi-class classification
    std::vector<da_int> class_sizes;

    // Utility pointer to column major allocated copy of user's data
    T *X_temp = nullptr;

    const T *X = nullptr;
    const T *y = nullptr;
    da_int nrow, ncol;

    da_int ldx_train;

    // Set true when user data is loaded
    bool loadingdone = false;
    // Set true when SVM is computed successfully
    bool iscomputed = false;
    bool ismulticlass = false;

    da_svm_model mod = svm_undefined;

    // Results
    std::vector<bool> is_sv; // only used for multiclass
    da_int n_sv = 0;
    std::vector<T> support_coefficients, support_vectors, bias;
    std::vector<da_int> support_indexes, n_sv_per_class;

  public:
    svm(da_errors::da_error_t &err) : basic_handle<T>(err) {
        // Initialize the options registry
        // Any error is stored err->status[.] and this needs to be checked
        // by the caller.
        register_svm_options<T>(this->opts, err);
    }
    ~svm() {
        // Destructor needs to handle arrays that were allocated due to row major storage of input data
        if (X_temp)
            delete[] (X_temp);
    };

    // Main functions
    da_status set_data(da_int n, da_int p, const T *X, da_int ldx_train, const T *y);
    da_status select_model(da_svm_model mod);
    da_status compute();
    da_status predict(da_int nsamples, da_int nfeat, const T *X_test, da_int ldx_test,
                      T *predictions);
    da_status decision_function(da_int nsamples, da_int nfeat, const T *X_test,
                                da_int ldx_test, T *decision_values, da_int ldd,
                                da_svm_decision_function_shape shape = ovr);
    da_status score(da_int nsamples, da_int nfeat, const T *X_test, da_int ldx_test,
                    const T *y_test, T *score);

    void refresh() { iscomputed = false; }

    /* get_result (required to be defined by basic_handle) */
    da_status get_result(da_result query, da_int *dim, T *result) {
        // Don't return anything if SVM has not been computed
        if (!iscomputed) {
            return da_warn(this->err, da_status_unknown_query,
                           "SVM has not yet been computed. Please call da_svm_compute_s "
                           "or da_svm_compute_d before extracting results.");
        }

        da_int rinfo_size = 3;
        da_int size;

        if (result == nullptr) {
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The results array has not been allocated.");
        }
        switch (query) {
        case da_result::da_rinfo:
            if (*dim < rinfo_size) {
                *dim = rinfo_size;
                return da_warn(this->err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(rinfo_size) + ".");
            }
            result[0] = (T)nrow;
            result[1] = (T)ncol;
            result[2] = (T)n_class;
            break;
        case da_result::da_svm_dual_coef:
            size = (n_class - 1) * n_sv;
            if (*dim < size) {
                *dim = size;
                return da_warn(this->err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(size) + ".");
            }
            this->copy_2D_results_array(n_class - 1, n_sv, support_coefficients.data(),
                                        n_class - 1, result);
            break;
        case da_result::da_svm_support_vectors:
            size = n_sv * ncol;
            if (*dim < size) {
                *dim = size;
                return da_warn(this->err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(size) + ".");
            }
            this->copy_2D_results_array(n_sv, ncol, support_vectors.data(), n_sv, result);
            break;
        case da_result::da_svm_bias:
            size = n_classifiers;
            if (*dim < size) {
                *dim = size;
                return da_warn(this->err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(size) + ".");
            }
            for (da_int i = 0; i < n_classifiers; i++)
                result[i] = bias[i];
            break;
        default:
            return da_warn(this->err, da_status_unknown_query,
                           "The requested result could not be found.");
        }
        return da_status_success;
    }

    da_status get_result(da_result query, da_int *dim, da_int *result) {
        // Don't return anything if SVM has not been computed
        if (!iscomputed) {
            return da_warn(this->err, da_status_unknown_query,
                           "SVM has not yet been computed. Please call da_svm_compute_s "
                           "or da_svm_compute_d before extracting results.");
        }
        da_int size;
        switch (query) {
        case da_result::da_svm_n_support_vectors:
            size = 1;
            if (*dim < size) {
                *dim = size;
                return da_warn(this->err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(size) + ".");
            }
            result[0] = n_sv;
            break;
        case da_result::da_svm_n_support_vectors_per_class:
            if (mod == da_svm_model::svc || mod == da_svm_model::nusvc)
                size = n_class;
            else
                size = 1;
            if (*dim < size) {
                *dim = size;
                return da_warn(this->err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(size) + ".");
            }
            if (mod == da_svm_model::svc || mod == da_svm_model::nusvc)
                for (da_int i = 0; i < size; i++)
                    result[i] = n_sv_per_class[i];
            else
                result[0] = n_sv;
            break;
        case da_result::da_svm_idx_support_vectors:
            size = n_sv;
            if (*dim < size) {
                *dim = size;
                return da_warn(this->err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(size) + ".");
            }
            for (da_int i = 0; i < size; i++)
                result[i] = support_indexes[i];
            break;
        default:
            return da_warn(this->err, da_status_unknown_query,
                           "The requested result could not be found.");
        }
        return da_status_success;
    };
};

/* Store the user's data matrix in preparation for SVM computation */
template <typename T>
da_status svm<T>::set_data(da_int n, da_int p, const T *X_in, da_int ldx_train,
                           const T *y_in) {
    nrow = n;
    ncol = p;
    try {
        is_sv.resize(n);
    } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }
    // Guard against memory leaks due to multiple calls using the same class instantiation
    if (X_temp) {
        delete[] (X_temp);
        X_temp = nullptr;
        X = nullptr;
    }

    if (mod == da_svm_model::svm_undefined)
        return da_error(this->err, da_status_unknown_query,
                        "SVM model has not been selected.");

    da_status status =
        this->store_2D_array(n, p, X_in, ldx_train, &X_temp, &X, this->ldx_train,
                             "n_samples", "n_features", "X", "ldx");
    if (status != da_status_success)
        return status;

    status = this->check_1D_array(n, y_in, "n", "y", 1);
    if (status != da_status_success)
        return status;

    y = y_in;

    if (mod == da_svm_model::svc || mod == da_svm_model::nusvc) {
        // y is assumed to only contain values from 0 to K-1 (K being the number of classes)
        n_class = (da_int)(std::round(*std::max_element(y, y + n)) + 1);
        n_classifiers = n_class * (n_class - 1) / 2;
        try {
            classifiers.resize(n_classifiers);
            class_sizes.resize(n_class);
            bias.resize(n_classifiers);
        } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation error");
        }
        da_int k = 0;
        // Layout of OVO problems in classifiers vector is: 0v1, 0v2, ..., 0v(k-1), 1v2, 1v3, ... etc.
        if (n_class > 2) {
            ismulticlass = true;
            for (da_int i = 0; i < n_class; i++)
                for (da_int j = i + 1; j < n_class; j++) {
                    if (mod == da_svm_model::svc)
                        classifiers[k] = std::make_unique<svc<T>>();
                    else
                        classifiers[k] = std::make_unique<nusvc<T>>();
                    da_int size = 0, pos_size = 0;
                    // Vector that will store all indexes whose label is i or j (reserve bigger size here and then copy only necessary)
                    std::vector<da_int> all_idx(n);
                    classifiers[k]->XUSR = X;
                    classifiers[k]->yusr = y;
                    classifiers[k]->ldx = this->ldx_train;
                    classifiers[k]->p = p;
                    classifiers[k]->ismulticlass = true;
                    classifiers[k]->pos_class = i;
                    classifiers[k]->neg_class = j;
                    // Pick all indexes where label is i or j
                    for (da_int row = 0; row < n; row++)
                        if (y[row] == i || y[row] == j)
                            all_idx[size++] = row;

                    // Among the picked indexes assign ones where class==i as positive
                    std::vector<bool> is_positive(size, false);
                    std::vector<da_int> pos_idx(size), neg_idx(size);
                    for (da_int idx = 0; idx < size; idx++) {
                        if (y[all_idx[idx]] == i) {
                            is_positive[idx] = true;
                            pos_idx[pos_size++] = all_idx[idx];
                        }
                    }
                    try {
                        classifiers[k]->idx_class.resize(size);
                        classifiers[k]->idx_is_positive.resize(size);
                    } catch (std::bad_alloc &) { // LCOV_EXCL_LINE
                        return da_error(this->err,
                                        da_status_memory_error, // LCOV_EXCL_LINE
                                        "Memory allocation error");
                    }
                    std::copy_n(all_idx.begin(), size, classifiers[k]->idx_class.begin());
                    std::copy_n(is_positive.begin(), size,
                                classifiers[k]->idx_is_positive.begin());
                    classifiers[k]->n = size;
                    class_sizes[i] = pos_size;
                    k++;
                }
        } else {
            ismulticlass = false;
            if (mod == da_svm_model::svc)
                classifiers[0] = std::make_unique<svc<T>>();
            else
                classifiers[0] = std::make_unique<nusvc<T>>();
            classifiers[0]->XUSR = X;
            classifiers[0]->yusr = y;
            classifiers[0]->ldx = this->ldx_train;
            classifiers[0]->n = n;
            classifiers[0]->p = p;
        }

    } else {
        ismulticlass = false;
        n_class = 2;
        n_classifiers = 1;
        try {
            classifiers.resize(1);
            bias.resize(1);
        } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation error");
        }
        if (mod == da_svm_model::svr)
            classifiers[0] = std::make_unique<svr<T>>();
        else
            classifiers[0] = std::make_unique<nusvr<T>>();
        classifiers[0]->XUSR = X;
        classifiers[0]->yusr = y;
        classifiers[0]->ldx = this->ldx_train;
        classifiers[0]->n = n;
        classifiers[0]->p = p;
    }
    // Record that initialization is complete but computation has not yet been performed
    loadingdone = true;
    iscomputed = false;

    return da_status_success;
}

template <typename T> da_status svm<T>::select_model(da_svm_model mod) {

    // reset model_trained only if the model is changed
    if (mod != this->mod) {
        if (mod == da_svm_model::svc || mod == da_svm_model::svr ||
            mod == da_svm_model::nusvc || mod == da_svm_model::nusvr) {
            this->mod = mod;
            iscomputed = false;
            loadingdone = false;
        } else {
            return da_error(this->err, da_status_unknown_query,
                            "Unknown model requested.");
        }
    }
    return da_status_success;
}

/* Compute SVM */
template <typename T> da_status svm<T>::compute() {
    da_status status;
    std::string kernel_string;
    std::string order_string;
    if (!loadingdone)
        return da_error(this->err, da_status_no_data,
                        "No data has been passed to the handle. Please call "
                        "da_svm_set_data_s or da_svm_set_data_d.");
    // Here is logic to get default gamma 1/(p*var(X))
    T gamma_temp;
    this->opts.get("gamma", gamma_temp);
    if (gamma_temp < 0) {
        T mean, variance = 1;
        da_basic_statistics::variance(column_major, da_axis_all, nrow, ncol, X, ldx_train,
                                      -1, &mean, &variance);
        gamma_temp = 1 / (ncol * variance);
    }

    // Necessary to reset when svm is called multiple times on the same handle
    std::fill(is_sv.begin(), is_sv.end(), false);
    n_sv = 0;
    try {
        n_sv_per_class.resize(n_class);
    } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }
    std::fill(n_sv_per_class.begin(), n_sv_per_class.end(), 0);

    for (da_int i = 0; i < n_classifiers; i++) {

        this->opts.get("kernel", kernel_string, classifiers[i]->kernel_function);
        this->opts.get("C", classifiers[i]->C);
        this->opts.get("epsilon", classifiers[i]->eps);
        this->opts.get("nu", classifiers[i]->nu);
        this->opts.get("tolerance", classifiers[i]->tol);
        this->opts.get("degree", classifiers[i]->degree);
        this->opts.get("coef0", classifiers[i]->coef0);
        this->opts.get("tau", classifiers[i]->tau);
        this->opts.get("max_iter", classifiers[i]->max_iter);
        classifiers[i]->gamma = gamma_temp;

        status = classifiers[i]->compute();

        if (status != da_status_success)
            return status; // Error message already loaded

        bias[i] = classifiers[i]->bias;

        if (ismulticlass) {
            // n_sv_per_class = [n_sv_0, n_sv_1, ..., n_sv_(k-1)] where k is class number
            for (auto &support_index : classifiers[i]->support_indexes_pos) {
                if (is_sv[classifiers[i]->idx_class[support_index]] == false) {
                    is_sv[classifiers[i]->idx_class[support_index]] = true;
                    n_sv++;
                    n_sv_per_class[classifiers[i]->pos_class]++;
                }
            }
            for (auto &support_index : classifiers[i]->support_indexes_neg) {
                if (is_sv[classifiers[i]->idx_class[support_index]] == false) {
                    is_sv[classifiers[i]->idx_class[support_index]] = true;
                    n_sv++;
                    n_sv_per_class[classifiers[i]->neg_class]++;
                }
            }
        } else {
            n_sv = classifiers[0]->n_support;
            n_sv_per_class = classifiers[0]->n_support_per_class;
            support_coefficients = classifiers[0]->support_coefficients;
            support_vectors = classifiers[0]->support_vectors;
            support_indexes = classifiers[0]->support_indexes;
        }
    }

    // Path to aggregate results for multiclass problem
    if (ismulticlass) {
        std::vector<da_int> starting_col_idx, starting_col_idx_copy, starting_row_idx;
        try {
            starting_col_idx.resize(n_class);
            starting_row_idx.resize(n_class);
            support_coefficients.resize((n_class - 1) * n_sv);
            support_indexes.resize(n_sv);
            support_vectors.resize(n_sv * ncol);
        } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation error");
        }
        // Cumulative sum of number of support vectors per class
        // f.e [n_sv_0, n_sv_1, n_sv_2] -> [0, n_sv_0, n_sv_0 + n_sv_1]
        partial_sum(n_sv_per_class.begin(), n_sv_per_class.end() - 1,
                    starting_col_idx.begin() + 1);

        // support_coefficients = nontrivial design
        da_int k = 0;
        for (da_int i = 0; i < n_class; i++) {
            for (da_int j = i + 1; j < n_class; j++) {
                da_int starting_col_i = starting_col_idx[i];
                da_int starting_col_j = starting_col_idx[j];
                for (da_int l = 0; l < classifiers[k]->n; l++) {
                    if (is_sv[classifiers[k]->idx_class[l]]) {
                        if (classifiers[k]->idx_is_positive[l]) {
                            support_coefficients[((n_class - 1) * starting_col_i++) +
                                                 starting_row_idx[i]] =
                                classifiers[k]->alpha[l];
                        } else {
                            support_coefficients[((n_class - 1) * starting_col_j++) +
                                                 starting_row_idx[j]] =
                                classifiers[k]->alpha[l];
                        }
                    }
                }
                k++;
                starting_row_idx[i]++;
                starting_row_idx[j]++;
            }
        }

        // support_indexes = [support_indexes_0, support_indexes_1, ..., support_indexes_(k-1)]
        // where each support_indexes_k is array of length n_sv_k
        for (da_int i = 0; i < nrow; i++) {
            if (is_sv[i]) {
                da_int class_ = y[i];
                support_indexes[starting_col_idx[class_]++] = i;
            }
        }
        // support_vectors = slices of X matrix along rows specified in support_indexes
        // can be optimised for row major
        for (da_int i = 0; i < n_sv; i++) {
            da_int current_idx = support_indexes[i];
            for (da_int j = 0; j < ncol; j++) {
                support_vectors[i + j * n_sv] = X[current_idx + j * nrow];
            }
        }
    }

    iscomputed = true;
    return da_status_success;
}

/* Predict SVM */
template <typename T>
da_status svm<T>::predict(da_int nsamples, da_int nfeat, const T *X_test, da_int ldx_test,
                          T *predictions) {
    if (predictions == nullptr) {
        return da_error_bypass(this->err, da_status_invalid_pointer,
                               "predictions is not valid pointers.");
    }

    const T *X_test_temp;
    T *utility_ptr1 = nullptr;
    da_int ldx_test_temp;

    if (nfeat != ncol) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "n_features = " + std::to_string(nfeat) +
                                   " doesn't match the expected value " +
                                   std::to_string(ncol) + ".");
    }

    if (!iscomputed)
        return da_error(this->err, da_status_out_of_date,
                        "The model has not been trained yet.");

    da_status status = this->store_2D_array(
        nsamples, nfeat, X_test, ldx_test, &utility_ptr1, &X_test_temp, ldx_test_temp,
        "n_samples", "n_features", "X_test", "ldx_test");
    if (status != da_status_success)
        return status;

    if (ismulticlass) {
        std::vector<da_int> votes_array;
        try {
            votes_array.resize(n_class * nsamples);
        } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation error");
        }
        for (da_int i = 0; i < n_classifiers; i++) {
            std::vector<T> votes_temp; // this can technically be int
            try {
                votes_temp.resize(nsamples);
            } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
                return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                                "Memory allocation error");
            }
            classifiers[i]->predict(nsamples, nfeat, X_test_temp, ldx_test_temp,
                                    votes_temp.data());
            da_int pos_class = classifiers[i]->pos_class,
                   neg_class = classifiers[i]->neg_class;
            for (da_int j = 0; j < nsamples; j++) {
                if (votes_temp[j] == 1)
                    votes_array[j * n_class + pos_class]++;
                else
                    votes_array[j * n_class + neg_class]++;
            }
        }
        for (da_int i = 0; i < nsamples; i++) {
            da_int max_votes = 0, max_idx = 0;
            for (da_int j = 0; j < n_class; j++) {
                if (votes_array[i * n_class + j] > max_votes) {
                    max_votes = votes_array[i * n_class + j];
                    max_idx = j;
                }
            }
            predictions[i] = max_idx;
        }
    } else {
        classifiers[0]->predict(nsamples, nfeat, X_test_temp, ldx_test_temp, predictions);
    }
    if (utility_ptr1)
        delete[] (utility_ptr1);
    return da_status_success;
}

/* Decision function SVM */
template <typename T>
da_status svm<T>::decision_function(da_int nsamples, da_int nfeat, const T *X_test,
                                    da_int ldx_test, T *decision_values, da_int ldd,
                                    da_svm_decision_function_shape shape) {
    if (decision_values == nullptr) {
        return da_error_bypass(this->err, da_status_invalid_pointer,
                               "decision_values is not valid pointers.");
    }
    if (mod == da_svm_model::svr || mod == da_svm_model::nusvr) {
        return da_error_bypass(
            this->err, da_status_invalid_input,
            "Decision function is not defined for regression. Use predict instead.");
    }
    if (nfeat != ncol) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "n_features = " + std::to_string(nfeat) +
                                   " doesn't match the expected value " +
                                   std::to_string(ncol) + ".");
    }
    if (!iscomputed)
        return da_error(this->err, da_status_out_of_date,
                        "The model has not been trained yet.");
    // We do OVR only for multi-class classification and when requested
    bool is_ovo = (shape == ovo) || (ismulticlass == false);
    da_int n_rows = nsamples, n_cols = is_ovo ? n_classifiers : n_class;
    // Auxiliary variables for storing array
    const T *X_test_temp;
    T *utility_ptr1 = nullptr;
    T *utility_ptr2 = nullptr;
    T *decision_values_temp = nullptr;
    da_int ldx_test_temp;
    da_int ldd_temp;

    da_status status = this->store_2D_array(
        nsamples, nfeat, X_test, ldx_test, &utility_ptr1, &X_test_temp, ldx_test_temp,
        "n_samples", "n_features", "X_test", "ldx_test");
    if (status != da_status_success)
        return status;
    status = this->store_2D_array(n_rows, n_cols, decision_values, ldd, &utility_ptr2,
                                  const_cast<const T **>(&decision_values_temp), ldd_temp,
                                  "n_rows", "n_cols", "decision_values", "ldd", 1);
    if (status != da_status_success)
        return status;

    std::vector<T> decision_values_ovo;
    try {
        decision_values_ovo.resize(nsamples * n_classifiers);
    } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }
    // Obtain OVO decision function values
    for (da_int i = 0; i < n_classifiers; i++)
        classifiers[i]->decision_function(nsamples, nfeat, X_test_temp, ldx_test_temp,
                                          decision_values_ovo.data() + i * nsamples);

    // Path where decision values are 1D - binary classification (have to be OVO)
    if (!ismulticlass) {
        for (da_int i = 0; i < nsamples; i++)
            decision_values[i] = decision_values_ovo[i];
    }
    // Path for 2D OVO
    else if (is_ovo) {
        // decision_values_ovo is internal array in column major and dense. Here we copy it with consideration of leading dimension
        // To consider ldd in row major we can call utils function, but for column major there is no column to column function

        if (this->order == row_major) {
            da_utils::copy_transpose_2D_array_column_to_row_major(
                n_rows, n_cols, decision_values_ovo.data(), n_rows, decision_values, ldd);
        } else {
            for (da_int i = 0; i < n_classifiers; i++) {
                for (da_int j = 0; j < nsamples; j++) {
                    decision_values[i * ldd + j] = decision_values_ovo[i * nsamples + j];
                }
            }
        }
    }
    // Path for 2D OVR
    else {
        std::vector<T> decision_values_ovr, confidence_sum;
        try {
            decision_values_ovr.resize(nsamples * n_class);
            confidence_sum.resize(nsamples * n_class);
        } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation error");
        }
        for (da_int i = 0; i < n_classifiers; i++) {
            for (da_int j = 0; j < nsamples; j++) {
                confidence_sum[nsamples * classifiers[i]->pos_class + j] +=
                    decision_values_ovo[i * nsamples + j];
                confidence_sum[nsamples * classifiers[i]->neg_class + j] -=
                    decision_values_ovo[i * nsamples + j];
                if (decision_values_ovo[i * nsamples + j] > 0)
                    decision_values_ovr[nsamples * classifiers[i]->pos_class + j]++;
                else
                    decision_values_ovr[nsamples * classifiers[i]->neg_class + j]++;
            }
        }
        for (da_int i = 0; i < n_class; i++) {
            for (da_int j = 0; j < nsamples; j++) {
                decision_values_ovr[i * nsamples + j] +=
                    confidence_sum[i * nsamples + j] /
                    (3 * (abs(confidence_sum[i * nsamples + j]) + 1));
            }
        }
        // Save computed values into output array with consideration of leading dimension
        if (this->order == row_major) {
            da_utils::copy_transpose_2D_array_column_to_row_major(
                n_rows, n_cols, decision_values_ovr.data(), n_rows, decision_values, ldd);
        } else {
            for (da_int i = 0; i < n_class; i++) {
                for (da_int j = 0; j < nsamples; j++) {
                    decision_values[i * ldd + j] = decision_values_ovr[i * nsamples + j];
                }
            }
        }
    }

    if (utility_ptr1)
        delete[] (utility_ptr1);
    if (utility_ptr2)
        delete[] (utility_ptr2);
    return da_status_success;
}

template <typename T>
da_status svm<T>::score(da_int nsamples, da_int nfeat, const T *X_test, da_int ldx_test,
                        const T *y_test, T *score) {
    if (score == nullptr) {
        return da_error_bypass(this->err, da_status_invalid_pointer,
                               "score is not valid pointers.");
    }
    if (nfeat != ncol) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "n_features = " + std::to_string(nfeat) +
                                   " doesn't match the expected value " +
                                   std::to_string(ncol) + ".");
    }
    if (!iscomputed)
        return da_error(this->err, da_status_out_of_date,
                        "The model has not been trained yet.");

    const T *X_test_temp;
    T *utility_ptr1 = nullptr;
    da_int ldx_test_temp;

    da_status status = this->store_2D_array(
        nsamples, nfeat, X_test, ldx_test, &utility_ptr1, &X_test_temp, ldx_test_temp,
        "n_samples", "n_features", "X_test", "ldx_test");
    if (status != da_status_success)
        return status;

    status = this->check_1D_array(nsamples, y_test, "n_samples", "y_test", 1);
    if (status != da_status_success)
        return status;

    std::vector<T> predictions;
    try {
        predictions.resize(nsamples);
    } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }
    status = predict(nsamples, nfeat, X_test, ldx_test, predictions.data());
    // Calculate accuracy for classification and R^2 for regression
    if (mod == da_svm_model::svc || mod == da_svm_model::nusvc) {
        *score = 0.;
        for (da_int i = 0; i < nsamples; i++) {
            if (predictions[i] == y_test[i]) {
                *score += (T)1.0;
            }
        }
        *score = *score / (T)nsamples;
    } else {
        T y_test_mean, rss = 0, tss = 0;
        da_basic_statistics::mean(column_major, da_axis_all, nsamples, 1, y_test,
                                  nsamples, &y_test_mean);
        for (da_int i = 0; i < nsamples; i++) {
            rss += pow(y_test[i] - predictions[i], 2);
            tss += pow(y_test[i] - y_test_mean, 2);
        }
        *score = T(1 - rss / tss);
    }
    if (utility_ptr1)
        delete[] (utility_ptr1);
    return da_status_success;
}

} // namespace da_svm
#endif