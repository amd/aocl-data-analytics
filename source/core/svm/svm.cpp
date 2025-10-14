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

#include "svm.hpp"
#include "aoclda.h"
#include "basic_statistics.hpp"
#include "da_error.hpp"
#include "da_std.hpp"
#include "macros.h"
#include "options.hpp"
#include "svm_options.hpp"
#include "svm_types.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

/*
 * SVM handle class that contains definitions to all the user
 * facing functionalities like set_data(),
 */

namespace ARCH {

namespace da_svm {

using namespace da_svm_types;

template <typename T> svm<T>::svm(da_errors::da_error_t &err) : basic_handle<T>(err) {
    // Initialize the options registry
    // Any error is stored err->status[.] and this needs to be checked
    // by the caller.
    register_svm_options<T>(this->opts, err);
}

template <typename T> svm<T>::~svm() {
    // Destructor needs to handle arrays that were allocated due to row major storage of input data
    if (X_temp)
        delete[] (X_temp);
};

template <typename T> void svm<T>::refresh() { iscomputed = false; }

/* get_result (required to be defined by basic_handle) */
template <typename T>
da_status svm<T>::get_result(da_result query, da_int *dim, T *result) {
    // Don't return anything if SVM has not been computed
    if (!iscomputed) {
        return da_warn(this->err, da_status_unknown_query,
                       "SVM has not yet been computed. Please call da_svm_compute_s "
                       "or da_svm_compute_d before extracting results.");
    }

    da_int rinfo_size = 100;
    da_int size;

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
    // support_vectors = slices of X matrix along rows specified in support_indexes
    case da_result::da_svm_support_vectors:
        size = n_sv * ncol;
        if (*dim < size) {
            *dim = size;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(size) + ".");
        }
        try {
            support_vectors.resize(size);
        } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation error");
        }
        // Construct a matrix consisting of only support vectors (can be optimised in row major)
        for (da_int i = 0; i < n_sv; i++) {
            da_int current_idx = support_indexes[i];
            for (da_int j = 0; j < ncol; j++) {
                support_vectors[i + j * n_sv] = X[current_idx + j * nrow];
            }
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
    case da_result::da_svm_probaA:
        if (mod == da_svm_model::svr || mod == da_svm_model::nusvr)
            return da_error(
                this->err, da_status_invalid_input,
                "predict probabilities are only available for classification problems");
        if (!predict_proba_opt) {
            return da_error(this->err, da_status_invalid_input,
                            "predict probabilities must have been set to 1 prior to "
                            "computing the SVM");
        }
        size = n_classifiers;
        if (*dim < size) {
            *dim = size;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(size) + ".");
        }
        for (da_int i = 0; i < n_classifiers; i++)
            result[i] = probaA[i];
        break;
    case da_result::da_svm_probaB:
        if (mod == da_svm_model::svr || mod == da_svm_model::nusvr)
            return da_error(
                this->err, da_status_invalid_input,
                "predict probabilities are only available for classification problems");
        if (!predict_proba_opt) {
            return da_error(this->err, da_status_invalid_input,
                            "predict probabilities must have been set to 1 prior to "
                            "computing the SVM");
        }
        size = n_classifiers;
        if (*dim < size) {
            *dim = size;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(size) + ".");
        }
        for (da_int i = 0; i < n_classifiers; i++)
            result[i] = probaB[i];
        break;
    default:
        return da_warn(this->err, da_status_unknown_query,
                       "The requested result could not be found.");
    }
    return da_status_success;
}

template <typename T>
da_status svm<T>::get_result(da_result query, da_int *dim, da_int *result) {
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
    case da_result::da_svm_n_iterations:
        size = n_classifiers;
        if (*dim < size) {
            *dim = size;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(size) + ".");
        }
        for (da_int i = 0; i < size; i++)
            result[i] = n_iteration[i];

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

/* Store the user's data matrix in preparation for SVM computation */
template <typename T>
da_status svm<T>::set_data(da_int n_samples, da_int n_features, const T *X_in,
                           da_int ldx_train, const T *y_in) {
    nrow = n_samples;
    ncol = n_features;
    ismulticlass = false;
    try {
        is_sv.resize(n_samples);
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
        this->store_2D_array(n_samples, n_features, X_in, ldx_train, &X_temp, &X,
                             this->ldx_train, "n_samples", "n_features", "X", "ldx");
    if (status != da_status_success)
        return status;

    status = this->check_1D_array(n_samples, y_in, "n_samples", "y", 1);
    if (status != da_status_success)
        return status;

    y = y_in;
    // Find n_class and validate y for classification
    if (mod == da_svm_model::svc || mod == da_svm_model::nusvc) {
        // Check if y contains only whole numbers
        for (da_int i = 0; i < n_samples; i++)
            if (y[i] != std::round(y[i]))
                return da_error(this->err, da_status_invalid_input,
                                "Labels must be whole numbers from 0 to K-1, where K is "
                                "the number of classes.");
        // y is assumed to only contain values from 0 to K-1 (K being the number of classes)
        n_class = (da_int)(std::round(*std::max_element(y, y + n_samples)) + 1);
        if (n_class < 2)
            return da_error(this->err, da_status_invalid_input,
                            "Number of classes must be at least 2.");
        n_classifiers = n_class * (n_class - 1) / 2;
    } else { // Regression
        n_class = 2;
        n_classifiers = 1;
    }
    try {
        classifiers.resize(n_classifiers);
        class_sizes.resize(n_class);
        bias.resize(n_classifiers);
        probaA.resize(n_classifiers);
        probaB.resize(n_classifiers);
        n_iteration.resize(n_classifiers);
    } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }
    for (da_int i = 0; i < n_classifiers; i++) {
        if (mod == da_svm_model::svc)
            classifiers[i] =
                std::make_unique<svc<T>>(X, y, n_samples, n_features, this->ldx_train);
        else if (mod == da_svm_model::nusvc)
            classifiers[i] =
                std::make_unique<nusvc<T>>(X, y, n_samples, n_features, this->ldx_train);
        else if (mod == da_svm_model::svr)
            classifiers[i] =
                std::make_unique<svr<T>>(X, y, n_samples, n_features, this->ldx_train);
        else if (mod == da_svm_model::nusvr)
            classifiers[i] =
                std::make_unique<nusvr<T>>(X, y, n_samples, n_features, this->ldx_train);
    }
    da_int k = 0;
    // Layout of OVO problems in classifiers vector is: 0v1, 0v2, ..., 0v(k-1), 1v2, 1v3, ... etc.
    if (n_class > 2) {
        ismulticlass = true;
        for (da_int i = 0; i < n_class; i++)
            for (da_int j = i + 1; j < n_class; j++) {
                da_int size = 0, pos_size = 0;
                // Vector that will store all indexes whose label is i or j (reserve bigger size here and then copy only necessary)
                std::vector<da_int> all_idx(n_samples);
                classifiers[k]->ismulticlass = true;
                classifiers[k]->pos_class = i;
                classifiers[k]->neg_class = j;
                // Pick all indexes where label is i or j
                for (da_int row = 0; row < n_samples; row++)
                    if (y[row] == i || y[row] == j)
                        all_idx[size++] = row;

                // Among the picked indexes assign ones where class==i is positive
                std::vector<bool> is_positive(size, false);
                std::vector<da_int> pos_idx(size), neg_idx(size);
                for (da_int idx = 0; idx < size; idx++) {
                    if (y[all_idx[idx]] == i) {
                        is_positive[idx] = true;
                        pos_idx[pos_size++] = all_idx[idx];
                    }
                }
                // Stop execution if one of the classes has no samples (triggered when y is in the wrong format, i.e not 0 to K-1)
                if (pos_size == 0 || size - pos_size == 0)
                    return da_error(
                        this->err, da_status_invalid_input,
                        "One of the classes has no samples. Check if your label "
                        "array is in the right format, i.e. 0 to K-1.");
                try {
                    classifiers[k]->idx_class.resize(size);
                    classifiers[k]->idx_is_positive.resize(size);
                } catch (std::bad_alloc &) {   // LCOV_EXCL_LINE
                    return da_error(this->err, // LCOV_EXCL_LINE
                                    da_status_memory_error, "Memory allocation error");
                }
                std::copy_n(all_idx.begin(), size, classifiers[k]->idx_class.begin());
                std::copy_n(is_positive.begin(), size,
                            classifiers[k]->idx_is_positive.begin());
                classifiers[k]->n = size;
                class_sizes[i] = pos_size;
                k++;
            }
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
    da_int kernel_enum;
    std::string order_string;
    if (!loadingdone)
        return da_error(this->err, da_status_no_data,
                        "No data has been passed to the handle. Please call "
                        "da_svm_set_data_s or da_svm_set_data_d.");

    // Here is logic to get default gamma 1/(ncol*var(X)), we only do that for kernels that use gamma
    T gamma_temp = 1;
    this->opts.get("kernel", kernel_string, kernel_enum);
    if (kernel_enum == rbf || kernel_enum == polynomial || kernel_enum == sigmoid) {
        this->opts.get("gamma", gamma_temp);
        if (gamma_temp < 0) {
            T mean, variance = 1;
            ARCH::da_basic_statistics::variance(column_major, da_axis_all, nrow, ncol, X,
                                                ldx_train, -1, &mean, &variance);
            if (variance == 0)
                return da_error(
                    this->err, da_status_invalid_input,
                    "Variance of the input data is zero. Use different gamma.");
            gamma_temp = 1 / (ncol * variance);
        }
    }

    // Necessary to reset some variables to ensure svm being called multiple times on the same handle works
    da_std::fill(is_sv.begin(), is_sv.end(), false);
    n_sv = 0;
    try {
        n_sv_per_class.resize(n_class);
    } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }
    da_std::fill(n_sv_per_class.begin(), n_sv_per_class.end(), 0);

    // Get the options set by user
    T C, epsilon, nu, tolerance, coef0, tau, cache_size;
    da_int degree, max_iter, n_fold, max_ws_size;
    this->opts.get("C", C);
    this->opts.get("epsilon", epsilon);
    this->opts.get("nu", nu);
    this->opts.get("coef0", coef0);
    this->opts.get("degree", degree);
    this->opts.get("tolerance", tolerance);
    this->opts.get("max_iter", max_iter);
    this->opts.get("tau", tau);
    this->opts.get("cache size", cache_size);
    this->opts.get("max_ws_size", max_ws_size);
    this->opts.get("predict probabilities", predict_proba_opt);
    this->opts.get("n_folds", n_fold);
    this->opts.get("seed", seed);

    if (predict_proba_opt) {
        if (seed == -1) {
            std::random_device r;
            seed = std::abs((da_int)r());
        }
        mt_gen.seed(seed);
    }
    // Compute each created classifier in the order 0v1, 0v2, ..., 0v(k-1), 1v2, 1v3, ... etc.
    for (da_int i = 0; i < n_classifiers; i++) {
        classifiers[i]->C = C;
        classifiers[i]->eps = epsilon;
        classifiers[i]->nu = nu;
        classifiers[i]->coef0 = coef0;
        classifiers[i]->degree = degree;
        classifiers[i]->tol = tolerance;
        classifiers[i]->max_iter = max_iter;
        classifiers[i]->tau = tau;
        classifiers[i]->gamma = gamma_temp;
        classifiers[i]->kernel_function = kernel_enum;
        classifiers[i]->cache_size = cache_size;
        classifiers[i]->max_ws_size = max_ws_size;

        if (predict_proba_opt) {
            status = compute_probabilities(*classifiers[i], n_fold, probaA[i], probaB[i]);
        }

        status = classifiers[i]->compute();

        if (status != da_status_success)
            return status; // Error message already loaded

        bias[i] = classifiers[i]->bias;
        n_iteration[i] = classifiers[i]->iter;

        // We increment the total number of new support vectors and support vectors in specific class
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
            // For non-multiclass problems, we just set the results of the first classifier
            n_sv = classifiers[0]->n_support;
            n_sv_per_class = classifiers[0]->n_support_per_class;
            support_coefficients = classifiers[0]->support_coefficients;
            support_indexes = classifiers[0]->support_indexes;
        }
    }
    // Check if any support vectors were found
    if (n_sv == 0)
        status = da_warn(
            this->err, da_status_numerical_difficulties,
            "No support vectors found. Check if your data is in the right format.");

    // Get support coefficients and support indexes for multiclass problem
    // NOTE: support coefficients have quite an unusual layout to match scikit-learn (LibSVM)
    // The shape of support_coefficients is (n_class-1, n_sv) where columns are filled from left to right
    // with coefficients corresponding to support vectors of the first class, then the second class, etc.
    // References: https://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f402 and
    // https://scikit-learn.org/stable/modules/svm.html#svm-multi-class (expand "details on multi-class strategies")
    if (ismulticlass) {
        // starting_col_idx - will tell at which position (column) to start filling support_coefficients for each class, effectively partial sum of n_sv_per_class
        // starting_row_idx - will tell at which row to start filling support_coefficients for each class
        std::vector<da_int> starting_col_idx, starting_row_idx;
        try {
            starting_col_idx.resize(n_class);
            starting_row_idx.resize(n_class);
            support_coefficients.resize((n_class - 1) * n_sv);
            support_indexes.resize(n_sv);
        } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation error");
        }
        // Cumulative sum of number of support vectors per class
        // f.e [n_sv_0, n_sv_1, n_sv_2] -> [0, n_sv_0, n_sv_0 + n_sv_1]
        partial_sum(n_sv_per_class.begin(), n_sv_per_class.end() - 1,
                    starting_col_idx.begin() + 1);

        // Approach to fill the support coefficients array:
        // For each classifier get starting column index for positive (i) and negative (j) class
        // and then iterate over all rows that are either class i or j and are support vectors
        // to fill the support coefficients array with alphas. Effectively we are filling the support_coefficients row-wise
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

        // support indexes is a vector which lists support indexes of each class one after another
        // support_indexes = [support_indexes_0, support_indexes_1, ..., support_indexes_(k-1)]
        // where each support_indexes_k is array of length n_sv_k
        for (da_int i = 0; i < nrow; i++) {
            if (is_sv[i]) {
                da_int class_ = y[i];
                support_indexes[starting_col_idx[class_]++] = i;
            }
        }
    }

    iscomputed = true;
    return status;
}

/* Predict SVM */
template <typename T>
da_status svm<T>::predict(da_int nsamples, da_int nfeat, const T *X_test, da_int ldx_test,
                          T *predictions) {
    if (predictions == nullptr) {
        return da_error_bypass(this->err, da_status_invalid_pointer,
                               "predictions is not a valid pointer.");
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
            status = classifiers[i]->predict(nsamples, nfeat, X_test_temp, ldx_test_temp,
                                             votes_temp.data());
            if (status != da_status_success)
                return status;
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
        status = classifiers[0]->predict(nsamples, nfeat, X_test_temp, ldx_test_temp,
                                         predictions);
    }
    if (utility_ptr1)
        delete[] (utility_ptr1);
    return status;
}

/* Decision function SVM */
template <typename T>
da_status svm<T>::decision_function(da_int nsamples, da_int nfeat, const T *X_test,
                                    da_int ldx_test, da_svm_decision_function_shape shape,
                                    T *decision_values, da_int ldd) {
    if (decision_values == nullptr) {
        return da_error_bypass(this->err, da_status_invalid_pointer,
                               "decision_values is not a valid pointer.");
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
    for (da_int i = 0; i < n_classifiers; i++) {
        status =
            classifiers[i]->decision_function(nsamples, nfeat, X_test_temp, ldx_test_temp,
                                              decision_values_ovo.data() + i * nsamples);
        if (status != da_status_success)
            return status;
    }

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
                               "score is not a valid pointer.");
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
        // Accuracy implementation
        *score = 0.;
        for (da_int i = 0; i < nsamples; i++) {
            if (predictions[i] == y_test[i]) {
                *score += (T)1.0;
            }
        }
        *score = *score / (T)nsamples;
    } else {
        // R^2 implementation
        T y_test_mean, rss = 0, tss = 0;
        ARCH::da_basic_statistics::mean(column_major, da_axis_all, nsamples, 1, y_test,
                                        nsamples, &y_test_mean);
        for (da_int i = 0; i < nsamples; i++) {
            rss += pow(y_test[i] - predictions[i], 2);
            tss += pow(y_test[i] - y_test_mean, 2);
        }
        // If 0 in numerator, return 1.0 as perfect score
        // If 0 in denominator and nonzero numerator, return 0.0 (sklearn behavior)
        if (rss == 0)
            *score = 1.0;
        else if (tss == 0)
            *score = 0.0;
        else
            *score = T(1 - rss / tss);
    }
    if (utility_ptr1)
        delete[] (utility_ptr1);
    return da_status_success;
}

template <typename T>
da_status svm<T>::predict_proba(da_int nsamples, da_int nfeat, const T *X_test,
                                da_int ldx_test, T *y_proba, da_int ldy) {
    if (y_proba == nullptr) {
        return da_error_bypass(this->err, da_status_invalid_pointer,
                               "y_proba is not a valid pointer.");
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
    if (!predict_proba_opt)
        return da_error(
            this->err, da_status_invalid_input,
            "Predict probabilities option was not set prior to computing SVM.");

    const T *X_test_temp;
    T *y_proba_temp;
    T *utility_ptr1 = nullptr;
    T *utility_ptr2 = nullptr;
    da_int ldx_test_temp;
    da_int ldy_temp;

    da_status status = this->store_2D_array(
        nsamples, nfeat, X_test, ldx_test, &utility_ptr1, &X_test_temp, ldx_test_temp,
        "n_samples", "n_features", "X_test", "ldx_test");
    if (status != da_status_success)
        return status;
    status = this->store_2D_array(nsamples, n_class, y_proba, ldy, &utility_ptr2,
                                  const_cast<const T **>(&y_proba_temp), ldy_temp,
                                  "n_rows", "n_cols", "y_proba", "ldy", 1);
    if (status != da_status_success)
        return status;
    std::vector<T> dec_values;
    try {
        dec_values.resize(nsamples * n_classifiers);
    } catch (std::bad_alloc &) {
        return da_error(this->err, da_status_memory_error, "Memory allocation error");
    }
    // Obtain OVO decision function values
    for (da_int i = 0; i < n_classifiers; i++) {
        status =
            classifiers[i]->decision_function(nsamples, nfeat, X_test_temp, ldx_test_temp,
                                              dec_values.data() + i * nsamples);
        if (status != da_status_success)
            return status;
    }
    // To match sklearn convention we need to flip the sign in binary case
    if (!ismulticlass)
        for (da_int i = 0; i < nsamples; i++)
            dec_values[i] = -dec_values[i];
    // One-vs-One decision values -> pairwise probabilities -> coupled multiclass probabilities
    const T min_prob = (T)1e-7;
    for (da_int i = 0; i < nsamples; i++) {
        // Temporary buffers per each sample
        std::vector<T> pairwise_prob, Q, Qp;
        try {
            pairwise_prob.resize(n_class * n_class);
            Q.resize(n_class * n_class);
            Qp.resize(n_class);
        } catch (std::bad_alloc &) {
            return da_error(this->err, da_status_memory_error, "Memory allocation error");
        }
        // Construct pairwise probabilities matrix
        da_int k = 0;
        for (da_int j = 0; j < n_class; j++) {
            for (da_int l = j + 1; l < n_class; l++) {
                T fApB = dec_values[i + k * nsamples] * probaA[k] + probaB[k];
                T proba;
                if (fApB >= 0)
                    proba = exp(-fApB) / (1.0 + exp(-fApB));
                else
                    proba = 1.0 / (1.0 + exp(fApB));
                proba = std::min(std::max(proba, min_prob), (T)1.0 - min_prob);
                pairwise_prob[j * n_class + l] = proba;
                pairwise_prob[l * n_class + j] = 1.0 - proba;
                k++;
            }
        }
        // Compute class probabilities from pairwise probabilities
        // Build Q matrix
        for (da_int t = 0; t < n_class; t++) {
            y_proba_temp[i + t * ldy_temp] = (T)1.0 / (T)n_class;
            Q[t * n_class + t] = 0;
            for (da_int j = 0; j < t; j++) {
                Q[t * n_class + t] +=
                    pairwise_prob[j * n_class + t] * pairwise_prob[j * n_class + t];
                Q[t * n_class + j] = Q[j * n_class + t];
            }
            for (da_int j = t + 1; j < n_class; j++) {
                Q[t * n_class + t] +=
                    pairwise_prob[j * n_class + t] * pairwise_prob[j * n_class + t];
                Q[t * n_class + j] =
                    -pairwise_prob[j * n_class + t] * pairwise_prob[t * n_class + j];
            }
        }

        da_int max_iter = std::max((da_int)100, n_class);
        T eps = (T)0.005 / (T)n_class;
        for (da_int iter = 0; iter < max_iter; iter++) {
            // Compute Qp and pQp
            T pQp = 0;
            for (da_int t = 0; t < n_class; t++) {
                Qp[t] = 0;
                da_int Q_offset = t * n_class;
                da_int y_proba_offset = t * ldy_temp;
                for (da_int j = 0; j < n_class; j++)
                    Qp[t] += Q[Q_offset + j] * y_proba_temp[i + j * ldy_temp];
                pQp += y_proba_temp[i + y_proba_offset] * Qp[t];
            }
            // Find max error
            T max_error = 0;
            for (da_int t = 0; t < n_class; t++) {
                T error = std::fabs(Qp[t] - pQp);
                max_error = std::max(max_error, error);
            }
            if (max_error < eps)
                break;

            // Update p
            for (da_int t = 0; t < n_class; t++) {
                T Qt = Q[t * n_class + t];
                if (Qt == 0)
                    continue;
                T diff = (-Qp[t] + pQp) / Qt;
                y_proba_temp[i + t * ldy_temp] += diff;
                // Safeguard to avoid unreasonable updates
                // (optional; keeps behavior closer to libsvm if omitted)
                T denom = (T)1 + diff;
                if (denom == 0)
                    continue;
                pQp = (pQp + diff * (diff * Qt + (T)2 * Qp[t])) / (denom * denom);
                for (da_int j = 0; j < n_class; j++) {
                    Qp[j] = (Qp[j] + diff * Q[t * n_class + j]) / denom;
                    y_proba_temp[i + j * ldy_temp] /= denom;
                }
            }
        }
    }
    // For row-major order, we need to transpose the output
    if (this->order == row_major) {
        da_utils::copy_transpose_2D_array_column_to_row_major(
            nsamples, n_class, y_proba_temp, ldy_temp, y_proba, ldy);
    }
    if (utility_ptr1)
        delete[] (utility_ptr1);
    if (utility_ptr2)
        delete[] (utility_ptr2);
    return status;
}

template <typename T>
da_status svm<T>::predict_log_proba(da_int nsamples, da_int nfeat, const T *X_test,
                                    da_int ldx_test, T *y_log_proba, da_int ldy) {
    da_status status = da_status_success;

    status = predict_proba(nsamples, nfeat, X_test, ldx_test, y_log_proba, ldy);
    if (status != da_status_success)
        return status;

    if (this->order == column_major) {
        for (da_int j = 0; j < n_class; j++) {
            for (da_int i = 0; i < nsamples; i++) {
                y_log_proba[ldy * j + i] = log(y_log_proba[ldy * j + i]);
            }
        }
    } else {
        for (da_int j = 0; j < nsamples; j++) {
            for (da_int i = 0; i < n_class; i++) {
                y_log_proba[j * ldy + i] = log(y_log_proba[j * ldy + i]);
            }
        }
    }
    return status;
}

// Compute probabilities using cross-validation
template <typename T>
da_status svm<T>::compute_probabilities(base_svm<T> &classifier, da_int n_fold, T &probaA,
                                        T &probaB) {
    std::vector<T> X_class, y_class;
    da_int n = classifier.n, p = classifier.p;
    if (n_fold > n) {
        return da_error(
            this->err, da_status_invalid_input,
            "Number of folds is larger than number of samples in a classifier.");
    }
    // Obtain slices from user data, containing relevant classes
    if (classifier.ismulticlass) {
        try {
            X_class.resize(n * p);
            y_class.resize(n);
        } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation error");
        }
        for (da_int i = 0; i < n; i++) {
            da_int current_idx = classifier.idx_class[i];
            for (da_int j = 0; j < p; j++) {
                X_class[i + j * n] = classifier.XUSR[current_idx + j * classifier.ldx];
            }
            y_class[i] = classifier.idx_is_positive[i] ? 1.0 : 0.0;
        }
    } else {
        X_class.assign(classifier.XUSR, classifier.XUSR + classifier.ldx * p);
        y_class.resize(n);
        // Flip the sign here to match sklearn convention (0.0 is positive class, 1.0 negative)
        for (da_int i = 0; i < n; i++) {
            y_class[i] = (classifier.yusr[i] == 1) ? 0 : 1;
        }
    }
    // For multiclass case X_class is column-major dense (no leading dimension needs to be considered)
    // For binary classification X_class is still column-major only but might be affected by leading dimension (it is raw user pointer)
    da_int ldx_class = (classifier.ismulticlass) ? n : classifier.ldx;
    // Create vector of random indices for shuffling
    std::vector<da_int> rand_indices;
    std::vector<T> decision_values;
    try {
        rand_indices.resize(n);
        decision_values.resize(n);
    } catch (std::bad_alloc &) {
        return da_error(this->err, da_status_memory_error, "Memory allocation error");
    }
    da_std::iota(rand_indices.begin(), rand_indices.end(), 0);
    da_std::shuffle(rand_indices.begin(), rand_indices.end(), mt_gen);

    // Start cross-validation
    for (da_int i = 0; i < n_fold; i++) {
        da_int fold_start = i * n / n_fold;
        da_int fold_end = (i == n_fold - 1) ? n : (i + 1) * n / n_fold;

        std::vector<T> X_train, y_train, X_val, y_val;
        try {
            X_train.resize((n - (fold_end - fold_start)) * p);
            y_train.resize(n - (fold_end - fold_start));
            X_val.resize((fold_end - fold_start) * p);
            y_val.resize(fold_end - fold_start);
        } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation error");
        }

        // Split data into training and validation sets using random indices (expensive in column major)
        da_int idx_train = 0, idx_val = 0;
        for (da_int sample_idx = 0; sample_idx < n; sample_idx++) {
            da_int j = rand_indices[sample_idx];
            if (sample_idx >= fold_start && sample_idx < fold_end) {
                for (da_int k = 0; k < p; k++) {
                    X_val[idx_val + (fold_end - fold_start) * k] =
                        X_class[j + k * ldx_class];
                }
                y_val[idx_val] = y_class[j];
                idx_val++;
            } else {
                for (da_int k = 0; k < p; k++) {
                    X_train[idx_train + (n - (fold_end - fold_start)) * k] =
                        X_class[j + k * ldx_class];
                }
                y_train[idx_train] = y_class[j];
                idx_train++;
            }
        }

        // Check for missing classes in the training fold
        bool missing_positive =
            std::find(y_train.begin(), y_train.end(), (T)0.0) == y_train.end();
        bool missing_negative =
            std::find(y_train.begin(), y_train.end(), (T)1.0) == y_train.end();
        std::vector<T> fold_decision;
        try {
            fold_decision.resize(y_val.size());
        } catch (std::bad_alloc &) {
            return da_error(this->err, da_status_memory_error, "Memory allocation error");
        }
        if (missing_positive && missing_negative) {
            // If both classes are missing, assign zero decision value
            std::fill(fold_decision.begin(), fold_decision.end(), (T)0.0);
        } else if (missing_positive || missing_negative) {
            // If one of the classes is missing, assign a default decision value:
            // positive decision if negatives are missing, negative otherwise.
            T default_decision = missing_positive ? (T)1.0 : (T)-1.0;
            std::fill(fold_decision.begin(), fold_decision.end(), default_decision);
        } else {
            // Create a fold classifier using the training fold
            std::unique_ptr<base_svm<T>> fold_classifier;
            switch (mod) {
            case da_svm_model::svc:
                fold_classifier = std::make_unique<svc<T>>(
                    X_train.data(), y_train.data(), static_cast<da_int>(y_train.size()),
                    p, static_cast<da_int>(y_train.size()));
                break;
            case da_svm_model::nusvc:
                fold_classifier = std::make_unique<nusvc<T>>(
                    X_train.data(), y_train.data(), static_cast<da_int>(y_train.size()),
                    p, static_cast<da_int>(y_train.size()));
                break;
            default:
                return da_error(this->err, da_status_unknown_query,
                                "Unsupported SVM model.");
            }
            fold_classifier->C = classifier.C;
            fold_classifier->eps = classifier.eps;
            fold_classifier->nu = classifier.nu;
            fold_classifier->coef0 = classifier.coef0;
            fold_classifier->degree = classifier.degree;
            fold_classifier->tol = classifier.tol;
            fold_classifier->max_iter = classifier.max_iter;
            fold_classifier->tau = classifier.tau;
            fold_classifier->gamma = classifier.gamma;
            fold_classifier->kernel_function = classifier.kernel_function;
            fold_classifier->cache_size = classifier.cache_size;
            fold_classifier->max_ws_size = classifier.max_ws_size;

            da_status fold_status = fold_classifier->compute();
            if (fold_status != da_status_success)
                return fold_status;
            fold_status = fold_classifier->decision_function(
                static_cast<da_int>(y_val.size()), p, X_val.data(),
                static_cast<da_int>(y_val.size()), fold_decision.data());
            if (fold_status != da_status_success)
                return fold_status;
        }
        // Copy the computed decision values for this fold back to the overall array.
        for (da_int j = 0; j < static_cast<da_int>(y_val.size()); j++) {
            decision_values[rand_indices[fold_start + j]] = fold_decision[j];
        }
    }

    // Improvement of Platt's binary SVM Probablistic Output, from Lin et al.
    // Adaptation of LibSVM code, following pseudocode from the original paper.
    // Reference: https://doi.org/10.1007/s10994-007-5018-6

    da_int pos_count = 0, neg_count = 0;
    for (da_int i = 0; i < n; i++) {
        if (y_class[i] == 1.0)
            pos_count++;
        else
            neg_count++;
    }

    // Parameters from the paper
    da_int maxiter = 100;
    T minstep = 1e-10, sigma = 1e-12;
    T eps = 1e-5;
    T hi_target = (pos_count + 1.0) / (pos_count + 2.0),
      lo_target = 1.0 / (neg_count + 2.0);
    probaA = 0.0, probaB = std::log((neg_count + 1.0) / (pos_count + 1.0));
    T fval = 0.0;

    std::vector<T> t;
    try {
        t.resize(n);
    } catch (std::bad_alloc &) {
        return da_error(this->err, da_status_memory_error, "Memory allocation error");
    }

    // Initialize t and fval
    for (da_int i = 0; i < n; i++) {
        t[i] = (y_class[i] == 1.0) ? hi_target : lo_target;
        T fApB = decision_values[i] * probaA + probaB;
        if (fApB >= 0)
            fval += t[i] * fApB + std::log(1.0 + std::exp(-fApB));
        else
            fval += (t[i] - 1.0) * fApB + std::log(1.0 + std::exp(fApB));
    }
    da_int iter = 0;
    for (; iter < maxiter; iter++) {
        T h11 = sigma, h22 = sigma, h21 = 0.0, g1 = 0.0, g2 = 0.0;
        for (da_int i = 0; i < n; i++) {
            T fApB = decision_values[i] * probaA + probaB;
            T p, q;
            if (fApB >= 0) {
                p = exp(-fApB) / (1.0 + std::exp(-fApB));
                q = 1.0 / (1.0 + std::exp(-fApB));
            } else {
                p = 1.0 / (1.0 + std::exp(fApB));
                q = exp(fApB) / (1.0 + std::exp(fApB));
            }
            T d2 = p * q;
            h11 += decision_values[i] * decision_values[i] * d2;
            h22 += d2;
            h21 += decision_values[i] * d2;
            T d1 = t[i] - p;
            g1 += decision_values[i] * d1;
            g2 += d1;
        }
        if (std::abs(g1) < eps && std::abs(g2) < eps)
            break; // Convergence reached
        T det = h11 * h22 - h21 * h21;
        T dA = (h21 * g2 - h22 * g1) / det;
        T dB = (h21 * g1 - h11 * g2) / det;
        T gd = g1 * dA + g2 * dB;
        T stepsize = 1.0;
        while (stepsize >= minstep) {
            T newA = probaA + stepsize * dA;
            T newB = probaB + stepsize * dB;
            T newfval = 0.0;
            for (da_int i = 0; i < n; i++) {
                T fApB = decision_values[i] * newA + newB;
                if (fApB >= 0)
                    newfval += t[i] * fApB + std::log(1.0 + std::exp(-fApB));
                else
                    newfval += (t[i] - 1.0) * fApB + std::log(1.0 + std::exp(fApB));
            }
            if (newfval < fval + 0.0001 * stepsize * gd) {
                probaA = newA;
                probaB = newB;
                fval = newfval;
                break; // Accept the step
            } else {
                stepsize *= 0.5; // Reduce step size
            }
        }

        if (stepsize < minstep) {
            // If stepsize is too small, we stop the optimization
            return da_warn(
                this->err, da_status_numerical_difficulties,
                "Line search had troubles converging during probability estimation.");
        }
    }
    if (iter >= maxiter) {
        return da_warn(
            this->err, da_status_maxit,
            "Maximum number of iterations reached during probability estimation.");
    }

    return da_status_success;
}

template class svm<float>;
template class svm<double>;

} // namespace da_svm

} // namespace ARCH