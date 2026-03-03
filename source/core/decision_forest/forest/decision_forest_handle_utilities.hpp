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

#ifndef FOREST_UTILITIES_HPP
#define FOREST_UTILITIES_HPP

#include "common/tree_options_types.hpp"
#include "decision_forest_options.hpp"
#include "macros.h"

namespace ARCH {
namespace da_decision_forest {

using namespace da_tree_options_types;
using namespace da_errors;

template <typename T>
decision_forest<T>::decision_forest(da_errors::da_error_t &err) : basic_handle<T>(err) {
    // Initialize the options registry
    // Any error is stored err->status[.] and this NEEDS to be checked
    // by the caller.
    register_forest_options<T>(this->opts, *this->err);
}

template <typename T> decision_forest<T>::~decision_forest() {
    // Destructor needs to handle arrays that were allocated due to row major storage of input data
    if (X_temp)
        delete[] (X_temp);
    if (X_binned)
        delete (X_binned);
}

template <typename T>
da_status decision_forest<T>::get_result([[maybe_unused]] da_result query,
                                         [[maybe_unused]] da_int *dim,
                                         [[maybe_unused]] da_int *result) {
    return da_warn(this->err, da_status_unknown_query,
                   "There are no integer results available for this API.");
};

template <typename T>
da_status decision_forest<T>::get_result(da_result query, da_int *dim, T *result) {

    if (!model_trained)
        return da_warn_bypass(
            this->err, da_status_unknown_query,
            "Handle does not contain data relevant to this query. Was the "
            "last call to the solver successful?");
    // Pointers were already tested in the generic get_result

    da_int rinfo_size = 5;
    switch (query) {
    case da_result::da_rinfo:
        if (*dim < rinfo_size) {
            *dim = rinfo_size;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(rinfo_size) + ".");
        }
        result[0] = (T)n_features;
        result[1] = (T)n_samples;
        result[2] = (T)n_obs;
        result[3] = (T)seed;
        result[4] = (T)n_tree;
        break;
    default:
        return da_warn_bypass(this->err, da_status_unknown_query,
                              "The requested result could not be found.");
    }
    return da_status_success;
}

template <typename T>
da_status decision_forest<T>::set_training_data(da_int n_samples, da_int n_features,
                                                const T *X, da_int ldx, const da_int *y,
                                                da_int n_class,
                                                const da_int *usr_cat_feat) {

    // Guard against errors due to multiple calls using the same class instantiation
    if (X_temp) {
        delete[] (X_temp);
        X_temp = nullptr;
    }

    da_status status =
        this->store_2D_array(n_samples, n_features, X, ldx, &X_temp, &this->X, this->ldx,
                             "n_samples", "n_features", "X", "ldx");
    if (status != da_status_success)
        return status;

    status = this->check_1D_array(n_samples, y, "n_samples", "y", 1);
    if (status != da_status_success)
        return status;

    this->refresh();
    this->y = y;
    this->n_samples = n_samples;
    this->n_features = n_features;
    this->n_class = n_class;
    if (n_class <= 0)
        this->n_class = *std::max_element(y, y + n_samples) + 1;

    usr_categorical_feat = usr_cat_feat;

    return da_status_success;
}

} // namespace da_decision_forest
} // namespace ARCH

#endif