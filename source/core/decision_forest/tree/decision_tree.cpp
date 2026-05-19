/*
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "decision_tree.hpp"
#include "aoclda.h"
#include "decision_tree_handle_utilities.hpp"
#include "decision_tree_inference.hpp"
#include "decision_tree_training.hpp"
#include "macros.h"
#include "model_persistence.hpp"

namespace ARCH {

namespace da_decision_forest {

using namespace da_model_persistence;

template <typename T> da_status node<T>::serialize(serialization_buffer &buffer) {
    da_status status = da_status_success;
    auto io_dispatch = [&buffer, &status](auto &data) -> void {
        if (status != da_status_success) {
            return;
        }
        status = buffer.dispatch_buffer_io(data);
        return;
    };

    io_dispatch(this->parent_idx);
    io_dispatch(this->right_child_idx);
    io_dispatch(this->left_child_idx);
    io_dispatch(this->is_leaf);
    io_dispatch(this->depth);
    io_dispatch(this->score);
    io_dispatch(this->prop);
    io_dispatch(this->y_pred);
    io_dispatch(this->feature);
    io_dispatch(this->x_threshold);
    io_dispatch(this->category);
    io_dispatch(this->start_idx);
    io_dispatch(this->end_idx);
    io_dispatch(this->n_samples);
    io_dispatch(this->const_feat_idx);
    io_dispatch(this->children_const_idx);

    return status;
}

template <typename T>
da_status
decision_tree<T>::tree_serialization(da_model_persistence::serialization_buffer &buffer) {
    da_status status = da_status_success;

    if (buffer.get_mode() == deserialize) {
        if (this->n_nodes <= 0)
            return da_status_invalid_file_data;

        try {
            this->tree.resize(this->n_nodes);
        } catch (std::bad_alloc const &) {
            return da_error_bypass(
                this->err, da_status_memory_error,
                "Failing to allocate enough memory."); // LCOV_EXCL_LINE
        }
    }

    for (da_int i = 0; i < this->n_nodes; ++i) {
        if (status != da_status_success)
            return status;
        status = this->tree[i].serialize(buffer);
    }
    return status;
}

template <typename T>
da_status decision_tree<T>::serialize(serialization_buffer &buffer) {
    da_status status = da_status_success;
    auto io_dispatch = [&buffer, &status](auto &data) -> void {
        if (status != da_status_success) {
            return;
        }
        status = buffer.dispatch_buffer_io(data);
        return;
    };

    io_dispatch(this->model_trained);
    io_dispatch(this->predict_proba_opt);
    io_dispatch(this->ldx);
    io_dispatch(this->n_samples);
    io_dispatch(this->n_features);
    io_dispatch(this->n_class);
    io_dispatch(this->n_obs);
    io_dispatch(this->n_obs_total);
    io_dispatch(this->depth);
    io_dispatch(this->n_nodes);
    io_dispatch(this->n_leaves);
    io_dispatch(this->class_props);
    io_dispatch(this->samples_idx);
    io_dispatch(this->count_classes);
    io_dispatch(this->count_left_classes);
    io_dispatch(this->count_right_classes);
    io_dispatch(this->bootstrap_sample_frequency);
    io_dispatch(this->feature_values);
    io_dispatch(this->cat_feat);
    io_dispatch(this->max_cat);
    io_dispatch(this->cat_feat_table);
    io_dispatch(this->internal_bins);
    io_dispatch(this->node_hist);
    io_dispatch(this->hist_count_samples);
    io_dispatch(this->hist_feat_values);
    io_dispatch(this->features_idx);
    io_dispatch(this->read_public_options);
    io_dispatch(this->max_depth);
    io_dispatch(this->min_node_sample);
    io_dispatch(this->method);
    io_dispatch(this->nfeat_split);
    io_dispatch(this->seed);
    io_dispatch(this->min_split_score);
    io_dispatch(this->feat_thresh);
    io_dispatch(this->min_improvement);
    io_dispatch(this->bootstrap);
    io_dispatch(this->check_cat_data);
    io_dispatch(this->opt_max_cat);
    io_dispatch(this->use_hist);
    io_dispatch(this->usr_max_bins);
    io_dispatch(this->cat_tol);
    io_dispatch(this->cat_split_strat);

    if (status != da_status_success)
        return status;

    status = tree_serialization(buffer);

    return status;
}

template <typename T>
da_status decision_tree<T>::save_model(serialization_buffer &buffer) {

    if (!this->model_trained) {
        return da_error(this->err, da_status_no_data,
                        "The model has not yet been trained or the data it is "
                        "associated with is out of date.");
    }

    da_status status = basic_handle<T>::save_model(buffer);
    if (status != da_status_success)
        return da_error_trace(this->err, status, "Failure serializing model.");

    return status;
}

template <typename T>
da_status decision_tree<T>::load_model(serialization_buffer &buffer) {
    da_status status = basic_handle<T>::load_model(buffer);
    if (status != da_status_success)
        return da_error_trace(this->err, status, "Failure deserializing model.");

    return status;
}

template class decision_tree<double>;
template class decision_tree<float>;
} // namespace da_decision_forest

} // namespace ARCH
