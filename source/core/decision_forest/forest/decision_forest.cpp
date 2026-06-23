/*
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "decision_forest.hpp"
#include "decision_forest_handle_utilities.hpp"
#include "decision_forest_inference.hpp"
#include "decision_forest_training.hpp"
#include "model_persistence.hpp"

namespace ARCH {

namespace da_decision_forest {

using namespace da_model_persistence;

template <typename T>
da_status decision_forest<T>::forest_serialization(
    da_model_persistence::serialization_buffer &buffer) {
    da_status status = da_status_success;

    if (buffer.get_mode() == deserialize) {
        try {
            this->forest.resize(n_tree);
            for (da_int i = 0; i < this->n_tree; ++i) {
                this->forest[i] = std::make_unique<decision_tree<T>>();
            }
        } catch (std::bad_alloc const &) {
            return da_error(this->err, da_status_memory_error,
                            "Failing to allocate enough memory."); // LCOV_EXCL_LINE
        }
    }

    for (da_int i = 0; i < this->n_tree; ++i) {
        if (status != da_status_success)
            return status;
        status = this->forest[i]->serialize(buffer);
    }
    return status;
}

template <typename T>
da_status
decision_forest<T>::serialize(da_model_persistence::serialization_buffer &buffer) {
    da_status status = da_status_success;
    auto io_dispatch = [&buffer, &status](auto &data) -> void {
        if (status != da_status_success) {
            return;
        }
        status = buffer.dispatch_buffer_io(data);
        return;
    };

    io_dispatch(this->model_trained);
    io_dispatch(this->n_samples);
    io_dispatch(this->ldx);
    io_dispatch(this->n_features);
    io_dispatch(this->n_class);
    io_dispatch(this->n_tree);
    io_dispatch(this->seed);
    io_dispatch(this->n_obs);
    io_dispatch(this->block_size);
    io_dispatch(this->use_hist);
    io_dispatch(this->usr_max_bins);

    if (status != da_status_success)
        return status;

    status = forest_serialization(buffer);

    return status;
}

template <typename T>
da_status
decision_forest<T>::save_model(da_model_persistence::serialization_buffer &buffer) {

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
da_status
decision_forest<T>::load_model(da_model_persistence::serialization_buffer &buffer) {

    da_status status = basic_handle<T>::load_model(buffer);
    if (status != da_status_success)
        return da_error_trace(this->err, status, "Failure deserializing model.");

    return status;
}

template class decision_forest<double>;
template class decision_forest<float>;

} // namespace da_decision_forest

} // namespace ARCH
