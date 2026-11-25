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

#include "basic_handle.hpp"
#include "aoclda_error.h"
#include "aoclda_result.h"
#include "aoclda_types.h"
#include "basic_handle_options.hpp"
#include "da_error.hpp"
#include "da_logger.hpp"
#include "model_persistence.hpp"
#include "options.hpp"
#include <iomanip>

#include <map>
#include <variant>

/*
 * Base handle class (basic_handle) that contains members that
 * are common for all specialized handle types, pca, linear
 * models, etc.
 *
 * This handle is inherited by all specialized (internal) handles.
 */

template <typename T> basic_handle<T>::basic_handle(da_errors::da_error_t *err) {
    this->err = err;
    this->serialization_version =
        (da_int)da_model_persistence::model_persistence_min_version;
}

template <typename T> basic_handle<T>::basic_handle(da_errors::da_error_t &err) {
    // Assumes that err is valid
    this->err = &err;
    this->serialization_version =
        (da_int)da_model_persistence::model_persistence_min_version;
    // Initialize the options registry with common options to all handles
    register_common_options<T>(this->opts, *this->err);
}

template <typename T> basic_handle<T>::~basic_handle() {}

template <typename T> da_options::OptionRegistry &basic_handle<T>::get_opts() {
    return this->opts;
}

template <typename T> void basic_handle<T>::refresh() {}

template <typename T>
da_status basic_handle<T>::check_1D_array(da_int n, const T *data,
                                          const std::string &n_name,
                                          const std::string &data_name, da_int n_min) {

    da_int check_data = 0;
    std::string check_data_str;
    opts.get("check data", check_data_str, check_data);
    return ARCH::da_utils::check_1D_array<T>(check_data != 0, this->err, n, data, n_name,
                                             data_name, n_min);
}

template <typename T>
da_status basic_handle<T>::check_1D_array(da_int n, const da_int *data,
                                          const std::string &n_name,
                                          const std::string &data_name, da_int n_min) {

    da_int check_data = 0;
    std::string check_data_str;
    opts.get("check data", check_data_str, check_data);
    return ARCH::da_utils::check_1D_array<da_int>(check_data != 0, this->err, n, data,
                                                  n_name, data_name, n_min);
}

template <typename T>
da_status basic_handle<T>::check_2D_array(da_order order, da_int n_rows, da_int n_cols,
                                          const T *data, da_int lddata,
                                          const std::string &n_rows_name,
                                          const std::string &n_cols_name,
                                          const std::string &data_name,
                                          const std::string &lddata_name,
                                          da_int n_rows_min, da_int n_cols_min) {

    da_int check_data = 0;
    std::string check_data_str;
    opts.get("check data", check_data_str, check_data);
    return ARCH::da_utils::check_2D_array(check_data != 0, order, this->err, n_rows,
                                          n_cols, data, lddata, n_rows_name, n_cols_name,
                                          data_name, lddata_name, n_rows_min, n_cols_min);
}

template <typename T>
void basic_handle<T>::copy_2D_results_array(da_int n_rows, da_int n_cols, const T *data,
                                            da_int lddata, T *results_arr) {
    if (this->order == column_major) {
        ARCH::da_utils::copy_array_convert_precision<T, T>(
            column_major, n_rows, n_cols, data, lddata, results_arr, n_rows);
    } else {
        ARCH::da_utils::copy_transpose_2D_array_column_to_row_major(
            n_rows, n_cols, data, lddata, results_arr, n_cols);
    }
}

template <typename T>
da_status
basic_handle<T>::serialize_registry(da_model_persistence::serialization_buffer &buffer) {
    if (buffer.get_mode() != da_model_persistence::buffer_mode::serialize &&
        buffer.get_mode() != da_model_persistence::buffer_mode::reserve)
        return da_status_serialization_error;

    std::map<std::string, da_options::option_variant_t> options = {};

    try {
        this->opts.export_registry(
            [&](const std::string &key, da_options::option_variant_t value) {
                options[key] = value;
            });
    } catch (std::bad_alloc const &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    }

    // Save number of options to be serialized.
    da_int n = static_cast<da_int>(options.size());
    da_status status = buffer.dispatch_buffer_io(n);
    if (status != da_status_success)
        return status;

    for (auto &[key, value] : options) {
        // Copy needed: map keys are const, but dispatch_buffer_io takes a non-const ref.
        std::string name;
        try {
            name = key;
        } catch (std::bad_alloc const &) {
            return da_status_memory_error; // LCOV_EXCL_LINE
        }
        status = buffer.dispatch_buffer_io(name);
        if (status != da_status_success)
            return status;

        status =
            std::visit([&](auto &&v) { return buffer.dispatch_buffer_io(v); }, value);
        if (status != da_status_success)
            return status;
    }

    return status;
}

template <typename T>
da_status basic_handle<T>::deserialize_registry(
    da_model_persistence::serialization_buffer &buffer) {
    if (buffer.get_mode() != da_model_persistence::buffer_mode::deserialize)
        return da_status_serialization_error;

    // Load number of options that have been serialized.
    da_int n;
    da_status status = buffer.dispatch_buffer_io(n);
    if (status != da_status_success)
        return status;

    std::map<std::string, da_options::option_variant_t> current;

    try {
        this->opts.export_registry(
            [&](const std::string &key, da_options::option_variant_t value) {
                current[key] = value;
            });
    } catch (std::bad_alloc const &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    }

    if (n < 0 || n > (da_int)current.size())
        return da_status_invalid_file_data;

    for (da_int i = 0; i < n; ++i) {
        std::string key;
        status = buffer.dispatch_buffer_io(key);
        if (status != da_status_success)
            return status;

        auto it = current.find(key);
        if (it == current.end())
            return da_status_invalid_file_data;

        status = std::visit(
            [&](auto &&default_value) {
                using V = std::decay_t<decltype(default_value)>;
                V stored_value;
                da_status s = buffer.dispatch_buffer_io(stored_value);
                if (s != da_status_success)
                    return s;
                return this->opts.set(key, stored_value);
            },
            it->second);
        if (status != da_status_success)
            return status;
    }

    return status;
}

template <typename T>
da_status
basic_handle<T>::preallocate_buffer(da_model_persistence::serialization_buffer &buffer) {
    buffer.set_mode(da_model_persistence::buffer_mode::reserve);

    da_status status = this->serialize_registry(buffer);
    if (status != da_status_success)
        return da_error_trace(this->err, status, "Failure while getting registry size.");

    status = this->serialize(buffer);
    if (status != da_status_success)
        return da_error_trace(this->err, status, "Failure while getting model size.");

    status = buffer.reserve();
    if (status != da_status_success)
        return da_error_trace(this->err, status, "Failure while resizing buffer.");

    return status;
}

template <typename T>
da_status
basic_handle<T>::save_model(da_model_persistence::serialization_buffer &buffer) {
    // Clear vector data so serialization starts at the beginning of it.
    da_status status = buffer.clear_data();
    if (status != da_status_success)
        return da_error_trace(this->err, status, "Failure clearing buffer.");

    // Gets exact size needed to serialize the whole model and allocates it.
    status = preallocate_buffer(buffer);
    if (status != da_status_success)
        return da_error_trace(this->err, status, "Failure resizing buffer.");

    buffer.set_mode(da_model_persistence::buffer_mode::serialize);

    status = buffer.serialize_metadata(sizeof(T), this->serialization_version);
    if (status != da_status_success)
        return da_error(this->err, status, "Failure serializing metadata.");

    status = this->serialize_registry(buffer);
    if (status != da_status_success)
        return da_error_trace(this->err, status, "Failure serializing registry.");

    status = this->serialize(buffer);
    if (status != da_status_success)
        return da_error(this->err, status, "Failure serializing model.");

    return status;
}

template <typename T>
da_status
basic_handle<T>::load_model(da_model_persistence::serialization_buffer &buffer) {

    if (buffer.get_saved_serialization_version() != this->serialization_version)
        return da_error(this->err, da_status_version_mismatch,
                        "Saved model's version is incompatible with the current "
                        "library version.");

    this->saved_aoclda_version = buffer.get_saved_aoclda_version();

    da_status status = this->deserialize_registry(buffer);
    if (status != da_status_success)
        return da_error_trace(this->err, status, "Failure deserializing registry.");

    status = this->serialize(buffer);
    if (status != da_status_success)
        return da_error_trace(this->err, status, "Failure deserializing model.");

    this->model_loaded = true;

    return status;
}

template <typename T> da_status basic_handle<T>::print_model_versions() {
    // serialization_version must match exactly saved_serialization version,
    // thus current serialization version can be printed out instead.
    if (serialization_version == 0 || saved_aoclda_version == "NA") {
        return da_error(this->err, da_status_invalid_input,
                        "The handle doesn't contain a valid loaded model.");
    }

    da_logger::logger out;
    out << "===== Loaded Model Versions =====\n";
    out << std::setw(30) << "Serialization version:" << this->serialization_version
        << "\n";
    out << std::setw(30) << "Saved AOCL-DA build version:" << this->saved_aoclda_version
        << "\n";
    out << "=================================\n";
    out << std::flush;
    return da_status_success;
};

// Return data that is common across the handle types this function should
// be called at the beginning of the get_result function of the derived classes
// If the query is not handled by this function, it returns da_status_unknown_query
template <typename T>
da_status basic_handle<T>::get_result_common(da_result query, da_int *dim,
                                             da_int *result) {
    switch (query) {
    case da_result::da_trained:
        if (*dim < 1) {
            *dim = 1;
            return da_warn(
                this->err, da_status_invalid_array_dimension,
                "Size of the result array is too small, provide an array of at "
                "least size: 1.");
        }
        result[0] = da_int(this->model_trained);
        return da_status_success;
    // case da_...
    default:
        // Inform the caller that the query was not handled by this function;
        // it is expected that the caller fills in correctly the error stack
        return da_status_unknown_query; // No error stack registry
    }
}

#ifdef DA_BASIC_HANDLE_COMMON
template class basic_handle<double>;
template class basic_handle<float>;
#endif
#if defined(DA_BASIC_HANDLE_FP16) && defined(__AVX512FP16__)
template class basic_handle<_Float16>;
#endif