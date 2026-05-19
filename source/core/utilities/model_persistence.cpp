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

#include "model_persistence.hpp"
#include "approximate_neighbors_types.hpp"
#include "common/tree_options_types.hpp"
#include "da_vector.hpp"
#include "linmod_types.hpp"
#include "svm_types.hpp"

#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <type_traits>

#ifndef AOCLDA_VERSION_INT
#define AOCLDA_VERSION_INT 0
#endif

namespace da_model_persistence {

using namespace da_linmod_types;
using namespace da_tree_options_types;
using namespace da_approx_nn_types;

// METADATA KERNELS

da_status serialization_buffer::set_buffer_data(const char *buffer_data,
                                                const size_t size) {
    if (buffer_data == nullptr)
        return da_status_invalid_pointer;
    if (size == 0)
        return da_status_invalid_input;
    this->read_ptr = buffer_data;
    this->size = size;
    this->mode = deserialize;
    return da_status_success;
}

da_status serialization_buffer::set_buffer_data(std::vector<char> *buffer_data) {
    if (buffer_data == nullptr)
        return da_status_invalid_pointer;
    this->write_buf = buffer_data;
    this->mode = buffer_mode::reserve;
    this->add_metadata_size();
    return da_status_success;
}

da_status serialization_buffer::add_size(size_t value) {
    // Size of buffer can be modified only in reserve mode
    if (this->mode != buffer_mode::reserve)
        return da_status_invalid_option;
    if (this->size + value > static_cast<size_t>(std::numeric_limits<da_int>::max()))
        return da_status_invalid_input;
    this->size += value;
    return da_status_success;
}

da_status serialization_buffer::clear_data() {
    if (this->write_buf == nullptr)
        return da_status_invalid_pointer;

    this->write_buf->clear();
    return da_status_success;
}

da_status serialization_buffer::reserve() {
    // Size of buffer can be modified only in reserve mode
    if (this->mode != buffer_mode::reserve)
        return da_status_invalid_option;
    if (this->write_buf == nullptr)
        return da_status_invalid_pointer;

    try {
        this->write_buf->reserve(this->size);
    } catch (std::bad_alloc const &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    }
    return da_status_success;
}

da_status serialization_buffer::serialize_metadata(size_t precision,
                                                   da_int min_lib_version) {
    da_int da_int_size = (da_int)sizeof(da_int);
    da_int prec = (da_int)precision;

    da_status status = da_status_success;
    auto serialize = [this, &status](const auto &data) -> void {
        if (status != da_status_success) {
            return;
        }
        status = serialize_data(data);
        return;
    };

    std::string str_header_keyword = std::string(header_keyword);
    serialize(str_header_keyword);
    serialize(da_int_size);
    serialize(min_lib_version);
    serialize(this->handle_type);
    serialize(prec);

    return status;
}

da_status serialization_buffer::deserialize_metadata(da_int &precision) {
    da_status status = da_status_success;
    auto deserialize = [this, &status](auto &data) -> void {
        if (status != da_status_success) {
            return;
        }
        status = deserialize_data(data);
        return;
    };

    std::string loaded_header_keyword;
    deserialize(loaded_header_keyword);
    if (header_keyword != loaded_header_keyword)
        return da_status_invalid_file_data;

    da_int saved_int_size, saved_min_lib_version;
    deserialize(saved_int_size);
    deserialize(saved_min_lib_version);
    deserialize(this->handle_type);
    deserialize(precision);

    if (status != da_status_success)
        return status;

    if (saved_min_lib_version > (da_int)AOCLDA_VERSION_INT) {
        return da_status_version_mismatch;
    } else if (da_int(sizeof(da_int)) != saved_int_size &&
               saved_int_size != da_int(sizeof(int32_t))) {
        // Allows to load int32 with int64 da_int, but not the opposite.
        return da_status_invalid_file_data;
    } else if (precision != da_int(sizeof(float)) &&
               precision != da_int(sizeof(double))) {
        return da_status_invalid_file_data;
    }

    return status;
}

// SAVING KERNELS

template <typename Container>
da_status serialization_buffer::serialize_container_impl(const Container &data) {
    da_status status = da_status_success;
    using ValT = typename Container::value_type;

    // Store container length first so loading can reserve before reading elements.
    // Use int_save_t directly to always use the range of int64.
    int_save_t vec_size = int_save_t(data.size());
    const char *bytes_size = reinterpret_cast<const char *>(&vec_size);
    this->write_buf->insert(this->write_buf->end(), bytes_size,
                            bytes_size + sizeof(vec_size));

    if constexpr (is_valid_scalar<ValT> || is_valid_container<ValT>) {
        for (size_t i = 0; i < data.size(); ++i) {
            status = serialize_data(data[i]);
            if (status != da_status_success)
                return status;
        }
    } else {
        static_assert(is_valid_container<ValT>,
                      "Unsupported element type for serialization.");
    }
    return status;
}

template <typename T> da_status serialization_buffer::serialize_data(const T &data) {
    if constexpr (is_valid_scalar<T>) {
        // Cast data if necessary to a save safe type
        save_type_t<T> val = static_cast<save_type_t<T>>(data);
        const char *bytes = reinterpret_cast<const char *>(&val);
        this->write_buf->insert(this->write_buf->end(), bytes, bytes + sizeof(val));
    } else if constexpr (is_valid_container<T>) {
        return serialize_container_impl(data);
    } else {
        static_assert(is_valid_scalar<T>, "Unsupported element type for serialization.");
    }
    return da_status_success;
}

// LOADING KERNELS

template <typename Container>
da_status serialization_buffer::deserialize_container_impl(Container &data) {
    da_status status = da_status_success;
    using ValT = typename Container::value_type;

    // Load container length which is stored first in the stream.
    if (this->offset + sizeof(int_save_t) > this->get_size()) {
        return da_status_invalid_file_data;
    }
    int_save_t vec_size;
    std::memcpy(&vec_size, this->read_ptr + this->offset, sizeof(vec_size));
    this->offset += sizeof(vec_size);

    if (vec_size < 0 || (size_t)vec_size > (this->get_size() - this->offset)) {
        return da_status_invalid_file_data;
    }

    try {
        data.resize(vec_size);
    } catch (std::bad_alloc const &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    }

    if constexpr (is_valid_scalar<ValT> || is_valid_container<ValT>) {
        // Load each element through read_ptr (scalar conversion or recursive container load).
        for (da_int i = 0; i < vec_size; ++i) {
            status = deserialize_data(data[i]);
            if (status != da_status_success)
                return status;
        }
    } else {
        static_assert(is_valid_container<ValT>,
                      "Unsupported element type for deserialization.");
    }
    return status;
}

template <typename T> da_status serialization_buffer::deserialize_data(T &data) {
    if constexpr (is_valid_scalar<T>) {
        using SavedT = save_type_t<T>;
        // Check bounds before reading to prevent buffer overflow.
        if (this->offset + sizeof(SavedT) > this->get_size()) {
            return da_status_invalid_file_data;
        }
        SavedT val;
        // Read normalized on-disk representation and cast to runtime type.
        std::memcpy(&val, this->read_ptr + this->offset, sizeof(SavedT));
        this->offset += sizeof(SavedT);
        data = static_cast<T>(val);
    } else if constexpr (is_valid_container<T>) {
        return deserialize_container_impl(data);
    } else {
        static_assert(is_valid_scalar<T>,
                      "Unsupported element type for deserialization.");
    }
    return da_status_success;
}

// REROUTE KERNELS

template <typename T> da_status serialization_buffer::dispatch_buffer_io(T &data) {
    da_status status = da_status_success;
    if (this->mode == buffer_mode::reserve) {
        status = this->add_size(get_type_size(data));
    } else if (this->mode == serialize) {
        status = serialize_data(data);
    } else {
        status = deserialize_data(data);
    }
    return status;
};

// USER DATA KERNELS

template <typename T>
void serialization_buffer::serialize_user_data_impl(const T *X, da_int inner_dim) {
    if constexpr (std::is_same_v<T, da_int>) {
        // Needs casting to save a safe type.
        for (da_int j = 0; j < inner_dim; ++j) {
            int_save_t val = static_cast<int_save_t>(X[j]);
            const char *bytes = reinterpret_cast<const char *>(&val);
            this->write_buf->insert(this->write_buf->end(), bytes, bytes + sizeof(val));
        }
    } else {
        const char *bytes = reinterpret_cast<const char *>(X);
        this->write_buf->insert(this->write_buf->end(), bytes,
                                bytes + (inner_dim * sizeof(T)));
    }
}

template <typename T>
da_status serialization_buffer::serialize_user_data(const T *X, da_order order, da_int m,
                                                    da_int n, da_int ldx) {
    da_status status;

    // For 1D input, force column-major handling so stride logic stays consistent.
    if (n == 1) {
        order = column_major;
    }

    // Save element count first so the payload can be loaded as a vector<T>.
    // If nullptr make sure that the vector size saved is 0.
    da_int vector_size = X != nullptr ? m * n : 0;
    status = dispatch_buffer_io(vector_size);
    if (status != da_status_success)
        return status;

    // If X == nullptr return success, so it is saved as 0 size vector
    if (X == nullptr)
        return status;

    if (this->mode == buffer_mode::reserve) {
        status = this->add_size(size_t(vector_size) * sizeof(save_type_t<T>));
        return status;
    }

    // Serialize only logical elements and skip any stride padding when ldx > m (ldx > n for row major).
    if (order == column_major) {
        for (da_int i = 0; i < n; ++i) {
            const T *col_ptr = X + i * ldx;
            serialize_user_data_impl(col_ptr, m);
        }
    } else if (order == row_major) {
        for (da_int i = 0; i < m; ++i) {
            const T *row_ptr = X + i * ldx;
            serialize_user_data_impl(row_ptr, n);
        }
    }
    return status;
}

// EXPLICIT INSTANTIATION

// SAVE

template da_status serialization_buffer::serialize_data(const bool &data);
template da_status serialization_buffer::serialize_data(const da_int &data);
template da_status serialization_buffer::serialize_data(const std::string &data);
template da_status serialization_buffer::serialize_data(const float &data);
template da_status serialization_buffer::serialize_data(const double &data);
template da_status serialization_buffer::serialize_data(const da_order &data);
template da_status serialization_buffer::serialize_data(const da_svm_model &data);
template da_status serialization_buffer::serialize_data(const da_metric &data);
template da_status serialization_buffer::serialize_data(const linmod_model &data);
template da_status serialization_buffer::serialize_data(const linmod_method &data);
template da_status serialization_buffer::serialize_data(const logistic_constraint &data);
template da_status serialization_buffer::serialize_data(const split_property &data);
template da_status serialization_buffer::serialize_data(const approx_nn_metric &data);
template da_status serialization_buffer::serialize_data(const da_handle_type &data);

template da_status serialization_buffer::serialize_data(const std::vector<da_int> &data);
template da_status serialization_buffer::serialize_data(const std::vector<float> &data);
template da_status serialization_buffer::serialize_data(const std::vector<double> &data);
template da_status
serialization_buffer::serialize_data(const da_vector::da_vector<da_int> &data);
template da_status
serialization_buffer::serialize_data(const da_vector::da_vector<float> &data);
template da_status
serialization_buffer::serialize_data(const da_vector::da_vector<double> &data);
template da_status serialization_buffer::serialize_data(
    const std::vector<da_vector::da_vector<da_int>> &data);
template da_status serialization_buffer::serialize_data(
    const std::vector<da_vector::da_vector<float>> &data);
template da_status serialization_buffer::serialize_data(
    const std::vector<da_vector::da_vector<double>> &data);

// LOAD
template da_status serialization_buffer::deserialize_data(bool &data);
template da_status serialization_buffer::deserialize_data(da_int &data);
template da_status serialization_buffer::deserialize_data(std::string &data);
template da_status serialization_buffer::deserialize_data(float &data);
template da_status serialization_buffer::deserialize_data(double &data);
template da_status serialization_buffer::deserialize_data(da_order &data);
template da_status serialization_buffer::deserialize_data(da_svm_model &data);
template da_status serialization_buffer::deserialize_data(da_metric &data);
template da_status serialization_buffer::deserialize_data(linmod_model &data);
template da_status serialization_buffer::deserialize_data(linmod_method &data);
template da_status serialization_buffer::deserialize_data(logistic_constraint &data);
template da_status serialization_buffer::deserialize_data(split_property &data);
template da_status serialization_buffer::deserialize_data(approx_nn_metric &data);
template da_status serialization_buffer::deserialize_data(da_handle_type &data);

template da_status serialization_buffer::deserialize_data(std::vector<da_int> &data);
template da_status serialization_buffer::deserialize_data(std::vector<float> &data);
template da_status serialization_buffer::deserialize_data(std::vector<double> &data);
template da_status
serialization_buffer::deserialize_data(da_vector::da_vector<da_int> &data);
template da_status
serialization_buffer::deserialize_data(da_vector::da_vector<float> &data);
template da_status
serialization_buffer::deserialize_data(da_vector::da_vector<double> &data);

template da_status
serialization_buffer::deserialize_data(std::vector<da_vector::da_vector<da_int>> &data);
template da_status
serialization_buffer::deserialize_data(std::vector<da_vector::da_vector<float>> &data);
template da_status
serialization_buffer::deserialize_data(std::vector<da_vector::da_vector<double>> &data);

// REROUTE

template da_status serialization_buffer::dispatch_buffer_io(bool &data);
template da_status serialization_buffer::dispatch_buffer_io(da_int &data);
template da_status serialization_buffer::dispatch_buffer_io(std::string &data);
template da_status serialization_buffer::dispatch_buffer_io(float &data);
template da_status serialization_buffer::dispatch_buffer_io(double &data);
template da_status serialization_buffer::dispatch_buffer_io(da_order &data);
template da_status serialization_buffer::dispatch_buffer_io(da_svm_model &data);
template da_status serialization_buffer::dispatch_buffer_io(da_metric &data);
template da_status serialization_buffer::dispatch_buffer_io(linmod_model &data);
template da_status serialization_buffer::dispatch_buffer_io(linmod_method &data);
template da_status serialization_buffer::dispatch_buffer_io(logistic_constraint &data);
template da_status serialization_buffer::dispatch_buffer_io(split_property &data);
template da_status serialization_buffer::dispatch_buffer_io(approx_nn_metric &data);

template da_status serialization_buffer::dispatch_buffer_io(std::vector<da_int> &data);
template da_status serialization_buffer::dispatch_buffer_io(std::vector<float> &data);
template da_status serialization_buffer::dispatch_buffer_io(std::vector<double> &data);

template da_status
serialization_buffer::dispatch_buffer_io(da_vector::da_vector<da_int> &data);
template da_status
serialization_buffer::dispatch_buffer_io(da_vector::da_vector<float> &data);
template da_status
serialization_buffer::dispatch_buffer_io(da_vector::da_vector<double> &data);

template da_status
serialization_buffer::dispatch_buffer_io(std::vector<da_vector::da_vector<da_int>> &data);
template da_status
serialization_buffer::dispatch_buffer_io(std::vector<da_vector::da_vector<float>> &data);
template da_status
serialization_buffer::dispatch_buffer_io(std::vector<da_vector::da_vector<double>> &data);

// USER DATA KERNELS

template da_status serialization_buffer::serialize_user_data(const da_int *X,
                                                             da_order order, da_int m,
                                                             da_int n, da_int ldx);
template da_status serialization_buffer::serialize_user_data(const float *X,
                                                             da_order order, da_int m,
                                                             da_int n, da_int ldx);
template da_status serialization_buffer::serialize_user_data(const double *X,
                                                             da_order order, da_int m,
                                                             da_int n, da_int ldx);

template void serialization_buffer::serialize_user_data_impl(const da_int *X,
                                                             da_int inner_dim);
template void serialization_buffer::serialize_user_data_impl(const float *X,
                                                             da_int inner_dim);
template void serialization_buffer::serialize_user_data_impl(const double *X,
                                                             da_int inner_dim);

} // namespace da_model_persistence