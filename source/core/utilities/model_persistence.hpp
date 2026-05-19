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

#ifndef AOCLDA_MODEL_PERSISTENCE_HPP
#define AOCLDA_MODEL_PERSISTENCE_HPP

#include "aoclda.h"
#include "da_vector.hpp"
#include "macros.h"
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace da_model_persistence {

constexpr std::string_view header_keyword = "AOCLDA_STORED_MODEL";

using int_save_t = int64_t;
using bool_save_t = uint8_t;

// Maps types to their on-disk serialization representation
// (bool/enums/da_int are normalized to fixed-width integral types).
template <typename T>
using save_type_t = std::conditional_t<
    std::is_same_v<T, bool>, bool_save_t,
    std::conditional_t<std::is_same_v<T, da_int> || std::is_enum_v<T>, int_save_t, T>>;

// Allowed scalar types for saving.
template <typename T>
constexpr bool is_valid_scalar =
    std::is_same_v<T, bool> || std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, da_int> || std::is_enum_v<T> || std::is_same_v<T, char>;

// Type trait indicating whether a container type is supported for serialization.
template <typename T> struct is_valid_container_type : std::false_type {};
template <typename T>
struct is_valid_container_type<da_vector::da_vector<T>> : std::true_type {};
template <typename T> struct is_valid_container_type<std::vector<T>> : std::true_type {};
template <> struct is_valid_container_type<std::string> : std::true_type {};

// Helper variable template for convenient usage (consistent with is_valid_scalar).
template <typename T>
constexpr bool is_valid_container = is_valid_container_type<T>::value;

template <typename T> constexpr size_t get_type_size(T const &data);

template <typename T> constexpr size_t get_container_size(T const &data) {
    using ValT = typename T::value_type;
    // Every serialized container starts with its element count.
    size_t total_size = sizeof(int_save_t);

    if constexpr (is_valid_container<ValT>) {
        // For containers of containers, compute size by checking each item one by one.
        for (const auto &item : data) {
            total_size += get_type_size(item);
        }
    } else if constexpr (is_valid_scalar<ValT>) {
        size_t elem_size = sizeof(save_type_t<ValT>);
        total_size += data.size() * elem_size;
    } else {
        static_assert(is_valid_scalar<ValT> || is_valid_container<ValT>,
                      "Unsupported element type.");
    }
    return total_size;
}

template <typename T> constexpr size_t get_type_size(T const &data) {
    if constexpr (is_valid_container<T>) {
        return get_container_size(data);
    } else if constexpr (is_valid_scalar<T>) {
        return sizeof(save_type_t<T>);
    } else {
        static_assert(is_valid_scalar<T>, "Unsupported element type.");
    }
}

enum buffer_mode { reserve = 0, serialize, deserialize };

class serialization_buffer {
  private:
    da_handle_type handle_type = da_handle_uninitialized;
    // Points to the caller-owned input buffer during deserialization.
    const char *read_ptr = nullptr;
    // Points to the output vector that accumulates serialized bytes.
    std::vector<char> *write_buf = nullptr;
    size_t size = 0;
    uint64_t offset = 0;
    buffer_mode mode = buffer_mode::reserve;

  public:
    serialization_buffer(da_handle_type handle_type) : handle_type(handle_type){};

    ~serialization_buffer(){};

    // Setter use when serialization will be completed
    da_status set_buffer_data(std::vector<char> *buffer_data);

    // Setter use when deserialization will be completed
    da_status set_buffer_data(const char *buffer_data, const size_t size);

    template <typename Container>
    da_status serialize_container_impl(const Container &data);

    template <typename T> da_status serialize_data(const T &data);

    template <typename Container> da_status deserialize_container_impl(Container &data);

    template <typename T> da_status deserialize_data(T &data);

    template <typename T> da_status dispatch_buffer_io(T &data);

    template <typename T> void serialize_user_data_impl(const T *X, da_int inner_dim);

    template <typename T>
    da_status serialize_user_data(const T *X, da_order order, da_int m, da_int n,
                                  da_int ldx);

    da_status serialize_metadata(std::size_t precision, da_int min_lib_version);
    da_status deserialize_metadata(da_int &precision);

    constexpr void add_metadata_size() {
        // header_keyword size
        this->size += sizeof(save_type_t<da_int>);
        this->size += header_keyword.size();
        // da_int_size
        this->size += sizeof(save_type_t<da_int>);
        // precision size
        this->size += sizeof(save_type_t<da_int>);
        // min_library_version (supported)
        this->size += sizeof(save_type_t<da_int>);
        this->size += sizeof(save_type_t<da_handle_type>);
        return;
    }

    // Clears any data inside the save_data vector
    da_status clear_data();

    void set_handle_type(da_handle_type type) { this->handle_type = type; }
    da_handle_type get_handle_type() { return this->handle_type; }

    void set_mode(buffer_mode mode) { this->mode = mode; }
    buffer_mode get_mode() { return this->mode; }

    size_t get_size() { return this->size; }
    // Addition method for the size member
    da_status add_size(size_t value);

    // Reserves memory for the write buffer.
    // Used when model is going to be saved and in reserve mode.
    da_status reserve();
};

} // namespace da_model_persistence

#endif