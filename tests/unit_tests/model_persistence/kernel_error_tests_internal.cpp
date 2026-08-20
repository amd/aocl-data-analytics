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

#include "aoclda.h"
#include "da_handle.hpp"
#include "model_persistence.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

/*
 * Serialization kernel error and edge case tests.
 *
 * Tests error conditions, boundary cases, and robustness of the serialization
 * buffer API. Covers null pointer handling, buffer overflows, invalid modes,
 * size limits, and malformed data.
*/

using namespace da_model_persistence;

class SerializationKernelErrorTests : public testing::Test {
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============================================================================
// set_buffer_data(const char*, size_t) Tests
// ============================================================================

TEST_F(SerializationKernelErrorTests, SetBufferDataReadNullptr) {
    serialization_buffer buffer(da_handle_uninitialized);
    da_status status = buffer.set_buffer_data(nullptr, 100);
    EXPECT_EQ(status, da_status_invalid_pointer);
}

TEST_F(SerializationKernelErrorTests, SetBufferDataReadZeroSize) {
    char dummy_data[10] = {0};
    serialization_buffer buffer(da_handle_uninitialized);
    da_status status = buffer.set_buffer_data(dummy_data, 0);
    EXPECT_EQ(status, da_status_invalid_input);
}

TEST_F(SerializationKernelErrorTests, SetBufferDataReadModeAndSize) {
    size_t data_size = 10;
    char dummy_data[10] = {0};
    serialization_buffer buffer(da_handle_uninitialized);
    da_status status = buffer.set_buffer_data(dummy_data, data_size);
    ASSERT_EQ(status, da_status_success);
    EXPECT_EQ(buffer.get_mode(), buffer_mode::deserialize);
    EXPECT_EQ(buffer.get_size(), data_size);
}

// ============================================================================
// set_buffer_data(std::vector<char>*) Tests
// ============================================================================

TEST_F(SerializationKernelErrorTests, SetBufferDataWriteNullptr) {
    serialization_buffer buffer(da_handle_uninitialized);
    da_status status = buffer.set_buffer_data(nullptr);
    EXPECT_EQ(status, da_status_invalid_pointer);
}

TEST_F(SerializationKernelErrorTests, SetBufferDataWriteModeAndMetadataSize) {
    std::vector<char> data;
    serialization_buffer buffer(da_handle_uninitialized);
    da_status status = buffer.set_buffer_data(&data);
    ASSERT_EQ(status, da_status_success);
    EXPECT_EQ(buffer.get_mode(), buffer_mode::reserve);
    // Metadata size should be added:
    // 1. keyword string size stored (8 bytes)
    // 2. keyword string (19 bytes)
    // 3. da_int_size (8 bytes)
    // 4. serialization_version (8 bytes)
    // 5. handle_type (8 bytes)
    // 6. precision (8 bytes)
    // 7. aoclda_version string size stored (8 bytes)
    // 8. aoclda_version string (strlen(da_get_version()) bytes)
    size_t expected_metadata_size = sizeof(int64_t) + header_keyword.size() +
                                    sizeof(int64_t) + sizeof(int64_t) + sizeof(int64_t) +
                                    sizeof(int64_t) + sizeof(int64_t) +
                                    std::strlen(da_get_version());
    EXPECT_EQ(buffer.get_size(), expected_metadata_size);
}

// ============================================================================
// add_size Tests
// ============================================================================

TEST_F(SerializationKernelErrorTests, AddSizeModeNotReserve) {
    // Set buffer to deserialize mode
    char dummy_data[10] = {0};
    serialization_buffer buffer(da_handle_uninitialized);
    buffer.set_buffer_data(dummy_data, 10);
    EXPECT_EQ(buffer.get_mode(), buffer_mode::deserialize);

    // Try to add size - should fail
    EXPECT_EQ(buffer.add_size(100), da_status_invalid_option);

    buffer.set_mode(buffer_mode::serialize);
    EXPECT_EQ(buffer.get_mode(), buffer_mode::serialize);
    EXPECT_EQ(buffer.add_size(100), da_status_invalid_option);
}

TEST_F(SerializationKernelErrorTests, AddSizeValueAboveLimit) {
    std::vector<char> data;
    serialization_buffer buffer(da_handle_uninitialized);
    buffer.set_buffer_data(&data);

    // Try to add a value larger than da_int max
    size_t big_value = static_cast<size_t>(std::numeric_limits<da_int>::max()) + 1;
    da_status status = buffer.add_size(big_value);
    EXPECT_EQ(status, da_status_invalid_input);
}

TEST_F(SerializationKernelErrorTests, AddSizeOverflow) {
    std::vector<char> data;
    serialization_buffer buffer(da_handle_uninitialized);
    buffer.set_buffer_data(&data);

    // Set size to near max, accounting for metadata size already in the buffer
    size_t metadata_size = buffer.get_size();
    size_t near_max =
        static_cast<size_t>(std::numeric_limits<da_int>::max()) - metadata_size;
    da_status status = buffer.add_size(near_max);
    EXPECT_EQ(status, da_status_success);

    // Try to add more - should overflow
    status = buffer.add_size(100);
    EXPECT_EQ(status, da_status_invalid_input);
}

TEST_F(SerializationKernelErrorTests, AddSizeSuccess) {
    std::vector<char> data;
    serialization_buffer buffer(da_handle_uninitialized);
    buffer.set_buffer_data(&data);

    size_t initial_size = buffer.get_size();
    da_status status = buffer.add_size(100);
    ASSERT_EQ(status, da_status_success);
    EXPECT_EQ(buffer.get_size(), initial_size + 100);
}

// ============================================================================
// clear_data Tests
// ============================================================================

TEST_F(SerializationKernelErrorTests, ClearDataNullptr) {
    // Create buffer without setting write_buf
    serialization_buffer buffer(da_handle_uninitialized);
    da_status status = buffer.clear_data();
    EXPECT_EQ(status, da_status_invalid_pointer);
}

TEST_F(SerializationKernelErrorTests, ClearDataSuccess) {
    std::vector<char> data = {1, 2, 3, 4, 5};
    serialization_buffer buffer(da_handle_uninitialized);
    buffer.set_buffer_data(&data);

    // Serialize something to add data
    buffer.serialize_data(da_int(42));
    ASSERT_FALSE(data.empty());

    // Clear should empty the vector
    da_status status = buffer.clear_data();
    ASSERT_EQ(status, da_status_success);
    EXPECT_TRUE(data.empty());
}

// ============================================================================
// reserve Tests
// ============================================================================

TEST_F(SerializationKernelErrorTests, ReserveModeNotReserve) {
    char dummy_data[10] = {0};
    serialization_buffer buffer(da_handle_uninitialized);
    buffer.set_buffer_data(dummy_data, 10);
    EXPECT_EQ(buffer.get_mode(), buffer_mode::deserialize);

    da_status status = buffer.reserve();
    EXPECT_EQ(status, da_status_invalid_option);

    buffer.set_mode(buffer_mode::serialize);
    EXPECT_EQ(buffer.get_mode(), buffer_mode::serialize);
    EXPECT_EQ(buffer.reserve(), da_status_invalid_option);
}

TEST_F(SerializationKernelErrorTests, ReserveNullptr) {
    serialization_buffer buffer(da_handle_uninitialized);
    // Don't set write_buf, so it's nullptr
    da_status status = buffer.reserve();
    EXPECT_EQ(status, da_status_invalid_pointer);
}

TEST_F(SerializationKernelErrorTests, ReserveSuccess) {
    std::vector<char> data;
    serialization_buffer buffer(da_handle_uninitialized);
    buffer.set_buffer_data(&data);

    buffer.add_size(1000);
    da_status status = buffer.reserve();
    ASSERT_EQ(status, da_status_success);
    EXPECT_GE(data.capacity(), buffer.get_size());
}

// ============================================================================
// serialize_container_impl Tests
// ============================================================================

TEST_F(SerializationKernelErrorTests, SerializeEmptyVector) {
    std::vector<char> buffer_data;
    serialization_buffer buffer(da_handle_uninitialized);
    buffer.set_buffer_data(&buffer_data);

    std::vector<da_int> empty_vec;
    da_status status = buffer.serialize_data(empty_vec);
    ASSERT_EQ(status, da_status_success);

    // Deserialize and verify it's empty
    serialization_buffer read_buffer(da_handle_uninitialized);
    read_buffer.set_buffer_data(buffer_data.data(), buffer_data.size());

    std::vector<da_int> result;
    status = read_buffer.deserialize_data(result);
    ASSERT_EQ(status, da_status_success);
    EXPECT_TRUE(result.empty());
}

TEST_F(SerializationKernelErrorTests, SerializeVectorWithReservedCapacity) {
    std::vector<char> buffer_data;
    serialization_buffer buffer(da_handle_uninitialized);
    buffer.set_buffer_data(&buffer_data);

    // Vector with capacity but no elements
    std::vector<float> vec_with_capacity;
    vec_with_capacity.reserve(100);
    ASSERT_EQ(vec_with_capacity.size(), (size_t)0);
    ASSERT_GE(vec_with_capacity.capacity(), (size_t)100);

    da_status status = buffer.serialize_data(vec_with_capacity);
    ASSERT_EQ(status, da_status_success);

    // Deserialize - should still be empty
    serialization_buffer read_buffer(da_handle_uninitialized);
    read_buffer.set_buffer_data(buffer_data.data(), buffer_data.size());

    std::vector<float> result;
    status = read_buffer.deserialize_data(result);
    ASSERT_EQ(status, da_status_success);
    EXPECT_TRUE(result.empty());
}

// ============================================================================
// deserialize_container_impl Tests
// ============================================================================

TEST_F(SerializationKernelErrorTests, DeserializeContainerBufferOverflow) {
    // Create buffer with size field but no actual data
    std::vector<char> data;
    int64_t vec_size = 10; // Claims 10 elements
    const char *bytes = reinterpret_cast<const char *>(&vec_size);
    data.insert(data.end(), bytes, bytes + sizeof(vec_size));
    // Don't add actual elements - buffer too small

    serialization_buffer buffer(da_handle_uninitialized);
    buffer.set_buffer_data(data.data(), data.size());

    std::vector<double> result;
    da_status status = buffer.deserialize_data(result);
    EXPECT_EQ(status, da_status_invalid_file_data);
}

TEST_F(SerializationKernelErrorTests, DeserializeContainerZeroSize) {
    std::vector<char> data;
    int64_t vec_size = 0;
    const char *bytes = reinterpret_cast<const char *>(&vec_size);
    data.insert(data.end(), bytes, bytes + sizeof(vec_size));

    serialization_buffer buffer(da_handle_uninitialized);
    buffer.set_buffer_data(data.data(), data.size());

    std::vector<da_int> result = {1, 2, 3}; // Start non-empty
    da_status status = buffer.deserialize_data(result);
    ASSERT_EQ(status, da_status_success);
    EXPECT_TRUE(result.empty());
}

// ============================================================================
// deserialize_data Tests
// ============================================================================

TEST_F(SerializationKernelErrorTests, DeserializeScalarBufferOverflow) {
    // Buffer too small for a double
    std::vector<char> data = {1, 2, 3}; // Only 3 bytes

    serialization_buffer buffer(da_handle_uninitialized);
    buffer.set_buffer_data(data.data(), data.size());

    double result;
    da_status status = buffer.deserialize_data(result);
    EXPECT_EQ(status, da_status_invalid_file_data);
}

TEST_F(SerializationKernelErrorTests, DeserializeMultipleScalarsOverflow) {
    // Buffer with exactly one int64
    std::vector<char> data(sizeof(int64_t), 0x00);

    serialization_buffer buffer(da_handle_uninitialized);
    buffer.set_buffer_data(data.data(), data.size());

    da_int first;
    da_status status = buffer.deserialize_data(first);
    ASSERT_EQ(status, da_status_success);

    // Try to read another - should fail
    da_int second;
    status = buffer.deserialize_data(second);
    EXPECT_EQ(status, da_status_invalid_file_data);
}

TEST_F(SerializationKernelErrorTests, DeserializeStringBufferOverflow) {
    // String with claimed length longer than buffer
    std::vector<char> data;
    int64_t str_len = 1000; // Claims 1000 chars
    const char *bytes = reinterpret_cast<const char *>(&str_len);
    data.insert(data.end(), bytes, bytes + sizeof(str_len));
    // Add only 5 chars instead of 1000
    data.insert(data.end(), {'h', 'e', 'l', 'l', 'o'});

    serialization_buffer buffer(da_handle_uninitialized);
    buffer.set_buffer_data(data.data(), data.size());

    std::string result;
    da_status status = buffer.deserialize_data(result);
    EXPECT_EQ(status, da_status_invalid_file_data);
}

// ============================================================================
// Edge Cases for Nested Containers
// ============================================================================

TEST_F(SerializationKernelErrorTests, SerializeNestedEmptyVectors) {
    std::vector<char> buffer_data;
    serialization_buffer buffer(da_handle_uninitialized);
    buffer.set_buffer_data(&buffer_data);

    // Outer vector with empty inner da_vectors (supported type)
    std::vector<da_vector::da_vector<da_int>> nested;
    nested.resize(3); // 3 empty da_vectors

    da_status status = buffer.serialize_data(nested);
    ASSERT_EQ(status, da_status_success);

    // Deserialize and verify structure
    serialization_buffer read_buffer(da_handle_uninitialized);
    read_buffer.set_buffer_data(buffer_data.data(), buffer_data.size());

    std::vector<da_vector::da_vector<da_int>> result;
    status = read_buffer.deserialize_data(result);
    ASSERT_EQ(status, da_status_success);
    ASSERT_EQ(result.size(), (size_t)3);
    for (const auto &inner : result) {
        EXPECT_EQ(inner.size(), (size_t)0);
    }
}

TEST_F(SerializationKernelErrorTests, DeserializeNestedBufferOverflow) {
    std::vector<char> data;

    // Outer size = 2
    int64_t outer_size = 2;
    const char *bytes = reinterpret_cast<const char *>(&outer_size);
    data.insert(data.end(), bytes, bytes + sizeof(outer_size));

    // First inner vector: size = 1, with one element
    int64_t inner1_size = 1;
    bytes = reinterpret_cast<const char *>(&inner1_size);
    data.insert(data.end(), bytes, bytes + sizeof(inner1_size));
    int64_t val1 = 42;
    bytes = reinterpret_cast<const char *>(&val1);
    data.insert(data.end(), bytes, bytes + sizeof(val1));

    // Second inner vector: claim size = 5 but provide no data
    int64_t inner2_size = 5;
    bytes = reinterpret_cast<const char *>(&inner2_size);
    data.insert(data.end(), bytes, bytes + sizeof(inner2_size));
    // No actual data for the 5 elements

    serialization_buffer buffer(da_handle_uninitialized);
    buffer.set_buffer_data(data.data(), data.size());

    std::vector<da_vector::da_vector<da_int>> result;
    da_status status = buffer.deserialize_data(result);
    EXPECT_EQ(status, da_status_invalid_file_data);
}

// ============================================================================
// de/serialize_metadata Tests
// ============================================================================

TEST_F(SerializationKernelErrorTests, De_SerializeMetadataSuccess) {
    da_int serialization_version = model_persistence_min_version;
    std::vector<char> data;
    serialization_buffer buffer(da_handle_uninitialized);
    buffer.set_buffer_data(&data);

    da_status status = buffer.serialize_metadata(sizeof(float), serialization_version);
    ASSERT_EQ(status, da_status_success);
    EXPECT_FALSE(data.empty());

    // Verify we can deserialize it back
    serialization_buffer read_buffer(da_handle_uninitialized);
    read_buffer.set_buffer_data(data.data(), data.size());

    da_int precision;
    status = read_buffer.deserialize_metadata(precision);
    ASSERT_EQ(status, da_status_success);
    EXPECT_EQ(precision, da_int(sizeof(float)));
    EXPECT_EQ(serialization_version, read_buffer.get_saved_serialization_version());
}

// ============================================================================
// Version Check Tests
// ============================================================================

// Helper: serialize valid metadata with a given version into a buffer
static std::vector<char> make_metadata_buffer(da_int version,
                                              da_handle_type handle_type = da_handle_pca,
                                              da_int precision = sizeof(double)) {
    std::vector<char> data;
    serialization_buffer buffer(handle_type);
    buffer.set_buffer_data(&data);
    buffer.set_mode(buffer_mode::serialize);
    buffer.serialize_metadata(precision, version);
    return data;
}

// saved_serialization_version < model_persistence_min_version
// → rejected at deserialize_metadata level
TEST_F(SerializationKernelErrorTests, VersionCheck_SavedBelowGlobalMinVersion) {
    da_int saved_version = model_persistence_min_version - 1;
    auto data = make_metadata_buffer(saved_version);

    serialization_buffer buffer(da_handle_pca);
    buffer.set_buffer_data(data.data(), data.size());

    da_int precision;
    da_status status = buffer.deserialize_metadata(precision);
    EXPECT_EQ(status, da_status_version_mismatch);
}

// saved_serialization_version < algo's serialization_version (both >= min_version)
// → passes deserialize_metadata, rejected at load_model (exact match fails)
TEST_F(SerializationKernelErrorTests, VersionCheck_SavedSmallerThanAlgoVer) {
    da_int algo_ver = model_persistence_min_version + 1;
    da_int saved_version = model_persistence_min_version;
    auto data = make_metadata_buffer(saved_version);

    serialization_buffer buffer(da_handle_pca);
    buffer.set_buffer_data(data.data(), data.size());

    da_int precision;
    da_status status = buffer.deserialize_metadata(precision);
    ASSERT_EQ(status, da_status_success);

    da_handle handle = nullptr;
    status = da_handle_init_d(&handle, da_handle_pca);
    ASSERT_EQ(status, da_status_success);

    auto *alg = handle->get_alg_handle<double>();
    alg->set_serialization_version(algo_ver);

    status = alg->load_model(buffer);
    EXPECT_EQ(status, da_status_version_mismatch);

    da_handle_destroy(&handle);
}

// saved_serialization_version > algo's serialization_version (both >= min_version)
// → passes deserialize_metadata, rejected at load_model (exact match fails)
TEST_F(SerializationKernelErrorTests, VersionCheck_SavedGreaterThanAlgoVer) {
    da_int algo_ver = model_persistence_min_version;
    da_int saved_version = model_persistence_min_version + 1;
    auto data = make_metadata_buffer(saved_version);

    serialization_buffer buffer(da_handle_pca);
    buffer.set_buffer_data(data.data(), data.size());

    da_int precision;
    da_status status = buffer.deserialize_metadata(precision);
    ASSERT_EQ(status, da_status_success);

    da_handle handle = nullptr;
    status = da_handle_init_d(&handle, da_handle_pca);
    ASSERT_EQ(status, da_status_success);

    auto *alg = handle->get_alg_handle<double>();
    alg->set_serialization_version(algo_ver);

    status = alg->load_model(buffer);
    EXPECT_EQ(status, da_status_version_mismatch);

    da_handle_destroy(&handle);
}

// ============================================================================
// Serialize-side overflow Tests (insert_data_in_buffer guard)
// ============================================================================

template <typename T> static void run_scalar_overflow() {
    std::vector<char> buffer_data;
    serialization_buffer buffer(da_handle_uninitialized);
    ASSERT_EQ(buffer.set_buffer_data(&buffer_data), da_status_success);

    size_t reserved = buffer.get_size();
    buffer.set_mode(buffer_mode::serialize);

    // Serialize scalars until the reserved size is exceeded.
    size_t max_fit = reserved / sizeof(save_type_t<T>);
    da_status status = da_status_success;
    for (size_t i = 0; i <= max_fit; ++i) {
        status = buffer.serialize_data(T(1));
        if (status != da_status_success)
            break;
    }
    EXPECT_EQ(status, da_status_internal_error);
}

TEST_F(SerializationKernelErrorTests, SerializeScalarOverflow) {
    run_scalar_overflow<da_int>();
    run_scalar_overflow<float>();
    run_scalar_overflow<double>();
}

template <typename T> static void run_container_overflow() {
    std::vector<char> buffer_data;
    serialization_buffer buffer(da_handle_uninitialized);
    ASSERT_EQ(buffer.set_buffer_data(&buffer_data), da_status_success);

    size_t reserved = buffer.get_size();
    buffer.set_mode(buffer_mode::serialize);

    std::vector<T> big(reserved / sizeof(save_type_t<T>) + 1, T(1));
    EXPECT_EQ(buffer.serialize_data(big), da_status_internal_error);
}

TEST_F(SerializationKernelErrorTests, SerializeContainerOverflow) {
    run_container_overflow<da_int>();
    run_container_overflow<float>();
    run_container_overflow<double>();
}

// ============================================================================
// serialize_user_data / serialize_user_data_impl Tests
// ============================================================================

template <typename T> static void run_user_data_overflow_1d(da_order order) {
    std::vector<char> buffer_data;
    serialization_buffer buffer(da_handle_uninitialized);
    ASSERT_EQ(buffer.set_buffer_data(&buffer_data), da_status_success);

    ASSERT_EQ(buffer.add_size(size_t(100)), da_status_success);
    size_t reserved = buffer.get_size();

    buffer.set_mode(buffer_mode::serialize);

    da_int count = static_cast<da_int>(reserved / sizeof(save_type_t<T>)) + 1;
    std::vector<T> data(count, T(1));
    da_status status = buffer.serialize_user_data(data.data(), order, count, 1, count);
    EXPECT_EQ(status, da_status_internal_error);
}

template <typename T> static void run_user_data_overflow_2d(da_order order) {
    std::vector<char> buffer_data;
    serialization_buffer buffer(da_handle_uninitialized);
    ASSERT_EQ(buffer.set_buffer_data(&buffer_data), da_status_success);

    // Add extra space to ensure matrix will be of useful size
    ASSERT_EQ(buffer.add_size(size_t(200)), da_status_success);
    size_t reserved = buffer.get_size();
    buffer.set_mode(buffer_mode::serialize);

    da_int n_elements = static_cast<da_int>(reserved / sizeof(save_type_t<T>));
    da_int n_rows = static_cast<da_int>(std::sqrt(n_elements));

    // Add more columns to ensure it overflows
    da_int n_cols = n_rows + 5;

    std::vector<T> data(n_cols * n_rows, T(1));
    da_status status = buffer.serialize_user_data(
        data.data(), order, n_rows, n_cols, order == column_major ? n_rows : n_cols);
    EXPECT_EQ(status, da_status_internal_error);
}

TEST_F(SerializationKernelErrorTests, SerializeUserDataOverflow) {
    run_user_data_overflow_1d<da_int>(column_major);
    run_user_data_overflow_1d<float>(column_major);
    run_user_data_overflow_1d<double>(column_major);

    run_user_data_overflow_1d<da_int>(row_major);
    run_user_data_overflow_1d<float>(row_major);
    run_user_data_overflow_1d<double>(row_major);

    run_user_data_overflow_2d<da_int>(column_major);
    run_user_data_overflow_2d<float>(column_major);
    run_user_data_overflow_2d<double>(column_major);

    run_user_data_overflow_2d<da_int>(row_major);
    run_user_data_overflow_2d<float>(row_major);
    run_user_data_overflow_2d<double>(row_major);
}

template <typename T>
static void run_user_data_reserve_size(da_order order, da_int extra_ldx) {
    std::vector<char> buffer_data;
    serialization_buffer buffer(da_handle_uninitialized);
    ASSERT_EQ(buffer.set_buffer_data(&buffer_data), da_status_success);
    ASSERT_EQ(buffer.get_mode(), buffer_mode::reserve);

    const da_int saved_data_size = 512;

    // Add extra space to ensure matrix will be of useful size
    size_t initial_size = buffer.get_size();
    ASSERT_EQ(
        buffer.add_size(size_t(saved_data_size) - initial_size + sizeof(int_save_t)),
        da_status_success);

    initial_size = buffer.get_size();
    da_int outer_dim = 4;
    da_int inner_dim = initial_size / (outer_dim * sizeof(save_type_t<T>));
    ASSERT_EQ(outer_dim * inner_dim * sizeof(save_type_t<T>) + sizeof(int_save_t),
              saved_data_size + sizeof(int_save_t));

    da_int ldx = inner_dim + extra_ldx;
    std::vector<T> data(ldx * outer_dim, T(1));

    da_int m = order == column_major ? inner_dim : outer_dim;
    da_int n = order == column_major ? outer_dim : inner_dim;

    // this adds another saved_data_size + sizeof(int_save_t) to the size
    da_status status = buffer.serialize_user_data(data.data(), order, m, n, ldx);
    ASSERT_EQ(status, da_status_success);

    EXPECT_EQ(buffer.get_size(), 2 * (saved_data_size + sizeof(int_save_t)));

    // ** Additional serialization to ensure everything is correct **
    buffer.set_mode(buffer_mode::serialize);
    status = buffer.serialize_user_data(data.data(), order, m, n, ldx);
    ASSERT_EQ(status, da_status_success);
    status = buffer.serialize_user_data(data.data(), order, m, n, ldx);
    ASSERT_EQ(status, da_status_success);
    EXPECT_EQ(buffer.get_size(), 2 * (saved_data_size + sizeof(int_save_t)));
}

TEST_F(SerializationKernelErrorTests, SerializeUserDataReserveSize) {
    run_user_data_reserve_size<da_int>(column_major, 0);
    run_user_data_reserve_size<float>(column_major, 0);
    run_user_data_reserve_size<double>(column_major, 0);

    run_user_data_reserve_size<da_int>(row_major, 0);
    run_user_data_reserve_size<float>(row_major, 0);
    run_user_data_reserve_size<double>(row_major, 0);

    run_user_data_reserve_size<da_int>(column_major, 2);
    run_user_data_reserve_size<float>(column_major, 5);
    run_user_data_reserve_size<double>(column_major, 4);

    run_user_data_reserve_size<da_int>(row_major, 3);
    run_user_data_reserve_size<float>(row_major, 8);
    run_user_data_reserve_size<double>(row_major, 2);
}

TEST_F(SerializationKernelErrorTests, SerializeUserDataReserveModeNullptr) {
    std::vector<char> buffer_data;
    serialization_buffer buffer(da_handle_uninitialized);
    ASSERT_EQ(buffer.set_buffer_data(&buffer_data), da_status_success);
    ASSERT_EQ(buffer.get_mode(), buffer_mode::reserve);

    size_t size_before = buffer.get_size();
    da_status status = buffer.serialize_user_data<float>(nullptr, column_major, 5, 5, 5);
    ASSERT_EQ(status, da_status_success);

    EXPECT_EQ(buffer.get_size() - size_before, sizeof(int_save_t));
}
