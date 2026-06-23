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
#include "model_persistence.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
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
    // Metadata size should be added: keyword length + 5 int64_t values
    // header_keyword = "AOCLDA_STORED_MODEL" (19 chars)
    // 1. string size (8 bytes)
    // 2. 6 chars for keyword
    // 3. da_int_size (8 bytes)
    // 4. min_lib_version (8 bytes)
    // 5. handle_type (8 bytes)
    // 6. precision (8 bytes)
    size_t expected_metadata_size = 8 + 19 + 8 + 8 + 8 + 8;
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

    // Set size to near max
    // Subtract enough to cover metadata size
    size_t near_max = static_cast<size_t>(std::numeric_limits<da_int>::max()) - 70;
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
    std::vector<char> data;
    serialization_buffer buffer(da_handle_uninitialized);
    buffer.set_buffer_data(&data);

    da_status status = buffer.serialize_metadata(sizeof(float), 100);
    ASSERT_EQ(status, da_status_success);
    EXPECT_FALSE(data.empty());

    // Verify we can deserialize it back
    serialization_buffer read_buffer(da_handle_uninitialized);
    read_buffer.set_buffer_data(data.data(), data.size());

    da_int precision;
    status = read_buffer.deserialize_metadata(precision);
    ASSERT_EQ(status, da_status_success);
    EXPECT_EQ(precision, da_int(sizeof(float)));
}
