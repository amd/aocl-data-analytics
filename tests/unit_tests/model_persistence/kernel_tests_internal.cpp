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

#include "../utest_utils.hpp"
#include "aoclda.h"
#include "approximate_neighbors_types.hpp"
#include "common/tree_options_types.hpp"
#include "da_vector.hpp"
#include "generate_test_data.hpp"
#include "linmod_types.hpp"
#include "model_persistence.hpp"
#include "persistence_test_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstdio>
#include <fstream>
#include <list>

/*
 * Serialization kernel tests for verifying binary format consistency.
 *
 * Tests:
 * - SerializationMatchesGenerator: Verifies that the serialization library produces
 *   identical binary output to the raw generator (generate_test_data.hpp).
 * - FileRoundTrip: Serializes data to file, reads it back, deserializes and verifies
 *   all values match expected. Also validates get_type_size() accuracy.
 *
 * Tested types: bool, da_int, float, double, std::string, std::vector<T>,
 * da_vector<T>, std::vector<da_vector<T>>, and various enums.
*/

using namespace da_model_persistence;

namespace {

// Helper to populate da_vector from initializer list (avoids return-by-value issues)
template <typename T, typename U = T>
void fill_da_vector(da_vector::da_vector<T> &vec, std::initializer_list<U> init) {
    vec.clear();
    for (const auto &val : init) {
        vec.push_back(static_cast<T>(val));
    }
}

// Expected values - must match generate_bin.cpp
// Scalars
constexpr bool exp_bol = true;
constexpr da_int exp_da_in = 42;
constexpr float exp_fl = 127.9f;
constexpr double exp_doubl = 83.22;
const std::string exp_str = "test";

// std::vector<T>
const std::vector<da_int> exp_v_da_int = {88, -72, 0};
const std::vector<float> exp_v_fl = {27.3f, -77.9f};
const std::vector<double> exp_v_doubl = {33.3, -66.9, -250.0, -89.1};

// da_vector::da_vector<T>
void get_exp_dv_int(da_vector::da_vector<da_int> &vec) {
    fill_da_vector(vec, {100, -200});
}
void get_exp_dv_fl(da_vector::da_vector<float> &vec) {
    fill_da_vector(vec, {1.5f, 2.5f, 3.5f});
}
void get_exp_dv_doubl(da_vector::da_vector<double> &vec) { fill_da_vector(vec, {10.1}); }

// std::vector<da_vector::da_vector<T>>
void get_exp_vdv_int(std::vector<da_vector::da_vector<da_int>> &result) {
    result.clear();
    result.resize(3);
    fill_da_vector(result[0], {1, 2});
    fill_da_vector(result[1], {3});
    fill_da_vector(result[2], {4, 5, 6, 7});
}
void get_exp_vdv_fl(std::vector<da_vector::da_vector<float>> &result) {
    result.clear();
    result.resize(2);
    fill_da_vector(result[0], {1.1f});
    fill_da_vector(result[1], {2.2f, 3.3f});
}
void get_exp_vdv_doubl(std::vector<da_vector::da_vector<double>> &result) {
    result.clear();
    result.resize(2);
    fill_da_vector(result[0], {-1.0, -2.0, -3.0});
    fill_da_vector(result[1], {99.9});
}

// Enums
constexpr da_order exp_ord = column_major;
constexpr da_svm_model exp_svm_mod = svr;
constexpr da_metric exp_metr = da_minkowski;
constexpr linmod_model exp_lin_mod = linmod_model_undefined;
constexpr da_linmod_types::logistic_constraint exp_log_con = da_linmod_types::rsc;
constexpr da_tree_options_types::split_property exp_spl_prop =
    da_tree_options_types::categorical_onevall;
constexpr da_approx_nn_types::approx_nn_metric exp_ann_metr =
    da_approx_nn_types::inner_product;
constexpr da_handle_type exp_han_type = da_handle_linmod;

// User data (serialize_user_data)
const std::vector<float> exp_user_data_fl = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
const std::vector<double> exp_user_data_doubl = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
const std::vector<float> exp_user_data_nullptr = {};

// Helper to load binary file into vector<char>
std::vector<char> load_binary_file(const std::string &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return {};
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    if (!file.good()) {
        return {};
    }
    return buffer;
}

// Helper to write vector<char> to binary file
bool write_binary_file(const std::string &path, const std::vector<char> &data) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file.write(data.data(), data.size());
    return file.good();
}

// Compare two binary files without loading into memory
bool compare_binary_files(const std::string &path1, const std::string &path2) {
    std::ifstream f1(path1, std::ios::binary);
    std::ifstream f2(path2, std::ios::binary);

    if (!f1.is_open() || !f2.is_open())
        return false;

    // Compare sizes first
    f1.seekg(0, std::ios::end);
    f2.seekg(0, std::ios::end);
    if (f1.tellg() != f2.tellg())
        return false;

    f1.seekg(0, std::ios::beg);
    f2.seekg(0, std::ios::beg);

    // Compare byte by byte
    char c1, c2;
    while (f1.get(c1) && f2.get(c2)) {
        if (c1 != c2)
            return false;
    }
    return true;
}

size_t get_buffer_size() {
    size_t size = 0;

    // Scalars
    size += get_type_size(exp_bol);
    size += get_type_size(exp_da_in);
    size += get_type_size(exp_fl);
    size += get_type_size(exp_doubl);
    size += get_type_size(exp_str);

    // std::vector<T>
    size += get_type_size(exp_v_da_int);
    size += get_type_size(exp_v_fl);
    size += get_type_size(exp_v_doubl);

    // da_vector::da_vector<T>
    da_vector::da_vector<da_int> dv_int;
    get_exp_dv_int(dv_int);
    size += get_type_size(dv_int);
    da_vector::da_vector<float> dv_fl;
    get_exp_dv_fl(dv_fl);
    size += get_type_size(dv_fl);
    da_vector::da_vector<double> dv_doubl;
    get_exp_dv_doubl(dv_doubl);
    size += get_type_size(dv_doubl);

    // std::vector<da_vector::da_vector<T>>
    std::vector<da_vector::da_vector<da_int>> vdv_int;
    get_exp_vdv_int(vdv_int);
    size += get_type_size(vdv_int);
    std::vector<da_vector::da_vector<float>> vdv_fl;
    get_exp_vdv_fl(vdv_fl);
    size += get_type_size(vdv_fl);
    std::vector<da_vector::da_vector<double>> vdv_doubl;
    get_exp_vdv_doubl(vdv_doubl);
    size += get_type_size(vdv_doubl);

    // Enums
    size += get_type_size(exp_ord);
    size += get_type_size(exp_svm_mod);
    size += get_type_size(exp_metr);
    size += get_type_size(exp_lin_mod);
    size += get_type_size(exp_log_con);
    size += get_type_size(exp_spl_prop);
    size += get_type_size(exp_ann_metr);
    size += get_type_size(exp_han_type);

    // User data
    size += get_type_size(exp_user_data_fl);
    size += get_type_size(exp_user_data_doubl);
    size += get_type_size(exp_user_data_nullptr);

    return size;
}

// Serialize all expected values into a buffer
da_status serialize_all(std::vector<char> &data) {
    da_model_persistence::serialization_buffer buffer(da_handle_uninitialized);
    da_status status = buffer.set_buffer_data(&data);
    if (status != da_status_success)
        return status;

    // Switch from reserve mode to serialize mode so serialize_user_data writes bytes
    buffer.set_mode(buffer_mode::serialize);

    auto serialize = [&buffer, &status](auto &data) -> void {
        if (status != da_status_success)
            return;
        status = buffer.serialize_data(data);
        return;
    };

    // Scalars
    serialize(exp_bol);
    serialize(exp_da_in);
    serialize(exp_fl);
    serialize(exp_doubl);
    serialize(exp_str);

    // std::vector<T>
    serialize(exp_v_da_int);
    serialize(exp_v_fl);
    serialize(exp_v_doubl);

    // da_vector::da_vector<T>
    da_vector::da_vector<da_int> dv_int;
    get_exp_dv_int(dv_int);
    serialize(dv_int);
    da_vector::da_vector<float> dv_fl;
    get_exp_dv_fl(dv_fl);
    serialize(dv_fl);
    da_vector::da_vector<double> dv_doubl;
    get_exp_dv_doubl(dv_doubl);
    serialize(dv_doubl);

    // std::vector<da_vector::da_vector<T>>
    std::vector<da_vector::da_vector<da_int>> vdv_int;
    get_exp_vdv_int(vdv_int);
    serialize(vdv_int);
    std::vector<da_vector::da_vector<float>> vdv_fl;
    get_exp_vdv_fl(vdv_fl);
    serialize(vdv_fl);
    std::vector<da_vector::da_vector<double>> vdv_doubl;
    get_exp_vdv_doubl(vdv_doubl);
    serialize(vdv_doubl);

    // Enums
    serialize(exp_ord);
    serialize(exp_svm_mod);
    serialize(exp_metr);
    serialize(exp_lin_mod);
    serialize(exp_log_con);
    serialize(exp_spl_prop);
    serialize(exp_ann_metr);
    serialize(exp_han_type);

    // User data (serialize_user_data)
    // 2x3 column-major float array
    if (status != da_status_success)
        return status;
    status = buffer.serialize_user_data(exp_user_data_fl.data(), column_major, 2, 3, 2);
    // 3x2 row-major double array
    if (status != da_status_success)
        return status;
    status = buffer.serialize_user_data(exp_user_data_doubl.data(), row_major, 3, 2, 2);
    // nullptr case
    if (status != da_status_success)
        return status;
    status = buffer.serialize_user_data<float>(nullptr, column_major, 2, 3, 2);

    return status;
}

// Deserialize and verify all values match expected
da_status deserialize_and_verify(std::vector<char> &buffer_data) {
    da_model_persistence::serialization_buffer buffer(da_handle_uninitialized);
    da_status status = buffer.set_buffer_data(buffer_data.data(), buffer_data.size());
    if (status != da_status_success)
        return status;

    auto deserialize = [&buffer, &status](auto &data) -> void {
        if (status != da_status_success)
            return;
        status = buffer.deserialize_data(data);
        return;
    };

    // Scalars
    bool bol;
    deserialize(bol);
    EXPECT_EQ(bol, exp_bol);

    da_int da_in;
    deserialize(da_in);
    EXPECT_EQ(da_in, exp_da_in);

    float fl;
    deserialize(fl);
    EXPECT_FLOAT_EQ(fl, exp_fl);

    double doubl;
    deserialize(doubl);
    EXPECT_DOUBLE_EQ(doubl, exp_doubl);

    std::string str;
    deserialize(str);
    EXPECT_EQ(str, exp_str);

    // std::vector<T>
    std::vector<da_int> v_da_int;
    deserialize(v_da_int);
    EXPECT_EQ(v_da_int.size(), exp_v_da_int.size());
    EXPECT_ARR_EQ((da_int)v_da_int.size(), v_da_int.data(), exp_v_da_int.data(), 1, 1, 0,
                  0);

    std::vector<float> v_fl;
    deserialize(v_fl);
    EXPECT_EQ(v_fl.size(), exp_v_fl.size());
    EXPECT_ARR_EQ((da_int)v_fl.size(), v_fl.data(), exp_v_fl.data(), 1, 1, 0, 0);

    std::vector<double> v_doubl;
    deserialize(v_doubl);
    EXPECT_EQ(v_doubl.size(), exp_v_doubl.size());
    EXPECT_ARR_EQ((da_int)v_doubl.size(), v_doubl.data(), exp_v_doubl.data(), 1, 1, 0, 0);

    // da_vector::da_vector<T>
    da_vector::da_vector<da_int> dv_int;
    deserialize(dv_int);
    da_vector::da_vector<da_int> exp_dv_int;
    get_exp_dv_int(exp_dv_int);
    EXPECT_EQ(dv_int.size(), exp_dv_int.size());
    EXPECT_ARR_EQ((da_int)dv_int.size(), dv_int.data(), exp_dv_int.data(), 1, 1, 0, 0);

    da_vector::da_vector<float> dv_fl;
    deserialize(dv_fl);
    da_vector::da_vector<float> exp_dv_fl;
    get_exp_dv_fl(exp_dv_fl);
    EXPECT_EQ(dv_fl.size(), exp_dv_fl.size());
    EXPECT_ARR_EQ((da_int)dv_fl.size(), dv_fl.data(), exp_dv_fl.data(), 1, 1, 0, 0);

    da_vector::da_vector<double> dv_doubl;
    deserialize(dv_doubl);
    da_vector::da_vector<double> exp_dv_doubl;
    get_exp_dv_doubl(exp_dv_doubl);
    EXPECT_EQ(dv_doubl.size(), exp_dv_doubl.size());
    EXPECT_ARR_EQ((da_int)dv_doubl.size(), dv_doubl.data(), exp_dv_doubl.data(), 1, 1, 0,
                  0);

    // std::vector<da_vector::da_vector<T>>
    std::vector<da_vector::da_vector<da_int>> vdv_int;
    deserialize(vdv_int);
    std::vector<da_vector::da_vector<da_int>> exp_vdv_int;
    get_exp_vdv_int(exp_vdv_int);
    EXPECT_EQ(vdv_int.size(), exp_vdv_int.size());
    for (size_t i = 0; i < vdv_int.size(); ++i) {
        EXPECT_EQ(vdv_int[i].size(), exp_vdv_int[i].size());
        EXPECT_ARR_EQ((da_int)vdv_int[i].size(), vdv_int[i].data(), exp_vdv_int[i].data(),
                      1, 1, 0, 0);
    }

    std::vector<da_vector::da_vector<float>> vdv_fl;
    deserialize(vdv_fl);
    std::vector<da_vector::da_vector<float>> exp_vdv_fl;
    get_exp_vdv_fl(exp_vdv_fl);
    EXPECT_EQ(vdv_fl.size(), exp_vdv_fl.size());
    for (size_t i = 0; i < vdv_fl.size(); ++i) {
        EXPECT_EQ(vdv_fl[i].size(), exp_vdv_fl[i].size());
        EXPECT_ARR_EQ((da_int)vdv_fl[i].size(), vdv_fl[i].data(), exp_vdv_fl[i].data(), 1,
                      1, 0, 0);
    }

    std::vector<da_vector::da_vector<double>> vdv_doubl;
    deserialize(vdv_doubl);
    std::vector<da_vector::da_vector<double>> exp_vdv_doubl;
    get_exp_vdv_doubl(exp_vdv_doubl);
    EXPECT_EQ(vdv_doubl.size(), exp_vdv_doubl.size());
    for (size_t i = 0; i < vdv_doubl.size(); ++i) {
        EXPECT_EQ(vdv_doubl[i].size(), exp_vdv_doubl[i].size());
        EXPECT_ARR_EQ((da_int)vdv_doubl[i].size(), vdv_doubl[i].data(),
                      exp_vdv_doubl[i].data(), 1, 1, 0, 0);
    }

    // Enums
    da_order ord;
    deserialize(ord);
    EXPECT_EQ(ord, exp_ord);

    da_svm_model svm_mod;
    deserialize(svm_mod);
    EXPECT_EQ(svm_mod, exp_svm_mod);

    da_metric metr;
    deserialize(metr);
    EXPECT_EQ(metr, exp_metr);

    linmod_model lin_mod;
    deserialize(lin_mod);
    EXPECT_EQ(lin_mod, exp_lin_mod);

    da_linmod_types::logistic_constraint log_con;
    deserialize(log_con);
    EXPECT_EQ(log_con, exp_log_con);

    da_tree_options_types::split_property spl_prop;
    deserialize(spl_prop);
    EXPECT_EQ(spl_prop, exp_spl_prop);

    da_approx_nn_types::approx_nn_metric ann_metr;
    deserialize(ann_metr);
    EXPECT_EQ(ann_metr, exp_ann_metr);

    da_handle_type han_type;
    deserialize(han_type);
    EXPECT_EQ(han_type, exp_han_type);

    // User data (deserialize as vectors since serialize_user_data uses same format)
    std::vector<float> user_data_fl;
    deserialize(user_data_fl);
    EXPECT_EQ(user_data_fl.size(), exp_user_data_fl.size());
    EXPECT_ARR_EQ((da_int)user_data_fl.size(), user_data_fl.data(),
                  exp_user_data_fl.data(), 1, 1, 0, 0);

    std::vector<double> user_data_doubl;
    deserialize(user_data_doubl);
    EXPECT_EQ(user_data_doubl.size(), exp_user_data_doubl.size());
    EXPECT_ARR_EQ((da_int)user_data_doubl.size(), user_data_doubl.data(),
                  exp_user_data_doubl.data(), 1, 1, 0, 0);

    std::vector<float> user_data_nullptr;
    deserialize(user_data_nullptr);
    EXPECT_TRUE(user_data_nullptr.empty());

    // Verify we consumed all data (to be added with internal method)
    // EXPECT_EQ(buffer.offset, buffer_data.get());

    return status;
}

} // anonymous namespace

class SerializationKernelTests : public testing::Test {
  protected:
    std::string generated_path;
    std::string serialized_path;
    std::string temp_path;

    void SetUp() override {
        // Create unique filenames per test to allow parallel execution
        const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string test_name = test_info->name();
        std::string test_dir = model_persistence_test_utils::get_test_file_dir();
        generated_path = test_dir + "/kernel_test_generated_" + test_name + ".bin";
        serialized_path = test_dir + "/kernel_test_serialized_" + test_name + ".bin";
        temp_path = test_dir + "/kernel_test_roundtrip_" + test_name + ".bin";
    }

    void TearDown() override {
        std::remove(generated_path.c_str());
        std::remove(serialized_path.c_str());
        std::remove(temp_path.c_str());
    }
};

// Test 1: Verify serialize_all produces same binary as generate_test_data
TEST_F(SerializationKernelTests, SerializationMatchesGenerator) {
    // Generate expected binary data using the test data generator to a temp file
    ASSERT_TRUE(test_data_generator::generate_kernel_data(generated_path))
        << "Failed to generate test data";

    // Serialize using serialization library
    std::vector<char> serialized_data;
    da_status status = serialize_all(serialized_data);
    ASSERT_EQ(status, da_status_success);

    // Write serialized data to file for comparison
    ASSERT_TRUE(write_binary_file(serialized_path, serialized_data))
        << "Failed to write serialized file";

    // Compare files directly
    EXPECT_TRUE(compare_binary_files(generated_path, serialized_path))
        << "Generated and serialized files differ";
}

// Test 2: Serialize to file, deserialize and verify values
TEST_F(SerializationKernelTests, FileRoundTrip) {
    // Serialize using serialization library
    std::vector<char> serialized_data;
    da_status status = serialize_all(serialized_data);
    ASSERT_EQ(status, da_status_success);

    // Check if get_type_size() returns the right size.
    // Note: This kernel test doesn't serialize metadata, only raw data types.
    size_t buffer_size = get_buffer_size();
    ASSERT_EQ(buffer_size, serialized_data.size());

    // Write to file
    ASSERT_TRUE(write_binary_file(temp_path, serialized_data))
        << "Failed to write temp file";

    // Read back from file
    std::vector<char> read_data = load_binary_file(temp_path);
    ASSERT_FALSE(read_data.empty()) << "Failed to read temp file";

    // Deserialize and verify values
    status = deserialize_and_verify(read_data);
    ASSERT_EQ(status, da_status_success);
}

// Test 3: Loading vector with negative size (invalid, should fail)
TEST_F(SerializationKernelTests, LoadVectorNegativeSize) {
    std::vector<char> data;
    int64_t neg_size = -5;
    const char *bytes = reinterpret_cast<const char *>(&neg_size);
    data.insert(data.end(), bytes, bytes + sizeof(neg_size));

    da_model_persistence::serialization_buffer buffer(da_handle_uninitialized);
    ASSERT_EQ(buffer.set_buffer_data(data.data(), data.size()), da_status_success);

    std::vector<da_int> result;
    da_status status = buffer.deserialize_data(result);
    EXPECT_EQ(status, da_status_invalid_file_data);
}

// Test 4: serialize_user_data with stride (column-major)
TEST_F(SerializationKernelTests, UserDataColumnMajorWithStride) {
    // Create a 2x2 array with stride (ldx=3, but only 2 rows used)
    // Memory layout: [1, 2, X, 3, 4, X] where X is padding
    std::vector<float> data_with_padding = {1.0f, 2.0f, 999.0f, 3.0f, 4.0f, 999.0f};
    // Expected serialized data (no padding): [1, 2, 3, 4]
    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};

    // Create buffer
    std::vector<char> buffer_data;
    da_model_persistence::serialization_buffer buffer(da_handle_uninitialized);
    ASSERT_EQ(buffer.set_buffer_data(&buffer_data), da_status_success);
    buffer.set_mode(buffer_mode::serialize);

    // Serialize 2x2 column-major with ldx=3
    da_status status =
        buffer.serialize_user_data(data_with_padding.data(), column_major, 2, 2, 3);
    ASSERT_EQ(status, da_status_success);

    // Deserialize as vector to verify padding was skipped
    da_model_persistence::serialization_buffer read_buffer(da_handle_uninitialized);
    ASSERT_EQ(read_buffer.set_buffer_data(buffer_data.data(), buffer_data.size()),
              da_status_success);

    std::vector<float> result;
    status = read_buffer.deserialize_data(result);
    ASSERT_EQ(status, da_status_success);
    EXPECT_EQ(result.size(), expected.size());
    EXPECT_ARR_EQ((da_int)result.size(), result.data(), expected.data(), 1, 1, 0, 0);
}

// Test 5: serialize_user_data with stride (row-major)
TEST_F(SerializationKernelTests, UserDataRowMajorWithStride) {
    // Create a 2x2 array with stride (ldx=3, but only 2 cols used per row)
    // Memory layout: [10, 20, X, 30, 40, X] where X is padding
    std::vector<double> data_with_padding = {10.0, 20.0, 999.0, 30.0, 40.0, 999.0};
    // Expected serialized data (no padding): [10, 20, 30, 40]
    std::vector<double> expected = {10.0, 20.0, 30.0, 40.0};

    std::vector<char> buffer_data;
    da_model_persistence::serialization_buffer buffer(da_handle_uninitialized);
    ASSERT_EQ(buffer.set_buffer_data(&buffer_data), da_status_success);
    buffer.set_mode(buffer_mode::serialize);

    // Serialize 2x2 row-major with ldx=3
    da_status status =
        buffer.serialize_user_data(data_with_padding.data(), row_major, 2, 2, 3);
    ASSERT_EQ(status, da_status_success);

    // Deserialize as vector to verify padding was skipped
    da_model_persistence::serialization_buffer read_buffer(da_handle_uninitialized);
    ASSERT_EQ(read_buffer.set_buffer_data(buffer_data.data(), buffer_data.size()),
              da_status_success);

    std::vector<double> result;
    status = read_buffer.deserialize_data(result);
    ASSERT_EQ(status, da_status_success);
    EXPECT_EQ(result.size(), expected.size());
    EXPECT_ARR_EQ((da_int)result.size(), result.data(), expected.data(), 1, 1, 0, 0);
}

// Test 6: serialize_user_data with nullptr
TEST_F(SerializationKernelTests, UserDataNullptr) {
    std::vector<char> buffer_data;
    da_model_persistence::serialization_buffer buffer(da_handle_uninitialized);
    ASSERT_EQ(buffer.set_buffer_data(&buffer_data), da_status_success);
    buffer.set_mode(buffer_mode::serialize);

    // Serialize nullptr
    da_status status = buffer.serialize_user_data<float>(nullptr, column_major, 5, 5, 5);
    ASSERT_EQ(status, da_status_success);

    // Deserialize - should get empty vector
    da_model_persistence::serialization_buffer read_buffer(da_handle_uninitialized);
    ASSERT_EQ(read_buffer.set_buffer_data(buffer_data.data(), buffer_data.size()),
              da_status_success);

    std::vector<float> result = {999.0f}; // Start non-empty
    status = read_buffer.deserialize_data(result);
    ASSERT_EQ(status, da_status_success);
    EXPECT_TRUE(result.empty());
}