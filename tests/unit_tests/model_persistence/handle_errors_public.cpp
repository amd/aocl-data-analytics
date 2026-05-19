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
#include "aoclda_cpp_overloads.hpp"
#include "persistence_test_utils.hpp"
#include "gtest/gtest.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

// ==================== HANDLE SERIALIZATION ERROR TESTS ====================
// Tests for error handling in da_handle_save_model and da_handle_load_model

class HandleSerializationErrorTest : public testing::Test {
  protected:
    std::string test_file;
    std::string empty_file;
    std::string corrupt_file;

    void SetUp() override {
        // Create unique filenames per test to allow parallel execution
        const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string test_name = test_info->name();
        std::string test_dir = model_persistence_test_utils::get_test_file_dir();
        test_file = test_dir + "/handle_error_" + test_name + ".bin";
        empty_file = test_dir + "/empty_" + test_name + ".bin";
        corrupt_file = test_dir + "/corrupt_" + test_name + ".bin";
    }

    void TearDown() override {
        std::remove(test_file.c_str());
        std::remove(empty_file.c_str());
        std::remove(corrupt_file.c_str());
    }
};

// ==================== SAVE HANDLE ERRORS ====================

TEST_F(HandleSerializationErrorTest, SaveWithNullHandleFilePath) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_save_model(handle, test_file.c_str()), da_status_invalid_pointer);
}

TEST_F(HandleSerializationErrorTest, SaveWithNullHandleBufferPath) {
    da_handle handle = nullptr;
    std::vector<char> buffer;
    EXPECT_EQ(da_handle_save_model(handle, buffer), da_status_invalid_pointer);
}

TEST_F(HandleSerializationErrorTest, SaveWithNullFile) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_s(&handle, da_handle_pca), da_status_success);
    char *filename = nullptr;
    EXPECT_EQ(da_handle_save_model(handle, filename), da_status_invalid_pointer);
    da_handle_destroy(&handle);
}

TEST_F(HandleSerializationErrorTest, SaveToBufferClearsGarbage) {
    // Create a fitted PCA model for testing
    da_handle handle = nullptr;
    ASSERT_EQ(da_handle_init_d(&handle, da_handle_pca), da_status_success);

    // Set up minimal data for PCA
    std::vector<double> X = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    da_int n_samples = 3, n_features = 2;
    ASSERT_EQ(da_pca_set_data_d(handle, n_samples, n_features, X.data(), n_samples),
              da_status_success);
    ASSERT_EQ(da_options_set_int(handle, "n_components", 2), da_status_success);
    ASSERT_EQ(da_pca_compute_d(handle), da_status_success);

    // Buffer with garbage data
    std::vector<char> buffer(500, 0xAB);
    ASSERT_FALSE(buffer.empty());

    // Save should clear garbage and produce valid data
    ASSERT_EQ(da_handle_save_model(handle, buffer), da_status_success);

    // Buffer should not start with garbage anymore
    EXPECT_NE(buffer[0], static_cast<char>(0xAB));

    // Verify the saved data can be loaded
    da_handle loaded = nullptr;
    EXPECT_EQ(da_handle_load_model(&loaded, buffer.data(), buffer.size()),
              da_status_success);

    da_handle_destroy(&handle);
    da_handle_destroy(&loaded);
}

// ==================== LOAD ERRORS ====================

TEST_F(HandleSerializationErrorTest, LoadFromNullPtrHandleBufferPath) {
    std::vector<char> buffer(100, 0xFF);
    EXPECT_EQ(da_handle_load_model(nullptr, buffer.data(), buffer.size()),
              da_status_invalid_pointer);
}

TEST_F(HandleSerializationErrorTest, LoadFromNullPtrHandleFilePath) {
    EXPECT_EQ(da_handle_load_model(nullptr, test_file.c_str()),
              da_status_invalid_pointer);
}

TEST_F(HandleSerializationErrorTest, LoadFromNonNullHandleBufferPath) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_pca), da_status_success);

    std::vector<char> buffer(100, 0x00);
    // Handle is already initialized - should fail
    EXPECT_EQ(da_handle_load_model(&handle, buffer.data(), buffer.size()),
              da_status_invalid_pointer);
    da_handle_destroy(&handle);
}

TEST_F(HandleSerializationErrorTest, LoadFromNotNullHandleFilePath) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_s(&handle, da_handle_pca), da_status_success);
    // Handle is already initialized - should fail
    EXPECT_EQ(da_handle_load_model(&handle, test_file.c_str()),
              da_status_invalid_pointer);
    da_handle_destroy(&handle);
}

TEST_F(HandleSerializationErrorTest, LoadFromNullBuffer) {
    da_handle handle = nullptr;
    char *buffer = nullptr;
    EXPECT_EQ(da_handle_load_model(&handle, buffer, 100), da_status_invalid_pointer);
    EXPECT_EQ(handle, nullptr);
}

TEST_F(HandleSerializationErrorTest, LoadFromNullFilename) {
    da_handle handle = nullptr;
    char *filename = nullptr;
    EXPECT_EQ(da_handle_load_model(&handle, filename), da_status_invalid_pointer);
    EXPECT_EQ(handle, nullptr);
}

TEST_F(HandleSerializationErrorTest, LoadFromBufferZeroSize) {
    da_handle handle = nullptr;
    std::vector<char> buffer(100, 0xFF);
    EXPECT_EQ(da_handle_load_model(&handle, buffer.data(), 0), da_status_invalid_input);
    EXPECT_EQ(handle, nullptr);
}

TEST_F(HandleSerializationErrorTest, LoadFromEmptyBuffer) {
    da_handle handle = nullptr;
    std::vector<char> buffer;
    da_status status = da_handle_load_model(&handle, buffer.data(), buffer.size());
    EXPECT_EQ(status, da_status_invalid_pointer);
    EXPECT_EQ(handle, nullptr);
}

TEST_F(HandleSerializationErrorTest, LoadFromNonExistentFile) {
    da_handle handle = nullptr;
    da_status status =
        da_handle_load_model(&handle, "this_file_does_not_exist_12345.bin");
    EXPECT_EQ(status, da_status_io_error);
    EXPECT_EQ(handle, nullptr);
}

TEST_F(HandleSerializationErrorTest, LoadFromEmptyFile) {
    // Create an empty file
    std::ofstream ofs(empty_file, std::ios::binary);
    ofs.close();

    da_handle handle = nullptr;
    da_status status = da_handle_load_model(&handle, empty_file.c_str());
    EXPECT_EQ(status, da_status_io_error);
    EXPECT_EQ(handle, nullptr);
}

// ==================== METADATA ERRORS ====================

TEST_F(HandleSerializationErrorTest, LoadFromCorruptBuffer) {
    da_handle handle = nullptr;
    std::vector<char> buffer(3, 0xFF);
    da_status status = da_handle_load_model(&handle, buffer.data(), buffer.size());
    EXPECT_EQ(status, da_status_invalid_file_data);
    EXPECT_EQ(handle, nullptr);
}

TEST_F(HandleSerializationErrorTest, LoadCorruptedFileMetadataValidation) {
    // Metadata layout:
    //   Bytes 0-7:    magic_keyword size (int64_t = 19)
    //   Bytes 8-26:   magic_keyword string "AOCLDA_STORED_MODEL" (19 bytes)
    //   Bytes 27-34:  da_int_size (int64_t, must be 8 or 4)
    //   Bytes 35-42:  lib_version (int64_t, must equal da_model_persistence::min_library_version)
    //   Bytes 43-50:  algorithm (da_handle_type enum as int64_t)
    //   Bytes 51-58:  precision (int64_t, must be 4 or 8)
    // Valid values
    const std::string_view valid_magic = "AOCLDA_STORED_MODEL";
    constexpr int64_t valid_lib_version = 50100;
    constexpr int64_t valid_int_size = 4;
    constexpr int64_t valid_algorithm = static_cast<int64_t>(da_handle_pca);
    constexpr int64_t valid_precision = 8; // sizeof(double)

    // Build metadata buffer with configurable magic keyword
    auto build_metadata = [](std::string_view magic, int64_t int_size, int64_t lib_ver,
                             int64_t algo, int64_t prec) {
        std::vector<char> buf;
        // Serialize magic keyword: size + content
        int64_t magic_size = static_cast<int64_t>(magic.size());
        auto *size_bytes = reinterpret_cast<const char *>(&magic_size);
        buf.insert(buf.end(), size_bytes, size_bytes + sizeof(magic_size));
        buf.insert(buf.end(), magic.begin(), magic.end());
        // Serialize metadata fields
        auto *int_size_bytes = reinterpret_cast<const char *>(&int_size);
        buf.insert(buf.end(), int_size_bytes, int_size_bytes + sizeof(int_size));
        auto *lib_ver_bytes = reinterpret_cast<const char *>(&lib_ver);
        buf.insert(buf.end(), lib_ver_bytes, lib_ver_bytes + sizeof(lib_ver));
        auto *algo_bytes = reinterpret_cast<const char *>(&algo);
        buf.insert(buf.end(), algo_bytes, algo_bytes + sizeof(algo));
        auto *prec_bytes = reinterpret_cast<const char *>(&prec);
        buf.insert(buf.end(), prec_bytes, prec_bytes + sizeof(prec));
        // Pad to ensure sufficient size
        buf.resize(buf.size() + 50, 0x00);
        return buf;
    };

    da_handle handle = nullptr;
    da_status status;

    // Test 1: Invalid da_int_size (not 8 or 4)
    {
        auto metadata = build_metadata(valid_magic, 16, valid_lib_version,
                                       valid_algorithm, valid_precision);
        std::ofstream ofs(corrupt_file, std::ios::binary);
        ofs.write(metadata.data(), metadata.size());
        ASSERT_TRUE(ofs.good());
        ofs.close();

        handle = nullptr;
        status = da_handle_load_model(&handle, corrupt_file.c_str());
        EXPECT_EQ(status, da_status_invalid_file_data);
        EXPECT_EQ(handle, nullptr);
        std::remove(corrupt_file.c_str());
    }

#ifndef AOCLDA_ILP64
    // Test 2: Different da_int_size between build and save
    // Only runs in LP64 builds (da_int is 32-bit); loading a file saved with
    // 8-byte int_size should fail because the opposite direction is not allowed.
    {
        da_int invalid_da_int_size = 8;

        auto metadata =
            build_metadata(valid_magic, invalid_da_int_size, valid_lib_version,
                           valid_algorithm, valid_precision);
        std::ofstream ofs(corrupt_file, std::ios::binary);
        ofs.write(metadata.data(), metadata.size());
        ASSERT_TRUE(ofs.good());
        ofs.close();

        handle = nullptr;
        status = da_handle_load_model(&handle, corrupt_file.c_str());
        EXPECT_EQ(status, da_status_invalid_file_data);
        EXPECT_EQ(handle, nullptr);
        std::remove(corrupt_file.c_str());
    }
#endif // AOCLDA_ILP64

    // Test 3: Invalid library version (version mismatch) (must be bigger than current lib version)
    {
        auto metadata = build_metadata(valid_magic, valid_int_size, 90000,
                                       valid_algorithm, valid_precision);
        std::ofstream ofs(corrupt_file, std::ios::binary);
        ofs.write(metadata.data(), metadata.size());
        ASSERT_TRUE(ofs.good());
        ofs.close();

        handle = nullptr;
        status = da_handle_load_model(&handle, corrupt_file.c_str());
        EXPECT_EQ(status, da_status_version_mismatch);
        EXPECT_EQ(handle, nullptr);
        std::remove(corrupt_file.c_str());
    }

    // Test 4: Invalid precision (not 4 or 8)
    {
        auto metadata = build_metadata(valid_magic, valid_int_size, valid_lib_version,
                                       valid_algorithm, 2);
        std::ofstream ofs(corrupt_file, std::ios::binary);
        ofs.write(metadata.data(), metadata.size());
        ASSERT_TRUE(ofs.good());
        ofs.close();

        handle = nullptr;
        status = da_handle_load_model(&handle, corrupt_file.c_str());
        EXPECT_EQ(status, da_status_invalid_file_data);
        EXPECT_EQ(handle, nullptr);
        std::remove(corrupt_file.c_str());
    }

    // Test 5: Invalid magic keyword size
    {
        std::vector<char> buf;
        // Invalid magic keyword size (5 instead of 19)
        int64_t invalid_magic_size = 5;
        auto *size_bytes = reinterpret_cast<const char *>(&invalid_magic_size);
        buf.insert(buf.end(), size_bytes, size_bytes + sizeof(invalid_magic_size));
        // Append valid magic keyword + valid metadata fields so failure is due to size mismatch
        const std::string_view magic = valid_magic;
        buf.insert(buf.end(), magic.begin(), magic.end());
        int64_t fields[] = {valid_int_size, valid_lib_version, valid_algorithm,
                            valid_precision};
        for (auto f : fields) {
            auto *fb = reinterpret_cast<const char *>(&f);
            buf.insert(buf.end(), fb, fb + sizeof(f));
        }
        buf.resize(buf.size() + 50, 0x00);

        std::ofstream ofs(corrupt_file, std::ios::binary);
        ofs.write(buf.data(), buf.size());
        ASSERT_TRUE(ofs.good());
        ofs.close();

        handle = nullptr;
        status = da_handle_load_model(&handle, corrupt_file.c_str());
        EXPECT_EQ(status, da_status_invalid_file_data);
        EXPECT_EQ(handle, nullptr);
        std::remove(corrupt_file.c_str());
    }

    // Test 6: Invalid magic keyword string (wrong content)
    {
        auto buf = build_metadata("WRONG_MAGIC_KEYWORD", valid_int_size,
                                  valid_lib_version, valid_algorithm, valid_precision);

        std::ofstream ofs(corrupt_file, std::ios::binary);
        ofs.write(buf.data(), buf.size());
        ASSERT_TRUE(ofs.good());
        ofs.close();

        handle = nullptr;
        status = da_handle_load_model(&handle, corrupt_file.c_str());
        EXPECT_EQ(status, da_status_invalid_file_data);
        EXPECT_EQ(handle, nullptr);
        std::remove(corrupt_file.c_str());
    }
}

TEST_F(HandleSerializationErrorTest, LoadCorruptedFileTooSmall) {
    // Create a file that's too small to contain valid metadata
    std::ofstream ofs(corrupt_file, std::ios::binary);
    char garbage[] = {0x01, 0x02, 0x03};
    ofs.write(garbage, sizeof(garbage));
    ofs.close();

    da_handle handle = nullptr;
    da_status status = da_handle_load_model(&handle, corrupt_file.c_str());
    EXPECT_EQ(status, da_status_invalid_file_data);
    // Handle should still be nullptr after failure
    EXPECT_EQ(handle, nullptr);
}