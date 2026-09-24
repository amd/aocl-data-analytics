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
 */

#include "../model_persistence/persistence_test_utils.hpp"
#include "../utest_utils.hpp"
#include "Au/Cpuid/X86Cpu.hh"
#include "ann_tests.hpp"
#include "aoclda.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {

struct QuantizationMode {
    std::string name;
    std::string scheme;
    std::string residual;
    std::string quantization = "int8";
};

const std::vector<QuantizationMode> quantization_modes{
    {"per_dimension_residual", "per_dimension", "yes"},
    {"per_dimension_raw", "per_dimension", "no"},
    {"uniform_residual", "uniform", "yes"},
    {"uniform_raw", "uniform", "no"},
    {"direct_signed_residual", "direct_signed", "yes"},
    {"direct_signed_raw", "direct_signed", "no"},
    // IVF residuals contain negative values, outside direct_unsigned's code range.
    {"direct_unsigned_raw", "direct_unsigned", "no"},
    {"int4_per_dimension_residual", "per_dimension", "yes", "int4"},
    {"int4_per_dimension_raw", "per_dimension", "no", "int4"},
    {"int4_uniform_residual", "uniform", "yes", "int4"},
    {"int4_uniform_raw", "uniform", "no", "int4"},
    {"int4_direct_signed_residual", "direct_signed", "yes", "int4"},
    {"int4_direct_signed_raw", "direct_signed", "no", "int4"},
    {"int4_direct_unsigned_raw", "direct_unsigned", "no", "int4"},
    {"bfloat16_residual", "direct_signed", "yes", "bfloat16"},
    {"bfloat16_raw", "direct_signed", "no", "bfloat16"},
};

void PrintTo(const QuantizationMode &mode, std::ostream *os) { *os << mode.name; }

template <typename T> struct ANNData {
    da_int n_samples = 32;
    da_int n_features = 5;
    da_int n_queries = 7;
    da_int ldx_train = 0;
    da_int ldx_test = 0;
    std::vector<T> train;
    std::vector<T> test;
};

template <typename T>
std::vector<T> materialize(const std::vector<T> &row_major_data, da_int n_rows,
                           da_int n_features, da_order order, da_int padding,
                           da_int &leading_dimension) {
    if (order == row_major) {
        leading_dimension = n_features + padding;
        std::vector<T> output(static_cast<size_t>(n_rows) * leading_dimension, T(0));
        for (da_int row = 0; row < n_rows; row++)
            for (da_int feature = 0; feature < n_features; feature++)
                output[static_cast<size_t>(row) * leading_dimension + feature] =
                    row_major_data[static_cast<size_t>(row) * n_features + feature];
        return output;
    }

    leading_dimension = n_rows + padding;
    std::vector<T> output(static_cast<size_t>(n_features) * leading_dimension, T(0));
    for (da_int row = 0; row < n_rows; row++)
        for (da_int feature = 0; feature < n_features; feature++)
            output[row + static_cast<size_t>(feature) * leading_dimension] =
                row_major_data[static_cast<size_t>(row) * n_features + feature];
    return output;
}

template <typename T>
ANNData<T> make_ann_data(const QuantizationMode &mode, da_order order,
                         da_int padding = 0) {
    ANNData<T> data;
    std::vector<T> train_row(static_cast<size_t>(data.n_samples) * data.n_features);
    std::vector<T> test_row(static_cast<size_t>(data.n_queries) * data.n_features);

    const bool direct_unsigned = mode.scheme == "direct_unsigned";
    for (da_int sample = 0; sample < data.n_samples; sample++) {
        const da_int cluster = sample % 4;
        for (da_int feature = 0; feature < data.n_features; feature++) {
            if (mode.quantization == "int4" && mode.scheme == "direct_unsigned") {
                train_row[static_cast<size_t>(sample) * data.n_features + feature] =
                    T((sample + 2 * feature) % 16);
                continue;
            }
            if (mode.quantization == "int4" && mode.scheme == "direct_signed") {
                train_row[static_cast<size_t>(sample) * data.n_features + feature] =
                    T((sample + 2 * feature) % 16 - 8);
                continue;
            }
            const T base = direct_unsigned ? T(25 + 45 * cluster) : T(-75 + 45 * cluster);
            train_row[static_cast<size_t>(sample) * data.n_features + feature] =
                base + T(3 * feature) + T(sample / 4) * T(0.37) +
                T(feature + 1) * T(sample + 1) * T(0.003);
        }
    }

    for (da_int query = 0; query < data.n_queries; query++) {
        const da_int source = (query * 5 + 1) % data.n_samples;
        for (da_int feature = 0; feature < data.n_features; feature++)
            test_row[static_cast<size_t>(query) * data.n_features + feature] =
                train_row[static_cast<size_t>(source) * data.n_features + feature] +
                T(query + 1) * T(feature + 2) * T(0.007);
    }

    if (mode.scheme == "direct_unsigned") {
        for (T value : train_row) {
            EXPECT_GE(value, T(0));
            EXPECT_LE(value, T(255));
        }
        for (T value : test_row) {
            EXPECT_GE(value, T(0));
            EXPECT_LE(value, T(255));
        }
    } else if (mode.scheme == "direct_signed") {
        for (T value : train_row) {
            EXPECT_GE(value, T(-128));
            EXPECT_LE(value, T(127));
        }
        for (T value : test_row) {
            EXPECT_GE(value, T(-128));
            EXPECT_LE(value, T(127));
        }
    }

    data.train = materialize(train_row, data.n_samples, data.n_features, order, padding,
                             data.ldx_train);
    data.test = materialize(test_row, data.n_queries, data.n_features, order, padding,
                            data.ldx_test);
    return data;
}

template <typename T> struct ANNResult {
    da_order order;
    da_int n_queries;
    da_int k;
    da_int n_samples;
    da_int n_features;
    da_int seed;
    std::vector<da_int> indices;
    std::vector<T> distances;
};

struct ANNRunConfig {
    da_order order = row_major;
    da_int padding = 0;
    bool train_and_add = false;
    da_int add_chunk_size = 0;
    da_int query_block_size = 0;
    da_int list_block_size = 0;
    da_int add_block_size = 0;
    bool save_and_load = false;
};

struct ANNRunState {
    bool reset_query_block_size = false;
    bool reset_list_block_size = false;
    bool reset_add_block_size = false;
    std::string model_file;

    ~ANNRunState() {
        if (reset_query_block_size)
            da_debug_set("ivf.query_blk_sz", "");
        if (reset_list_block_size)
            da_debug_set("ivf.list_blk_sz", "");
        if (reset_add_block_size)
            da_debug_set("ivf.add_blk_sz", "");
        if (!model_file.empty())
            std::remove(model_file.c_str());
    }
};

size_t result_offset(da_order order, da_int n_queries, da_int k, da_int query,
                     da_int rank) {
    // ANN output follows the configured storage order.
    return order == row_major ? static_cast<size_t>(query) * k + rank
                              : query + static_cast<size_t>(rank) * n_queries;
}

template <typename T>
ANNResult<T> run_ann_case(const QuantizationMode &mode, const ANNRunConfig &config) {
    constexpr da_int k = 4;
    ANNData<T> data = make_ann_data<T>(mode, config.order, config.padding);
    ANNResult<T> result{config.order,
                        data.n_queries,
                        k,
                        data.n_samples,
                        data.n_features,
                        42,
                        std::vector<da_int>(data.n_queries * k),
                        std::vector<T>(data.n_queries * k)};
    da_handle handle = nullptr;
    ANNRunState run_state;

    if (config.query_block_size > 0) {
        const da_status status = da_debug_set(
            "ivf.query_blk_sz", std::to_string(config.query_block_size).c_str());
        EXPECT_EQ(status, da_status_success);
        run_state.reset_query_block_size = status == da_status_success;
    }
    if (config.list_block_size > 0) {
        const da_status status = da_debug_set(
            "ivf.list_blk_sz", std::to_string(config.list_block_size).c_str());
        EXPECT_EQ(status, da_status_success);
        run_state.reset_list_block_size = status == da_status_success;
    }
    if (config.add_block_size > 0) {
        const da_status status =
            da_debug_set("ivf.add_blk_sz", std::to_string(config.add_block_size).c_str());
        EXPECT_EQ(status, da_status_success);
        run_state.reset_add_block_size = status == da_status_success;
    }

    const da_status init_status = da_handle_init<T>(&handle, da_handle_approx_nn);
    EXPECT_EQ(init_status, da_status_success);
    if (init_status != da_status_success)
        return result;
    EXPECT_EQ(da_options_set_string(handle, "algorithm", "ivfsq"), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "metric", "sqeuclidean"), da_status_success);
    EXPECT_EQ(
        da_options_set_string(handle, "storage order",
                              config.order == row_major ? "row-major" : "column-major"),
        da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "number of neighbors", k), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_list", 4), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_probe", 4), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "seed", 42), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "k-means_iter", 20), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "quantization", mode.quantization.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "quantization_scheme", mode.scheme.c_str()),
              da_status_success);
    EXPECT_EQ(
        da_options_set_string(handle, "quantization_residual", mode.residual.c_str()),
        da_status_success);
    EXPECT_EQ(da_approx_nn_set_training_data(handle, data.n_samples, data.n_features,
                                             data.train.data(), data.ldx_train),
              da_status_success);

    if (config.train_and_add) {
        EXPECT_EQ(da_approx_nn_train_and_add<T>(handle), da_status_success);
    } else {
        EXPECT_EQ(da_approx_nn_train<T>(handle), da_status_success);
        if (config.add_chunk_size > 0) {
            for (da_int start = 0; start < data.n_samples;
                 start += config.add_chunk_size) {
                const da_int chunk_size =
                    std::min(config.add_chunk_size, data.n_samples - start);
                const T *chunk =
                    config.order == row_major
                        ? data.train.data() + static_cast<size_t>(start) * data.ldx_train
                        : data.train.data() + start;
                EXPECT_EQ(da_approx_nn_add<T>(handle, chunk_size, data.n_features, chunk,
                                              data.ldx_train),
                          da_status_success);
            }
        } else {
            EXPECT_EQ(da_approx_nn_add<T>(handle, data.n_samples, data.n_features,
                                          data.train.data(), data.ldx_train),
                      da_status_success);
        }
    }

    if (config.save_and_load) {
        run_state.model_file = model_persistence_test_utils::get_test_file_dir() +
                               "/ann_quantized_" + mode.name + "_" +
                               std::to_string(sizeof(T)) + ".bin";
        const da_status save_status =
            da_handle_save_model(handle, run_state.model_file.c_str());
        EXPECT_EQ(save_status, da_status_success);
        if (save_status != da_status_success) {
            da_handle_destroy(&handle);
            return result;
        }
        da_handle_destroy(&handle);
        const da_status load_status =
            da_handle_load_model(&handle, run_state.model_file.c_str());
        EXPECT_EQ(load_status, da_status_success);
        if (load_status != da_status_success)
            return result;
    }

    EXPECT_EQ(da_approx_nn_kneighbors<T>(
                  handle, data.n_queries, data.n_features, data.test.data(),
                  data.ldx_test, result.indices.data(), result.distances.data(), k, true),
              da_status_success);
    da_handle_destroy(&handle);
    return result;
}

template <typename T>
void expect_same_results(const ANNResult<T> &expected, const ANNResult<T> &actual,
                         const QuantizationMode &mode, const char *variant) {
    ASSERT_EQ(expected.n_queries, actual.n_queries);
    ASSERT_EQ(expected.k, actual.k);
    ASSERT_TRUE(validate_indices<T>(expected.indices, expected.n_queries, expected.k,
                                    expected.n_samples, expected.order));
    ASSERT_TRUE(validate_indices<T>(actual.indices, actual.n_queries, actual.k,
                                    actual.n_samples, actual.order));
    const T absolute_tolerance = T(5e-3);
    const T relative_tolerance = T(2e-5);

    for (da_int query = 0; query < expected.n_queries; query++) {
        for (da_int rank = 0; rank < expected.k; rank++) {
            const size_t expected_pos = result_offset(expected.order, expected.n_queries,
                                                      expected.k, query, rank);
            const size_t actual_pos =
                result_offset(actual.order, actual.n_queries, actual.k, query, rank);
            EXPECT_EQ(expected.indices[expected_pos], actual.indices[actual_pos])
                << mode.name << " " << variant << " query=" << query << " rank=" << rank
                << " seed=" << actual.seed << " n_features=" << actual.n_features;
            const T expected_distance = expected.distances[expected_pos];
            const T actual_distance = actual.distances[actual_pos];
            const T tolerance = absolute_tolerance +
                                relative_tolerance * std::max(std::abs(expected_distance),
                                                              std::abs(actual_distance));
            EXPECT_NEAR(expected_distance, actual_distance, tolerance)
                << mode.name << " " << variant << " query=" << query << " rank=" << rank
                << " seed=" << actual.seed << " n_features=" << actual.n_features;
        }
    }
}

template <typename T> void test_layout_and_training_paths(const QuantizationMode &mode) {
    const ANNRunConfig baseline_config;
    const ANNResult<T> baseline = run_ann_case<T>(mode, baseline_config);

    ANNRunConfig column_major_config = baseline_config;
    column_major_config.order = column_major;
    const ANNResult<T> column_major_result = run_ann_case<T>(mode, column_major_config);
    expect_same_results(baseline, column_major_result, mode, "column-major");

    ANNRunConfig padded_config = baseline_config;
    padded_config.padding = 3;
    const ANNResult<T> padded_result = run_ann_case<T>(mode, padded_config);
    expect_same_results(baseline, padded_result, mode, "padded");

    ANNRunConfig train_and_add_config = baseline_config;
    train_and_add_config.train_and_add = true;
    const ANNResult<T> train_and_add_result = run_ann_case<T>(mode, train_and_add_config);
    expect_same_results(baseline, train_and_add_result, mode, "train-and-add");

    ANNRunConfig chunked_add_config = baseline_config;
    chunked_add_config.add_chunk_size = 7;
    const ANNResult<T> chunked_add_result = run_ann_case<T>(mode, chunked_add_config);
    expect_same_results(baseline, chunked_add_result, mode, "chunked-add");

    // Different SGEMM block shapes can change accumulation order enough to reorder
    // near-tied int8 neighbors in the expanded squared-L2 calculation.
    if (mode.quantization != "int8") {
        ANNRunConfig blocked_config = baseline_config;
        blocked_config.query_block_size = 3;
        blocked_config.list_block_size = 5;
        blocked_config.add_block_size = 7;
        const ANNResult<T> blocked_result = run_ann_case<T>(mode, blocked_config);
        expect_same_results(baseline, blocked_result, mode, "forced-blocking");
    }

    ANNRunConfig loaded_config = baseline_config;
    loaded_config.save_and_load = true;
    const ANNResult<T> loaded_result = run_ann_case<T>(mode, loaded_config);
    expect_same_results(baseline, loaded_result, mode, "save-and-load");
}

class FloatQuantizedANNTest : public testing::TestWithParam<QuantizationMode> {};
class DoubleQuantizedANNTest : public testing::TestWithParam<QuantizationMode> {};

TEST_P(FloatQuantizedANNTest, EquivalentExecutionPaths) {
    test_layout_and_training_paths<float>(GetParam());
}

TEST_P(DoubleQuantizedANNTest, EquivalentExecutionPaths) {
    test_layout_and_training_paths<double>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(ANN_Quantized_Float, FloatQuantizedANNTest,
                         testing::ValuesIn(quantization_modes),
                         [](const testing::TestParamInfo<QuantizationMode> &info) {
                             return info.param.name;
                         });
INSTANTIATE_TEST_SUITE_P(ANN_Quantized_Double, DoubleQuantizedANNTest,
                         testing::ValuesIn(quantization_modes),
                         [](const testing::TestParamInfo<QuantizationMode> &info) {
                             return info.param.name;
                         });

template <typename T> void test_quality(const ANNRecallPair<T> &pair) {
    const T baseline_recall = test_ann_recall(pair.baseline);
    const T quantized_recall = test_ann_recall(pair.quantized);
    EXPECT_GE(quantized_recall, pair.minimum_recall);
    EXPECT_GE(quantized_recall, baseline_recall - pair.maximum_recall_drop);
}

template <typename T> std::vector<ANNRecallPair<T>> get_quality_cases() {
    std::vector<ANNRecallPair<T>> pairs;
    GetANNQuantizedQualityData(pairs);
    return pairs;
}

class FloatQuantizedQualityTest : public testing::TestWithParam<ANNRecallPair<float>> {};
class DoubleQuantizedQualityTest : public testing::TestWithParam<ANNRecallPair<double>> {
};

TEST_P(FloatQuantizedQualityTest, Recall) { test_quality<float>(GetParam()); }
TEST_P(DoubleQuantizedQualityTest, Recall) { test_quality<double>(GetParam()); }

INSTANTIATE_TEST_SUITE_P(ANN_Quantized_Quality_Float, FloatQuantizedQualityTest,
                         testing::ValuesIn(get_quality_cases<float>()),
                         [](const testing::TestParamInfo<ANNRecallPair<float>> &info) {
                             return info.param.name;
                         });
INSTANTIATE_TEST_SUITE_P(ANN_Quantized_Quality_Double, DoubleQuantizedQualityTest,
                         testing::ValuesIn(get_quality_cases<double>()),
                         [](const testing::TestParamInfo<ANNRecallPair<double>> &info) {
                             return info.param.name;
                         });

TEST(QuantizedNumericalErrors, QueryBiasOverflow) {
    const float training_data = 1.0f;
    const float query = std::numeric_limits<float>::max() / 2.0f;
    da_int index;
    float distance;
    da_handle handle = nullptr;

    ASSERT_EQ(da_handle_init_s(&handle, da_handle_approx_nn), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "algorithm", "ivfsq"), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "metric", "inner product"),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "storage order", "row-major"),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "quantization", "int8"), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "quantization_scheme", "direct_unsigned"),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "quantization_residual", "no"),
              da_status_success);
    EXPECT_EQ(da_approx_nn_set_training_data_s(handle, 1, 1, &training_data, 1),
              da_status_success);
    ASSERT_EQ(da_approx_nn_train_and_add_s(handle), da_status_success);

    EXPECT_EQ(
        da_approx_nn_kneighbors_s(handle, 1, 1, &query, 1, &index, &distance, 1, true),
        da_status_numerical_difficulties);

    da_handle_destroy(&handle);
}

} // namespace
