/******************************************************************************
* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef QUANTIZED_BACKEND_HPP
#define QUANTIZED_BACKEND_HPP

#include "bfloat16_helpers.hpp"
#include "da_cblas.hh"
#include "da_std.hpp"
#include "scalar_quantizer.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ARCH {
namespace da_quantized_backend {

inline bool use_optimized_backend(scalar_quantizer::code_kind code_kind,
                                  bool has_avx512_bf16) {
#ifndef _WIN32
    switch (code_kind) {
    case scalar_quantizer::code_kind::int4:
        return has_avx512_bf16;
    case scalar_quantizer::code_kind::bfloat16:
        return true;
    default:
        return false;
    }
#else
    (void)code_kind;
    (void)has_avx512_bf16;
    return false;
#endif
}

struct int8_workspace {
    bool use_optimized_kernel = false;
    std::vector<float> dot_products;
    std::vector<float> block_queries;
    std::vector<float> list_queries;
    std::vector<float> list_codes;
};

struct bfloat16_workspace {
    bool use_optimized_kernel = false;
    std::vector<float> dot_products;
    std::vector<da_bf16::storage_type> block_queries;
    std::vector<da_bf16::storage_type> list_queries;
};

struct int4_workspace {
    bool use_optimized_kernel = false;
    std::vector<float> dot_products;
    std::vector<da_bf16::storage_type> block_queries;
    std::vector<da_bf16::storage_type> list_queries;
    std::vector<int8_t> reordered_list_codes;
};

template <typename T> struct unquantized_backend {
    static constexpr bool is_quantized = false;
    static constexpr scalar_quantizer::code_kind code_kind = scalar_quantizer::none;
    using stored_type = T;
    struct workspace_type {};
};

// Quantized search calls each backend in this order:
//
//   resize_workspace -> set_query_value -> gather_list_queries -> compute_products
//
// The first two stages prepare a complete query block. The final two gather the rows
// probing one IVF list and multiply them by blocks of encoded list vectors.
// set_query_value receives query[d] * database_quantizer_scale[d]; individual
// backends may convert that value further for their GEMM datatype.
template <typename Q> struct backend;

template <> struct backend<int8_t> {
    static constexpr bool is_quantized = true;
    static constexpr scalar_quantizer::code_kind code_kind = scalar_quantizer::int8;
    using T = float;
    using Q = int8_t;
    using stored_type = Q;
    using workspace_type = int8_workspace;

    static size_t stored_row_offset(da_int row, da_int n_features) {
        return static_cast<size_t>(row) * n_features;
    }

    static void resize_workspace(workspace_type &ws, da_int query_block_size,
                                 da_int list_block_size, da_int n_features) {
        const size_t query_elements = static_cast<size_t>(query_block_size) * n_features;
        ws.block_queries.resize(query_elements);
        ws.list_queries.resize(query_elements);
        ws.list_codes.resize(static_cast<size_t>(list_block_size) * n_features);
        ws.dot_products.resize(static_cast<size_t>(query_block_size) * list_block_size);
    }

    static void set_query_value(workspace_type &ws, size_t index, T value) {
        ws.block_queries[index] = value;
    }

    static void gather_list_queries(workspace_type &ws, const da_int *query_indices,
                                    da_int block_start, da_int n_rows,
                                    da_int n_features) {
        for (da_int row = 0; row < n_rows; row++) {
            const da_int block_row = query_indices[row] - block_start;
            const T *source =
                ws.block_queries.data() + static_cast<size_t>(block_row) * n_features;
            T *destination =
                ws.list_queries.data() + static_cast<size_t>(row) * n_features;
            da_std::copy(source, source + n_features, destination);
        }
    }

    static bool compute_products(workspace_type &ws, const Q *codes, da_int n_rows,
                                 da_int n_cols, da_int n_features) {
        const size_t code_elements = static_cast<size_t>(n_cols) * n_features;
#pragma omp simd
        for (size_t code_index = 0; code_index < code_elements; code_index++)
            ws.list_codes[code_index] = static_cast<T>(codes[code_index]);
        da_blas::cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_rows, n_cols,
                            n_features, /*alpha=*/1.0f, ws.list_queries.data(),
                            n_features, ws.list_codes.data(), n_features, /*beta=*/0.0f,
                            ws.dot_products.data(), n_cols);
        return true;
    }
};

template <> struct backend<da_bf16::storage_type> {
    static constexpr bool is_quantized = true;
    static constexpr scalar_quantizer::code_kind code_kind = scalar_quantizer::bfloat16;
    using T = float;
    using Q = da_bf16::storage_type;
    using stored_type = Q;
    using workspace_type = bfloat16_workspace;

    // Convert a logical list row to an offset measured in stored bfloat16 elements.
    static size_t stored_row_offset(da_int row, da_int n_features) {
        return static_cast<size_t>(row) * n_features;
    }

    static void resize_workspace(workspace_type &ws, da_int query_block_size,
                                 da_int list_block_size, da_int n_features) {
        const size_t query_elements = static_cast<size_t>(query_block_size) * n_features;
        ws.block_queries.resize(query_elements);
        ws.list_queries.resize(query_elements);
        ws.dot_products.resize(static_cast<size_t>(query_block_size) * list_block_size);
    }

    static void set_query_value(workspace_type &ws, size_t index, T value) {
        ws.block_queries[index] = da_bf16::from_float(value);
    }

    static void gather_list_queries(workspace_type &ws, const da_int *query_indices,
                                    da_int block_start, da_int n_rows,
                                    da_int n_features) {
        for (da_int row = 0; row < n_rows; row++) {
            const da_int block_row = query_indices[row] - block_start;
            const auto *source =
                ws.block_queries.data() + static_cast<size_t>(block_row) * n_features;
            auto *destination =
                ws.list_queries.data() + static_cast<size_t>(row) * n_features;
            da_std::copy(source, source + n_features, destination);
        }
    }

    static bool compute_products(workspace_type &ws, const Q *codes, da_int n_rows,
                                 da_int n_cols, da_int n_features) {
        da_blas::cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_rows, n_cols,
                            n_features, /*alpha=*/1.0f, ws.list_queries.data(),
                            n_features, codes, n_features, /*beta=*/0.0f,
                            ws.dot_products.data(), n_cols);
        return true;
    }
};

template <> struct backend<uint8_t> {
    static constexpr bool is_quantized = true;
    static constexpr scalar_quantizer::code_kind code_kind = scalar_quantizer::int4;
    using T = float;
    using Q = uint8_t;
    using stored_type = Q;
    using workspace_type = int4_workspace;

    // Packed int4 stores two feature values per byte, so a logical list row does
    // not start at row*n_features as it does for dense formats. Return the byte
    // offset of the requested row using the canonical packed-storage contract.
    static size_t stored_row_offset(da_int row, da_int n_features) {
        return scalar_quantizer::int4_packing::row_offset(row, n_features);
    }

    static void resize_workspace(workspace_type &ws, da_int query_block_size,
                                 da_int list_block_size, da_int n_features) {
        const size_t query_elements = static_cast<size_t>(query_block_size) * n_features;
        ws.block_queries.resize(query_elements);
        ws.list_queries.resize(query_elements);
        ws.dot_products.resize(static_cast<size_t>(query_block_size) * list_block_size);
#ifndef _WIN32
        if (!ws.use_optimized_kernel)
            return;
        // Reserve the largest reordered B buffer this workspace will need. Actual
        // list blocks may be smaller, so compute_products queries the exact size again.
        dlp_metadata_t reorder_metadata{};
        const msz_t reorder_bytes = aocl_get_reorder_buf_size_bf16s4f32of32(
            'r', 't', 'b', n_features, list_block_size, &reorder_metadata);
        if (reorder_metadata.error_hndl.error_code == DLP_CLSC_SUCCESS &&
            reorder_bytes > 0)
            ws.reordered_list_codes.resize(reorder_bytes);
#endif
    }

    static void set_query_value(workspace_type &ws, size_t index, T value) {
        ws.block_queries[index] = da_bf16::from_float(value);
    }

    static void gather_list_queries(workspace_type &ws, const da_int *query_indices,
                                    da_int block_start, da_int n_rows,
                                    da_int n_features) {
        for (da_int row = 0; row < n_rows; row++) {
            const da_int block_row = query_indices[row] - block_start;
            const auto *source =
                ws.block_queries.data() + static_cast<size_t>(block_row) * n_features;
            auto *destination =
                ws.list_queries.data() + static_cast<size_t>(row) * n_features;
            da_std::copy(source, source + n_features, destination);
        }
    }

    static bool compute_products(workspace_type &ws, const Q *codes, da_int n_rows,
                                 da_int n_cols, da_int n_features) {
        const auto compute_portable = [&]() {
            const size_t row_bytes =
                scalar_quantizer::int4_packing::packed_size(n_features);
            for (da_int row = 0; row < n_rows; row++) {
                for (da_int col = 0; col < n_cols; col++) {
                    T product = T(0);
                    const uint8_t *code_row =
                        codes + static_cast<size_t>(col) * row_bytes;
                    for (da_int feature = 0; feature < n_features; feature++) {
                        product +=
                            da_bf16::to_float(
                                ws.list_queries[static_cast<size_t>(row) * n_features +
                                                feature]) *
                            static_cast<T>(
                                scalar_quantizer::int4_packing::load(code_row, feature));
                    }
                    ws.dot_products[static_cast<size_t>(row) * n_cols + col] = product;
                }
            }
            return true;
        };
#ifndef _WIN32
        if (!ws.use_optimized_kernel)
            return compute_portable();

        // ANN stores one packed row per vector, so the list block is physically
        // n_cols-by-n_features. Tell DLP to transpose that matrix while reordering B
        // into its required opaque n_features-by-n_cols kernel layout.
        const size_t row_bytes = scalar_quantizer::int4_packing::packed_size(n_features);
        const da_int stored_row_stride = static_cast<da_int>(2 * row_bytes);

        // Request the exact reorder size for this list block. The workspace capacity
        // was allocated for list_block_size, so a larger request indicates an invalid
        // shape or an unexpected DLP contract change.
        dlp_metadata_t reorder_metadata{};
        const msz_t reorder_bytes = aocl_get_reorder_buf_size_bf16s4f32of32(
            'r', 't', 'b', n_features, n_cols, &reorder_metadata);
        if (reorder_metadata.error_hndl.error_code != DLP_CLSC_SUCCESS ||
            reorder_bytes == 0 || reorder_bytes > ws.reordered_list_codes.capacity())
            return false;
        ws.reordered_list_codes.resize(reorder_bytes);
        aocl_reorder_bf16s4f32of32('r', 't', 'b', reinterpret_cast<const int8_t *>(codes),
                                   ws.reordered_list_codes.data(), n_features, n_cols,
                                   stored_row_stride, &reorder_metadata);
        if (reorder_metadata.error_hndl.error_code != DLP_CLSC_SUCCESS)
            return false;

        // dlp_qparam_t describes the unit per-tensor scale; dlp_quant_op_t tells DLP
        // to dequantize signed S4 B to BF16. Actual per-feature scales are already
        // folded into the queries, so a unit scale and no zero point preserve codes.
        float scale = 1.0f;
        dlp_qparam_t scale_parameters{&scale, 1, DLP_F32, DLP_PARAM_DIM_PER_TENSOR};
        dlp_quant_op_t b_quantization{
            DLP_QUANT_OP_DEQUANTIZE, DLP_S4, DLP_BF16, 0, nullptr,
            &scale_parameters,       nullptr};
        dlp_metadata_t gemm_metadata{};
        gemm_metadata.b_quant_op = &b_quantization;
        aocl_gemm_bf16s4f32of32('r', 'n', 'n', n_rows, n_cols, n_features, 1.0f,
                                ws.list_queries.data(), n_features, 'n',
                                ws.reordered_list_codes.data(), n_cols, 'r', 0.0f,
                                ws.dot_products.data(), n_cols, &gemm_metadata);
        if (gemm_metadata.error_hndl.error_code != DLP_CLSC_SUCCESS)
            return false;
        return true;
#else
        return compute_portable();
#endif
    }
};

} // namespace da_quantized_backend
} // namespace ARCH

#endif