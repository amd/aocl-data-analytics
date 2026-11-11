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

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wpass-failed"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wopenmp-simd"
#endif

#include "scalar_quantizer.hpp"
#include "bfloat16_helpers.hpp"
#include "da_std.hpp"
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>
#include <vector>

namespace ARCH {

namespace scalar_quantizer {

namespace {
template <typename Q, typename T> inline Q clamp_round_cast(T v, T lo, T hi) {
    return static_cast<Q>(da_std::min(da_std::max(v, lo), hi) + std::copysign(T(0.5), v));
}

template <typename T>
inline void quantize_and_pack_int4(T integer_coordinate, T lower_bound, T upper_bound,
                                   da_int element_index, uint8_t *packed_values) {
    // `integer_coordinate` is not yet an integer. The scalar quantizer has
    // transformed an original data value x onto the quantization grid using
    // (x - bias) / scale (or the corresponding direct scheme). This function
    // finishes that conversion by rounding to the nearest integer and clamping
    // it to the signed four-bit range. A signed four-bit integer can represent
    // exactly the sixteen values -8, -7, ..., 6, 7. int8_t is used temporarily
    // because C++ does not provide an independently addressable int4_t type.
    const int8_t quantized_integer =
        clamp_round_cast<int8_t>(integer_coordinate, lower_bound, upper_bound);
    int4_packing::store(packed_values, element_index, quantized_integer);
}
} // namespace

template <typename T> void scalar_quantizer<T>::code_range(code_kind k, T &lo, T &hi) {
    switch (k) {
    case code_kind::int4:
        lo = T(-8);
        hi = T(7);
        break;
    default:
        lo = static_cast<T>(std::numeric_limits<int8_t>::lowest());
        hi = static_cast<T>(std::numeric_limits<int8_t>::max());
        break;
    }
}

template <typename T>
void scalar_quantizer<T>::range_to_scale_bias(T vmin, T vmax, T code_lo, T code_hi,
                                              T &scale, T &bias) {
    T vdiff = vmax - vmin;
    if (vdiff == T(0)) {
        // Constant range: nonzero placeholder scale (quantize() would otherwise
        // divide by zero); bias = vmin maps every value to shifted-code 0.
        scale = T(1);
        bias = vmin;
    } else {
        scale = vdiff / (code_hi - code_lo);
        bias = vmin + (-code_lo + T(0.5)) * scale;
    }
}

template <typename T>
da_status scalar_quantizer<T>::train(da_order order, const T *X, da_int n_samples,
                                     da_int n_features, da_int ldx) {
    std::vector<T> vmin, vmax;
    try {
        _scale.assign(n_features, T(1));
        _bias.assign(n_features, T(0));
        if (_code_kind != code_kind::bfloat16 && _scheme == scheme::per_dimension) {
            vmin.resize(n_features);
            vmax.resize(n_features);
        }
    } catch (std::bad_alloc const &) {
        return da_status_memory_error;
    }

    if (_code_kind == code_kind::bfloat16)
        return da_status_success;

    T code_lo, code_hi;
    code_range(_code_kind, code_lo, code_hi);

    switch (_scheme) {
    case scheme::direct_signed:
        break;
    case scheme::direct_unsigned:
        da_std::fill(_bias.begin(), _bias.end(), -code_lo);
        break;
    case scheme::uniform: {
        T vmin = X[0], vmax = X[0];
        if (order == column_major) {
            for (da_int d = 0; d < n_features; d++) {
                const T *col = X + d * ldx;
#pragma omp simd reduction(min : vmin) reduction(max : vmax)
                for (da_int i = 0; i < n_samples; i++) {
                    vmin = da_std::min(vmin, col[i]);
                    vmax = da_std::max(vmax, col[i]);
                }
            }
        } else {
            for (da_int i = 0; i < n_samples; i++) {
                const T *row = X + i * ldx;
#pragma omp simd reduction(min : vmin) reduction(max : vmax)
                for (da_int d = 0; d < n_features; d++) {
                    vmin = da_std::min(vmin, row[d]);
                    vmax = da_std::max(vmax, row[d]);
                }
            }
        }
        T sc, bs;
        range_to_scale_bias(vmin, vmax, code_lo, code_hi, sc, bs);
        da_std::fill(_scale.begin(), _scale.end(), sc);
        da_std::fill(_bias.begin(), _bias.end(), bs);
        break;
    }
    default: {
        if (order == column_major) {
            for (da_int d = 0; d < n_features; d++) {
                const T *col = X + d * ldx;
                T lo = col[0], hi = col[0];
#pragma omp simd reduction(min : lo) reduction(max : hi)
                for (da_int i = 1; i < n_samples; i++) {
                    lo = da_std::min(lo, col[i]);
                    hi = da_std::max(hi, col[i]);
                }
                vmin[d] = lo;
                vmax[d] = hi;
            }
        } else {
            for (da_int d = 0; d < n_features; d++) {
                vmin[d] = X[d];
                vmax[d] = X[d];
            }
            for (da_int i = 1; i < n_samples; i++) {
                const T *row = X + i * ldx;
#pragma omp simd
                for (da_int d = 0; d < n_features; d++) {
                    vmin[d] = da_std::min(vmin[d], row[d]);
                    vmax[d] = da_std::max(vmax[d], row[d]);
                }
            }
        }
        for (da_int d = 0; d < n_features; d++)
            range_to_scale_bias(vmin[d], vmax[d], code_lo, code_hi, _scale[d], _bias[d]);
        break;
    }
    }
    return da_status_success;
}

template <typename T> T scalar_quantizer<T>::map_to_code(T value, da_int feature) const {
    switch (_scheme) {
    case scheme::direct_signed:
        return value;
    case scheme::direct_unsigned:
        return value - _bias[feature];
    default:
        return (value - _bias[feature]) / _scale[feature];
    }
}

template <typename T>
template <typename Q>
void scalar_quantizer<T>::quantize(da_order order, const T *X, da_int n_samples,
                                   da_int n_features, da_int ldx, Q *codes,
                                   da_int ldc) const {
    static_assert(std::is_same_v<Q, uint8_t> || std::is_same_v<Q, int8_t> ||
                      std::is_same_v<Q, da_bf16::storage_type>,
                  "unsupported scalar_quantizer storage type");

    if (_code_kind == code_kind::bfloat16) {
        if constexpr (std::is_same_v<Q, da_bf16::storage_type>) {
            if (order == column_major) {
                for (da_int d = 0; d < n_features; d++) {
                    const T *xc = X + d * ldx;
                    Q *cc = codes + d * ldc;
                    for (da_int i = 0; i < n_samples; i++)
                        cc[i] = da_bf16::from_float(static_cast<float>(xc[i]));
                }
            } else {
                for (da_int i = 0; i < n_samples; i++) {
                    const T *xr = X + i * ldx;
                    Q *cr = codes + i * ldc;
                    for (da_int d = 0; d < n_features; d++)
                        cr[d] = da_bf16::from_float(static_cast<float>(xr[d]));
                }
            }
        }
        return;
    }

    T lo, hi;
    code_range(_code_kind, lo, hi);

    if constexpr (std::is_same_v<Q, uint8_t>) {
        if (order == column_major) {
            for (da_int d = 0; d < n_features; d++) {
                const T *xc = X + d * ldx;
                Q *cc = codes + d * ldc;
                da_std::fill(cc, cc + int4_packing::packed_size(n_samples), uint8_t(0));
                for (da_int i = 0; i < n_samples; i++)
                    quantize_and_pack_int4(map_to_code(xc[i], d), lo, hi, i, cc);
            }
        } else {
            for (da_int i = 0; i < n_samples; i++) {
                const T *xr = X + i * ldx;
                Q *cr = codes + i * ldc;
                da_std::fill(cr, cr + int4_packing::packed_size(n_features), uint8_t(0));
                for (da_int d = 0; d < n_features; d++)
                    quantize_and_pack_int4(map_to_code(xr[d], d), lo, hi, d, cr);
            }
        }
    } else if (order == column_major) {
        for (da_int d = 0; d < n_features; d++) {
            const T *xc = X + d * ldx;
            Q *cc = codes + d * ldc;
#pragma omp simd
            for (da_int i = 0; i < n_samples; i++)
                cc[i] = clamp_round_cast<Q>(map_to_code(xc[i], d), lo, hi);
        }
    } else {
        for (da_int i = 0; i < n_samples; i++) {
            const T *xr = X + i * ldx;
            Q *cr = codes + i * ldc;
#pragma omp simd
            for (da_int d = 0; d < n_features; d++)
                cr[d] = clamp_round_cast<Q>(map_to_code(xr[d], d), lo, hi);
        }
    }
}

template <typename T>
da_status
scalar_quantizer<T>::serialize(da_model_persistence::serialization_buffer &buffer) {
    da_status status = da_status_success;
    da_int scheme_value = static_cast<da_int>(this->_scheme);
    da_int code_kind_value = static_cast<da_int>(this->_code_kind);
    auto io_dispatch = [&buffer, &status](auto &data) -> void {
        if (status != da_status_success) {
            return;
        }
        status = buffer.dispatch_buffer_io(data);
        return;
    };

    io_dispatch(scheme_value);
    io_dispatch(code_kind_value);
    io_dispatch(this->_scale);
    io_dispatch(this->_bias);

    if (status == da_status_success &&
        buffer.get_mode() == da_model_persistence::deserialize) {
        this->_scheme = static_cast<scheme>(scheme_value);
        this->_code_kind = static_cast<code_kind>(code_kind_value);
    }

    return status;
}

template class scalar_quantizer<float>;
template class scalar_quantizer<double>;

template void scalar_quantizer<float>::quantize<uint8_t>(da_order, const float *, da_int,
                                                         da_int, da_int, uint8_t *,
                                                         da_int) const;
template void scalar_quantizer<float>::quantize<int8_t>(da_order, const float *, da_int,
                                                        da_int, da_int, int8_t *,
                                                        da_int) const;
template void scalar_quantizer<float>::quantize<da_bf16::storage_type>(
    da_order, const float *, da_int, da_int, da_int, da_bf16::storage_type *,
    da_int) const;
template void scalar_quantizer<double>::quantize<uint8_t>(da_order, const double *,
                                                          da_int, da_int, da_int,
                                                          uint8_t *, da_int) const;
template void scalar_quantizer<double>::quantize<int8_t>(da_order, const double *, da_int,
                                                         da_int, da_int, int8_t *,
                                                         da_int) const;
template void scalar_quantizer<double>::quantize<da_bf16::storage_type>(
    da_order, const double *, da_int, da_int, da_int, da_bf16::storage_type *,
    da_int) const;

} // namespace scalar_quantizer

} // namespace ARCH

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
