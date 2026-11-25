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

#ifndef SCALAR_QUANTIZER_HPP
#define SCALAR_QUANTIZER_HPP

#include "aoclda_types.h"
#include "macros.h"
#include "model_persistence.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ARCH {

namespace scalar_quantizer {

// How each scheme maps a value to its (signed) integer code.
enum scheme {
    per_dimension,
    uniform,
    direct_unsigned,
    direct_signed,
};

// Storage format of the produced codes; none means quantization is disabled.
enum code_kind {
    none = 0,
    int4 = 1,
    int8 = 2,
    bfloat16 = 3,
};

// Physical representation used for signed int4 values throughout ANN storage.
// C++ cannot address a four-bit object directly, so two consecutive values
// share one uint8_t. Even indices occupy bits 0..3; odd indices occupy bits
// 4..7. Each four-bit field stores the low four bits of the signed value's two's-
// complement representation, giving the range [-8, 7].
namespace int4_packing {

// Select the low four bits: 0000 1111.
constexpr uint8_t value_mask = 0x0f;
constexpr unsigned bits_per_value = 4;

inline size_t packed_size(da_int value_count) {
    return (static_cast<size_t>(value_count) + 1) / 2;
}

inline size_t row_offset(da_int row, da_int values_per_row) {
    return static_cast<size_t>(row) * packed_size(values_per_row);
}

inline void store(uint8_t *packed_values, da_int element_index, int8_t value) {
    // Select the byte containing this value. Two consecutive values share one byte.
    const size_t byte_index = static_cast<size_t>(element_index) / 2;
    // Select the low four bits or high four bits: bit_offset is 0 or 4.
    const unsigned bit_offset = static_cast<unsigned>(element_index % 2) * bits_per_value;
    // Shift 0000 1111 over the four bits that will be replaced.
    const uint8_t shifted_mask = static_cast<uint8_t>(value_mask << bit_offset);
    // Keep the low four bits of the value's two's-complement representation.
    const uint8_t four_bit_pattern = static_cast<uint8_t>(value) & value_mask;
    // Clear the destination four bits, preserve the other four bits, then write the value.
    packed_values[byte_index] = static_cast<uint8_t>(
        (packed_values[byte_index] & static_cast<uint8_t>(~shifted_mask)) |
        static_cast<uint8_t>(four_bit_pattern << bit_offset));
}

inline uint8_t load_bits(const uint8_t *packed_values, da_int element_index) {
    // Leave the low four bits in place or shift the high four bits down by four positions.
    const unsigned bit_offset = static_cast<unsigned>(element_index % 2) * bits_per_value;
    return static_cast<uint8_t>((packed_values[element_index / 2] >> bit_offset) &
                                value_mask);
}

inline int8_t load(const uint8_t *packed_values, da_int element_index) {
    uint8_t bits = load_bits(packed_values, element_index);
    // Bit 3 is the sign bit of the four-bit value: 0x08 = 0000 1000.
    // To widen a two's-complement value, copy the sign bit into all new high positions.
    if (bits & 0x08)
        // Set bits 4..7: 0xf0 = 1111 0000.
        bits |= 0xf0;
    return static_cast<int8_t>(bits);
}

} // namespace int4_packing

template <typename T> class scalar_quantizer {
  public:
    explicit scalar_quantizer(scheme s = scheme::per_dimension,
                              code_kind k = code_kind::int8)
        : _scheme(s), _code_kind(k) {}

    da_status train(da_order order, const T *X, da_int n_samples, da_int n_features,
                    da_int ldx);

    // ldc is in code-element units. For packed int4 (Q = uint8_t), it is in bytes.
    // Q must match get_code_kind().
    template <typename Q>
    void quantize(da_order order, const T *X, da_int n_samples, da_int n_features,
                  da_int ldx, Q *codes, da_int ldc) const;

    const std::vector<T> &scale() const { return _scale; }
    const std::vector<T> &bias() const { return _bias; }
    scheme get_scheme() const { return _scheme; }
    code_kind get_code_kind() const { return _code_kind; }

    da_status serialize(da_model_persistence::serialization_buffer &buffer);

  private:
    static void code_range(code_kind k, T &lo, T &hi);
    static void range_to_scale_bias(T vmin, T vmax, T code_lo, T code_hi, T &scale,
                                    T &bias);
    T map_to_code(T value, da_int feature) const;
    scheme _scheme;
    code_kind _code_kind;
    std::vector<T> _scale{};
    std::vector<T> _bias{};
};

} // namespace scalar_quantizer

} // namespace ARCH

#endif
