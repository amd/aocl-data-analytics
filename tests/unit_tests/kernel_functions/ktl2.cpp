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
#include "da_std.hpp"
#include "kt.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstring>

namespace {

using namespace TEST_ARCH;
using namespace kernel_templates;

template <typename T> class kt_level2_test : public testing::Test {};

using KTL2Types = ::testing::Types<float, double
#ifdef __AVX512FP16__
                                   ,
                                   _Float16
#endif
                                   >;
TYPED_TEST_SUITE(kt_level2_test, KTL2Types);

template <bsz SZ, typename SUF> void test_exp_kt() {
    constexpr size_t sz{tsz_v<SZ, SUF>};
    constexpr SUF data[32] = {
        (SUF)1,     (SUF)2,     (SUF)3,     (SUF)4,     (SUF)-1,   (SUF)-2,   (SUF)-3,
        (SUF)-4,    (SUF)5,     (SUF)-5,    (SUF)6,     (SUF)-6,   (SUF)7,    (SUF)-7,
        (SUF)8,     (SUF)-8,    (SUF)1.5,   (SUF)2.5,   (SUF)3.5,  (SUF)4.5,  (SUF)-1.5,
        (SUF)-2.5,  (SUF)-3.5,  (SUF)-4.5,  (SUF)1.25,  (SUF)2.25, (SUF)3.25, (SUF)4.25,
        (SUF)-1.25, (SUF)-2.25, (SUF)-3.25, (SUF)-4.25,
    };
    avxvector_t<SZ, SUF> a = kt_loadu_p<SZ>(data);
    avxvector_t<SZ, SUF> r = kt_exp_p<SZ, SUF>(a);

    SUF expected[sz];
    for (size_t i = 0; i < sz; ++i) {
#ifdef __AVX512FP16__
        if constexpr (std::is_same_v<SUF, _Float16>)
            expected[i] = da_std::exp(data[i]);
        else
#endif
            expected[i] = std::exp(data[i]);
    }

    SUF result[sz];
    kt_storeu_p<SZ>(result, r);

    for (size_t i = 0; i < sz; ++i) {
#ifdef VERBOSE
        std::cout << "Vector; got[" << i << "] = " << got << "     ref[" << i
                  << "] = " << ref << "   adiff = " << std::abs(got - ref) << "\n";
#endif
        if constexpr (std::is_same_v<SUF, float>) {
            EXPECT_FLOAT_EQ(result[i], expected[i]);
        } else if constexpr (std::is_same_v<SUF, double>) {
            EXPECT_DOUBLE_EQ(result[i], expected[i]);
        } else {
            // _Float16
            float resi = static_cast<float>(result[i]);
            float expi = static_cast<float>(expected[i]);
            EXPECT_FLOAT_EQ(resi, expi);
        }
    }
}

#ifdef DA_KT_TEST_FALLBACK
TYPED_TEST(kt_level2_test, exp_kt_fallback)
#else
TYPED_TEST(kt_level2_test, exp_kt)
#endif
{
    using SUF = TypeParam;

    std::cout << "KT Headers: [ ";
#ifdef DA_KT_TEST_FALLBACK
    std::cout << "DA_KT_TEST_FALLBACK ";
#endif
#ifdef _KT_L2_CLANG_
    std::cout << "_KT_L2_CLANG_ ";
#endif
#ifdef _KT_L2_GCC_
    std::cout << "_KT_L2_GCC_ ";
#endif
#ifdef _KT_L2_FALLBACK_
    std::cout << "_KT_L2_FALLBACK_ ";
#endif
    std::cout << "]\n";

    test_exp_kt<bsz::b128, SUF>();
    test_exp_kt<bsz::b256, SUF>();
#ifdef __AVX512F__
    test_exp_kt<bsz::b512, SUF>();
#endif
}
} // namespace
