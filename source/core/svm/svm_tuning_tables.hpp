/* ************************************************************************
 * Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */

#ifndef SVM_TUNING_TABLES_HPP
#define SVM_TUNING_TABLES_HPP

#include "aoclda.h"
#include "svm_types.hpp"

namespace da_svm_tuning_tables {
using namespace da_svm_types;
using namespace da_dispatch;

// clang-format off

// ------ WSSI/WSSJ TUNING TABLES ----------------------------------------------
constexpr TBL<KernelSelection>::type wssi_tuning = {{
    {generic,       tid<float>(),   {{{4, scalar}, {1024,  avx}, {  avx2}              }}},
    {generic,       tid<double>(),  {{{4, scalar}, { 512,  avx}, {  avx2}              }}},
    {generic,       tid<_Float16>(),{{{scalar}                                         }}},
    {zen2,          tid<float>(),   {{{4, scalar}, {1024,  avx}, {  avx2}              }}},
    {zen2,          tid<double>(),  {{{4, scalar}, { 512,  avx}, {  avx2}              }}},
    {zen2,          tid<_Float16>(),{{{scalar}                                         }}},
    {zen3,          tid<float>(),   {{{4, scalar}, {1024,  avx}, {  avx2}              }}},
    {zen3,          tid<double>(),  {{{4, scalar}, { 512,  avx}, {  avx2}              }}},
    {zen3,          tid<_Float16>(),{{{scalar}                                         }}},
    {zen4,          tid<float>(),   {{{4, scalar}, {1024,  avx}, {  avx2}              }}},
    {zen4,          tid<double>(),  {{{4, scalar}, { 512,  avx}, {  avx2}              }}},
    {zen4,          tid<_Float16>(),{{{scalar}                                         }}},
    {zen5,          tid<float>(),   {{{8, scalar}, {1024,  avx}, {  avx2}              }}},
    {zen5,          tid<double>(),  {{{8,    avx}, {1024, avx2}, {avx512}              }}},
    {zen5,          tid<_Float16>(),{{{scalar}                                         }}},
    {zen6,          tid<float>(),   {{{8, scalar}, {1024,  avx}, {  avx2}              }}},
    {zen6,          tid<double>(),  {{{8,    avx}, {1024, avx2}, {avx512}              }}},
    {zen6,          tid<_Float16>(),{{{8, scalar}, {1024,  avx}, {2048, avx2}, {avx512}}}},
    {generic_avx512,tid<float>(),   {{{8, scalar}, {1024,  avx}, {  avx2}              }}},
    {generic_avx512,tid<double>(),  {{{8,    avx}, {1024, avx2}, {avx512}              }}},
    {generic_avx512,tid<_Float16>(),{{{8, scalar}, {1024,  avx}, {2048, avx2}, {avx512}}}}
}};
constexpr TBL<KernelSelection>::type wssj_tuning = {{
    {generic,       tid<float>(),   {{{128, avx}, {       avx2}          }}},
    {generic,       tid<double>(),  {{{ 64, avx}, {       avx2}          }}},
    {generic,       tid<_Float16>(),{{{scalar}                           }}},
    {zen2,          tid<float>(),   {{{128, avx}, {       avx2}          }}},
    {zen2,          tid<double>(),  {{{ 64, avx}, {       avx2}          }}},
    {zen2,          tid<_Float16>(),{{{scalar}                           }}},
    {zen3,          tid<float>(),   {{{128, avx}, {       avx2}          }}},
    {zen3,          tid<double>(),  {{{ 64, avx}, {       avx2}          }}},
    {zen3,          tid<_Float16>(),{{{scalar}                           }}},
    {zen4,          tid<float>(),   {{{ 64, avx}, {128,   avx2}, {avx512}}}},
    {zen4,          tid<double>(),  {{{ 64, avx2},{     avx512}          }}},
    {zen4,          tid<_Float16>(),{{{scalar}                           }}},
    {zen5,          tid<float>(),   {{{128, avx2},{     avx512}          }}},
    {zen5,          tid<double>(),  {{{ 64, avx2},{     avx512}          }}},
    {zen5,          tid<_Float16>(),{{{scalar}                           }}},
    {zen6,          tid<float>(),   {{{128, avx2},{     avx512}          }}},
    {zen6,          tid<double>(),  {{{ 64, avx2},{     avx512}          }}},
    {zen6,          tid<_Float16>(),{{{256, avx}, {512, avx2},   {avx512}}}},
    {generic_avx512,tid<float>(),   {{{128, avx2},{     avx512}          }}},
    {generic_avx512,tid<double>(),  {{{ 64, avx2},{     avx512}          }}},
    {generic_avx512,tid<_Float16>(),{{{256, avx}, {512, avx2},   {avx512}}}}
}};

// ----- WS SIZE TUNING TABLES -------------------------------------------------
struct OptMapping {
    using thr_t = da_int; // threshold type
    using optv_t = da_int; // optimal value type
    thr_t threshold;
    optv_t optv; // if oracle(p, threshold) { return optimal value optv }
    OptMapping() = default;
    constexpr OptMapping(thr_t t, optv_t optv) : threshold(t), optv(optv) {};
    constexpr OptMapping(optv_t v) : threshold(std::numeric_limits<thr_t>::max()), optv(v) {};
};
using TBL_T = typename tuning::TBL<tuning::tblRow<svm_kernel, OptMapping, 3>, 4>::type;
// Note table does not need to define all enum types
constexpr TBL_T optimal_ws_size = {{
{ svm_kernel::rbf,        {{{ 5000, 512}, {50000, 1024}, {2048}}}},
{ svm_kernel::linear,     {{{20000, 256}, {50000,  512}, {1024}}}},
{ svm_kernel::sigmoid,    {{{ 5000, 512}, {50000, 1024}, {2048}}}},
{ svm_kernel::polynomial, {{{ 5000, 512}, {50000, 1024}, {2048}}}},
}};
// clang-format on

} // namespace da_svm_tuning_tables

#endif //SVM_TUNING_TABLES_HPP
