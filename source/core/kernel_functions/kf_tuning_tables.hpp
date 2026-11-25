/* ************************************************************************
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
 * ************************************************************************ */

#ifndef KF_TUNING_TABLES_HPP
#define KF_TUNING_TABLES_HPP

#include "da_kernel_utils.hpp"

namespace da_kernel_functions {

// clang-format off

// ------ KERNEL FUNCTIONS TUNING TABLE ----------------------------------------
constexpr TBL<KernelSelection>::type kf_tuning = {{
  {generic,       tid<float>(),   {{{4, scalar}, {8,  avx}, {avx2}              }}},
  {generic,       tid<double>(),  {{{2, scalar}, {4,  avx}, {avx2}              }}},
  {generic,       tid<_Float16>(),{{{   scalar}                                 }}},
  {zen2,          tid<float>(),   {{{4, scalar}, {8,  avx}, {avx2}              }}},
  {zen2,          tid<double>(),  {{{2, scalar}, {4,  avx}, {avx2}              }}},
  {zen2,          tid<_Float16>(),{{{   scalar}                                 }}},
  {zen3,          tid<float>(),   {{{4, scalar}, {8,  avx}, {avx2}              }}},
  {zen3,          tid<double>(),  {{{2, scalar}, {4,  avx}, {avx2}              }}},
  {zen3,          tid<_Float16>(),{{{   scalar}                                 }}},
  {zen4,          tid<float>(),   {{{4, scalar}, {8,  avx}, {16, avx2}, {avx512}}}},
  {zen4,          tid<double>(),  {{{2, scalar}, {4,  avx}, { 8, avx2}, {avx512}}}},
  {zen4,          tid<_Float16>(),{{{   scalar}                                 }}},
  {zen5,          tid<float>(),   {{{4, scalar}, {8,  avx}, {16, avx2}, {avx512}}}},
  {zen5,          tid<double>(),  {{{2, scalar}, {4,  avx}, { 8, avx2}, {avx512}}}},
  {zen5,          tid<_Float16>(),{{{   scalar}                                 }}},
  {zen6,          tid<float>(),   {{{4, scalar}, {8,  avx}, {16, avx2}, {avx512}}}},
  {zen6,          tid<double>(),  {{{2, scalar}, {4,  avx}, { 8, avx2}, {avx512}}}},
  {zen6,          tid<_Float16>(),{{{8, scalar}, {16, avx}, {32, avx2}, {avx512}}}},
  {generic_avx512,tid<float>(),   {{{4, scalar}, {8,  avx}, {16, avx2}, {avx512}}}},
  {generic_avx512,tid<double>(),  {{{2, scalar}, {4,  avx}, { 8, avx2}, {avx512}}}},
  {generic_avx512,tid<_Float16>(),{{{8, scalar}, {16, avx}, {32, avx2}, {avx512}}}}
}};

// clang-format on

} // namespace da_kernel_functions

#endif // KF_TUNING_TABLES_HPP