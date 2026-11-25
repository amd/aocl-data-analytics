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

#ifndef KMEANS_TYPES_HPP
#define KMEANS_TYPES_HPP

template <typename T>
inline constexpr da_int KMEANS_LLOYD_BLOCK_SIZE =
    std::is_same<T, double>::value ? 256 : (std::is_same<T, _Float16>::value ? 768 : 512);

template <typename T>
inline constexpr da_int KMEANS_ELKAN_BLOCK_SIZE =
    std::is_same<T, double>::value ? 256 : (std::is_same<T, _Float16>::value ? 768 : 512);

template <typename T>
inline constexpr da_int KMEANS_MACQUEEN_BLOCK_SIZE =
    std::is_same<T, double>::value ? 128 : (std::is_same<T, _Float16>::value ? 768 : 256);

namespace da_kmeans_types {

enum kmeans_method { lloyd = 0, elkan, hartigan_wong, macqueen };
enum kmeans_init { random_samples = 0, kmeanspp, supplied, random_partitions, afk_mcmc };
enum empty_cluster_method { ignore = 0, error, split };

} // namespace da_kmeans_types

#endif // KMEANS_TYPES_HPP
