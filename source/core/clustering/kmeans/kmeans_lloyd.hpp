/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef KMEANS_LLOYD_HPP
#define KMEANS_LLOYD_HPP

#include "aoclda.h"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "da_omp.hpp"
#include "da_std.hpp"
#include "kmeans.hpp"
#include "kmeans_kernels.hpp"
#include "kmeans_tuning_tables.hpp"
#include "kmeans_types.hpp"
#include "kt.hpp"
#include "lapack_templates.hpp"
#include "macros.h"
#include "miscellaneous.hpp"
#include "pairwise_distances.hpp"
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>

namespace ARCH {

namespace da_kmeans {
// clang-format off
// LLOYD KERNEL IMPLEMENTATIONS =========================================
namespace {
using ES = std::function<void(bool, da_int, float *, da_int *, da_int *, float *, da_int, da_int)>;
using ED = std::function<void(bool, da_int, double *, da_int *, da_int *, double *, da_int, da_int)>;
}
inline const kernel_implementations<ES, ED> lloyd_implementations = {
{{ // float map
            /* scalar    */ lloyd_iteration_kernel_scalar<float>,
            /* avx (sse) */ lloyd_iteration_kernel<float, avx>,
            /* avx2      */ lloyd_iteration_kernel<float, avx2>,
ORL_AVX512F(/* avx512    */ lloyd_iteration_kernel<float, avx512>)
}},
{{ // double map
            /* scalar    */ lloyd_iteration_kernel_scalar<double>,
            /* avx (sse) */ lloyd_iteration_kernel<double, avx>,
            /* avx2      */ lloyd_iteration_kernel<double, avx2>,
ORL_AVX512F(/* avx512    */ lloyd_iteration_kernel<double, avx512>)
}}
};
// clang-format on

using namespace da_kmeans_types;
using namespace std::literals::string_literals;

template <typename T>
void kmeans<T>::assign_lloyd_kernel(
    std::function<void(bool, da_int, T *, da_int *, da_int *, T *, da_int, da_int)>
        &kernel,
    da_int &padding, da_int n_clusters) {
    vectorization_type isa{undefined};
    using namespace ::da_kmeans; // External ns

    isa = Oracle<KernelSelection>(lloyd_tuning, tid<T>(), n_clusters, "kmeans.isa");

    kernel = lloyd_implementations.get<T>(isa);
    padding = get_padding<T>(isa);

    // Add telemetry
    context_set_hidden_settings("kmeans.setup"s,
                                "kernel=lloyd,kernel.type="s + std::to_string(isa) +
                                    ",kernel.padding="s + std::to_string(padding));
}

/* Perform a single iteration of Lloyd's method */
template <typename T>
void kmeans<T>::lloyd_iteration(bool update_centres, da_int n_threads) {

    if (update_centres) {
        da_std::fill(cluster_count.begin(), cluster_count.end(), 0);
        da_std::fill(current_cluster_centres->begin(), current_cluster_centres->end(),
                     (T)0.0);

        if (n_threads > 1) {
            for (da_int j = 0; j < n_threads; j++) {
                auto &local_cluster_centres = thd_cluster_centres[j];
                auto &local_work_int = thd_work_int[j];
                da_std::fill(local_cluster_centres.begin(),
                             local_cluster_centres.begin() + n_clusters * n_features,
                             (T)0.0);
                da_std::fill(local_work_int.begin(), local_work_int.begin() + n_clusters,
                             0);
            }
        }
    }

    // Compute the squared norms of the previous cluster centres to avoid recomputing them repeatedly in the blocked section
    for (da_int i = 0; i < n_clusters; i++) {
        workc1[i] = (T)0.0;
    }

    // Need to leave workc1 as zeros if we are doing spherical k-means
    if (!do_spherical) {
        da_utils::compute_squared_row_norms(column_major, n_clusters, n_features,
                                            (*previous_cluster_centres).data(),
                                            n_clusters, workc1.data());
    }

    // Distance matrix part of the computation needs to be done in blocks since it is memory intensive
    da_int block_index;
    da_int block_size = max_block_size;

    // For row-major storage of A, we use a trick which treats A as column-major storage of A^T in gemm calls
    auto A_blas_trans = (this->A_order == column_major) ? CblasTrans : CblasNoTrans;
    // if we are doing spherical k-means, the "distance computation" is just gemm, w/ a minus
    // sign so comparisons are done appropriately
    T gemm_scalar = (this->do_spherical) ? -1.0 : -2.0;

    // Precompute pointer to inverse data norms for normalized spherical k-means centre updates
    const T *inv_norm_ptr =
        (do_spherical && normalize_data) ? data_inv_norms.data() : nullptr;

    if (n_threads > 1) {

        omp_lock_t cluster_count_lock, cluster_centres_lock;
        omp_init_lock(&cluster_count_lock);
        omp_init_lock(&cluster_centres_lock);

#pragma omp parallel shared(                                                             \
        n_blocks, block_rem, update_centres, A, lda, gemm_scalar,                        \
            previous_cluster_centres, current_cluster_centres, cluster_count,            \
            current_labels, workc1, workcs1, ldworkcs1, max_block_size,                  \
            thd_cluster_centres, thd_work_int, cluster_centres_lock, cluster_count_lock, \
            A_blas_trans, thd_work1, thd_work2, thd_work3, thd_work4, inv_norm_ptr)      \
    firstprivate(block_size) private(block_index) default(none) num_threads(n_threads)
        {
            da_int this_thread = (da_int)omp_get_thread_num();
            auto &local_work_int = thd_work_int[this_thread];
            auto &local_cluster_centres = thd_cluster_centres[this_thread];
            auto &local_work1 = thd_work1[this_thread];
            auto &local_work2 = thd_work2[this_thread];
            auto &local_work3 = thd_work3[this_thread];
            auto &local_work4 = thd_work4[this_thread];
            da_int workcs1_index = this_thread * max_block_size * ldworkcs1;
#pragma omp for nowait schedule(dynamic)
            for (da_int i = 0; i < n_blocks; i++) {
                if (i == n_blocks - 1 && block_rem > 0) {
                    block_index = n_samples - block_rem;
                    block_size = block_rem;
                } else {
                    block_index = i * max_block_size;
                    block_size = max_block_size;
                }
                da_int A_index =
                    (this->A_order == column_major) ? block_index : block_index * lda;
                // Compute the matrix D where D_{ij} = ||C_j||^2 - 2 A C^T
                // Don't form it explicitly though: just form -2AC^T and add the ||C_j||^2 as and when we need them
                // Array access patterns mean for this loop it is quicker to form -2CA^T
                // For spherical kmeans we form -CA^T
                da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, A_blas_trans, n_clusters,
                                    block_size, n_features, gemm_scalar,
                                    (*previous_cluster_centres).data(), n_clusters,
                                    &A[A_index], lda, 0.0, &workcs1[workcs1_index],
                                    ldworkcs1);

                // Loop through the samples and find the closest cluster centre and its label
                lloyd_kernel(update_centres, block_size, workc1.data(),
                             &local_work_int[0], &(*current_labels)[block_index],
                             &workcs1[workcs1_index], ldworkcs1, n_clusters);

                if (update_centres)
                    lloyd_iteration_update_centres(
                        block_size, &A[A_index], lda, &local_cluster_centres[0],
                        &(*current_labels)[block_index], &local_work1[0], &local_work2[0],
                        &local_work3[0], &local_work4[0],
                        inv_norm_ptr ? inv_norm_ptr + block_index : nullptr);
            }
            // Now aggregate local_work_int into cluster_count and local_cluster_centres into current_cluster_centres
            // The while loop is used because we don't mind what order each thread executes the two critical regions
            bool reduced_cluster_count = false, reduced_cluster_centres = false;
            while (!reduced_cluster_count || !reduced_cluster_centres) {
                if (!reduced_cluster_count) {

                    omp_set_lock(&cluster_count_lock);
                    for (da_int i = 0; i < n_clusters; i++) {
                        cluster_count[i] += local_work_int[i];
                    }
                    omp_unset_lock(&cluster_count_lock);
                    reduced_cluster_count = true;
                }
                if (!reduced_cluster_centres) {
                    omp_set_lock(&cluster_centres_lock);
                    for (da_int i = 0; i < n_clusters * n_features; i++) {
                        (*current_cluster_centres)[i] += local_cluster_centres[i];
                    }
                    omp_unset_lock(&cluster_centres_lock);
                    reduced_cluster_centres = true;
                }
            }
        } // end of parallel region
        omp_destroy_lock(&cluster_count_lock);
        omp_destroy_lock(&cluster_centres_lock);
    } else {

        for (da_int i = 0; i < n_blocks; i++) {
            if (i == n_blocks - 1 && block_rem > 0) {
                block_index = n_samples - block_rem;
                block_size = block_rem;
            } else {
                block_index = i * max_block_size;
            }
            da_int A_index =
                (this->A_order == column_major) ? block_index : block_index * lda;
            // Compute the matrix D where D_{ij} = ||C_j||^2 - 2 A C^T
            // Don't form it explicitly though: just form -2AC^T and add the ||C_j||^2 as and when we need them
            // Array access patterns mean for this loop it is quicker to form -2CA^T
            // For spherical kmeans we form -CA^T
            da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, A_blas_trans, n_clusters,
                                block_size, n_features, gemm_scalar,
                                (*previous_cluster_centres).data(), n_clusters,
                                &A[A_index], lda, 0.0, workcs1.data(), ldworkcs1);

            // Loop through the samples and find the closest cluster centre and its label
            lloyd_kernel(update_centres, block_size, workc1.data(), cluster_count.data(),
                         &(*current_labels)[block_index], workcs1.data(), ldworkcs1,
                         n_clusters);

            if (update_centres)
                lloyd_iteration_update_centres(
                    block_size, &A[A_index], lda, (*current_cluster_centres).data(),
                    &(*current_labels)[block_index], thd_work1[0].data(),
                    thd_work2[0].data(), thd_work3[0].data(), thd_work4[0].data(),
                    inv_norm_ptr ? inv_norm_ptr + block_index : nullptr);
        }
    }

    if (update_centres) {
        scale_current_cluster_centres();
        // Compute change in centres in this iteration
        compute_centre_shift();
    }
}

/* During the Lloyd iteration, update the centres of the computed clusters */
template <typename T>
void kmeans<T>::lloyd_iteration_update_centres(da_int block_size, const T *data,
                                               da_int lddata, T *new_cluster_centres,
                                               da_int *labels, T *work1, T *work2,
                                               T *work3, T *work4,
                                               const T *block_data_inv_norms) {

    // Spherical k-means with normalized data: accumulate normalized data points
    if (do_spherical && normalize_data && block_data_inv_norms != nullptr) {
        if (this->A_order == column_major) {
            T *dst = new_cluster_centres;
            const T *src = data;
            for (da_int j = 0; j < n_features; j++) {
                for (da_int i = 0; i < block_size; i++) {
                    dst[labels[i]] += src[i] * block_data_inv_norms[i];
                }
                dst += n_clusters;
                src += lddata;
            }
        } else {
            for (da_int i = 0; i < block_size; i++) {
                T *dst = new_cluster_centres + labels[i];
                const T *src = data + i * lddata;
                for (da_int j = 0; j < n_features; j++) {
                    dst[j * n_clusters] += src[j] * block_data_inv_norms[i];
                }
            }
        }
        return;
    }

    if (this->A_order == column_major) {

        T *dst = new_cluster_centres;
        const T *src = data;

        if (n_clusters >= KMEANS_LLOYD_BLOCK_SIZE<T>) {
            for (da_int j = 0; j < n_features; j++) {
                for (da_int i = 0; i < block_size; i++) {
                    dst[labels[i]] += src[i];
                }
                dst += n_clusters;
                src += lddata;
            }
        } else {

            for (da_int j = 0; j < n_features; j++) {

                // The vector scatters into the cluster centres are not efficient, so we do a manual
                // accumulation 4 at a time using temporary work arrays to enable vectorization and amortize the cost
                da_std::fill(work1, work1 + n_clusters, (T)0.0);
                da_std::fill(work2, work2 + n_clusters, (T)0.0);
                da_std::fill(work3, work3 + n_clusters, (T)0.0);
                da_std::fill(work4, work4 + n_clusters, (T)0.0);

                da_int i = 0;
                for (; i + 3 < block_size; i += 4) {
                    work1[labels[i]] += src[i];
                    work2[labels[i + 1]] += src[i + 1];
                    work3[labels[i + 2]] += src[i + 2];
                    work4[labels[i + 3]] += src[i + 3];
                }

                // Deal with with remaining elements
                for (; i < block_size; i++) {
                    work1[labels[i]] += src[i];
                }

                for (da_int k = 0; k < n_clusters; k++) {
                    dst[k] += work1[k] + work2[k] + work3[k] + work4[k];
                }

                dst += n_clusters;
                src += lddata;
            }
        }
    } else {
        // A is row-major but cluster centres are column-major
        for (da_int i = 0; i < block_size; i++) {
            T *dst = new_cluster_centres + labels[i];
            const T *src = data + i * lddata;
            // Add this sample to the cluster mean
            for (da_int j = 0; j < n_features; j++) {
                dst[j * n_clusters] += src[j];
            }
        }
    }
}

/* Scaling phase for the current cluster centres; part of both the Elkan and Lloyd algorithms */
template <typename T> void kmeans<T>::scale_current_cluster_centres() {

    // Guard against empty clusters - avoid division by zero below
    for (da_int i = 0; i < n_clusters; i++) {
        if (cluster_count[i] == 0)
            cluster_count[i] = 1;
    }

    // Scale to get proper column means (cluster_count contains the number of data points in each cluster)
    if (!do_spherical) {
        if (this->algorithm == lloyd) {
// Clusters are stored column-major
#pragma omp simd collapse(2)
            for (da_int j = 0; j < n_features; j++) {
                for (da_int i = 0; i < n_clusters; i++) {
                    (*current_cluster_centres)[i + j * n_clusters] /= cluster_count[i];
                }
            }
        } else {
            // Clusters are stored row-major
#pragma omp simd collapse(2)
            for (da_int i = 0; i < n_clusters; i++) {
                for (da_int j = 0; j < n_features; j++) {
                    (*current_cluster_centres)[i * n_features + j] /= cluster_count[i];
                }
            }
        }
    } else {
        // For spherical k-means we need to normalize cluster centres to unit length
        if (this->algorithm == lloyd) {
            // Centres stored column-major
            da_utils::normalize_rows_inplace(column_major, n_clusters, n_features,
                                             (*current_cluster_centres).data(),
                                             n_clusters, workc1.data());
        } else {
            // Centres stored row-major (Elkan)
            da_utils::normalize_rows_inplace(row_major, n_clusters, n_features,
                                             (*current_cluster_centres).data(),
                                             n_features, workc1.data());
        }
    }
}

} // namespace da_kmeans

} // namespace ARCH

#endif // KMEANS_LLOYD_HPP
