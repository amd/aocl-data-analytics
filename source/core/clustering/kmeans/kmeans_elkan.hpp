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

#ifndef KMEANS_ELKAN_HPP
#define KMEANS_ELKAN_HPP

#include "aoclda.h"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "da_omp.hpp"
#include "da_std.hpp"
#include "da_utils.hpp"
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

using arch = dispatch_architecture;
using kernel_templates::bsz;

// clang-format off
// ELKAN UPDATE KERNEL IMPLEMENTATIONS =========================================
namespace {
using US = std::function<void(da_int, float *, da_int, float *, float *, da_int *, da_int)>;
using UD = std::function<void(da_int, double *, da_int, double *, double *, da_int *, da_int)>;
}
inline const kernel_implementations<US, UD> elkan_update_implementations = {
{{ // float map
            /* scalar    */ elkan_iteration_kernel_scalar<float>,
            /* avx (sse) */ elkan_iteration_kt<bsz::b128, float>,
            /* avx2      */ elkan_iteration_kt<bsz::b256, float>,
ORL_AVX512F(/* avx512    */ elkan_iteration_kt<bsz::b512, float>)
}},
{{ // double map
            /* scalar    */ elkan_iteration_kernel_scalar<double>,
            /* avx (sse) */ elkan_iteration_kt<bsz::b128, double>,
            /* avx2      */ elkan_iteration_kt<bsz::b256, double>,
ORL_AVX512F(/* avx512    */ elkan_iteration_kt<bsz::b512, double>)
}}
};
// ELKAN REDUCE KERNEL IMPLEMENTATIONS =========================================
namespace {
using RS = std::function<float(da_int, const float *, float *)>;
using RD = std::function<double(da_int, const double *, double *)>;
}
inline const kernel_implementations<RS, RD> elkan_reduction_implementations = {
{{ // float map
            /* scalar    */ elkan_reduction_kernel_scalar<float>,
            /* avx (sse) */ elkan_reduction_kt<bsz::b128, float>,
            /* avx2      */ elkan_reduction_kt<bsz::b256, float>,
ORL_AVX512F(/* avx512    */ elkan_reduction_kt<bsz::b512, float>)
}},
{{ // double map
            /* scalar    */ elkan_reduction_kernel_scalar<double>,
            /* avx (sse) */ elkan_reduction_kt<bsz::b128, double>,
            /* avx2      */ elkan_reduction_kt<bsz::b256, double>,
ORL_AVX512F(/* avx512    */ elkan_reduction_kt<bsz::b512, double>)
}}
};
// clang-format on

using namespace da_kmeans_types;
using namespace std::literals::string_literals;

// Elkan dispatcher
template <typename T>
void kmeans<T>::assign_elkan_kernels(
    std::function<void(da_int, T *, da_int, T *, T *, da_int *, da_int)> &update_kernel,
    std::function<T(da_int, const T *, T *)> &reduce_kernel, da_int &padding,
    da_int n_clusters, da_int n_features) {
    using namespace ::da_kmeans; // External ns
    vectorization_type u_isa{undefined}, r_isa{undefined};

    u_isa = Oracle<KernelSelection>(elkan_update, tid<T>(), n_clusters, "kmeans.isa");
    r_isa = Oracle<KernelSelection>(elkan_reduce, tid<T>(), n_features, "kmeans.isa");

    update_kernel = elkan_update_implementations.get<T>(u_isa);
    reduce_kernel = elkan_reduction_implementations.get<T>(r_isa);
    padding = get_padding<T>(u_isa);

    // Add telemetry
    context_set_hidden_settings(
        "kmeans.setup"s, "kernel=elkan,kernel.update_kernel.type="s +
                             std::to_string(u_isa) + ",kernel.reduce_kernel.type="s +
                             std::to_string(r_isa) + ",kernel.padding="s +
                             std::to_string(padding));
}

/* Initialize the upper and lower bounds for Elkan's method; stored in works1 and workcs1 */
template <typename T> void kmeans<T>::init_elkan() {

    // Elkan's method works best with A stored as row-major (which is already done) and cluster
    // centres in row-major, so we'll transpose the cluster centres to row-major format, using
    // previous_cluster_centres as temporary storage, just for use in the iterative phase of the algorithm
    da_utils::copy_transpose_2D_array_column_to_row_major(
        n_clusters, n_features, (*current_cluster_centres).data(), n_clusters,
        (*previous_cluster_centres).data(), n_features);
    da_std::fill(current_cluster_centres->begin(), current_cluster_centres->end(),
                 (T)0.0);

    std::swap(current_cluster_centres, previous_cluster_centres);

    compute_centre_half_distances();
    da_int label;
    da_int tmp_int;
    T smallest_dist, dist, tmp;

// For every sample, set upper bound (works1) to be distance to closest centre and update label
// Lower bound (workcs1) will contain distance from each sample to each cluster centre, if computed
#pragma omp parallel for schedule(static) private(label, tmp_int, smallest_dist, dist,   \
                                                      tmp)                               \
    shared(A, lda, current_cluster_centres, n_clusters, workcc1, workcs1, ldworkcs1,     \
               works1, current_labels, data_inv_norms) default(none)
    for (da_int i = 0; i < n_samples; i++) {

        da_int index = i * ldworkcs1;
        label = 0;

        if (do_spherical) {
            // Angular distance: arccos(dot(a,c) / ||a||)  (centres are unit-norm)
            T dot = (T)0.0, cos_val = (T)0.0;
#pragma omp simd reduction(+ : dot)
            for (da_int k = 0; k < n_features; k++) {
                dot += A[i * lda + k] * (*current_cluster_centres)[k];
            }
            if (normalize_data) {
                cos_val = dot * data_inv_norms[i];
            } else {
                cos_val = dot;
            }
            cos_val = std::max((T)-1.0, std::min((T)1.0, cos_val));
            smallest_dist = std::acos(cos_val);
        } else {
            smallest_dist = (T)0.0;
#pragma omp simd reduction(+ : smallest_dist)
            for (da_int k = 0; k < n_features; k++) {
                tmp = A[i * lda + k] - (*current_cluster_centres)[k];
                smallest_dist += tmp * tmp;
            }
            smallest_dist = std::sqrt(smallest_dist);
        }
        workcs1[index] = smallest_dist;

        for (da_int j = 1; j < n_clusters; j++) {
            // Compute distance between the ith sample and the jth centre only if needed
            workcs1[index + j] = (T)0.0;
            tmp_int = label * n_clusters + j;
            if (smallest_dist > workcc1[tmp_int]) {

                if (do_spherical) {
                    T dot = (T)0.0, cos_val = (T)0.0;
#pragma omp simd reduction(+ : dot)
                    for (da_int k = 0; k < n_features; k++) {
                        dot += A[i * lda + k] *
                               (*current_cluster_centres)[j * n_features + k];
                    }
                    if (normalize_data) {
                        cos_val = dot * data_inv_norms[i];
                    } else {
                        cos_val = dot;
                    }
                    cos_val = std::max((T)-1.0, std::min((T)1.0, cos_val));
                    dist = std::acos(cos_val);
                } else {
                    dist = (T)0.0;
#pragma omp simd reduction(+ : dist)
                    for (da_int k = 0; k < n_features; k++) {
                        tmp = A[i * lda + k] -
                              (*current_cluster_centres)[j * n_features + k];
                        dist += tmp * tmp;
                    }
                    dist = std::sqrt(dist);
                }
                workcs1[index + j] = dist;

                if (dist < smallest_dist) {
                    label = j;
                    smallest_dist = dist;
                }
            }
        }

        (*current_labels)[i] = label;
        works1[i] = smallest_dist;
    }
}

/* Perform a single iteration of Elkan's method */
template <typename T>
void kmeans<T>::elkan_iteration(bool update_centres, da_int n_threads) {

    if (update_centres) {
        da_std::fill(cluster_count.begin(), cluster_count.end(), 0);
        da_std::fill(current_cluster_centres->begin(), current_cluster_centres->end(),
                     (T)0.0);

        if (n_threads > 1) {
            for (da_int t = 0; t < n_threads; t++) {
                auto &local_cluster_centres = thd_cluster_centres[t];
                auto &local_work_int = thd_work_int[t];
                da_std::fill(local_cluster_centres.begin(),
                             local_cluster_centres.begin() + n_clusters * n_features,
                             (T)0.0);
                da_std::fill(local_work_int.begin(), local_work_int.begin() + n_clusters,
                             0);
            }
        }
    }

    // At this point workc1 contains distance of each cluster centre to the next nearest
    // The latest labels and centres are in 'previous' so we can update them to current

    da_int block_size = max_block_size;
    da_int block_index;
    if (n_threads > 1) {

        omp_lock_t cluster_count_lock, cluster_centres_lock;
        omp_init_lock(&cluster_count_lock);
        omp_init_lock(&cluster_centres_lock);

#pragma omp parallel shared(                                                             \
        thd_cluster_centres, thd_work_int, n_blocks, block_rem, update_centres, A, lda,  \
            previous_cluster_centres, current_cluster_centres, cluster_count, workc1,    \
            workcc1, ldworkcs1, max_block_size, current_labels, previous_labels, works1, \
            workcs1, cluster_count_lock, cluster_centres_lock, data_inv_norms)           \
    firstprivate(block_size) private(block_index) default(none) num_threads(n_threads)
        {
            da_int this_thread = omp_get_thread_num();
            auto &local_cluster_centres = thd_cluster_centres[this_thread];
            auto &local_work_int = thd_work_int[this_thread];
#pragma omp for schedule(dynamic) nowait
            for (da_int i = 0; i < n_blocks; i++) {
                if (i == n_blocks - 1 && block_rem > 0) {
                    block_index = n_samples - block_rem;
                    block_size = block_rem;
                } else {
                    block_index = i * max_block_size;
                }
                elkan_iteration_assign_block(
                    update_centres, block_size, &A[block_index * lda], lda,
                    (*previous_cluster_centres).data(), &local_cluster_centres[0],
                    &works1[block_index], &workcs1[block_index * ldworkcs1], ldworkcs1,
                    &(*previous_labels)[block_index], &(*current_labels)[block_index],
                    workcc1.data(), workc1.data(), &local_work_int[0],
                    normalize_data ? data_inv_norms.data() + block_index : nullptr);
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
        } // end parallel region
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
            elkan_iteration_assign_block(
                update_centres, block_size, &A[block_index * lda], lda,
                (*previous_cluster_centres).data(), (*current_cluster_centres).data(),
                &works1[block_index], &workcs1[block_index * ldworkcs1], ldworkcs1,
                &(*previous_labels)[block_index], &(*current_labels)[block_index],
                workcc1.data(), workc1.data(), cluster_count.data(),
                normalize_data ? data_inv_norms.data() + block_index : nullptr);
        }
    }

    if (update_centres) {
        T tmp;

        scale_current_cluster_centres();

        // Update upper and lower bounds and compute shift in centres
        if (do_spherical) {
            // For spherical: compute angular shift between old and new unit-norm centres
            // previous_cluster_centres still holds the old centres at this point
            for (da_int i = 0; i < n_clusters; i++) {
                T dot = (T)0.0;
                for (da_int j = 0; j < n_features; j++) {
                    dot += (*previous_cluster_centres)[i * n_features + j] *
                           (*current_cluster_centres)[i * n_features + j];
                }
                T cos_val = std::max((T)-1.0, std::min((T)1.0, dot));
                workc1[i] = std::acos(cos_val);
            }
            // We still need to call compute_centre_shift for the convergence test
            compute_centre_shift();
        } else {
            compute_centre_shift();
            for (da_int i = 0; i < n_clusters; i++) {
                T tmp2 = 0.0;
#pragma omp simd reduction(+ : tmp2)
                for (da_int j = 0; j < n_features; j++) {
                    tmp = (*previous_cluster_centres)[i * n_features + j];
                    tmp2 += tmp * tmp;
                }
                workc1[i] = std::sqrt(tmp2);
            }
        }

        if (n_threads > 1) {
            block_size = max_block_size;
#pragma omp parallel for default(none) schedule(dynamic)                                 \
    shared(n_blocks, n_samples, workcs1, ldworkcs1, works1, workc1, current_labels)      \
    firstprivate(block_size) private(block_index)
            for (da_int i = 0; i < n_blocks; i++) {
                if (i == n_blocks - 1 && block_rem > 0) {
                    block_index = n_samples - block_rem;
                    block_size = block_rem;
                } else {
                    block_index = i * max_block_size;
                }
                elkan_update_kernel(block_size, &workcs1[block_index * ldworkcs1],
                                    ldworkcs1, &works1[block_index], workc1.data(),
                                    &(*current_labels)[block_index], n_clusters);
            }
        } else {
            elkan_update_kernel(n_samples, workcs1.data(), ldworkcs1, works1.data(),
                                workc1.data(), (*current_labels).data(), n_clusters);
        }
    }

    compute_centre_half_distances();
}

/* Within Elkan iteration, assign a block of the labels */
template <typename T>
void kmeans<T>::elkan_iteration_assign_block(
    bool update_centres, da_int block_size, const T *data, da_int lddata,
    T *old_cluster_centres, T *new_cluster_centres, T *u_bounds, T *l_bounds,
    da_int ldl_bounds, da_int *old_labels, da_int *new_labels, T *centre_half_distances,
    T *next_centre_distances, da_int *cluster_counts, const T *block_data_inv_norms) {

    // Recall that for Elkan, data is stored row-major and cluster centres are stored row-major

    da_int l_bounds_index = 0;

    for (da_int i = 0; i < block_size; i++) {

        // New labels remain the same until we change them
        da_int label = old_labels[i];
        T u_bound = u_bounds[i];

        // This will be true if the upper and lower bounds are equal
        bool tight_bounds = false;

        // Use precomputed inverse sample norm for spherical distance calculations
        T data_inv_norm_i =
            (do_spherical && block_data_inv_norms) ? block_data_inv_norms[i] : (T)0.0;

        // Compute distance from sample i to centre c_idx (angular for spherical, Euclidean otherwise)
        auto sample_centre_dist = [&](da_int c_idx) -> T {
            if (do_spherical) {
                T dot = (T)0.0;
                for (da_int k = 0; k < n_features; k++) {
                    dot += data[i * lddata + k] *
                           old_cluster_centres[c_idx * n_features + k];
                }
                T cos_val;
                if (normalize_data) {
                    cos_val = dot * data_inv_norm_i;
                } else {
                    cos_val = dot;
                }
                cos_val = std::max((T)-1.0, std::min((T)1.0, cos_val));
                return std::acos(cos_val);
            } else {
                return std::sqrt(
                    elkan_reduce_kernel(n_features, &data[i * lddata],
                                        &old_cluster_centres[c_idx * n_features]));
            }
        };

        // Only proceed if distance to closest centre exceeds 0.5* distance to next centre
        if (u_bound > next_centre_distances[label]) {

            for (da_int j = 0; j < n_clusters; j++) {
                // Check if this centre is a good candidate for relabelling the sample
                da_int centre_half_distances_index = label * n_clusters + j;
                T l_bound = l_bounds[l_bounds_index + j];
                T centre_half_distance =
                    centre_half_distances[centre_half_distances_index];

                if (j != label && u_bound > l_bound && u_bound > centre_half_distance) {

                    if (tight_bounds == false) {
                        // Get distance from sample point to currently assigned centre
                        u_bound = sample_centre_dist(label);

                        l_bounds[l_bounds_index + label] = u_bound;
                        tight_bounds = true;
                    }

                    // If condition still holds then compute distance to candidate centre and check
                    if (u_bound > l_bound || u_bound > centre_half_distance) {
                        T dist = sample_centre_dist(j);

                        l_bounds[l_bounds_index + j] = dist;
                        if (dist < u_bound) {
                            u_bound = dist;
                            label = j;
                        }
                    }
                }
            }
        }

        u_bounds[i] = u_bound;
        new_labels[i] = label;

        if (update_centres) {
            cluster_counts[label] += 1;
            // Add this sample to the cluster mean
            if (do_spherical && normalize_data) {
                for (da_int j = 0; j < n_features; j++) {
                    new_cluster_centres[label * n_features + j] +=
                        data[i * lddata + j] * data_inv_norm_i;
                }
            } else {
                for (da_int j = 0; j < n_features; j++) {
                    new_cluster_centres[label * n_features + j] += data[i * lddata + j];
                }
            }
        }
        l_bounds_index += ldl_bounds;
    }
}

/* In the Elkan algorithm, compute the half distances between centres in current_cluster_centres and
   the distance to next closest centre. This matrix is symmetric so only the upper triangle is computed
   and stored. */
template <typename T> void kmeans<T>::compute_centre_half_distances() {
    T *dummy = nullptr;

    if (do_spherical) {
        // For spherical k-means, compute angular distances between unit-norm row-major centres
        // First compute the dot product matrix C * C^T using GEMM
        // Centres are row-major (n_clusters x n_features)
        da_blas::cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_clusters,
                            n_clusters, n_features, (T)1.0,
                            (*current_cluster_centres).data(), n_features,
                            (*current_cluster_centres).data(), n_features, (T)0.0,
                            workcc1.data(), n_clusters);
        // Convert dot products to angular distances
        for (da_int j = 0; j < n_clusters; j++) {
            for (da_int i = 0; i < n_clusters; i++) {
                if (i != j) {
                    T cos_val =
                        std::max((T)-1.0, std::min((T)1.0, workcc1[j * n_clusters + i]));
                    workcc1[j * n_clusters + i] = std::acos(cos_val);
                } else {
                    workcc1[j * n_clusters + i] = (T)0.0;
                }
            }
        }
    } else {
        ARCH::euclidean_gemm_distance(row_major, n_clusters, n_clusters, n_features,
                                      (*current_cluster_centres).data(), n_features,
                                      dummy, 0, workcc1.data(), n_clusters, workc1.data(),
                                      2, dummy, 0, false, true);
    }
    // For each centre, compute the half distance to next closest centre and store in workc1
    da_std::fill(workc1.begin(), workc1.begin() + n_clusters,
                 std::numeric_limits<T>::infinity());

    for (da_int j = 0; j < n_clusters; j++) {
        for (da_int i = 0; i < j; i++) {
            T tmp = (T)0.5 * workcc1[j + i * n_clusters];
            // Store half-distance in both triangles so assign_block reads consistent values
            workcc1[j + i * n_clusters] = tmp;
            workcc1[i + j * n_clusters] = tmp;
            if (tmp < workc1[i])
                workc1[i] = tmp;
            if (tmp < workc1[j])
                workc1[j] = tmp;
        }
    }
}

} // namespace da_kmeans

} // namespace ARCH

#endif // KMEANS_ELKAN_HPP
