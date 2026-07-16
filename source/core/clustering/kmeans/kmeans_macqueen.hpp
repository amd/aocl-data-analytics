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

#ifndef KMEANS_MACQUEEN_HPP
#define KMEANS_MACQUEEN_HPP

#include "aoclda.h"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "da_omp.hpp"
#include "da_std.hpp"
#include "kmeans.hpp"
#include "kmeans_types.hpp"
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

using namespace da_kmeans_types;
using namespace std::literals::string_literals;

/* Initialization for MacQueen's method */
template <typename T> void kmeans<T>::init_macqueen() {

    for (da_int j = 0; j < n_clusters; j++) {
        cluster_count[j] = 0; // Initialize to zero for use later
    }

    for (da_int i = 0; i < n_clusters * n_features; i++)
        (*previous_cluster_centres)[i] = (*current_cluster_centres)[i];

    // Compute the squared norms of the initial cluster centres to avoid recomputing them repeatedly in the blocked section; store in workc1
    // For spherical k-means, centres are unit-normalized so norms are 1 — leave workc1 as zeros for GEMM distance trick
    for (da_int i = 0; i < n_clusters; i++) {
        workc1[i] = (T)0.0;
    }

    T tmp;
    if (!do_spherical) {
        for (da_int j = 0; j < n_features; j++) {
            for (da_int i = 0; i < n_clusters; i++) {
                tmp = (*current_cluster_centres)[i + j * n_clusters];
                (*previous_cluster_centres)[i + j * n_clusters] = tmp;
                (*current_cluster_centres)[i + j * n_clusters] = (T)0.0;
                workc1[i] += tmp * tmp;
            }
        }
    } else {
        for (da_int j = 0; j < n_features; j++) {
            for (da_int i = 0; i < n_clusters; i++) {
                tmp = (*current_cluster_centres)[i + j * n_clusters];
                (*previous_cluster_centres)[i + j * n_clusters] = tmp;
                (*current_cluster_centres)[i + j * n_clusters] = (T)0.0;
            }
        }
    }

    // Distance matrix computation needs to be done in blocks due to memory use
    for (da_int i = 0; i < n_blocks; i++) {
        if (i == n_blocks - 1 && block_rem > 0) {
            init_macqueen_block(block_rem, n_samples - block_rem);
        } else {
            init_macqueen_block(max_block_size, i * max_block_size);
        }
    }

    // Finish updating cluster centres - being careful to guard against zero division in empty clusters
    if (do_spherical) {
        // Normalize to unit length
        da_utils::normalize_rows_inplace(column_major, n_clusters, n_features,
                                         (*current_cluster_centres).data(), n_clusters,
                                         workc1.data());
    } else {
        for (da_int j = 0; j < n_features; j++) {
            for (da_int i = 0; i < n_clusters; i++) {
                if (cluster_count[i] > 0)
                    (*current_cluster_centres)[i + j * n_clusters] /= cluster_count[i];
            }
        }
    }

    // Re-zero previous clusters, which were used temporarily here
    for (da_int i = 0; i < n_clusters * n_features; i++)
        (*previous_cluster_centres)[i] = 0;
}

/* Chunked part of MacQueen's method initialization */
template <typename T>
void kmeans<T>::init_macqueen_block(da_int block_size, da_int block_index) {

    // Compute the matrix D where D_{ij} = ||C_j||^2 - 2 A C^T
    // Don't form it explicitly though: just form -2AC^T and add the ||C_j||^2 as and when we need them
    // Array access patterns mean for this loop it is quicker to form -2CA^T
    // For spherical: use -CA^T (workc1 is zeros, centres are unit-norm)

    T tmp_dist;
    T gemm_scalar = do_spherical ? (T)-1.0 : (T)-2.0;

    // If A is row-major we use a trick which treats it as column-major storage of A^T
    da_int A_index = (this->A_order == column_major) ? block_index : block_index * lda;
    auto A_blas_trans = (this->A_order == column_major) ? CblasTrans : CblasNoTrans;

    da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, A_blas_trans, n_clusters, block_size,
                        n_features, gemm_scalar, (*previous_cluster_centres).data(),
                        n_clusters, &A[A_index], lda, (T)0.0, workcs1.data(), ldworkcs1);

    for (da_int i = block_index; i < block_index + block_size; i++) {
        T smallest_dist = workcs1[i - block_index] + workc1[0];
        da_int index = (i - block_index) * ldworkcs1;
        da_int label = 0;
        for (da_int j = 1; j < n_clusters; j++) {
            tmp_dist = workcs1[index + j] + workc1[j];
            if (tmp_dist < smallest_dist) {
                label = j;
                smallest_dist = tmp_dist;
            }
        }
        (*current_labels)[i] = label;
        // Also want to be counting number of points in each initial cluster
        cluster_count[label] += 1;

        // Update clusters now that we have assigned points to them
        T inv_norm_i = (do_spherical && normalize_data) ? data_inv_norms[i] : (T)1.0;
        if (this->A_order == column_major) {
            for (da_int j = 0; j < n_features; j++) {
                (*current_cluster_centres)[label + j * n_clusters] +=
                    A[i + j * lda] * inv_norm_i;
            }
        } else {
            // A is row-major but cluster centres are still column-major
            for (da_int j = 0; j < n_features; j++) {
                (*current_cluster_centres)[label + j * n_clusters] +=
                    A[i * lda + j] * inv_norm_i;
            }
        }
    }
}

/* Perform single iteration of MacQueen's method */
template <typename T>
void kmeans<T>::macqueen_iteration(bool update_centres,
                                   [[maybe_unused]] da_int n_threads) {

    // Copy data from previous iteration since it's updated in place; no way round this since we need previous iteration for convergence test
    for (da_int i = 0; i < n_clusters * n_features; i++)
        (*current_cluster_centres)[i] = (*previous_cluster_centres)[i];

    for (da_int i = 0; i < n_samples; i++)
        (*current_labels)[i] = (*previous_labels)[i];

    for (da_int i = 0; i < n_samples; i++) {
        // For sample point i, compute the cluster centre distances in workc2

        T *dummy = nullptr;
        T tmp;
        da_int A_index = (this->A_order == column_major) ? i : i * lda;
        da_int A_stride = (this->A_order == column_major) ? lda : 1;

        if (do_spherical) {
            // For spherical: compute cosine distance (1 - cos_sim) to each centre
            // Centres are column-major, unit-normalized
            for (da_int j = 0; j < n_clusters; j++) {
                T dot = (T)0.0, cos_sim = (T)0.0;
                for (da_int k = 0; k < n_features; k++) {
                    dot += A[A_index + k * A_stride] *
                           (*current_cluster_centres)[j + k * n_clusters];
                }
                if (normalize_data) {
                    cos_sim = dot * data_inv_norms[i];
                } else {
                    cos_sim = dot;
                }
                workc2[j] = (T)1.0 - cos_sim;
            }
        } else {
            ARCH::euclidean_gemm_distance(
                column_major, 1, n_clusters, n_features, &A[A_index], A_stride,
                (*current_cluster_centres).data(), n_clusters, workc2.data(), 1, dummy, 0,
                workc1.data(), 1, true, false);
        }

        T smallest_dist = da_std::isfinite(workc2[0])
                              ? workc2[0]
                              : da_std::numeric_limits<T>::infinity();
        da_int closest_centre = 0;
        for (da_int j = 1; j < n_clusters; j++) {
            if (da_std::isfinite(workc2[j]) && workc2[j] < smallest_dist) {
                smallest_dist = workc2[j];
                closest_centre = j;
            }
        }

        if ((*current_labels)[i] != closest_centre) {
            da_int old_centre = (*current_labels)[i];
            (*current_labels)[i] = closest_centre;

            if (update_centres) {
                // Now need to update the two affected centres: closest_centre and old_centre
                cluster_count[closest_centre] += 1;
                cluster_count[old_centre] -= 1;
                workc1[old_centre] = (T)0.0;
                workc1[closest_centre] = (T)0.0;

                // Clear closest_centre and old_centre cluster centres ahead of recomputation
                for (da_int j = 0; j < n_features; j++) {
                    (*current_cluster_centres)[old_centre + j * n_clusters] = (T)0.0;
                    (*current_cluster_centres)[closest_centre + j * n_clusters] = (T)0.0;
                }
                if (this->A_order == column_major) {
                    for (da_int k = 0; k < n_samples; k++) {
                        if ((*current_labels)[k] == closest_centre) {
                            T scale = (do_spherical && normalize_data) ? data_inv_norms[k]
                                                                       : (T)1.0;
                            for (da_int j = 0; j < n_features; j++) {
                                (*current_cluster_centres)[closest_centre +
                                                           j * n_clusters] +=
                                    A[k + j * lda] * scale;
                            }
                        } else if ((*current_labels)[k] == old_centre) {
                            T scale = (do_spherical && normalize_data) ? data_inv_norms[k]
                                                                       : (T)1.0;
                            for (da_int j = 0; j < n_features; j++) {
                                (*current_cluster_centres)[old_centre + j * n_clusters] +=
                                    A[k + j * lda] * scale;
                            }
                        }
                    }
                } else {
                    for (da_int k = 0; k < n_samples; k++) {
                        if ((*current_labels)[k] == closest_centre) {
                            T scale = (do_spherical && normalize_data) ? data_inv_norms[k]
                                                                       : (T)1.0;
                            for (da_int j = 0; j < n_features; j++) {
                                (*current_cluster_centres)[closest_centre +
                                                           j * n_clusters] +=
                                    A[k * lda + j] * scale;
                            }
                        } else if ((*current_labels)[k] == old_centre) {
                            T scale = (do_spherical && normalize_data) ? data_inv_norms[k]
                                                                       : (T)1.0;
                            for (da_int j = 0; j < n_features; j++) {
                                (*current_cluster_centres)[old_centre + j * n_clusters] +=
                                    A[k * lda + j] * scale;
                            }
                        }
                    }
                }

                // Scale to get proper mean and update the squared centre norms
                if (do_spherical) {
                    // Normalize old_centre
                    T norm_sq = (T)0.0;
                    for (da_int j = 0; j < n_features; j++) {
                        tmp = (*current_cluster_centres)[old_centre + j * n_clusters];
                        norm_sq += tmp * tmp;
                    }
                    if (norm_sq > (T)0.0) {
                        T inv_norm = (T)1.0 / da_std::sqrt(norm_sq);
                        for (da_int j = 0; j < n_features; j++)
                            (*current_cluster_centres)[old_centre + j * n_clusters] *=
                                inv_norm;
                    }
                    // Normalize closest_centre
                    norm_sq = (T)0.0;
                    for (da_int j = 0; j < n_features; j++) {
                        tmp = (*current_cluster_centres)[closest_centre + j * n_clusters];
                        norm_sq += tmp * tmp;
                    }
                    if (norm_sq > (T)0.0) {
                        T inv_norm = (T)1.0 / da_std::sqrt(norm_sq);
                        for (da_int j = 0; j < n_features; j++)
                            (*current_cluster_centres)[closest_centre + j * n_clusters] *=
                                inv_norm;
                    }
                    // workc1 stays at zero for spherical (not used for distance computation)
                } else {
                    for (da_int j = 0; j < n_features; j++) {
                        if (cluster_count[old_centre] > 0) {
                            (*current_cluster_centres)[old_centre + j * n_clusters] /=
                                cluster_count[old_centre];
                            tmp = (*current_cluster_centres)[old_centre + j * n_clusters];
                            workc1[old_centre] += tmp * tmp;
                        }
                        if (cluster_count[closest_centre] > 0) {
                            (*current_cluster_centres)[closest_centre + j * n_clusters] /=
                                cluster_count[closest_centre];
                            tmp = (*current_cluster_centres)[closest_centre +
                                                             j * n_clusters];
                            workc1[closest_centre] += tmp * tmp;
                        }
                    }
                }
            }
        }
    }

    if (update_centres) {
        // Compute change in centres in this iteration
        compute_centre_shift();
    }
}

} // namespace da_kmeans

} // namespace ARCH

#endif // KMEANS_MACQUEEN_HPP