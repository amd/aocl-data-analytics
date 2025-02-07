/* ************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc.
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

#ifndef KMEANS_LOOP_UNROLLS_HPP
#define KMEANS_LOOP_UNROLLS_HPP

#include "da_cblas.hh"
#include "macros.h"

namespace ARCH {

namespace da_kmeans {

/* These functions contain performance-critical loops which must vectorize for performance, but often this
can only be done by manually unrolling. The amount of unrolling and even the ordering of array elements
depends on the machine and the number of clusters. */

/* Within Elkan iteration update a block of the lower and upper bound matrices*/
template <typename T>
void kmeans<T>::elkan_iteration_update_block_no_unroll(da_int block_size, T *l_bound,
                                                       da_int ldl_bound, T *u_bound,
                                                       T *centre_shift, da_int *labels) {

    da_int index = 0;
    for (da_int i = 0; i < block_size; i++) {
        u_bound[i] += centre_shift[labels[i]];
#pragma omp simd
        for (da_int j = 0; j < n_clusters; j++) {
            l_bound[index + j] -= centre_shift[j];
            if (l_bound[index + j] < 0) {
                l_bound[index + j] = (T)0.0;
            }
        }
        index += ldl_bound;
    }
}

// LCOV_EXCL_START

template <typename T>
void kmeans<T>::elkan_iteration_update_block_unroll_4(da_int block_size, T *l_bound,
                                                      da_int ldl_bound, T *u_bound,
                                                      T *centre_shift, da_int *labels) {

#pragma omp simd
    for (da_int i = 0; i < block_size; i++) {
        da_int col_index = i * ldl_bound;
#pragma omp simd
        for (da_int j = 0; j < n_clusters; j += 4) {
            da_int index1 = col_index + j;
            da_int index2 = col_index + j + 1;
            da_int index3 = col_index + j + 2;
            da_int index4 = col_index + j + 3;
            l_bound[index1] -= centre_shift[j];
            l_bound[index2] -= centre_shift[j + 1];
            l_bound[index3] -= centre_shift[j + 2];
            l_bound[index4] -= centre_shift[j + 3];
            if (l_bound[index1] < 0) {
                l_bound[index1] = (T)0.0;
            }
            if (l_bound[index2] < 0) {
                l_bound[index2] = (T)0.0;
            }
            if (l_bound[index3] < 0) {
                l_bound[index3] = (T)0.0;
            }
            if (l_bound[index4] < 0) {
                l_bound[index4] = (T)0.0;
            }
        }
    }

    for (da_int i = 0; i < block_size; i++) {
        u_bound[i] += centre_shift[labels[i]];
    }
}

template <typename T>
void kmeans<T>::elkan_iteration_update_block_unroll_8(da_int block_size, T *l_bound,
                                                      da_int ldl_bound, T *u_bound,
                                                      T *centre_shift, da_int *labels) {

#pragma omp simd
    for (da_int i = 0; i < block_size; i++) {
        da_int col_index = i * ldl_bound;
#pragma omp simd
        for (da_int j = 0; j < n_clusters; j += 8) {
            da_int index1 = col_index + j;
            da_int index2 = col_index + j + 1;
            da_int index3 = col_index + j + 2;
            da_int index4 = col_index + j + 3;
            da_int index5 = col_index + j + 4;
            da_int index6 = col_index + j + 5;
            da_int index7 = col_index + j + 6;
            da_int index8 = col_index + j + 7;
            l_bound[index1] -= centre_shift[j];
            l_bound[index2] -= centre_shift[j + 1];
            l_bound[index3] -= centre_shift[j + 2];
            l_bound[index4] -= centre_shift[j + 3];
            l_bound[index5] -= centre_shift[j + 4];
            l_bound[index6] -= centre_shift[j + 5];
            l_bound[index7] -= centre_shift[j + 6];
            l_bound[index8] -= centre_shift[j + 7];
            if (l_bound[index1] < 0) {
                l_bound[index1] = (T)0.0;
            }
            if (l_bound[index2] < 0) {
                l_bound[index2] = (T)0.0;
            }
            if (l_bound[index3] < 0) {
                l_bound[index3] = (T)0.0;
            }
            if (l_bound[index4] < 0) {
                l_bound[index4] = (T)0.0;
            }
            if (l_bound[index5] < 0) {
                l_bound[index5] = (T)0.0;
            }
            if (l_bound[index6] < 0) {
                l_bound[index6] = (T)0.0;
            }
            if (l_bound[index7] < 0) {
                l_bound[index7] = (T)0.0;
            }
            if (l_bound[index8] < 0) {
                l_bound[index8] = (T)0.0;
            }
        }
    }

    for (da_int i = 0; i < block_size; i++) {
        u_bound[i] += centre_shift[labels[i]];
    }
}

// LCOV_EXCL_START

template <typename T>
void kmeans<T>::lloyd_iteration_block_no_unroll(bool update_centres, da_int block_size,
                                                const T *data, da_int lddata,
                                                T *cluster_centres,
                                                T *new_cluster_centres, T *centre_norms,
                                                da_int *cluster_count, da_int *labels,
                                                T *work, da_int ldwork) {

    // Compute the matrix D where D_{ij} = ||C_j||^2 - 2 A C^T
    // Don't form it explicitly though: just form -2AC^T and add the ||C_j||^2 as and when we need them
    // Array access patterns mean for this loop it is quicker to form -2CA^T
    T tmp2;
    da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, n_clusters, block_size,
                        n_features, -2.0, cluster_centres, n_clusters, data, lddata, 0.0,
                        work, ldwork);
    // Go through each sample in work and find argmin

    tmp2 = centre_norms[0];

#pragma omp simd
    for (da_int i = 0; i < block_size; i++) {
        da_int ind = i * ldwork;
        T smallest_dist = work[ind] + tmp2;
        da_int label = 0;
        for (da_int j = 1; j < n_clusters; j++) {
            da_int index = ind + j;
            T tmp = work[index] + centre_norms[j];
            if (tmp < smallest_dist) {
                label = j;
                smallest_dist = tmp;
            }
        }
        labels[i] = label;
    }

    if (update_centres) {

        for (da_int i = 0; i < block_size; i++) {
            da_int label = labels[i];
            cluster_count[label] += 1;
            // Add this sample to the cluster mean
            for (da_int j = 0; j < n_features; j++) {
                new_cluster_centres[label + j * n_clusters] += data[i + j * lddata];
            }
        }
    }
}

// Exclude unrolled loops from coverage as these are used in the benchmark tests
// LCOV_EXCL_START

template <typename T>
void kmeans<T>::lloyd_iteration_block_unroll_2(bool update_centres, da_int block_size,
                                               const T *data, da_int lddata,
                                               T *cluster_centres, T *new_cluster_centres,
                                               T *centre_norms, da_int *cluster_count,
                                               da_int *labels, T *work, da_int ldwork) {

    // Compute the matrix D where D_{ij} = ||C_j||^2 - 2 A C^T
    // Don't form it explicitly though: just form -2AC^T and add the ||C_j||^2 as and when we need them
    // Array access patterns mean for this loop it is quicker to form -2CA^T

    da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, n_clusters, block_size,
                        n_features, -2.0, cluster_centres, n_clusters, data, lddata, 0.0,
                        work, ldwork);
    // Go through each sample in work and find argmin

    T smallest_dists[2];
    da_int tmp_labels[2];

#pragma omp simd
    for (da_int i = 0; i < block_size; i++) {
        da_int ind = i * ldwork;
        smallest_dists[0] = work[ind] + centre_norms[0];
        smallest_dists[1] = work[ind + 1] + centre_norms[1];
        tmp_labels[0] = 0;
        tmp_labels[1] = 1;
        for (da_int j = 2; j < n_clusters; j += 2) {
            da_int index1 = ind + j;
            da_int index2 = ind + j + 1;
            T tmp1 = work[index1] + centre_norms[j];
            T tmp2 = work[index2] + centre_norms[j + 1];
            if (tmp1 < smallest_dists[0]) {
                tmp_labels[0] = j;
                smallest_dists[0] = tmp1;
            }
            if (tmp2 < smallest_dists[1]) {
                tmp_labels[1] = j + 1;
                smallest_dists[1] = tmp2;
            }
        }

        da_int label = tmp_labels[0];
        if (smallest_dists[1] < smallest_dists[0]) {
            label = tmp_labels[1];
        }

        labels[i] = label;
    }

    if (update_centres) {

        for (da_int i = 0; i < block_size; i++) {
            da_int label = labels[i];
            cluster_count[label] += 1;
            // Add this sample to the cluster mean
            for (da_int j = 0; j < n_features; j++) {
                new_cluster_centres[label + j * n_clusters] += data[i + j * lddata];
            }
        }
    }
}

template <typename T>
void kmeans<T>::lloyd_iteration_block_unroll_4(bool update_centres, da_int block_size,
                                               const T *data, da_int lddata,
                                               T *cluster_centres, T *new_cluster_centres,
                                               T *centre_norms, da_int *cluster_count,
                                               da_int *labels, T *work, da_int ldwork) {

    // Compute the matrix D where D_{ij} = ||C_j||^2 - 2 A C^T
    // Don't form it explicitly though: just form -2AC^T and add the ||C_j||^2 as and when we need them
    // Array access patterns mean for this loop it is quicker to form -2CA^T

    da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, n_clusters, block_size,
                        n_features, -2.0, cluster_centres, n_clusters, data, lddata, 0.0,
                        work, ldwork);
    // Go through each sample in works and find argmin

    T smallest_dists[4];
    da_int tmp_labels[4];

#pragma omp simd
    for (da_int i = 0; i < block_size; i++) {
        da_int ind = i * ldwork;
        smallest_dists[0] = work[ind] + centre_norms[0];
        smallest_dists[1] = work[ind + 1] + centre_norms[1];
        smallest_dists[2] = work[ind + 2] + centre_norms[2];
        smallest_dists[3] = work[ind + 3] + centre_norms[3];
        tmp_labels[0] = 0;
        tmp_labels[1] = 1;
        tmp_labels[2] = 2;
        tmp_labels[3] = 3;
        for (da_int j = 4; j < n_clusters; j += 4) {
            da_int index1 = ind + j;
            da_int index2 = ind + j + 1;
            da_int index3 = ind + j + 2;
            da_int index4 = ind + j + 3;
            T tmp1 = work[index1] + centre_norms[j];
            T tmp2 = work[index2] + centre_norms[j + 1];
            T tmp3 = work[index3] + centre_norms[j + 2];
            T tmp4 = work[index4] + centre_norms[j + 3];
            if (tmp1 < smallest_dists[0]) {
                tmp_labels[0] = j;
                smallest_dists[0] = tmp1;
            }
            if (tmp2 < smallest_dists[1]) {
                tmp_labels[1] = j + 1;
                smallest_dists[1] = tmp2;
            }
            if (tmp3 < smallest_dists[2]) {
                tmp_labels[2] = j + 2;
                smallest_dists[2] = tmp3;
            }
            if (tmp4 < smallest_dists[3]) {
                tmp_labels[3] = j + 3;
                smallest_dists[3] = tmp4;
            }
        }

        da_int label = tmp_labels[0];
        for (da_int j = 1; j < 4; j++) {
            if (smallest_dists[j] < smallest_dists[0]) {
                smallest_dists[0] = smallest_dists[j];
                label = tmp_labels[j];
            }
        }
        labels[i] = label;
    }

    if (update_centres) {

        for (da_int i = 0; i < block_size; i++) {
            da_int label = labels[i];
            cluster_count[label] += 1;
            // Add this sample to the cluster mean
            for (da_int j = 0; j < n_features; j++) {
                new_cluster_centres[label + j * n_clusters] += data[i + j * lddata];
            }
        }
    }
}

template <typename T>
void kmeans<T>::lloyd_iteration_block_unroll_4_T(bool update_centres, da_int block_size,
                                                 const T *data, da_int lddata,
                                                 T *cluster_centres,
                                                 T *new_cluster_centres, T *centre_norms,
                                                 da_int *cluster_count, da_int *labels,
                                                 T *work, da_int ldwork) {

    // Compute the matrix D where D_{ij} = ||C_j||^2 - 2 A C^T
    // Don't form it explicitly though: just form -2AC^T and add the ||C_j||^2 as and when we need them

    da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, block_size, n_clusters,
                        n_features, -2.0, data, lddata, cluster_centres, n_clusters, 0.0,
                        work, ldwork);

    // Go through each sample in work and find argmin

    T smallest_dists[4];
    da_int tmp_labels[4];
    da_int ldx2 = ldwork * 2;
    da_int ldx3 = ldwork * 3;

#pragma omp simd
    for (da_int i = 0; i < block_size; i++) {
        smallest_dists[0] = work[i] + centre_norms[0];
        smallest_dists[1] = work[i + ldwork] + centre_norms[1];
        smallest_dists[2] = work[i + ldx2] + centre_norms[2];
        smallest_dists[3] = work[i + ldx3] + centre_norms[3];
        tmp_labels[0] = 0;
        tmp_labels[1] = 1;
        tmp_labels[2] = 2;
        tmp_labels[3] = 3;
        for (da_int j = 4; j < n_clusters; j += 4) {
            da_int index1 = i + ldwork * j;
            da_int index2 = i + ldwork * (j + 1);
            da_int index3 = i + ldwork * (j + 2);
            da_int index4 = i + ldwork * (j + 3);
            T tmp1 = work[index1] + centre_norms[j];
            T tmp2 = work[index2] + centre_norms[j + 1];
            T tmp3 = work[index3] + centre_norms[j + 2];
            T tmp4 = work[index4] + centre_norms[j + 3];
            if (tmp1 < smallest_dists[0]) {
                tmp_labels[0] = j;
                smallest_dists[0] = tmp1;
            }
            if (tmp2 < smallest_dists[1]) {
                tmp_labels[1] = j + 1;
                smallest_dists[1] = tmp2;
            }
            if (tmp3 < smallest_dists[2]) {
                tmp_labels[2] = j + 2;
                smallest_dists[2] = tmp3;
            }
            if (tmp4 < smallest_dists[3]) {
                tmp_labels[3] = j + 3;
                smallest_dists[3] = tmp4;
            }
        }

        da_int label = tmp_labels[0];
        for (da_int j = 1; j < 4; j++) {
            if (smallest_dists[j] < smallest_dists[0]) {
                smallest_dists[0] = smallest_dists[j];
                label = tmp_labels[j];
            }
        }
        labels[i] = label;
    }

    if (update_centres) {

        for (da_int i = 0; i < block_size; i++) {
            da_int label = labels[i];
            cluster_count[label] += 1;
            // Add this sample to the cluster mean
            for (da_int j = 0; j < n_features; j++) {
                new_cluster_centres[label + j * n_clusters] += data[i + j * lddata];
            }
        }
    }
}

template <typename T>
void kmeans<T>::lloyd_iteration_block_unroll_8(bool update_centres, da_int block_size,
                                               const T *data, da_int lddata,
                                               T *cluster_centres, T *new_cluster_centres,
                                               T *centre_norms, da_int *cluster_count,
                                               da_int *labels, T *work, da_int ldwork) {

    // Compute the matrix D where D_{ij} = ||C_j||^2 - 2 A C^T
    // Don't form it explicitly though: just form -2AC^T and add the ||C_j||^2 as and when we need them
    // Array access patterns mean for this loop it is quicker to form -2CA^T

    da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, n_clusters, block_size,
                        n_features, -2.0, cluster_centres, n_clusters, data, lddata, 0.0,
                        work, ldwork);
    // Go through each sample in work and find argmin

    T smallest_dists[8];
    da_int tmp_labels[8];
#pragma omp simd
    for (da_int i = 0; i < block_size; i++) {
        da_int ind = i * ldwork;
        smallest_dists[0] = work[ind] + centre_norms[0];
        smallest_dists[1] = work[ind + 1] + centre_norms[1];
        smallest_dists[2] = work[ind + 2] + centre_norms[2];
        smallest_dists[3] = work[ind + 3] + centre_norms[3];
        smallest_dists[4] = work[ind + 4] + centre_norms[4];
        smallest_dists[5] = work[ind + 5] + centre_norms[5];
        smallest_dists[6] = work[ind + 6] + centre_norms[6];
        smallest_dists[7] = work[ind + 7] + centre_norms[7];
        tmp_labels[0] = 0;
        tmp_labels[1] = 1;
        tmp_labels[2] = 2;
        tmp_labels[3] = 3;
        tmp_labels[4] = 4;
        tmp_labels[5] = 5;
        tmp_labels[6] = 6;
        tmp_labels[7] = 7;
        for (da_int j = 8; j < n_clusters; j += 8) {
            da_int index1 = ind + j;
            da_int index2 = ind + j + 1;
            da_int index3 = ind + j + 2;
            da_int index4 = ind + j + 3;
            da_int index5 = ind + j + 4;
            da_int index6 = ind + j + 5;
            da_int index7 = ind + j + 6;
            da_int index8 = ind + j + 7;
            T tmp1 = work[index1] + centre_norms[j];
            T tmp2 = work[index2] + centre_norms[j + 1];
            T tmp3 = work[index3] + centre_norms[j + 2];
            T tmp4 = work[index4] + centre_norms[j + 3];
            T tmp5 = work[index5] + centre_norms[j + 4];
            T tmp6 = work[index6] + centre_norms[j + 5];
            T tmp7 = work[index7] + centre_norms[j + 6];
            T tmp8 = work[index8] + centre_norms[j + 7];
            if (tmp1 < smallest_dists[0]) {
                tmp_labels[0] = j;
                smallest_dists[0] = tmp1;
            }
            if (tmp2 < smallest_dists[1]) {
                tmp_labels[1] = j + 1;
                smallest_dists[1] = tmp2;
            }
            if (tmp3 < smallest_dists[2]) {
                tmp_labels[2] = j + 2;
                smallest_dists[2] = tmp3;
            }
            if (tmp4 < smallest_dists[3]) {
                tmp_labels[3] = j + 3;
                smallest_dists[3] = tmp4;
            }
            if (tmp5 < smallest_dists[4]) {
                tmp_labels[4] = j + 4;
                smallest_dists[4] = tmp5;
            }
            if (tmp6 < smallest_dists[5]) {
                tmp_labels[5] = j + 5;
                smallest_dists[5] = tmp6;
            }
            if (tmp7 < smallest_dists[6]) {
                tmp_labels[6] = j + 6;
                smallest_dists[6] = tmp7;
            }
            if (tmp8 < smallest_dists[7]) {
                tmp_labels[7] = j + 7;
                smallest_dists[7] = tmp8;
            }
        }

        da_int label = tmp_labels[0];
        for (da_int j = 1; j < 8; j++) {
            if (smallest_dists[j] < smallest_dists[0]) {
                smallest_dists[0] = smallest_dists[j];
                label = tmp_labels[j];
            }
        }
        labels[i] = label;
    }

    if (update_centres) {

        for (da_int i = 0; i < block_size; i++) {
            da_int label = labels[i];
            cluster_count[label] += 1;
            // Add this sample to the cluster mean
            for (da_int j = 0; j < n_features; j++) {
                new_cluster_centres[label + j * n_clusters] += data[i + j * lddata];
            }
        }
    }
}

// LCOV_EXCL_STOP

} // namespace da_kmeans

} // namespace ARCH

#endif // KMEANS_LOOP_UNROLLS_HPP