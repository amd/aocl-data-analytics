/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc.
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

#ifndef KMEANS_LLOYD_ITERATION_UNROLLS_HPP
#define KMEANS_LLOYD_ITERATION_UNROLLS_HPP

#include "da_cblas.hh"
//#include <omp.h>

namespace da_kmeans {

/* These functions update a chunk of the labels within a lloyd iteration. There is a critical loop
which must vectorize for performance, but this can only be done by manually unrolling. The amount of
unrolling depends on the machine and the number of clusters */

template <typename T>
void da_kmeans<T>::lloyd_iteration_chunk_no_unroll(bool update_centres, da_int chunk_size,
                                                   da_int chunk_index, da_int *labels) {

    // Compute the matrix D where D_{ij} = ||C_j||^2 - 2 A C^T
    // Don't form it explicitly though: just form -2AC^T and add the ||C_j||^2 as and when we need them

    T tmp2;
    //double t0 = omp_get_wtime();
    da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, chunk_size, n_clusters,
                        n_features, -2.0, &A[chunk_index], lda,
                        (*previous_cluster_centres).data(), n_clusters, 0.0,
                        worksc1.data(), ldworksc1);
    //double t1 = omp_get_wtime();
    //t_euclidean_it += t1 - t0;
    // Go through each sample (row) in worksc1 and find argmin

    tmp2 = workc1[0];

#pragma omp simd
    for (da_int i = 0; i < chunk_size; i++) {
        T smallest_dist = worksc1[i] + tmp2;
        da_int label = 0;
        for (da_int j = 1; j < n_clusters; j++) {
            da_int index = i + ldworksc1 * j;
            T tmp = worksc1[index] + workc1[j];
            if (tmp < smallest_dist) {
                label = j;
                smallest_dist = tmp;
            }
        }
        labels[i] = label;
    }

    //double t2 = omp_get_wtime();
    //t_assign += t2 - t1;
    if (update_centres) {

        for (da_int i = 0; i < chunk_size; i++) {
            da_int label = labels[i];
            work_int1[label] += 1;
            // Add this sample to the cluster mean
            for (da_int j = 0; j < n_features; j++) {
                (*current_cluster_centres)[label + j * n_clusters] +=
                    A[i + chunk_index + j * lda];
            }
        }
    }
    //double t3 = omp_get_wtime();
    //t_update += t3 - t2;
}

// Exclude unrolled loops from coverage as these are time consuming in benchmark tests
// LCOV_EXCL_START

template <typename T>
void da_kmeans<T>::lloyd_iteration_chunk_unroll_4(bool update_centres, da_int chunk_size,
                                                  da_int chunk_index, da_int *labels) {

    // Compute the matrix D where D_{ij} = ||C_j||^2 - 2 A C^T
    // Don't form it explicitly though: just form -2AC^T and add the ||C_j||^2 as and when we need them

    //T tmp2;
    //double t0 = omp_get_wtime();
    da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, chunk_size, n_clusters,
                        n_features, -2.0, &A[chunk_index], lda,
                        (*previous_cluster_centres).data(), n_clusters, 0.0,
                        worksc1.data(), ldworksc1);
    //double t1 = omp_get_wtime();
    //t_euclidean_it += t1 - t0;
    // Go through each sample (row) in worksc1 and find argmin

    double smallest_dists[4];
    da_int tmp_labels[4];
    da_int ldx2 = ldworksc1 * 2;
    da_int ldx3 = ldworksc1 * 3;

#pragma omp simd
    for (int i = 0; i < chunk_size; i++) {
        smallest_dists[0] = worksc1[i] + workc1[0];
        smallest_dists[1] = worksc1[i + ldworksc1] + workc1[1];
        smallest_dists[2] = worksc1[i + ldx2] + workc1[2];
        smallest_dists[3] = worksc1[i + ldx3] + workc1[3];
        tmp_labels[0] = 0;
        tmp_labels[1] = 0;
        tmp_labels[2] = 0;
        tmp_labels[3] = 0;
        for (int j = 4; j < n_clusters; j += 4) {
            int index1 = i + ldworksc1 * j;
            int index2 = i + ldworksc1 * (j + 1);
            int index3 = i + ldworksc1 * (j + 2);
            int index4 = i + ldworksc1 * (j + 3);
            double tmp1 = worksc1[index1] + workc1[j];
            double tmp2 = worksc1[index2] + workc1[j + 1];
            double tmp3 = worksc1[index3] + workc1[j + 2];
            double tmp4 = worksc1[index4] + workc1[j + 3];
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

        int label = tmp_labels[0];
        for (int j = 1; j < 4; j++) {
            if (smallest_dists[j] < smallest_dists[0]) {
                smallest_dists[0] = smallest_dists[j];
                label = tmp_labels[j];
            }
        }
        labels[i] = label;
    }

    //double t2 = omp_get_wtime();
    //t_assign += t2 - t1;
    if (update_centres) {

        for (da_int i = 0; i < chunk_size; i++) {
            da_int label = labels[i];
            work_int1[label] += 1;
            // Add this sample to the cluster mean
            for (da_int j = 0; j < n_features; j++) {
                (*current_cluster_centres)[label + j * n_clusters] +=
                    A[i + chunk_index + j * lda];
            }
        }
    }
    //double t3 = omp_get_wtime();
    //t_update += t3 - t2;
}

template <typename T>
void da_kmeans<T>::lloyd_iteration_chunk_unroll_8(bool update_centres, da_int chunk_size,
                                                  da_int chunk_index, da_int *labels) {

    // Compute the matrix D where D_{ij} = ||C_j||^2 - 2 A C^T
    // Don't form it explicitly though: just form -2AC^T and add the ||C_j||^2 as and when we need them

    //double t0 = omp_get_wtime();
    da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, chunk_size, n_clusters,
                        n_features, -2.0, &A[chunk_index], lda,
                        (*previous_cluster_centres).data(), n_clusters, 0.0,
                        worksc1.data(), ldworksc1);
    //double t1 = omp_get_wtime();
    //t_euclidean_it += t1 - t0;
    // Go through each sample (row) in worksc1 and find argmin

    da_int ldx2 = ldworksc1 * 2;
    da_int ldx3 = ldworksc1 * 3;
    da_int ldx4 = ldworksc1 * 4;
    da_int ldx5 = ldworksc1 * 5;
    da_int ldx6 = ldworksc1 * 6;
    da_int ldx7 = ldworksc1 * 7;
    double smallest_dists[8];
    int tmp_labels[8];
#pragma omp simd
    for (int i = 0; i < chunk_size; i++) {
        smallest_dists[0] = worksc1[i] + workc1[0];
        smallest_dists[1] = worksc1[i + ldworksc1] + workc1[1];
        smallest_dists[2] = worksc1[i + ldx2] + workc1[2];
        smallest_dists[3] = worksc1[i + ldx3] + workc1[3];
        smallest_dists[4] = worksc1[i + ldx4] + workc1[4];
        smallest_dists[5] = worksc1[i + ldx5] + workc1[5];
        smallest_dists[6] = worksc1[i + ldx6] + workc1[6];
        smallest_dists[7] = worksc1[i + ldx7] + workc1[7];
        tmp_labels[0] = 0;
        tmp_labels[1] = 0;
        tmp_labels[2] = 0;
        tmp_labels[3] = 0;
        tmp_labels[4] = 0;
        tmp_labels[5] = 0;
        tmp_labels[6] = 0;
        tmp_labels[7] = 0;
        for (int j = 8; j < n_clusters; j += 8) {
            int index1 = i + ldworksc1 * j;
            int index2 = i + ldworksc1 * (j + 1);
            int index3 = i + ldworksc1 * (j + 2);
            int index4 = i + ldworksc1 * (j + 3);
            int index5 = i + ldworksc1 * (j + 4);
            int index6 = i + ldworksc1 * (j + 5);
            int index7 = i + ldworksc1 * (j + 6);
            int index8 = i + ldworksc1 * (j + 7);
            double tmp1 = worksc1[index1] + workc1[j];
            double tmp2 = worksc1[index2] + workc1[j + 1];
            double tmp3 = worksc1[index3] + workc1[j + 2];
            double tmp4 = worksc1[index4] + workc1[j + 3];
            double tmp5 = worksc1[index5] + workc1[j + 4];
            double tmp6 = worksc1[index6] + workc1[j + 5];
            double tmp7 = worksc1[index7] + workc1[j + 6];
            double tmp8 = worksc1[index8] + workc1[j + 7];
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

        int label = tmp_labels[0];
        for (int j = 1; j < 8; j++) {
            if (smallest_dists[j] < smallest_dists[0]) {
                smallest_dists[0] = smallest_dists[j];
                label = tmp_labels[j];
            }
        }
        labels[i] = label;
    }

    //double t2 = omp_get_wtime();
    //t_assign += t2 - t1;
    if (update_centres) {

        for (da_int i = 0; i < chunk_size; i++) {
            da_int label = labels[i];
            work_int1[label] += 1;
            // Add this sample to the cluster mean
            for (da_int j = 0; j < n_features; j++) {
                (*current_cluster_centres)[label + j * n_clusters] +=
                    A[i + chunk_index + j * lda];
            }
        }
    }
    //double t3 = omp_get_wtime();
    //t_update += t3 - t2;
}

// LCOV_EXCL_STOP

} // namespace da_kmeans

#endif // KMEANS_LLOYD_ITERATION_UNROLLS_HPP