/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclda_cpp_overloads.hpp"
#include <iostream>

/* da_handle overloaded functions */
template <>
da_status da_handle_init<double>(da_handle *handle, da_handle_type handle_type) {
    return da_handle_init_d(handle, handle_type);
}
template <>
da_status da_handle_init<float>(da_handle *handle, da_handle_type handle_type) {
    return da_handle_init_s(handle, handle_type);
}
template <>
da_status da_linmod_fit_start<double>(da_handle handle, da_int ncoef,
                                      const double *coefs) {
    return da_linmod_fit_start_d(handle, ncoef, coefs);
}
template <>
da_status da_linmod_fit_start<float>(da_handle handle, da_int ncoef, const float *coefs) {
    return da_linmod_fit_start_s(handle, ncoef, coefs);
}
template <> da_status da_linmod_select_model<double>(da_handle handle, linmod_model mod) {
    return da_linmod_select_model_d(handle, mod);
}
template <> da_status da_linmod_select_model<float>(da_handle handle, linmod_model mod) {
    return da_linmod_select_model_s(handle, mod);
}
template <> da_status da_linmod_fit<double>(da_handle handle) {
    return da_linmod_fit_d(handle);
}
template <> da_status da_linmod_fit<float>(da_handle handle) {
    return da_linmod_fit_s(handle);
}
template <> da_status da_pca_compute<double>(da_handle handle) {
    return da_pca_compute_d(handle);
}
template <> da_status da_pca_compute<float>(da_handle handle) {
    return da_pca_compute_s(handle);
}
template <> da_status da_kmeans_compute<double>(da_handle handle) {
    return da_kmeans_compute_d(handle);
}
template <> da_status da_kmeans_compute<float>(da_handle handle) {
    return da_kmeans_compute_s(handle);
}
template <> da_status da_tree_fit<double>(da_handle handle) {
    return da_tree_fit_d(handle);
}
template <> da_status da_tree_fit<float>(da_handle handle) {
    return da_tree_fit_s(handle);
}
template <> da_status da_forest_fit<double>(da_handle handle) {
    return da_forest_fit_d(handle);
}
template <> da_status da_forest_fit<float>(da_handle handle) {
    return da_forest_fit_s(handle);
}
template <>
da_status da_knn_classes<double>(da_handle handle, da_int *n_classes, da_int *classes) {
    return da_knn_classes_d(handle, n_classes, classes);
}
template <>
da_status da_knn_classes<float>(da_handle handle, da_int *n_classes, da_int *classes) {
    return da_knn_classes_s(handle, n_classes, classes);
}
template <> da_status da_dbscan_compute<double>(da_handle handle) {
    return da_dbscan_compute_d(handle);
}
template <> da_status da_dbscan_compute<float>(da_handle handle) {
    return da_dbscan_compute_s(handle);
}

template <> da_status da_svm_select_model<double>(da_handle handle, da_svm_model mod) {
    return da_svm_select_model_d(handle, mod);
}
template <> da_status da_svm_select_model<float>(da_handle handle, da_svm_model mod) {
    return da_svm_select_model_s(handle, mod);
}

template <> da_status da_svm_compute<double>(da_handle handle) {
    return da_svm_compute_d(handle);
}
template <> da_status da_svm_compute<float>(da_handle handle) {
    return da_svm_compute_s(handle);
}