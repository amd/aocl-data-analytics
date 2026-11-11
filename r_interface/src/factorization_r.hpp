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

#ifndef FACTORIZATION_R_HPP
#define FACTORIZATION_R_HPP

#include <Rcpp.h>
#include <vector>

#include "aoclda.h"
#include "internal_utilities_r.hpp"

using namespace Rcpp;

class PCADecomposition : public rda_handle {
  public:
    PCADecomposition(da_int n_components = 1, std::string bias = "unbiased",
                     std::string method = "covariance", std::string solver = "gesdd",
                     bool store_U = false, bool whiten = false,
                     da_int n_oversamples = 10, da_int power_iterations = -1,
                     std::string power_normalization = "qr", da_int seed = -1,
                     bool check_data = false) {

        da_status status = da_handle_init_d(&handle, da_handle_pca);
        exception_check(status);

        status = da_options_set(handle, "n_components", n_components);
        exception_check(status);
        status = da_options_set(handle, "pca method", method.c_str());
        exception_check(status);
        status = da_options_set(handle, "degrees of freedom", bias.c_str());
        exception_check(status);
        status = da_options_set(handle, "svd solver", solver.c_str());
        exception_check(status);
        if (store_U) {
            status = da_options_set(handle, "store u", 1);
            exception_check(status);
        }
        if (whiten) {
            status = da_options_set(handle, "whiten", 1);
            exception_check(status);
        }
        status = da_options_set_int(handle, "n_oversamples", n_oversamples);
        exception_check(status);
        status = da_options_set_int(handle, "power iterations", power_iterations);
        exception_check(status);
        status = da_options_set_string(handle, "power normalization",
                                       power_normalization.c_str());
        exception_check(status);
        status = da_options_set_int(handle, "seed", seed);
        exception_check(status);
        if (check_data) {
            status = da_options_set(handle, "check data", "yes");
            exception_check(status);
        }
    }

    void fit(NumericMatrix X) {
        da_int n_samples = X.nrow();
        da_int n_features = X.ncol();

        da_status status;
        status = da_options_set(handle, "storage order", "column-major");
        exception_check(status);

        status = da_pca_set_data_d(handle, n_samples, n_features, X.begin(), n_samples);
        exception_check(status);
        status = da_pca_compute_d(handle);
        exception_check(status);
    }

    NumericMatrix transform(NumericMatrix X) {
        da_int n_samples = X.nrow();
        da_int n_features = X.ncol();

        da_int dim = 3;
        std::vector<double> result(dim);
        da_status status = da_handle_get_result_d(handle, da_rinfo, &dim, result.data());
        exception_check(status);

        da_int result_components = static_cast<da_int>(result[2]);
        NumericMatrix X_transformed(n_samples, result_components);

        status = da_pca_transform_d(handle, n_samples, n_features, X.begin(), n_samples,
                                    X_transformed.begin(), n_samples);
        exception_check(status);

        return X_transformed;
    }

    NumericMatrix inverse_transform(NumericMatrix X) {
        da_int n_samples = X.nrow();
        da_int n_components = X.ncol();

        da_int dim = 3;
        std::vector<double> result(dim);
        da_status status = da_handle_get_result_d(handle, da_rinfo, &dim, result.data());
        exception_check(status);

        da_int n_features = static_cast<da_int>(result[1]);
        NumericMatrix X_inv_transform(n_samples, n_features);

        status =
            da_pca_inverse_transform_d(handle, n_samples, n_components, X.begin(),
                                       n_samples, X_inv_transform.begin(), n_samples);
        exception_check(status);

        return X_inv_transform;
    }

    void get_rinfo(da_int *n_samples, da_int *n_features, da_int *n_components) {
        da_status status;
        da_int dim = 3;
        std::vector<double> rinfo(dim);

        status = da_handle_get_result_d(handle, da_rinfo, &dim, rinfo.data());
        exception_check(status);

        *n_samples = static_cast<da_int>(rinfo[0]);
        *n_features = static_cast<da_int>(rinfo[1]);
        *n_components = static_cast<da_int>(rinfo[2]);
    }

    NumericMatrix get_result(da_result result) {
        da_status status = da_status_success;
        da_int n_samples, n_features, n_components;
        da_int dim1 = 0, dim2 = 0;

        get_rinfo(&n_samples, &n_features, &n_components);

        switch (result) {
        case da_pca_principal_components:
            dim1 = n_components;
            dim2 = n_features;
            break;
        case da_pca_scores:
            dim1 = n_samples;
            dim2 = n_components;
            break;
        case da_pca_variance:
            dim1 = n_components;
            dim2 = 1;
            break;
        case da_pca_total_variance:
            dim1 = 1;
            dim2 = 1;
            break;
        case da_pca_u:
            dim1 = n_samples;
            dim2 = n_components;
            break;
        case da_pca_sigma:
            dim1 = n_components;
            dim2 = 1;
            break;
        case da_pca_vt:
            dim1 = n_components;
            dim2 = n_features;
            break;
        case da_pca_column_means:
            dim1 = n_features;
            dim2 = 1;
            break;
        case da_pca_column_sdevs:
            dim1 = n_features;
            dim2 = 1;
            break;
        case da_rinfo:
            dim1 = 3;
            dim2 = 1;
            break;
        default:
            status = da_status_invalid_input;
            break;
        }
        exception_check(status, "Unexpected result input");

        da_int dim = dim1 * dim2;

        NumericMatrix result_mat(dim1, dim2);
        status = da_handle_get_result_d(handle, result, &dim, result_mat.begin());
        exception_check(status);
        return result_mat;
    }

    auto get_principal_components() { return get_result(da_pca_principal_components); }
    auto get_scores() { return get_result(da_pca_scores); }
    auto get_variance() {
        NumericMatrix result_mat = get_result(da_pca_variance);
        return NumericVector(result_mat.begin(), result_mat.end());
    }
    auto get_total_variance() {
        NumericMatrix result_mat = get_result(da_pca_total_variance);
        NumericVector total_var(result_mat.begin(), result_mat.end());
        return total_var[0];
    }
    auto get_u() { return get_result(da_pca_u); }
    auto get_sigma() {
        NumericMatrix result_mat = get_result(da_pca_sigma);
        return NumericVector(result_mat.begin(), result_mat.end());
    }
    auto get_vt() { return get_result(da_pca_vt); }
    auto get_column_means() {
        NumericMatrix result_mat = get_result(da_pca_column_means);
        return NumericVector(result_mat.begin(), result_mat.end());
    }
    auto get_column_sdevs() {
        NumericMatrix result_mat = get_result(da_pca_column_sdevs);
        return NumericVector(result_mat.begin(), result_mat.end());
    }

    auto get_n_samples() {
        da_int n_samples, n_features, n_components;
        get_rinfo(&n_samples, &n_features, &n_components);
        return n_samples;
    }

    auto get_n_components() {
        da_int n_samples, n_features, n_components;
        get_rinfo(&n_samples, &n_features, &n_components);
        return n_components;
    }

    auto get_n_features() {
        da_int n_samples, n_features, n_components;
        get_rinfo(&n_samples, &n_features, &n_components);
        return n_features;
    }
};

class KernelPCADecomposition : public rda_handle {
  public:
    KernelPCADecomposition(da_int n_components = 0, std::string kernel = "linear",
                           std::string eigensolver = "syevd", da_int degree = 3,
                           bool fit_inverse_transform = false,
                           bool remove_zero_eig = false, bool copy_X = true,
                           da_int n_oversamples = 10, da_int power_iterations = -1,
                           std::string power_normalization = "qr", da_int seed = -1,
                           bool check_data = false) {

        da_status status = da_handle_init_d(&handle, da_handle_kernel_pca);
        exception_check(status);

        status = da_options_set_int(handle, "n_components", n_components);
        exception_check(status);
        status = da_options_set_string(handle, "kernel", kernel.c_str());
        exception_check(status);
        status = da_options_set_string(handle, "eigensolver", eigensolver.c_str());
        exception_check(status);
        status = da_options_set_int(handle, "degree", degree);
        exception_check(status);
        status = da_options_set_string(handle, "fit inverse transform",
                                       fit_inverse_transform ? "yes" : "no");
        exception_check(status);
        status = da_options_set_string(handle, "remove zero eig",
                                       remove_zero_eig ? "yes" : "no");
        exception_check(status);
        status = da_options_set_string(handle, "copy data", copy_X ? "yes" : "no");
        exception_check(status);
        status = da_options_set_int(handle, "n_oversamples", n_oversamples);
        exception_check(status);
        status = da_options_set_int(handle, "power iterations", power_iterations);
        exception_check(status);
        status = da_options_set_string(handle, "power normalization",
                                       power_normalization.c_str());
        exception_check(status);
        status = da_options_set_int(handle, "seed", seed);
        exception_check(status);
        if (check_data) {
            status = da_options_set(handle, "check data", "yes");
            exception_check(status);
        }
    }

    void fit(NumericMatrix A, double gamma, double coef0, double alpha) {
        da_int n_samples = A.nrow();
        da_int n_features = A.ncol();

        da_status status;
        status = da_options_set(handle, "gamma", gamma);
        exception_check(status);
        status = da_options_set(handle, "coef0", coef0);
        exception_check(status);
        status = da_options_set(handle, "alpha", alpha);
        exception_check(status);
        status = da_options_set(handle, "storage order", "column-major");
        exception_check(status);

        status =
            da_kernel_pca_set_data_d(handle, n_samples, n_features, A.begin(), n_samples);
        exception_check(status);
        status = da_kernel_pca_compute_d(handle);
        exception_check(status);
    }

    NumericMatrix transform(NumericMatrix X) {
        da_int m_samples = X.nrow();
        da_int m_features = X.ncol();
        da_int n_components = get_n_components();

        NumericMatrix X_transform(m_samples, n_components);

        da_status status = da_kernel_pca_transform_d(
            handle, m_samples, m_features, X.begin(), m_samples, X_transform.begin(),
            m_samples);
        exception_check(status);

        return X_transform;
    }

    NumericMatrix inverse_transform(NumericMatrix Y) {
        da_int k_samples = Y.nrow();
        da_int k_components = Y.ncol();

        da_int n_samples, n_features;
        get_rinfo(&n_samples, &n_features);

        NumericMatrix Y_inv_transform(k_samples, n_features);

        da_status status = da_kernel_pca_inverse_transform_d(
            handle, k_samples, k_components, Y.begin(), k_samples,
            Y_inv_transform.begin(), k_samples);
        exception_check(status);

        return Y_inv_transform;
    }

    da_int get_n_components() {
        da_int dim = 1;
        da_int n_components;
        da_status status =
            da_handle_get_result_int(handle, da_kernel_pca_n_components, &dim,
                                     &n_components);
        exception_check(status);
        return n_components;
    }

    da_int get_n_samples() {
        da_int n_samples, n_features;
        get_rinfo(&n_samples, &n_features);
        return n_samples;
    }

    da_int get_n_features() {
        da_int n_samples, n_features;
        get_rinfo(&n_samples, &n_features);
        return n_features;
    }

    NumericVector get_eigenvalues() {
        NumericMatrix result_mat = get_result(da_kernel_pca_eigenvalues);
        return NumericVector(result_mat.begin(), result_mat.end());
    }

    NumericMatrix get_eigenvectors() {
        return get_result(da_kernel_pca_eigenvectors);
    }

    NumericMatrix get_scores() {
        return get_result(da_kernel_pca_scores);
    }

    NumericMatrix get_dual_coef() {
        return get_result(da_kernel_pca_dual_coef);
    }

    NumericMatrix get_X_fit() {
        return get_result(da_kernel_pca_X_fit);
    }

    double get_gamma() {
        NumericMatrix result_mat = get_result(da_kernel_pca_gamma);
        return result_mat[0];
    }

  private:
    void get_rinfo(da_int *n_samples, da_int *n_features) {
        da_int dim = 2;
        double rinfo[2];
        da_status status = da_handle_get_result_d(handle, da_rinfo, &dim, rinfo);
        exception_check(status);
        *n_samples = static_cast<da_int>(rinfo[0]);
        *n_features = static_cast<da_int>(rinfo[1]);
    }

    NumericMatrix get_result(da_result result) {
        da_status status = da_status_success;
        da_int n_samples, n_features;
        da_int n_components = get_n_components();
        da_int dim1 = 0, dim2 = 0;

        get_rinfo(&n_samples, &n_features);

        switch (result) {
        case da_kernel_pca_eigenvalues:
            dim1 = n_components;
            dim2 = 1;
            break;
        case da_kernel_pca_eigenvectors:
            dim1 = n_samples;
            dim2 = n_components;
            break;
        case da_kernel_pca_scores:
            dim1 = n_samples;
            dim2 = n_components;
            break;
        case da_kernel_pca_dual_coef:
            dim1 = n_samples;
            dim2 = n_features;
            break;
        case da_kernel_pca_gamma:
            dim1 = 1;
            dim2 = 1;
            break;
        case da_kernel_pca_X_fit:
            dim1 = n_samples;
            dim2 = n_features;
            break;
        case da_rinfo:
            dim1 = 2;
            dim2 = 1;
            break;
        default:
            status = da_status_invalid_input;
            break;
        }
        exception_check(status, "Unexpected result input");

        da_int dim = dim1 * dim2;
        NumericMatrix result_mat(dim1, dim2);
        status = da_handle_get_result_d(handle, result, &dim, result_mat.begin());
        exception_check(status);
        return result_mat;
    }
};

class NMFDecomposition : public rda_handle {
  public:
    NMFDecomposition(da_int n_components = 0, std::string solver = "mu",
                     std::string beta_loss = "frobenius", da_int max_iter = 200,
                     double tol = 1.0e-4, std::string init = "nndsvda",
                     da_int random_state = -1, da_int check_every = 10,
                     bool check_data = false) {

        da_status status = da_handle_init_d(&handle, da_handle_nmf);
        exception_check(status);

        status = da_options_set_int(handle, "n_components", n_components);
        exception_check(status);
        status = da_options_set_int(handle, "max_iter", max_iter);
        exception_check(status);
        status = da_options_set_int(handle, "random_state", random_state);
        exception_check(status);
        status = da_options_set_int(handle, "check_every", check_every);
        exception_check(status);
        status = da_options_set_string(handle, "solver", solver.c_str());
        exception_check(status);
        status = da_options_set_string(handle, "beta_loss", beta_loss.c_str());
        exception_check(status);
        status = da_options_set_string(handle, "init", init.c_str());
        exception_check(status);
        status = da_options_set(handle, "tol", tol);
        exception_check(status);
        if (check_data) {
            status = da_options_set(handle, "check data", "yes");
            exception_check(status);
        }
    }

    void set_n_components(da_int n_components) {
        da_status status = da_options_set_int(handle, "n_components", n_components);
        exception_check(status);
    }

    void fit(NumericMatrix A) {
        da_int n_samples = A.nrow();
        da_int n_features = A.ncol();

        da_status status = da_options_set(handle, "storage order", "column-major");
        exception_check(status);

        status = da_nmf_init_d(handle, n_samples, n_features, A.begin(), n_samples);
        exception_check(status);
        status = da_nmf_compute_d(handle);
        exception_check(status);
    }

    /* Fit with a user-supplied (W, H) pair for init='custom'. */
    void fit_custom(NumericMatrix A, NumericMatrix W, NumericMatrix H) {
        da_int n_samples = A.nrow();
        da_int n_features = A.ncol();
        da_int n_rows_w = W.nrow();
        da_int k_w = W.ncol();
        da_int k_h = H.nrow();
        da_int n_cols_h = H.ncol();

        if (k_w != k_h)
            Rcpp::stop("W and H have incompatible inner dimensions: W has %d columns "
                       "but H has %d rows.",
                       static_cast<int>(k_w), static_cast<int>(k_h));

        da_status status = da_options_set(handle, "storage order", "column-major");
        exception_check(status);

        status = da_nmf_init_d(handle, n_samples, n_features, A.begin(), n_samples);
        exception_check(status);
        status = da_nmf_set_data_d(handle, n_rows_w, k_w, W.begin(), n_rows_w, n_cols_h,
                                   H.begin(), k_h);
        exception_check(status);
        status = da_nmf_compute_d(handle);
        exception_check(status);
    }

    NumericMatrix transform(NumericMatrix X) {
        da_int m_samples = X.nrow();
        da_int m_features = X.ncol();
        da_int n_components = get_n_components();

        NumericMatrix X_transform(m_samples, n_components);
        da_status status = da_nmf_transform_d(handle, m_samples, m_features, X.begin(),
                                              m_samples, X_transform.begin(), m_samples);
        exception_check(status);

        return X_transform;
    }

    NumericMatrix inverse_transform(NumericMatrix Xt) {
        da_int k_samples = Xt.nrow();
        da_int k_features = Xt.ncol();
        da_int n_features = get_n_features();

        NumericMatrix X(k_samples, n_features);
        da_status status = da_nmf_inverse_transform_d(handle, k_samples, k_features,
                                                      Xt.begin(), k_samples,
                                                      X.begin(), k_samples);
        exception_check(status);

        return X;
    }

    da_int get_n_components() {
        da_int dim = 1;
        da_int n_components = 0;
        da_status status =
            da_handle_get_result_int(handle, da_nmf_n_components, &dim, &n_components);
        exception_check(status);
        return n_components;
    }

    da_int get_n_iter() {
        da_int dim = 1;
        da_int n_iter;
        da_status status = da_handle_get_result_int(handle, da_nmf_n_iter, &dim, &n_iter);
        exception_check(status);
        return n_iter;
    }

    da_int get_n_samples() {
        da_int n_samples, n_features;
        get_rinfo(&n_samples, &n_features);
        return n_samples;
    }

    da_int get_n_features() {
        da_int n_samples, n_features;
        get_rinfo(&n_samples, &n_features);
        return n_features;
    }

    /* H matrix: (n_components, n_features). */
    NumericMatrix get_components() {
        return get_2d_result(da_nmf_h, get_n_components(), get_n_features());
    }

    /* W matrix: (n_samples, n_components). */
    NumericMatrix get_transform() {
        return get_2d_result(da_nmf_w, get_n_samples(), get_n_components());
    }

    double get_reconstruction_error() {
        da_int dim = 1;
        double v = 0.0;
        da_status status =
            da_handle_get_result_d(handle, da_nmf_reconstruction_err, &dim, &v);
        exception_check(status);
        return v;
    }

  private:
    void get_rinfo(da_int *n_samples, da_int *n_features) {
        da_int dim = 2;
        double rinfo[2];
        da_status status = da_handle_get_result_d(handle, da_rinfo, &dim, rinfo);
        exception_check(status);
        *n_samples = static_cast<da_int>(rinfo[0]);
        *n_features = static_cast<da_int>(rinfo[1]);
    }

    /* Retrieve a 2D result (W or H) with caller-supplied dimensions. */
    NumericMatrix get_2d_result(da_result query, da_int dim1, da_int dim2) {
        da_int dim = dim1 * dim2;
        NumericMatrix result_mat(dim1, dim2);
        da_status status =
            da_handle_get_result_d(handle, query, &dim, result_mat.begin());
        exception_check(status);
        return result_mat;
    }
};

#endif // FACTORIZATION_R_HPP
