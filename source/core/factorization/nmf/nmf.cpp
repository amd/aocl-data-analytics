/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

#include "nmf.hpp"
#include "aoclda.h"
#include "aoclda.hpp"
#include "aoclda_nmf.h"
#include "boost/random/uniform_real_distribution.hpp"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "da_randsvd.hpp"
#include "da_utils.hpp"
#include "macros.h"
#include "model_persistence.hpp"
#include "nmf/nmf_options.hpp"
#include "nmf/nmf_types.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <new>
#include <string>
#include <vector>

namespace ARCH {

namespace da_nmf {

using namespace da_model_persistence;

template <typename T> nmf<T>::nmf(da_errors::da_error_t &err) : basic_handle<T>(err) {
    // Initialize the options registry
    // Any error is stored err->status[.] and this NEEDS to be checked
    // by the caller.
    register_nmf_options<T>(this->opts, *this->err);
}

template <typename T> nmf<T>::~nmf() {
    // Clean up any row-major to column-major conversion buffer
    if (this->A_temp)
        delete[] (this->A_temp);
    this->A_temp = nullptr;
    this->A = nullptr;
}

template <typename T>
da_status nmf<T>::get_result(da_result query, da_int *dim, da_int *result) {
    /* Common results (e.g. da_trained) come from basic_handle. */
    da_status status = this->get_result_common(query, dim, result);
    if (status != da_status_unknown_query)
        return status;

    if (!this->model_trained)
        return da_warn(this->err, da_status_no_data,
                       "NMF has not yet been computed. Please call da_nmf_compute_s "
                       "or da_nmf_compute_d before extracting results.");

    if (result == nullptr || dim == nullptr)
        return da_warn(this->err, da_status_invalid_input,
                       "Output pointers must not be null.");

    switch (query) {
    case da_result::da_nmf_n_iter:
        if (*dim < 1) {
            *dim = 1;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size 1.");
        }
        // Return actual number of iterations performed, which could be less than max_iter.
        result[0] = this->n_iter;
        return da_status_success;
    case da_result::da_nmf_n_components:
        if (*dim < 1) {
            *dim = 1;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size 1.");
        }
        result[0] = this->n_components;
        return da_status_success;
    default:
        return da_warn(this->err, da_status_unknown_query,
                       "The requested integer result could not be found.");
    }
}

template <typename T>
da_status nmf<T>::get_result(da_result query, da_int *dim, T *result) {
    if (!this->model_trained)
        return da_warn(this->err, da_status_no_data,
                       "NMF has not yet been computed. Please call da_nmf_compute_s "
                       "or da_nmf_compute_d before extracting results.");

    if (result == nullptr || dim == nullptr)
        return da_warn(this->err, da_status_invalid_input,
                       "Output pointers must not be null.");

    const da_int W_size = this->n_samples * this->n_components;
    const da_int H_size = this->n_components * this->n_features;

    switch (query) {
    case da_result::da_rinfo: {
        if (*dim < 2) {
            *dim = 2;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The results array is too small. Please provide an array of "
                           "size at least 2.");
        }
        result[0] = static_cast<T>(this->n_samples);
        result[1] = static_cast<T>(this->n_features);
        return da_status_success;
    }
    case da_result::da_nmf_w:
        if (*dim < W_size) {
            *dim = W_size;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(W_size) + ".");
        }
        /* W is stored column-major (lda = n). Convert back to user's order. */
        this->copy_2D_results_array(this->n_samples, this->n_components, this->W.data(),
                                    this->n_samples, result);
        return da_status_success;
    case da_result::da_nmf_h:
        if (*dim < H_size) {
            *dim = H_size;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(H_size) + ".");
        }
        /* H is stored column-major (lda = n_components). */
        this->copy_2D_results_array(this->n_components, this->n_features, this->H.data(),
                                    this->n_components, result);
        return da_status_success;
    case da_result::da_nmf_reconstruction_err:
        if (*dim < 1) {
            *dim = 1;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size 1.");
        }
        // Return the Frobenius reconstruction error ||A - W H||_F.
        result[0] = this->reconstruction_err;
        return da_status_success;
    default:
        return da_warn(this->err, da_status_unknown_query,
                       "The requested result could not be found.");
    }
}

template <typename T>
da_status nmf<T>::check_nonneg_matrix(const T *mat, da_int lda, da_int n_rows,
                                      da_int n_cols) {
    da_int has_negative = 0;
    if (this->order == row_major) {
        for (da_int i = 0; i < n_rows; ++i) {
            const T *row = mat + i * lda;
#pragma omp simd reduction(| : has_negative)
            for (da_int j = 0; j < n_cols; ++j)
                has_negative |= (row[j] < T{0});
        }
    } else {
        for (da_int j = 0; j < n_cols; ++j) {
            const T *col = mat + j * lda;
#pragma omp simd reduction(| : has_negative)
            for (da_int i = 0; i < n_rows; ++i)
                has_negative |= (col[i] < T{0});
        }
    }
    if (!has_negative)
        return da_status_success;

    return da_error(this->err, da_status_invalid_input,
                    "Input matrix contains a negative value.");
}

/* Store the user's data matrix in preparation for NMF computation */
template <typename T>
da_status nmf<T>::init(da_int n_in, da_int p_in, const T *A_in, da_int lda_in) {
    /* Guard against errors due to multiple calls using the same class instantiation. */
    if (this->A_temp) {
        delete[] (this->A_temp);
        this->A_temp = nullptr;
        this->A = nullptr;
    }

    /* Resolve storage order from common options. */
    std::string opt_order;
    da_int iorder;
    this->opts.get("storage order", opt_order, iorder);
    this->order = da_order(iorder);

    /* Validate dimensions and pointer. */
    da_status status = this->check_2D_array(this->order, n_in, p_in, A_in, lda_in,
                                            "n_samples", "n_features", "A", "lda", 1, 1);
    if (status != da_status_success)
        return status;

    /* Reject negative entries: NMF requires A >= 0. */
    status = check_nonneg_matrix(A_in, lda_in, n_in, p_in);
    if (status != da_status_success)
        return status;

    /* Internally we always work with column-major data.
     * If row-major: convert and store in A_temp (lda = n_in); active pointer is A_temp.
     * If column-major: active pointer is A_in directly (lda = lda_in). */
    if (this->order == row_major) {
        this->A_temp = new T[n_in * p_in];
        da_utils::copy_transpose_2D_array_row_to_column_major<T>(n_in, p_in, A_in, lda_in,
                                                                 this->A_temp, n_in);
        this->A = this->A_temp;
        this->lda = n_in;
    } else {
        this->A = A_in;
        this->lda = lda_in;
    }

    // Store dimensions of A
    this->n_samples = n_in;
    this->n_features = p_in;

    // Set those so that if the same handle is used but init() is called again,
    // the old results are not incorrectly reported as valid.
    this->model_trained = false;
    this->n_iter = 0;
    this->reconstruction_err = T{0};
    this->W.resize(0);
    this->H.resize(0);
    /* Any previously-uploaded custom init is invalidated by new data. */
    this->k_init = 0;
    this->custom_w_set = false;
    this->custom_h_set = false;
    this->custom_init_set = false;

    return da_status_success;
}

/*
 * Optional function used to set the initial W and H factors when init='custom'.
 */
template <typename T>
da_status nmf<T>::nmf_set_data(da_int n_samples, da_int n_components, const T *W_in,
                               da_int ldw, da_int n_features, const T *H_in, da_int ldh) {
    if (this->n_samples <= 0 || this->n_features <= 0 || !this->A)
        return da_error(this->err, da_status_no_data,
                        "Input data has not been set. Call nmf_init before "
                        "nmf_set_data.");
    if (n_samples != this->n_samples)
        return da_error(
            this->err, da_status_invalid_input,
            "n_samples does not match the number of rows in the data matrix.");
    if (n_features != this->n_features)
        return da_error(this->err, da_status_invalid_input,
                        "n_features does not match the number of columns in the data "
                        "matrix.");
    if (n_components <= 0)
        return da_error(this->err, da_status_invalid_input,
                        "Number of components must be >= 1.");
    if (n_components > std::min(this->n_samples, this->n_features))
        return da_error(this->err, da_status_invalid_input,
                        "Number of components must be <= min(n_samples, n_features).");
    const bool has_W = (W_in != nullptr);
    const bool has_H = (H_in != nullptr);
    if (!has_W && !has_H)
        return da_error(this->err, da_status_invalid_input,
                        "At least one of W or H must be provided.");

    /* Check that if this function is called twice to set W and H separately,
     * the number of components matches. */
    if ((this->custom_w_set || this->custom_h_set) && this->k_init > 0 &&
        n_components != this->k_init)
        return da_error(this->err, da_status_invalid_input,
                        "All custom-factor uploads must use the same number of "
                        "components.");

    /* Validate leading dimensions against the active storage order. */
    da_status status = da_status_success;
    if (has_W) {
        status = this->check_2D_array(this->order, this->n_samples, n_components, W_in,
                                      ldw, "n_samples", "n_components", "W", "ldw", 1, 1);
        if (status != da_status_success)
            return status;
    }
    if (has_H) {
        status =
            this->check_2D_array(this->order, n_components, this->n_features, H_in, ldh,
                                 "n_components", "n_features", "H", "ldh", 1, 1);
        if (status != da_status_success)
            return status;
    }

    try {
        if (has_W)
            this->W.resize(this->n_samples * n_components);
        if (has_H)
            this->H.resize(n_components * this->n_features);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed for NMF custom init.");
    }

    if (has_W) {
        status = check_nonneg_matrix(W_in, ldw, this->n_samples, n_components);
        if (status != da_status_success)
            return status;
    }
    if (has_H) {
        status = check_nonneg_matrix(H_in, ldh, n_components, this->n_features);
        if (status != da_status_success)
            return status;
    }

    /* Internally W is n x n_components column-major (lda = n) and H is
     * n_components x p column-major (lda = n_components). Convert from the
     * user's storage order. */
    if (this->order == row_major) {
        if (has_W)
            da_utils::copy_transpose_2D_array_row_to_column_major<T>(
                this->n_samples, n_components, W_in, ldw, this->W.data(),
                this->n_samples);
        if (has_H)
            da_utils::copy_transpose_2D_array_row_to_column_major<T>(
                n_components, this->n_features, H_in, ldh, this->H.data(), n_components);
    } else {
        if (has_W) {
            for (da_int j = 0; j < n_components; ++j) {
                const T *src = W_in + j * ldw;
                T *dst = this->W.data() + j * this->n_samples;
                for (da_int i = 0; i < this->n_samples; ++i)
                    dst[i] = src[i];
            }
        }
        if (has_H) {
            for (da_int j = 0; j < this->n_features; ++j) {
                const T *src = H_in + j * ldh;
                T *dst = this->H.data() + j * n_components;
                for (da_int i = 0; i < n_components; ++i)
                    dst[i] = src[i];
            }
        }
    }

    this->model_trained = false;
    this->n_iter = 0;
    this->reconstruction_err = T{0};
    this->k_init = n_components;
    this->custom_w_set = this->custom_w_set || has_W;
    this->custom_h_set = this->custom_h_set || has_H;
    this->custom_init_set = this->custom_w_set && this->custom_h_set;

    if (!this->custom_init_set)
        return da_warn(this->err, da_status_missing_block,
                       "Only one custom factor was provided. Upload the missing "
                       "factor before calling da_nmf_compute with init='custom'.");

    return da_status_success;
}

/*
 * Initialization algorithm for W and H when init='nnsvd' or init='nnsvda'.
 * First, extract U, S and Vt using randomized SVD. Then, use those matrices
 * to construct nonnegative initial W and H.
 */
template <typename T> da_status nmf<T>::do_nndsvd_init(da_nmf_types::nmf_init variant) {
    /* Get seed for randomized SVD from options, for reproducibility. */
    da_int seed;
    this->opts.get("random_state", seed);

    /* Workspace for the randomized SVD: only k triplets are needed. */
    std::vector<T> U, S, Vt;
    try {
        U.resize(this->n_samples * this->n_components);
        S.resize(this->n_components);
        Vt.resize(this->n_components * this->n_features);
        /* Initialize to zero, for those elements that depend on negative entries
         * in U and Vt respectively. */
        this->W.assign(this->n_samples * this->n_components, T{0});
        this->H.assign(this->n_components * this->n_features, T{0});
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed for NMF initializer.");
    }

    /* p=10 oversampling is clamped inside da_random_svd to min(m,n)-k, so
     * sketch_size never exceeds min(n_samples, n_features) even for small matrices.
     * q=-1 signals to auto-select the number of power iterations. */
    da_status status = da_random_svd<T>(this->n_samples, this->n_features, this->A,
                                        this->lda, U.data(), this->n_samples, S.data(),
                                        Vt.data(), this->n_components, this->n_components,
                                        10, -1, seed, power_normalizer_t::qr, *this->err);
    if (status != da_status_success)
        return da_error(this->err, status,
                        "Randomized SVD failed during NMF initialization.");

    /* First component: straight magnitude, no pos/neg selection.
     * Updating only the first column of W and first row of H. */
    const T scale0 = std::sqrt(S[0]);
    /* W[:, 0] = sqrt(S[0]) * abs(U[:, 0]) */
    for (da_int i = 0; i < this->n_samples; ++i)
        this->W[i] = scale0 * std::abs(U[i]);
    /* Vt is this->n_components x n_features column-major (ldvt = this->n_components),
     * so Vt[0, j] (row 0, feature j) is at index j*this->n_components. */
    for (da_int j = 0; j < this->n_features; ++j)
        this->H[j * this->n_components] =
            scale0 * std::abs(Vt[j * this->n_components]); /* H[0, j] */

    /* Remaining components via Boutsidis-Gallopoulos pos/neg-part selection. */
    std::vector<T> xp, xn, yp, yn;
    try {
        xp.resize(this->n_samples);
        xn.resize(this->n_samples);
        yp.resize(this->n_features);
        yn.resize(this->n_features);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed for NMF initializer.");
    }
    /* Iterate through the rest of the components and save the following values:
     * xp holds abs(U[:, j]) where U[:, j] > 0 (zero for negative values).
     * xn holds abs(U[:, j]) where U[:, j] < 0 (zero for positive values).
     * yp holds abs(Vt[j, :]) where Vt[j, :] > 0 (zero for negative values)
     * yn holds abs(Vt[j, :]) where Vt[j, :] < 0 (zero for positive values).
     * The squared 2-norms of these parts are accumulated in xp_n2, xn_n2, yp_n2, and yn_n2.
     */
    for (da_int j = 1; j < this->n_components; ++j) {
        /* x = U[:, j], length n */
        const T *x = U.data() + j * this->n_samples;
        T xp_n2 = T{0}, xn_n2 = T{0};
        for (da_int i = 0; i < this->n_samples; ++i) {
            xp[i] = (x[i] > T{0}) ? x[i] : T{0};
            xn[i] = (x[i] < T{0}) ? -x[i] : T{0};
            xp_n2 += xp[i] * xp[i];
            xn_n2 += xn[i] * xn[i];
        }
        xp_n2 = std::sqrt(xp_n2);
        xn_n2 = std::sqrt(xn_n2);

        /* y = Vt[j, :], length p, stride this->n_components */
        T yp_n2 = T{0}, yn_n2 = T{0};
        for (da_int jj = 0; jj < this->n_features; ++jj) {
            const T yi = Vt[j + jj * this->n_components];
            yp[jj] = (yi > T{0}) ? yi : T{0};
            yn[jj] = (yi < T{0}) ? -yi : T{0};
            yp_n2 += yp[jj] * yp[jj];
            yn_n2 += yn[jj] * yn[jj];
        }
        yp_n2 = std::sqrt(yp_n2);
        yn_n2 = std::sqrt(yn_n2);

        const T m_p = xp_n2 * yp_n2;
        const T m_n = xn_n2 * yn_n2;

        T x_norm, y_norm, sigma_jj;
        const T *x_part, *y_part;
        if (m_p > m_n) {
            x_norm = xp_n2;
            y_norm = yp_n2;
            sigma_jj = m_p;
            x_part = xp.data();
            y_part = yp.data();
        } else {
            x_norm = xn_n2;
            y_norm = yn_n2;
            sigma_jj = m_n;
            x_part = xn.data();
            y_part = yn.data();
        }

        /* If the chosen pos/neg side is degenerate (norm ~ 0), leave the
         * column and row at zero; the nndsvda/nndsvdar post-step (if requested)
         * will fill them. */
        if (x_norm <= T{0} || y_norm <= T{0})
            continue;

        const T factor = std::sqrt(S[j] * sigma_jj);
        const T wscale = factor / x_norm;
        const T hscale = factor / y_norm;

        T *W_col = this->W.data() + j * this->n_samples;
        for (da_int i = 0; i < this->n_samples; ++i)
            W_col[i] = wscale * x_part[i];
        for (da_int jj = 0; jj < this->n_features; ++jj)
            this->H[j + jj * this->n_components] = hscale * y_part[jj];
    }

    /* Post-step: replace near-zero entries depending on variant. */
    if (variant == da_nmf_types::nndsvda || variant == da_nmf_types::nndsvdar) {
        T avg;
        status = da_mean<T>(column_major, da_axis_all, this->n_samples, this->n_features,
                            this->A, this->lda, &avg);
        if (status != da_status_success)
            return status;
        const T eps = std::numeric_limits<T>::epsilon();
        if (variant == da_nmf_types::nndsvda) {
            for (std::size_t i = 0; i < this->W.size(); ++i)
                if (this->W[i] < eps)
                    this->W[i] = avg;
            for (std::size_t i = 0; i < this->H.size(); ++i)
                if (this->H[i] < eps)
                    this->H[i] = avg;
        } else {
            /* nndsvdar: fill with U(0, mean(A) * 0.01).
             * Use seed+1 so the fill RNG is independent from the SVD sketch RNG. */
            boost::random::mt19937 rng;
            const da_int fill_seed = (seed == -1) ? -1 : seed + 1;
            if (fill_seed == -1) {
                std::random_device rd;
                rng.seed(rd());
            } else {
                rng.seed(static_cast<uint32_t>(fill_seed));
            }
            boost::random::uniform_real_distribution<T> uniform_dist(T{0}, avg * T{0.01});
            for (std::size_t i = 0; i < this->W.size(); ++i)
                if (this->W[i] < eps)
                    this->W[i] = uniform_dist(rng);
            for (std::size_t i = 0; i < this->H.size(); ++i)
                if (this->H[i] < eps)
                    this->H[i] = uniform_dist(rng);
        }
    }

    return da_status_success;
}

template <typename T> da_status nmf<T>::do_random_init() {
    const da_int k = this->n_components;

    da_int seed;
    this->opts.get("random_state", seed);

    const da_int nk = this->n_samples * k;
    const da_int kp = k * this->n_features;
    try {
        this->W.resize(nk);
        this->H.resize(kp);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed for NMF random initializer.");
    }

    /* Scale = sqrt(mean(A) / k). */
    T mean_a;
    da_status status = da_mean<T>(column_major, da_axis_all, this->n_samples,
                                  this->n_features, this->A, this->lda, &mean_a);
    if (status != da_status_success)
        return status;
    const T scale = std::sqrt(mean_a / static_cast<T>(k));

    generate_random_normal_matrix(this->W.data(), this->n_samples, k, seed);
    /* Use a different seed for H so W and H are independent. Add 1 so that
     * seed==0 still produces two distinct sequences. */
    generate_random_normal_matrix(this->H.data(), k, this->n_features,
                                  seed == -1 ? -1 : seed + 1);

    for (da_int i = 0; i < nk; ++i)
        this->W[i] = scale * std::abs(this->W[i]);
    for (da_int i = 0; i < kp; ++i)
        this->H[i] = scale * std::abs(this->H[i]);

    return da_status_success;
}

template <typename T>
da_status nmf<T>::fro_residual(da_int n_rows, const T *W, const T *X, da_int ldx,
                               double &result) {
    /* Block the columns so that W lies contiguously in memory while we walk
     * across the feature dimension. */
    const da_int block_size = std::min<da_int>(this->n_features, 256);
    std::vector<T> WH_block;
    try {
        WH_block.resize(n_rows * block_size);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed for NMF residual workspace.");
    }
    double err_sq_hp = 0.0;
    for (da_int j0 = 0; j0 < this->n_features; j0 += block_size) {
        const da_int jb = std::min<da_int>(block_size, this->n_features - j0);
        /* WH_block = W * H[:, j0:j0+jb]  (n_rows x jb) */
        da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n_rows, jb,
                            this->n_components, T{1}, W, n_rows,
                            this->H.data() + j0 * this->n_components, this->n_components,
                            T{0}, WH_block.data(), n_rows);
        /* Accumulate ||X[:, j0:j0+jb] - WH_block||_F^2 without forming the
         * difference matrix. Iterate column by column to respect ldx (X may
         * have ldx > n_rows padding). WH_block is always tightly packed with
         * stride n_rows. */
        for (da_int jj = 0; jj < jb; ++jj) {
            const T *Xcol = X + (j0 + jj) * ldx;
            const T *WH_col = WH_block.data() + jj * n_rows;
            for (da_int i = 0; i < n_rows; ++i) {
                const double d =
                    static_cast<double>(Xcol[i]) - static_cast<double>(WH_col[i]);
                err_sq_hp += d * d;
            }
        }
    }
    /* Return square root of the accumulated squared Frobenius norm. */
    result = std::sqrt(err_sq_hp);
    return da_status_success;
}

/* Use Multiplicative Update (MU) iterations to refine W and H. Returns when either the
  * maximum number of iterations is reached or the relative change in the Frobenius norm of the
  * reconstruction error is less than tol.
  * The Frobenius norm of the reconstruction error is computed as:
  * min_{W >= 0, H >= 0} ||A - W H||_F = sum_{i=1}^{n_samples} sum_{j=1}^{n_features} (A[i, j] - (W H)[i, j])^2, for W>=0 and H>=0.
  * and the Gauss-Seidel updates at iteration k are:
  *     W^{k+1} = W^k * (A (H^k)^T) / (W^k H^k (H^k)^T)
  *     H^{k+1} = H^k * ((W^{k+1})^T A) / ((W^{k+1})^T W^{k+1} H^k)
  */
template <typename T> da_status nmf<T>::do_mu_iterations(da_int max_iter, T tol) {
    /* Persistent workspaces; all column-major. */
    std::vector<T> WtA, WtW, WtWH, AHt, HHt, WHHt;
    try {
        WtA.resize(this->n_components * this->n_features);
        WtW.resize(this->n_components * this->n_components);
        WtWH.resize(this->n_components * this->n_features);
        AHt.resize(this->n_samples * this->n_components);
        HHt.resize(this->n_components * this->n_components);
        WHHt.resize(this->n_samples * this->n_components);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed for NMF MU workspaces.");
    }

    /* Use float32 eps as the safe-divide threshold for both
     * float and double inputs. */
    const T eps = static_cast<T>(std::numeric_limits<float>::epsilon());
    da_int check_every;
    this->opts.get("check_every", check_every);

    /* Compute initial residual, just in case W and H are already close to the solution. */
    double err;
    da_status status =
        this->fro_residual(this->n_samples, this->W.data(), this->A, this->lda, err);
    if (status != da_status_success)
        return status;
    /* If the initial factorization is already exact (A is the zero matrix),
     * there is nothing to do. Also guards against division by zero in the
     * relative-improvement check below. */
    if (err == 0.0) {
        this->reconstruction_err = T{0};
        this->n_iter = 0;
        return da_status_success;
    }
    /* MU is not guaranteed to converge to a global minimum and the residual depends on the magnitude of A,
     * so we check the relative improvement against the initial error rather than the previous error.
     * This way, the iteration will stop on a "good enough" solution even if the error decreases very
     * slowly in later iterations.
     */
    const double err_init = err;
    double err_prev = err;

    da_int iter = 0;
    bool converged = false;

    // Iterate up to max_iter, but may break early if converged. The check is done every check_every iterations for efficiency.
    for (iter = 1; iter <= max_iter; ++iter) {
        /* W update: W^{k+1} = W^k * (A (H^k)^T) / (W^k H^k (H^k)^T)*/
        /* AHt = A * H^T  (n_samples x n_components) */
        da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, this->n_samples,
                            this->n_components, this->n_features, T{1}, this->A,
                            this->lda, this->H.data(), this->n_components, T{0},
                            AHt.data(), this->n_samples);
        /* HHt = H * H^T  (n_components x n_components) -- current H */
        da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, this->n_components,
                            this->n_components, this->n_features, T{1}, this->H.data(),
                            this->n_components, this->H.data(), this->n_components, T{0},
                            HHt.data(), this->n_components);
        /* WHHt = W * HHt  (n_samples x n_components) */
        da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, this->n_samples,
                            this->n_components, this->n_components, T{1}, this->W.data(),
                            this->n_samples, HHt.data(), this->n_components, T{0},
                            WHHt.data(), this->n_samples);
        /* We know that W has leading dimensions of n_samples so we can treat the operation
         * W_{ik} = W_{ik} * (AHt_{ik} / (WHHt_{ik} + eps)) as a vectorized operation on the entire matrix. */
        const da_int nk = this->n_samples * this->n_components;
#pragma omp simd
        for (da_int i = 0; i < nk; ++i)
            this->W[i] *= AHt[i] / (WHHt[i] + eps);

        /* H update: H^{k+1} = H^k * ((W^{k+1})^T A) / ((W^{k+1})^T W^{k+1} H^k) (using updated W) */
        /* WtA = W^T * A  (n_components x n_features) */
        da_blas::cblas_gemm(CblasColMajor, CblasTrans, CblasNoTrans, this->n_components,
                            this->n_features, this->n_samples, T{1}, this->W.data(),
                            this->n_samples, this->A, this->lda, T{0}, WtA.data(),
                            this->n_components);
        /* WtW = W^T * W  (n_components x n_components) */
        da_blas::cblas_gemm(CblasColMajor, CblasTrans, CblasNoTrans, this->n_components,
                            this->n_components, this->n_samples, T{1}, this->W.data(),
                            this->n_samples, this->W.data(), this->n_samples, T{0},
                            WtW.data(), this->n_components);
        /* WtWH = WtW * H  (n_components x n_features) */
        da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, this->n_components,
                            this->n_features, this->n_components, T{1}, WtW.data(),
                            this->n_components, this->H.data(), this->n_components, T{0},
                            WtWH.data(), this->n_components);
        /* We know that H has leading dimensions of n_components so we can treat the operation
         * H_{kj} = H_{kj} * (WtA_{kj} / (WtWH_{kj} + eps)) as a vectorized operation on the entire matrix. */
        const da_int kp = this->n_components * this->n_features;
#pragma omp simd
        for (da_int i = 0; i < kp; ++i)
            this->H[i] *= WtA[i] / (WtWH[i] + eps);

        /* Convergence check: every check_every iterations, only when tol > 0 */
        if (tol > T{0} && (iter % check_every == 0)) {
            status = this->fro_residual(this->n_samples, this->W.data(), this->A,
                                        this->lda, err);
            if (status != da_status_success)
                return status;
            if ((err_prev - err) / err_init < static_cast<double>(tol)) {
                converged = true;
                break;
            }
            err_prev = err;
        }
    }

    /* Reuse the residual already computed at the last convergence check when
     * converged, otherwise recompute (tol==0 means no check ran at all, and
     * if max_iter is not a multiple of check_every the last check is stale). */
    if (!converged) {
        status =
            this->fro_residual(this->n_samples, this->W.data(), this->A, this->lda, err);
        if (status != da_status_success)
            return status;
    }
    this->reconstruction_err = static_cast<T>(err);

    /* When converged, iter is the iteration that triggered the break (no post-increment).
     * When not converged, the for-loop increments iter to max_iter+1 before the condition fails. */
    this->n_iter = converged ? iter : max_iter;
    return da_status_success;
}

/* Compute the NMF factors W and H using the specified initialization and MU iterations. */
template <typename T> da_status nmf<T>::compute() {
    if (!this->A || this->n_samples <= 0 || this->n_features <= 0)
        return da_error(this->err, da_status_no_data,
                        "Input data has not been set. Call da_nmf_init before "
                        "da_nmf_compute.");

    /* Resolve options. */
    da_int k_opt;
    this->opts.get("n_components", k_opt);
    if (k_opt == 0) {
        k_opt = std::min(this->n_samples, this->n_features);
    }
    if (k_opt > std::min(this->n_samples, this->n_features)) {
        return da_error(this->err, da_status_invalid_input,
                        "Option 'n_components' must be <= min(n_samples, n_features).");
    }
    this->n_components = k_opt;

    std::string init_str;
    da_int init_algo;
    this->opts.get("init", init_str, init_algo);
    /* Select initialization algorithm. */
    da_status status = da_status_success;
    auto init = static_cast<da_nmf_types::nmf_init>(init_algo);
    switch (init) {
    case da_nmf_types::nndsvd:
    case da_nmf_types::nndsvda:
    case da_nmf_types::nndsvdar:
        status = do_nndsvd_init(init);
        break;
    case da_nmf_types::random:
        status = do_random_init();
        break;
    case da_nmf_types::automatic: {
        /* Choose 'nndsvda' when n_components < min(n_samples, n_features),
         * 'random' otherwise. */
        if (this->n_components < std::min(this->n_samples, this->n_features))
            status = do_nndsvd_init(da_nmf_types::nndsvda);
        else
            status = do_random_init();
        break;
    }
    case da_nmf_types::custom:
        if (!this->custom_init_set)
            return da_error(this->err, da_status_no_data,
                            "init='custom' requires a prior call to "
                            "da_nmf_set_data to provide initial W and H.");
        if (this->k_init != this->n_components)
            return da_error(this->err, da_status_invalid_input,
                            "The number of components in the supplied initial W, H "
                            "does not match the 'n_components' option.");
        /* Custom init is used directly as the mutable working state.
         * If the caller needs to replay the exact same initial estimates, it must call da_nmf_set_data again before compute(). */
        this->custom_w_set = false;
        this->custom_h_set = false;
        this->custom_init_set = false;
        break;
    }
    if (status != da_status_success)
        return status;

    /* Read MU options now that init succeeded. */
    da_int max_iter_opt;
    this->opts.get("max_iter", max_iter_opt);
    T tol_opt;
    this->opts.get("tol", tol_opt);

    // Run the MU iterations, which update this->W and this->H in-place.
    status = do_mu_iterations(max_iter_opt, tol_opt);
    if (status != da_status_success)
        return status;
    // By this point W and H contain the final NMF factors.
    this->model_trained = true;
    return da_status_success;
}

/* Use the NMF factor H to transform new data X into NMF space, returning the NMF
 * coefficient matrix W such that X ~= W H. */
template <typename T>
da_status nmf<T>::transform(da_int m_samples, da_int m_features, const T *X, da_int ldx,
                            T *Xt, da_int ldxt) {
    if (!this->model_trained)
        return da_warn(this->err, da_status_no_data,
                       "The NMF model has not been computed. Please call "
                       "da_nmf_compute_s or da_nmf_compute_d.");
    if (m_features != this->n_features)
        return da_error(
            this->err, da_status_invalid_input,
            "The function was called with m_features = " + std::to_string(m_features) +
                " but the NMF was computed with n_features = " +
                std::to_string(this->n_features) + ".");

    da_status status = this->check_2D_array(this->order, m_samples, m_features, X, ldx,
                                            "m_samples", "m_features", "X", "ldx", 1, 1);
    if (status != da_status_success)
        return status;
    status = this->check_2D_array(this->order, m_samples, this->n_components, Xt, ldxt,
                                  "m_samples", "n_components", "Xt", "ldxt", 1, 1);
    if (status != da_status_success)
        return status;

    /* Reject negative entries: NMF requires X >= 0. */
    status = check_nonneg_matrix(X, ldx, m_samples, m_features);
    if (status != da_status_success)
        return status;

    /* If row-major, transpose X into a tightly-packed col-major buffer (lda = m_samples).
     * If col-major, point directly at X (lda = ldx), no copy needed. */
    std::vector<T> Xcol;
    const T *Xdata;
    da_int ldx_col;
    if (this->order == row_major) {
        try {
            Xcol.resize(m_samples * this->n_features);
        } catch (std::bad_alloc const &) {
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed for NMF transform workspaces.");
        }
        da_utils::copy_transpose_2D_array_row_to_column_major<T>(
            m_samples, m_features, X, ldx, Xcol.data(), m_samples);
        Xdata = Xcol.data();
        ldx_col = m_samples;
    } else {
        Xdata = X;
        ldx_col = ldx;
    }

    std::vector<T> Wnew, XHt, HHt_local, WHHt_local;
    try {
        Wnew.resize(m_samples * this->n_components);
        XHt.resize(m_samples * this->n_components);
        HHt_local.resize(this->n_components * this->n_components);
        WHHt_local.resize(m_samples * this->n_components);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed for NMF transform workspaces.");
    }

    /* Compute mean(X) over all elements, Xdata is always column-major at this point. */
    T mean_x;
    status = da_mean<T>(column_major, da_axis_all, m_samples, this->n_features, Xdata,
                        ldx_col, &mean_x);
    if (status != da_status_success)
        return status;

    /* Initial guess W_new = sqrt(mean(X) / n_components), entry-wise constant. This is
     * a fall-back when only W needs to be initialized (H fixed). */
    T w0 = (mean_x > T{0}) ? std::sqrt(mean_x / static_cast<T>(this->n_components))
                           : std::sqrt(std::numeric_limits<T>::epsilon());
    da_std::fill(Wnew.begin(), Wnew.end(), w0);

    /* MU stopping options. */
    da_int max_iter_opt;
    this->opts.get("max_iter", max_iter_opt);
    T tol_opt;
    this->opts.get("tol", tol_opt);

    const T eps = static_cast<T>(std::numeric_limits<float>::epsilon());
    da_int check_every;
    this->opts.get("check_every", check_every);

    /* Precompute constants XHt = X H^T, HHt = H H^T.
     * Both are constant since H is fixed during transform(). */
    da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, m_samples,
                        this->n_components, this->n_features, T{1}, Xdata, ldx_col,
                        this->H.data(), this->n_components, T{0}, XHt.data(), m_samples);
    da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, this->n_components,
                        this->n_components, this->n_features, T{1}, this->H.data(),
                        this->n_components, this->H.data(), this->n_components, T{0},
                        HHt_local.data(), this->n_components);

    /* ||X - Wnew H||_F : blocked explicit computation via fro_residual. */
    double err;
    status = this->fro_residual(m_samples, Wnew.data(), Xdata, ldx_col, err);
    if (status != da_status_success)
        return status;
    const double err_init = (err > 0.0) ? err : 1.0;
    double err_prev = err;
    /* Iterate to update Wnew. */
    for (da_int iter = 1; iter <= max_iter_opt; ++iter) {
        /* Wnew *= (X H^T) / (Wnew H H^T)  with H fixed. */
        da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_samples,
                            this->n_components, this->n_components, T{1}, Wnew.data(),
                            m_samples, HHt_local.data(), this->n_components, T{0},
                            WHHt_local.data(), m_samples);
        const da_int mk = m_samples * this->n_components;
#pragma omp simd
        for (da_int i = 0; i < mk; ++i)
            Wnew[i] *= XHt[i] / (WHHt_local[i] + eps);

        if (tol_opt > T{0} && (iter % check_every == 0)) {
            status = this->fro_residual(m_samples, Wnew.data(), Xdata, ldx_col, err);
            if (status != da_status_success)
                return status;
            if ((err_prev - err) / err_init < static_cast<double>(tol_opt))
                break;
            err_prev = err;
        }
    }

    /* Copy Wnew (m_samples x k col-major) into Xt in the user's order. */
    if (this->order == row_major) {
        da_utils::copy_transpose_2D_array_column_to_row_major<T>(
            m_samples, this->n_components, Wnew.data(), m_samples, Xt, ldxt);
    } else {
        for (da_int j = 0; j < this->n_components; ++j) {
            const T *src = Wnew.data() + j * m_samples;
            da_std::copy(src, src + m_samples, Xt + j * ldxt);
        }
    }

    return da_status_success;
}

/* Compute X = Xt * H (in this case Xt is a previously-computed W, where H was kept fixed) */
template <typename T>
da_status nmf<T>::inverse_transform(da_int k_samples, da_int k_components, const T *Xt,
                                    da_int ldxt, T *X, da_int ldx) {
    if (!this->model_trained)
        return da_warn(this->err, da_status_no_data,
                       "The NMF model has not been computed. Please call "
                       "da_nmf_compute_s or da_nmf_compute_d.");
    if (k_components != this->n_components)
        return da_error(this->err, da_status_invalid_input,
                        "The function was called with k_components = " +
                            std::to_string(k_components) +
                            " but the NMF was computed with n_components = " +
                            std::to_string(this->n_components) + ".");

    da_status status =
        this->check_2D_array(this->order, k_samples, k_components, Xt, ldxt, "k_samples",
                             "k_components", "Xt", "ldxt", 1, 1);
    if (status != da_status_success)
        return status;
    status = this->check_2D_array(this->order, k_samples, this->n_features, X, ldx,
                                  "k_samples", "n_features", "X", "ldx", 1, 1);
    if (status != da_status_success)
        return status;

    /* Compute X = Xt * H.
     * H is always stored col-major order (n_components × n_features, lda = n_components).
     * For row-major inputs, we can use the transpose identity.
     * Col-major: straightforward NoTrans/NoTrans GEMM.
     *   X[k_samples × n_features] = Xt[k_samples × n_components] * H[n_components × n_features]
     *
     * Row-major: apply the transpose identity
     *   X_rm = Xt_rm * H  ⟺  X_rm^T = H^T * Xt_rm^T  (col-major GEMM)
     *   Xt_rm viewed as col-major is Xt_rm^T  (shape n_components × k_samples, lda = ldxt).
     *   X_rm  viewed as col-major is X_rm^T   (shape n_features  × k_samples, lda = ldx).
     *   H is already stored col-major (n_components × n_features, lda = n_components);
     *   CblasTrans gives H^T (n_features × n_components) without any data movement.
     */
    if (this->order == column_major) {
        da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, k_samples,
                            this->n_features, this->n_components, T{1}, Xt, ldxt,
                            this->H.data(), this->n_components, T{0}, X, ldx);
    } else {
        da_blas::cblas_gemm(CblasColMajor, CblasTrans, CblasNoTrans, this->n_features,
                            k_samples, this->n_components, T{1}, this->H.data(),
                            this->n_components, Xt, ldxt, T{0}, X, ldx);
    }

    return da_status_success;
}

template <typename T> da_status nmf<T>::serialize(serialization_buffer &buffer) {

    da_status status = da_status_success;
    auto io_dispatch = [&buffer, &status](auto &data) -> void {
        if (status != da_status_success) {
            return;
        }
        status = buffer.dispatch_buffer_io(data);
        return;
    };

    io_dispatch(this->model_trained);
    io_dispatch(this->order);
    io_dispatch(this->n_samples);
    io_dispatch(this->n_features);
    io_dispatch(this->n_components);
    io_dispatch(this->n_iter);
    io_dispatch(this->reconstruction_err);
    io_dispatch(this->W);
    io_dispatch(this->H);

    return status;
}

template <typename T> da_status nmf<T>::save_model(serialization_buffer &buffer) {

    if (!this->model_trained)
        return da_error(this->err, da_status_no_data,
                        "The NMF has not been computed. Please call da_nmf_compute_s or "
                        "da_nmf_compute_d.");

    da_status status = basic_handle<T>::save_model(buffer);
    if (status != da_status_success)
        return da_error_trace(this->err, status, "Failure serializing NMF model.");

    return status;
}

template <typename T> da_status nmf<T>::load_model(serialization_buffer &buffer) {
    da_status status = basic_handle<T>::load_model(buffer);
    if (status != da_status_success)
        return da_error_trace(this->err, status, "Failure deserializing NMF model.");

    return status;
}

template class nmf<double>;
template class nmf<float>;

} // namespace da_nmf
} // namespace ARCH
