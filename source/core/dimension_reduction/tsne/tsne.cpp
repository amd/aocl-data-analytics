/* ************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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

#include "tsne.hpp"
#include "basic_statistics.hpp"
#include "da_omp.hpp"
#include "da_std.hpp"
#include "da_utils.hpp"
#include "da_vector.hpp"
#include "miscellaneous.hpp"
#include "nearest_neighbors.hpp"
#include "pairwise_distances.hpp"
#include "pca/pca.hpp"
#include "tsne_kernels.hpp"
#include "tsne_options.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <type_traits>
#include <utility>

using namespace std::literals::string_literals;

namespace ARCH {

namespace da_tsne {

namespace {
using v = vectorization_type;
using namespace kernel_templates;
} // namespace

/***************************************************************************************************
 * Attractive-forces kernels - ISA mapping
 * Kernel Implementations Table (KIT)
 *
 * d=1 uses scalar_impl directly (not dispatched via the kernel templates).
 *
 * Implementations:
 *   scalar_impl  — plain C++ loop, no SIMD
 *   kt<W>        — single-neighbor SIMD via kernel_templates (1 neigh/iter)
 *   multi<W>     — multi-neighbor d=2 packing (W/2 per type neigh/iter)
 *
 * ISA      | Type   | d=1            | d=2                        | d=3
 * ---------+--------+-------------+-------------------------------+-------------------------
 * scalar   | float  | scalar_impl<1> | scalar_impl<2>             | scalar_impl<3>
 * scalar   | double | scalar_impl<1> | scalar_impl<2>             | scalar_impl<3>
 * avx      | float  | scalar_impl<1> | kt<b128,2>    1 neigh/iter | kt<b128,3>    1 neigh/iter
 * avx      | double | scalar_impl<1> | kt<b128,2>    1 neigh/iter | -> scalar_impl<3>
 * avx2     | float  | scalar_impl<1> | multi<b256,2> 4 neigh/iter | -> avx
 * avx2     | double | scalar_impl<1> | multi<b256,2> 2 neigh/iter | kt<b256,3>    1 neigh/iter
 * avx512   | float  | scalar_impl<1> | multi<b512,2> 8 neigh/iter | -> avx
 * avx512   | double | scalar_impl<1> | multi<b512,2> 4 neigh/iter | -> avx2
 *
 **************************************************************************************************/

// clang-format off
// d=2: multi-neighbor packing for avx2 / avx512.
static const kernel_implementations<KFS, KFD> tsne_d2_impls = {
/*scalar*/ {{          attractive_forces_scalar_impl<float, 2>,
/*   avx*/             attractive_forces_kt<bsz::b128, float, 2>,
/*  avx2*/             attractive_forces_multi_d2<bsz::b256, float>,
/*avx512*/ ORL_AVX512F(attractive_forces_multi_d2<bsz::b512, float>) }},
/*scalar*/ {{          attractive_forces_scalar_impl<double,2>,
/*   avx*/             attractive_forces_kt<bsz::b128, double, 2>,
/*  avx2*/             attractive_forces_multi_d2<bsz::b256, double>,
/*avx512*/ ORL_AVX512F(attractive_forces_multi_d2<bsz::b512, double>) }}
};

// d=3: single-neighbor kt kernels.
static const kernel_implementations<KFS, KFD> tsne_d3_impls = {
/*scalar*/ {{          attractive_forces_scalar_impl<float, 3>,
/*   avx*/             attractive_forces_kt<bsz::b128, float, 3>,
/*  avx2*/             attractive_forces_kt<bsz::b128, float, 3>,
/*avx512*/ ORL_AVX512F(attractive_forces_kt<bsz::b128, float, 3>) }},
/*scalar*/ {{          attractive_forces_scalar_impl<double,3>,
/*   avx*/             attractive_forces_scalar_impl<double, 3>,
/*  avx2*/             attractive_forces_kt<bsz::b256, double, 3>,
/*avx512*/ ORL_AVX512F(attractive_forces_kt<bsz::b256, double, 3>) }}
};
// clang-format on

static const std::array<const kernel_implementations<KFS, KFD> *, 2>
    attractive_forces_implementations = {{&tsne_d2_impls, &tsne_d3_impls}};

namespace testing {
const std::array<const kernel_implementations<KFS, KFD> *, 2> &
get_attractive_forces_implementations() {
    return attractive_forces_implementations;
}
} // namespace testing

template <typename T> tsne<T>::tsne(da_errors::da_error_t &err) : basic_handle<T>(err) {
    register_tsne_options<T>(this->opts, *this->err);
}

template <typename T>
tsne<T>::tsne(da_errors::da_error_t &err, da_int n_samples, da_int n_features,
              da_int n_components, da_int max_iter, T learning_rate, T early_exaggeration,
              T theta, T min_grad_norm, da_int n_iter_without_progress,
              std::vector<da_int> &&P_row_ptr, std::vector<da_int> &&P_col_idx,
              std::vector<T> &&P_values, std::vector<T> &&embedding,
              const std::string &init_method, da_int seed)
    : basic_handle<T>(err), n_samples(n_samples), n_features(n_features),
      n_components(n_components), max_iter(max_iter), initdone(true),
      check_options(false), embedding(std::move(embedding)),
      P_row_ptr(std::move(P_row_ptr)), P_col_idx(std::move(P_col_idx)),
      P_values(std::move(P_values)), learning_rate(learning_rate),
      early_exaggeration(early_exaggeration), theta(theta), min_grad_norm(min_grad_norm),
      n_iter_without_progress(n_iter_without_progress), init_method(init_method),
      seed(seed){};

template <typename T> void tsne<T>::refresh() {
    model_trained = false;
    kl_divergence = 0;
    lp_n_iter = 0;
    start_iter = 0;
    embedding.clear();
    iY.clear();
    gains.clear();
    P_row_ptr.clear();
    P_col_idx.clear();
    P_values.clear();
}

template <typename T> void tsne<T>::assign_attractive_force_kernel() {
    if (n_components == 1) {
        this->attractive_force_kernel_fn = attractive_forces_scalar_impl<T, 1>;
        return;
    }
    // Get best AVX ISA
    vectorization_type isa = Oracle("tsne.isa");
    da_int n_comp = std::clamp(n_components, da_int(2), da_int(3)) - 2;
    // Get best kernel
    this->attractive_force_kernel_fn =
        attractive_forces_implementations[n_comp]->get<T>(isa);
}

template <typename T>
da_status tsne<T>::set_data(da_int n_samples_in, da_int n_features_in, const T *X_in,
                            da_int ldx_in) {
    refresh();

    // Reset input state; will be repopulated below if validation passes
    X = nullptr;
    X_copy.clear();
    initdone = false;
    supplied_embedding.clear();
    supplied_n_components = 0;
    has_supplied_embedding = false;

    std::string opt_order;
    da_int iorder;
    da_status status = this->opts.get("storage order", opt_order, iorder);
    if (status != da_status_success)
        return status; // LCOV_EXCL_LINE
    this->order = da_order(iorder);

    status = this->check_2D_array(this->order, n_samples_in, n_features_in, X_in, ldx_in,
                                  "n_samples", "n_features", "X", "ldx", 2, 1);
    if (status != da_status_success)
        return status;

    n_samples = n_samples_in;
    n_features = n_features_in;

    // Copy if not compact row-major format
    if (this->order == row_major && ldx_in == n_features) {
        X = X_in;
    } else {
        try {
            X_copy.resize(n_samples * n_features);
        } catch (std::bad_alloc &) {
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation error.");
        }
        if (this->order == row_major) {
            for (da_int i = 0; i < n_samples; ++i)
                for (da_int j = 0; j < n_features; ++j)
                    X_copy[i * n_features + j] = X_in[i * ldx_in + j];
        } else {
            da_utils::copy_transpose_2D_array_column_to_row_major(
                n_samples, n_features, X_in, ldx_in, X_copy.data(), n_features);
        }
        X = X_copy.data();
    }

    initdone = true;
    model_trained = false;

    // Save user-set values before re-registering options (which resets to defaults)
    da_int temp_components = 2;
    this->opts.get("n_components", temp_components);
    T temp_perplexity = (T)30.0;
    this->opts.get("perplexity", temp_perplexity);

    reregister_tsne_options<T>(this->opts, n_samples, n_features);

    // Restore user values, clamped to valid range
    da_int max_components = std::min<da_int>(3, n_features);
    this->opts.set("n_components", std::min(temp_components, max_components));

    T max_perplexity = (T)(std::max<da_int>(1, n_samples - 1));
    this->opts.set("perplexity", std::min(temp_perplexity, max_perplexity));

    if (temp_components > max_components)
        return da_warn(this->err, da_status_incompatible_options,
                       "The requested number of components has been decreased from " +
                           std::to_string(temp_components) + " to " +
                           std::to_string(max_components) +
                           " due to the size of the data array.");

    if (temp_perplexity > max_perplexity)
        return da_warn(this->err, da_status_incompatible_options,
                       "The requested perplexity has been decreased from " +
                           std::to_string(temp_perplexity) + " to " +
                           std::to_string(max_perplexity) +
                           " due to the size of the data array.");

    return da_status_success;
}

template <typename T>
da_status tsne<T>::set_init_embedding(const T *Y_in, da_int ldy_in) {
    if (!initdone)
        return da_error(this->err, da_status_no_data,
                        "No data has been passed to the handle. Please call "
                        "da_tsne_set_data_s or da_tsne_set_data_d.");

    da_int n_comp = 2;
    this->opts.get("n_components", n_comp);

    da_status status =
        this->check_2D_array(this->order, n_samples, n_comp, Y_in, ldy_in, "n_samples",
                             "n_components", "Y", "ldy", n_samples, n_comp);
    if (status != da_status_success)
        return status;

    try {
        supplied_embedding.resize(n_samples * n_comp, (T)0);
    } catch (std::bad_alloc &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error.");
    }

    if (this->order == column_major) {
        da_utils::copy_transpose_2D_array_column_to_row_major(
            n_samples, n_comp, Y_in, ldy_in, supplied_embedding.data(), n_comp);
    } else {
        for (da_int i = 0; i < n_samples; ++i) {
            for (da_int j = 0; j < n_comp; ++j) {
                supplied_embedding[i * n_comp + j] = Y_in[i * ldy_in + j];
            }
        }
    }
    supplied_n_components = n_comp;
    has_supplied_embedding = true;
    return da_status_success;
}

// Binary search for bandwidth (beta = 1/(2*sigma^2)) to match target perplexity.
// H(p_i) = log(sum_j exp(-beta*d_ij)) + beta * sum_j d_ij * p_ij = log_perplexity
template <typename T>
void compute_row_probabilities(const T *sq_distances, da_int k, T log_perplexity,
                               T *row_prob) {
    if (k == 0)
        return;

    constexpr da_int max_iter = 100;
    constexpr T binary_search_tol = (T)1e-5;

    T beta = (T)1;
    T betamin = -std::numeric_limits<T>::infinity();
    T betamax = std::numeric_limits<T>::infinity();

    for (da_int iter = 0; iter < max_iter; ++iter) {
        T sum_p = (T)0, sum_dp = (T)0;
        for (da_int j = 0; j < k; ++j) {
            const T p = std::exp(-beta * sq_distances[j]);
            row_prob[j] = p;
            sum_p += p;
            sum_dp += sq_distances[j] * p;
        }
        if (sum_p <= (T)0) {
            betamax = beta;
            beta = std::isinf(betamin) ? beta / (T)2 : (beta + betamin) / (T)2;
            continue;
        }
        const T H = std::log(sum_p) + beta * sum_dp / sum_p;
        if (std::abs(H - log_perplexity) < binary_search_tol)
            break;
        if (H > log_perplexity) {
            betamin = beta;
            beta = std::isinf(betamax) ? beta * (T)2 : (beta + betamax) / (T)2;
        } else {
            betamax = beta;
            beta = std::isinf(betamin) ? beta / (T)2 : (beta + betamin) / (T)2;
        }
    }

    // Normalise to a proper distribution; fall back to uniform if underflowed.
    // Guard against denormalised sum_p where 1/sum_p would overflow.
    T sum_p = (T)0;
    for (da_int j = 0; j < k; ++j)
        sum_p += row_prob[j];
    if (sum_p >= std::numeric_limits<T>::min()) {
        const T inv = (T)1 / sum_p;
        for (da_int j = 0; j < k; ++j)
            row_prob[j] *= inv;
    } else {
        const T uniform = (T)1 / (T)k;
        for (da_int j = 0; j < k; ++j)
            row_prob[j] = uniform;
    }
}

// P_ij = (p(j|i) + p(i|j)) / (2n), then renormalised so sum P_ij = 1.
// neighbor_indices and neighbor_probs are flat n-by-k row-major arrays
// Output CSR has sorted column indices per row.
template <typename T>
da_status symmetrize_to_csr(da_int n, da_int k, const da_int *neighbor_indices,
                            const T *neighbor_probs, std::vector<da_int> &P_row_ptr,
                            std::vector<da_int> &P_col_idx, std::vector<T> &P_values) {
    // Collect (col, value) entries per row from both (i->j) and (j->i).
    const T norm = (T)(2 * n);
    std::vector<std::vector<std::pair<da_int, T>>> row_contributions;
    try {
        row_contributions.resize(n);
        P_row_ptr.resize(n + 1);
    } catch (std::bad_alloc &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    }
    for (da_int i = 0; i < n; ++i) {
        const da_int base = i * k;
        for (da_int t = 0; t < k; ++t) {
            const da_int j = neighbor_indices[base + t];
            const T contrib = neighbor_probs[base + t] / norm;
            row_contributions[i].emplace_back(j, contrib);
            row_contributions[j].emplace_back(i, contrib);
        }
    }

    // Sort each row by column, merge duplicates, and build CSR directly.
    P_row_ptr[0] = 0;
    P_col_idx.clear();
    P_values.clear();
    T total = (T)0;
    for (da_int i = 0; i < n; ++i) {
        std::vector<std::pair<da_int, T>> &row = row_contributions[i];
        std::sort(row.begin(), row.end());
        for (size_t t = 0; t < row.size();) {
            const da_int col = row[t].first;
            T val = (T)0;
            while (t < row.size() && row[t].first == col)
                val += row[t++].second;
            if (val > (T)0) {
                P_col_idx.push_back(col);
                P_values.push_back(val);
                total += val;
            }
        }
        P_row_ptr[i + 1] = (da_int)P_col_idx.size();
    }
    if (total > (T)0) {
        const T inv_total = (T)1 / total;
        for (T &v : P_values)
            v *= inv_total;
    }
    return da_status_success;
}

// Compute squared distances using pairwise distance matrix (exact method).
// Outputs flat n-by-(n-1) row-major arrays, excluding the self-distance diagonal.
template <typename T>
static da_status compute_distances_exact(da_int n, da_int n_features, const T *X,
                                         da_errors::da_error_t *err,
                                         da_vector::da_vector<da_int> &neighbor_indices,
                                         da_vector::da_vector<T> &sq_distances) {
    const da_int k = n - 1;
    da_vector::da_vector<T> dist_matrix;
    try {
        dist_matrix.resize(n * n);
        neighbor_indices.resize(n * k);
        sq_distances.resize(n * k);
    } catch (std::bad_alloc &) {
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error.");
    }
    da_status status = da_metrics::pairwise_distances::sqeuclidean(
        row_major, n, n, n_features, X, n_features, X, n_features, dist_matrix.data(), n);
    if (status != da_status_success)
        return da_error_bypass( // LCOV_EXCL_LINE
            err, status, "Failed to compute pairwise distances for affinities.");

    for (da_int i = 0; i < n; ++i) {
        for (da_int j = i + 1; j < n; ++j) {
            const T d = dist_matrix[i * n + j];
            neighbor_indices[i * k + (j - 1)] = j;
            sq_distances[i * k + (j - 1)] = d;
            neighbor_indices[j * k + i] = i;
            sq_distances[j * k + i] = d;
        }
    }
    return da_status_success;
}

// Compute squared distances using k-nearest neighbors (approximate method).
// Outputs flat n-by-k row-major arrays, excluding the self-neighbor.
template <typename T>
static da_status compute_distances_knn(da_int k, da_int n, da_int n_features, const T *X,
                                       da_errors::da_error_t *err,
                                       da_vector::da_vector<da_int> &neighbor_indices,
                                       da_vector::da_vector<T> &sq_distances) {
    // use k+1 to ensure we get k neighbors (we will skip the self-neighbor)
    da_int k_query = k + 1;

    da_neighbors::neighbors<T> nn(*err);
    da_status status = nn.get_opts().set("storage order", "row-major");
    if (status != da_status_success)
        return status; // LCOV_EXCL_LINE
    status = nn.get_opts().set("algorithm", "auto");
    if (status != da_status_success)
        return status; // LCOV_EXCL_LINE
    status = nn.get_opts().set("metric", "euclidean");
    if (status != da_status_success)
        return status; // LCOV_EXCL_LINE

    da_vector::da_vector<da_int> n_ind;
    da_vector::da_vector<T> n_dist;
    try {
        n_ind.resize(n * k_query);
        n_dist.resize(n * k_query);
        neighbor_indices.resize(n * k);
        sq_distances.resize(n * k);
    } catch (std::bad_alloc &) {
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error.");
    }

    status = nn.set_data(n, n_features, X, n_features);
    if (status != da_status_success)
        return status; // LCOV_EXCL_LINE

    status = nn.kneighbors(n, n_features, X, n_features, n_ind.data(), n_dist.data(),
                           k_query, true);
    if (status != da_status_success)
        return da_error_bypass( // LCOV_EXCL_LINE
            err, status, "Failed to compute nearest neighbors for affinities.");

    for (da_int i = 0; i < n; ++i) {
        const da_int in_base = i * k_query;
        const da_int out_base = i * k;
        da_int filled = 0;
        for (da_int t = 0; t < k_query && filled < k; ++t) {
            const da_int j = n_ind[in_base + t];
            if (j == i)
                continue;
            neighbor_indices[out_base + filled] = j;
            const T d = n_dist[in_base + t];
            sq_distances[out_base + filled] = d * d;
            ++filled;
        }
    }

    return da_status_success;
}

template <typename T>
da_status compute_affinities(T perplexity, bool use_exact, da_int n, da_int n_features,
                             const T *X, da_errors::da_error_t *err,
                             std::vector<da_int> &P_row_ptr,
                             std::vector<da_int> &P_col_idx, std::vector<T> &P_values) {
    const da_int k =
        use_exact ? (n - 1) : std::min<da_int>(n - 1, (da_int)(3 * perplexity + 1));

    const T log_perplexity = std::log(perplexity);

    // Compute squared distances into flat n-by-k arrays
    da_vector::da_vector<da_int> neighbor_indices;
    da_vector::da_vector<T> sq_distances;

    da_status status = use_exact ? compute_distances_exact(n, n_features, X, err,
                                                           neighbor_indices, sq_distances)
                                 : compute_distances_knn(k, n, n_features, X, err,
                                                         neighbor_indices, sq_distances);
    if (status != da_status_success)
        return status; // LCOV_EXCL_LINE

    // Compute conditional probabilities for each row
    da_vector::da_vector<T> neighbor_probs;
    try {
        neighbor_probs.resize(n * k);
    } catch (std::bad_alloc &) {
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error.");
    }
#pragma omp parallel for schedule(static) default(none)                                  \
    shared(sq_distances, neighbor_probs, n, k, log_perplexity)
    for (da_int i = 0; i < n; ++i)
        compute_row_probabilities(&sq_distances[i * k], k, log_perplexity,
                                  &neighbor_probs[i * k]);

    // Symmetrize and convert to CSR
    status = symmetrize_to_csr(n, k, neighbor_indices.data(), neighbor_probs.data(),
                               P_row_ptr, P_col_idx, P_values);
    if (status != da_status_success)
        return da_error( // LCOV_EXCL_LINE
            err, status, "Memory allocation error during symmetrization.");

    return da_status_success;
}

template <typename T>
da_status tsne<T>::initialize_embedding(const std::string &init_method, da_int seed) {
    if (init_method == "supplied") {
        if (!has_supplied_embedding)
            return da_error(
                this->err, da_status_no_data,
                "The initialization method was set to 'supplied' but no initial "
                "embedding has been provided. Call da_tsne_set_init_embedding_s or "
                "da_tsne_set_init_embedding_d.");
        if (supplied_n_components != n_components)
            return da_error(this->err, da_status_invalid_input,
                            "The supplied embedding dimension does not match "
                            "n_components.");
        embedding = supplied_embedding;
        return da_status_success;
    }

    try {
        embedding.resize(n_samples * n_components, (T)(0));
    } catch (std::bad_alloc &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error.");
    }

    if (init_method == "pca") {
        // PCA initialization: project onto first n_components principal components.
        da_pca::pca<T> pca(*this->err);
        auto &opts = pca.get_opts();
        opts.set("storage order", "row-major");
        opts.set("n_components", n_components);
        opts.set("pca method", "covariance");
        opts.set("seed", seed);

        da_status status = pca.init(n_samples, n_features, X, n_features);
        if (status != da_status_success)
            return status; // LCOV_EXCL_LINE
        status = pca.compute();
        if (status != da_status_success)
            return status; // LCOV_EXCL_LINE

        status = pca.transform(n_samples, n_features, X, n_features, embedding.data(),
                               n_components);
        if (status != da_status_success)
            return status; // LCOV_EXCL_LINE

        // Scale to small variance for stable optimization.
        // sklearn uses std of the first column: X_embedded / std(X_embedded[:,0]) * 1e-4
        T mean = 0, var = 0;
        constexpr T target_stdv = (T)1e-4;
        da_basic_statistics::variance(row_major, da_axis_all, n_samples, 1,
                                      embedding.data(), n_components, -1, &mean, &var);
        T stdv = std::sqrt(var);
        // Guard against near-zero std dev (e.g. constant-column input)
        constexpr T eps_safety_factor = (T)100;
        const T min_stdv = std::numeric_limits<T>::epsilon() * eps_safety_factor;

        if (stdv > min_stdv) {
            T scale = target_stdv / stdv;
            for (T &v : embedding)
                v *= scale;
        }
        return da_status_success;
    }

    // Set up random number generator
    std::mt19937_64 rng;
    if (seed == -1) {
        std::random_device rd;
        rng.seed(std::abs((da_int)rd()));
    } else {
        rng.seed(seed);
    }

    // Random initialization
    std::normal_distribution<T> normal((T)(0), (T)(1.0e-4));
    for (T &v : embedding)
        v = normal(rng);

    return da_status_success;
}

template <typename T>
void compute_attractive_forces(da_int n, da_int d, T exaggeration,
                               const std::vector<da_int> &row_ptr,
                               const std::vector<da_int> &col_idx,
                               const std::vector<T> &p_vals, const std::vector<T> &emb,
                               const std::vector<T> &repulsive, T sum_q_total,
                               std::vector<T> &grad,
                               attractive_forces_kernel_fn<T> kernel) {
#pragma omp parallel for schedule(static) default(none)                                  \
    shared(n, d, row_ptr, col_idx, p_vals, emb, grad, repulsive, exaggeration,           \
               sum_q_total, kernel)
    for (da_int i = 0; i < n; ++i) {
        T *grad_i = grad.data() + i * d;
        const T *emb_i = emb.data() + i * d;
        da_int start = row_ptr[i];
        da_int end = row_ptr[i + 1];
        kernel(emb_i, col_idx.data(), p_vals.data(), emb.data(), exaggeration, start, end,
               grad_i);
        for (da_int k = 0; k < d; ++k)
            grad_i[k] = (T)4 * (grad_i[k] - repulsive[i * d + k] / sum_q_total);
    }
}

// KL(P || Q) = sum_{i!=j} P_ij * log(P_ij / Q_ij)
// where Q_ij = (1 + ||y_i - y_j||^2)^{-1} / sum_{k!=l} (1 + ||y_k - y_l||^2)^{-1}
// P is provided in CSR format; sum_q_total = sum_{k!=l} q_{kl} may be
// precomputed (> 0) or will be computed here in O(n^2) if not available.
template <typename T>
T compute_kl_divergence(da_int n, da_int d, const std::vector<da_int> &row_ptr,
                        const std::vector<da_int> &col_idx, const std::vector<T> &p_vals,
                        const std::vector<T> &emb, T sum_q_total, std::vector<T> &work) {
    if ((da_int)work.size() < n)
        work.resize(n);
    if (sum_q_total <= (T)0) {
#pragma omp parallel for schedule(static) default(none) shared(n, d, emb, work)
        for (da_int i = 0; i < n; ++i) {
            const T *yi = &emb[i * d];
            T local = (T)0;
            for (da_int j = i + 1; j < n; ++j) {
                const T *yj = &emb[j * d];
                T dist2 = (T)0;
                for (da_int k = 0; k < d; ++k) {
                    const T diff = yi[k] - yj[k];
                    dist2 += diff * diff;
                }
                local += (T)2 / ((T)1 + dist2);
            }
            work[i] = local;
        }
        sum_q_total = (T)0;
        for (da_int i = 0; i < n; ++i)
            sum_q_total += work[i];
    }
    if (sum_q_total <= (T)0)
        sum_q_total = (T)1;
    const T eps = std::numeric_limits<T>::epsilon();
#pragma omp parallel for schedule(static) default(none)                                  \
    shared(n, d, row_ptr, col_idx, p_vals, emb, sum_q_total, eps, work)
    for (da_int i = 0; i < n; ++i) {
        const T *yi = &emb[i * d];
        T local = (T)0;
        for (da_int idx = row_ptr[i]; idx < row_ptr[i + 1]; ++idx) {
            const da_int j = col_idx[idx];
            const T *yj = &emb[j * d];
            T dist2 = (T)0;
            for (da_int k = 0; k < d; ++k) {
                const T diff = yi[k] - yj[k];
                dist2 += diff * diff;
            }
            const T Pij = std::max(p_vals[idx], eps);
            const T Qij = std::max((T)1 / (((T)1 + dist2) * sum_q_total), eps);
            local += Pij * std::log(Pij / Qij);
        }
        work[i] = local;
    }
    T kl = (T)0;
    for (da_int i = 0; i < n; ++i)
        kl += work[i];
    return kl;
}

template <typename T>
void update_embedding(da_int n, da_int d, T momentum, T learning_rate,
                      std::vector<T> &grad, std::vector<T> &iY, std::vector<T> &gains,
                      std::vector<T> &emb) {
    const da_int nd = n * d;
#pragma omp parallel for schedule(static)
    for (da_int idx = 0; idx < nd; ++idx) {
        if (iY[idx] * grad[idx] < (T)0)
            gains[idx] += (T)0.2;
        else
            gains[idx] *= (T)0.8;
        gains[idx] = std::max(gains[idx], (T)0.01);
        iY[idx] = momentum * iY[idx] - learning_rate * gains[idx] * grad[idx];
        emb[idx] += iY[idx];
    }
}

template <typename T, int8_t D>
void compute_repulsive_forces(BarnesHutTree<T, D> &tree, da_int n, T *repulsive,
                              T &sum_q_total, std::vector<T> &thread_sum_q) {
    const da_int *pt_order = tree.sorted_indices.data();
    const da_int n_batches = (da_int)(std::ceil((double)n / 4));
    // Per-batch accumulation (rather than per-thread) makes the serial sum
    // order-deterministic and independent of the OMP thread count.
    if ((da_int)thread_sum_q.size() < n_batches)
        thread_sum_q.resize(n_batches);

#pragma omp parallel for schedule(guided, 32) default(none)                              \
    shared(n_batches, n, tree, pt_order, repulsive, thread_sum_q)
    for (da_int batch = 0; batch < n_batches; ++batch) {
        int32_t node_stack[BarnesHutTree<T, D>::STACK_SIZE];
        int32_t depth_stack[BarnesHutTree<T, D>::STACK_SIZE];
        const da_int k = batch * 4;
        const da_int count = std::min((da_int)4, n - k);
        da_int pidx[4] = {};
        for (da_int b = 0; b < count; ++b)
            pidx[b] = pt_order[k + b];
        // Pad the last batch to make it a multiple of 4
        for (da_int b = count; b < 4; ++b)
            pidx[b] = pidx[0];
        T force[4][D], sq[4];
        // Walk the tree for 4 points simultaneously
        tree.compute_repulsive_batch4(pidx, force, sq, node_stack, depth_stack);
        T local = (T)0;
        for (da_int b = 0; b < count; ++b) {
            const da_int i = pidx[b];
            for (da_int dd = 0; dd < D; ++dd)
                repulsive[i * D + dd] = force[b][dd];
            local += sq[b];
        }
        thread_sum_q[batch] = local;
    }
    sum_q_total = (T)0;
    for (da_int batch = 0; batch < n_batches; ++batch)
        sum_q_total += thread_sum_q[batch];

    if (sum_q_total == (T)0)
        sum_q_total = (T)1;
}

template <typename T>
da_status tsne<T>::gradient_descent(T learning_rate, T early_exaggeration, T theta,
                                    da_int n_iter_without_progress, T min_grad_norm) {
    const da_int d = n_components;
    if (d == 1)
        return gradient_descent_impl<1>(learning_rate, early_exaggeration, theta,
                                        n_iter_without_progress, min_grad_norm);
    else if (d == 2)
        return gradient_descent_impl<2>(learning_rate, early_exaggeration, theta,
                                        n_iter_without_progress, min_grad_norm);
    else
        return gradient_descent_impl<3>(learning_rate, early_exaggeration, theta,
                                        n_iter_without_progress, min_grad_norm);
}

template <typename T>
template <int8_t D>
da_status tsne<T>::gradient_descent_impl(T learning_rate, T early_exaggeration, T theta,
                                         da_int n_iter_without_progress,
                                         T min_grad_norm) {
    const da_int n = n_samples;
    constexpr da_int d = D;
    const da_int early_iters = std::min<da_int>(250, max_iter);
    constexpr da_int mom_switch_iter = 250;
    const da_int check_interval = 50;

    // Allocate optimization state and per-thread workspaces
    std::vector<T> grad, repulsive, thread_work;
    const da_int state_size = n * d;
    try {
        if ((da_int)iY.size() != state_size)
            iY.assign(state_size, (T)0); // Momentum
        if ((da_int)gains.size() != state_size)
            gains.assign(state_size, (T)1); // Adaptive gains
        grad.resize(state_size, (T)0);      // Gradient
        repulsive.resize(state_size, (T)0); // Repulsive forces
        thread_work.resize(n, (T)0);        // Per-i reduction workspace
    } catch (std::bad_alloc &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error.");
    }

    // Barnes-Hut tree (memory reused across iterations)
    std::unique_ptr<BarnesHutTree<T, D>> tree;
    if (theta > (T)0)
        tree = std::make_unique<BarnesHutTree<T, D>>(embedding.data(), n, theta);

    // For exact mode: dense P avoids CSR indirection in the O(n²) gradient loop.
    // P_dense[i*n+i] == 0, so the j-loop is branch-free (j==i contributes 0).
    std::vector<T> P_dense;
    if (theta == (T)0) {
        try {
            P_dense.resize(n * n, (T)0);
        } catch (std::bad_alloc &) {
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation error.");
        }
        for (da_int i = 0; i < n; ++i)
            for (da_int idx = P_row_ptr[i]; idx < P_row_ptr[i + 1]; ++idx)
                P_dense[i * n + P_col_idx[idx]] = P_values[idx];
    }

    // Main optimization loop
    T best_error = std::numeric_limits<T>::infinity();
    da_int best_iter = 0;
    T sum_q_total = 0;
    for (da_int iter = 0; iter < max_iter; ++iter) {
        const da_int effective_iter = iter + start_iter;
        T exaggeration = (effective_iter < early_iters) ? early_exaggeration : (T)1;

        if (effective_iter == early_iters) {
            da_std::fill(gains.begin(), gains.end(), (T)1);
            da_std::fill(iY.begin(), iY.end(), (T)0);
        }

        sum_q_total = 0;
        T error = std::numeric_limits<T>::infinity();
        const bool check_convergence =
            (((iter + 1) % check_interval) == 0) || (iter == max_iter - 1);
        const bool in_early_phase = (effective_iter < early_iters);

        if (theta > 0) {
            // Barnes-Hut approximation: O(n log n)
            try {
                tree->build();
            } catch (std::bad_alloc &) {
                return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                                "Memory allocation error.");
            }
            da_tsne::compute_repulsive_forces(*tree, n, repulsive.data(), sum_q_total,
                                              thread_work);

            da_tsne::compute_attractive_forces(
                n, d, exaggeration, P_row_ptr, P_col_idx, P_values, embedding, repulsive,
                sum_q_total, grad, attractive_force_kernel_fn);

        } else {
            // Exact O(n²): fused attractive + repulsive, fully parallel.
            // Pass 1 – sum_q via rectangular j=[0,n), self-pair subtracted.
#pragma omp parallel for schedule(static) default(none) shared(n, embedding, thread_work)
            for (da_int i = 0; i < n; ++i) {
                const T *yi = &embedding[i * d];
                T partial = (T)0;
                for (da_int j = 0; j < n; ++j) {
                    T dist2 = (T)0;
                    for (da_int k = 0; k < d; ++k) {
                        const T diff = yi[k] - embedding[j * d + k];
                        dist2 += diff * diff;
                    }
                    partial += (T)1 / ((T)1 + dist2);
                }
                thread_work[i] = partial - (T)1;
            }
            sum_q_total = (T)0;
            for (da_int i = 0; i < n; ++i)
                sum_q_total += thread_work[i];
            if (sum_q_total <= (T)0)
                sum_q_total = (T)1;

            // Pass 2 – fused gradient
            // grad_i = 4 * sum_j (P_ij - Q_ij) * q_ij * (y_i - y_j)
            const T inv_sum_q = (T)1 / sum_q_total;
#pragma omp parallel for schedule(static) default(none)                                  \
    shared(n, embedding, P_dense, grad, exaggeration, inv_sum_q)
            for (da_int i = 0; i < n; ++i) {
                T *grad_i = &grad[i * d];
                const T *P_row = &P_dense[i * n];
                const T *yi = &embedding[i * d];
                for (da_int k = 0; k < d; ++k)
                    grad_i[k] = (T)0;
                for (da_int j = 0; j < n; ++j) {
                    T dist2 = (T)0;
                    for (da_int k = 0; k < d; ++k) {
                        const T diff = yi[k] - embedding[j * d + k];
                        dist2 += diff * diff;
                    }
                    const T q = (T)1 / ((T)1 + dist2);
                    const T f = (P_row[j] * exaggeration - q * inv_sum_q) * q;
                    for (da_int k = 0; k < d; ++k)
                        grad_i[k] += f * (yi[k] - embedding[j * d + k]);
                }
                for (da_int k = 0; k < d; ++k)
                    grad_i[k] *= (T)4;
            }
        }

        // Reset best_error tracking at the early/late phase boundary
        if (effective_iter == early_iters) {
            best_error = std::numeric_limits<T>::infinity();
            best_iter = iter;
        }

        if (check_convergence) {
#pragma omp parallel for schedule(static) default(none)                                  \
    shared(n, grad, gains, thread_work)
            for (da_int i = 0; i < n; ++i) {
                const da_int base = i * d;
                T local = (T)0;
                for (da_int k = 0; k < d; ++k) {
                    const T g = grad[base + k] * gains[base + k];
                    local += g * g;
                }
                thread_work[i] = local;
            }
            T grad_norm_sq = (T)0;
            for (da_int i = 0; i < n; ++i)
                grad_norm_sq += thread_work[i];
            if (min_grad_norm > 0 && std::sqrt(grad_norm_sq) <= min_grad_norm) {
                n_iter_performed = iter + 1;
                break;
            }

            if (!in_early_phase && n_iter_without_progress > 0) {
                error =
                    da_tsne::compute_kl_divergence(n, d, P_row_ptr, P_col_idx, P_values,
                                                   embedding, sum_q_total, thread_work);
                if (error < best_error) {
                    best_error = error;
                    best_iter = iter;
                } else if (iter - best_iter > n_iter_without_progress) {
                    n_iter_performed = iter + 1;
                    break;
                }
            }
        }
        const T momentum = (effective_iter < mom_switch_iter) ? (T)0.5 : (T)0.8;
        da_tsne::update_embedding(n, d, momentum, learning_rate, grad, iY, gains,
                                  embedding);
    }

    // n_iter_performed is set only in early convergence path
    if (n_iter_performed == 0)
        n_iter_performed = max_iter;

    kl_divergence = da_tsne::compute_kl_divergence(n, d, P_row_ptr, P_col_idx, P_values,
                                                   embedding, (T)0, thread_work);

    return da_status_success;
}

/* Iterative refinement: run gradient descent in lower precision, then use the
   resulting embedding as the starting point for the working precision phase. */
template <> da_status tsne<double>::lower_precision_init() {

    da_int lp_max_iter = 200;
    double lp_min_grad_norm = 1.0e-4;
    this->opts.get("low precision max_iter", lp_max_iter);
    this->opts.get("low precision min_grad_norm", lp_min_grad_norm);

    const da_int nnz = (da_int)P_values.size();
    const da_int nd = n_samples * n_components;

    // Allocate and convert P_values to float; copy integer CSR arrays
    std::vector<float> P_values_lp;
    std::vector<da_int> P_row_ptr_copy;
    std::vector<da_int> P_col_idx_copy;
    try {
        P_values_lp.resize(nnz);
        P_row_ptr_copy = P_row_ptr;
        P_col_idx_copy = P_col_idx;
    } catch (std::bad_alloc &) {
        return da_error(this->err, da_status_memory_error,
                        "Memory allocation error in lower_precision_init.");
    }
    for (da_int i = 0; i < nnz; ++i)
        P_values_lp[i] = static_cast<float>(P_values[i]);

    // Handle embedding for LP phase: if user supplied one, convert it;
    // otherwise pass an empty vector and let the LP phase initialize its own.
    std::vector<float> embedding_lp;
    if (has_supplied_embedding) {
        try {
            embedding_lp.resize(nd);
        } catch (std::bad_alloc &) {
            return da_error(this->err, da_status_memory_error,
                            "Memory allocation error in lower_precision_init.");
        }

        for (da_int i = 0; i < nd; ++i)
            embedding_lp[i] = static_cast<float>(supplied_embedding[i]);
    }

    // Create a float t-SNE instance via bypass constructor
    tsne<float> lp_tsne(
        *this->err, n_samples, n_features, n_components, lp_max_iter,
        static_cast<float>(learning_rate), static_cast<float>(early_exaggeration),
        static_cast<float>(theta), static_cast<float>(lp_min_grad_norm),
        n_iter_without_progress, std::move(P_row_ptr_copy), std::move(P_col_idx_copy),
        std::move(P_values_lp), std::move(embedding_lp), init_method, seed);

    // If no user-supplied embedding, the LP instance needs X data (in float)
    // so that it can initialize its own embedding (PCA or random).
    if (!has_supplied_embedding) {
        const da_int n_total = n_samples * n_features;
        try {
            lp_tsne.X_copy.resize(n_total);
        } catch (std::bad_alloc &) {
            return da_error(this->err, da_status_memory_error,
                            "Memory allocation error in lower_precision_init.");
        }
        for (da_int i = 0; i < n_total; ++i)
            lp_tsne.X_copy[i] = static_cast<float>(X[i]);
        lp_tsne.X = lp_tsne.X_copy.data();
    }

    // Run the low precision t-SNE
    da_status status = lp_tsne.compute();
    if (status != da_status_success)
        return status;

    lp_n_iter = lp_tsne.n_iter_performed;

    // Copy the refined embedding back to double precision
    try {
        embedding.resize(nd);
        iY.resize(nd);
        gains.resize(nd);
    } catch (std::bad_alloc &) {
        return da_error(this->err, da_status_memory_error,
                        "Memory allocation error in lower_precision_init.");
    }
    for (da_int i = 0; i < nd; ++i) {
        embedding[i] = static_cast<double>(lp_tsne.embedding[i]);
        iY[i] = static_cast<double>(lp_tsne.iY[i]);
        gains[i] = static_cast<double>(lp_tsne.gains[i]);
    }
    start_iter = lp_n_iter;
    return da_status_success;
}

template <> da_status tsne<float>::lower_precision_init() {
    // No lower precision available for float yet (future: half precision)
    return da_error(this->err, da_status_invalid_option,
                    "Mixed precision is not supported for single precision data. "
                    "It is only available when the working precision is double.");
}

template <typename T> da_status tsne<T>::compute() {
    if (!initdone)
        return da_error(this->err, da_status_no_data,
                        "No data has been passed to the handle. Please call "
                        "da_tsne_set_data_s or da_tsne_set_data_d.");

    T perplexity = (T)30;

    if (check_options) {
        // Needed to not falsely warm start on multiple calls on the same handle
        refresh();
        // Read options
        da_int n_comp = 2;
        da_int max_iter_opt = 1000;

        this->opts.get("n_components", n_comp);
        this->opts.get("max_iter", max_iter_opt);
        this->opts.get("seed", seed);
        this->opts.get("perplexity", perplexity);
        this->opts.get("learning rate", learning_rate);
        this->opts.get("early exaggeration", early_exaggeration);
        this->opts.get("theta", theta);
        this->opts.get("init", init_method);
        this->opts.get("min_grad_norm", min_grad_norm);
        this->opts.get("n_iter_without_progress", n_iter_without_progress);

        std::string opt_mp;
        da_int int_mp;
        this->opts.get("mixed precision", opt_mp, int_mp);
        use_mixed_precision = (int_mp == 1);

        // Auto learning rate: max(N / early_exaggeration / 4, 50)
        if (learning_rate <= (T)0) {
            learning_rate = std::max(n_samples / early_exaggeration / (T)4, (T)50);
        }

        n_components = n_comp;
        max_iter = max_iter_opt;
    }

    // At least 4 samples per thread: parallelism needs to be more thoroughly checked
    da_int n_threads = omp_get_max_threads();
    da_int thread_limit = std::min(n_threads, std::max<da_int>(1, n_samples / 4));
    omp_set_num_threads(thread_limit);

    n_iter_performed = 0;

    assign_attractive_force_kernel();

    // Execute t-SNE pipeline
    // Skip affinity computation if P matrix is already populated (bypass constructor)
    if (P_row_ptr.empty()) {
        bool use_exact = (theta == (T)0);
        da_status status =
            da_tsne::compute_affinities(perplexity, use_exact, n_samples, n_features, X,
                                        this->err, P_row_ptr, P_col_idx, P_values);
        if (status != da_status_success) {
            omp_set_num_threads(n_threads); // LCOV_EXCL_LINE
            return status;                  // LCOV_EXCL_LINE
        }
    }

    // If mixed precision is enabled, run a low-precision phase first.
    // This populates the embedding with the LP result, so initialize_embedding is skipped.
    if (use_mixed_precision) {
        da_status status = lower_precision_init();
        if (status != da_status_success) {
            omp_set_num_threads(n_threads);
            return status;
        }
    }

    // Skip embedding initialization if embedding is already populated
    // (by lower_precision_init above, or by the bypass constructor)
    if (embedding.empty()) {
        da_status status = initialize_embedding(init_method, seed);
        if (status != da_status_success) {
            omp_set_num_threads(n_threads);
            return status;
        }
    }

    da_status status = gradient_descent(learning_rate, early_exaggeration, theta,
                                        n_iter_without_progress, min_grad_norm);
    if (status != da_status_success) {
        omp_set_num_threads(n_threads); // LCOV_EXCL_LINE
        return status;                  // LCOV_EXCL_LINE
    }

    model_trained = true;
    omp_set_num_threads(n_threads);
    return da_status_success;
}

template <typename T>
da_status tsne<T>::get_result(da_result query, da_int *dim, T *result) {
    if (!model_trained)
        return da_warn(this->err, da_status_no_data,
                       "t-SNE has not yet been computed. Please call da_tsne_compute_s "
                       "or da_tsne_compute_d before extracting results.");
    switch (query) {
    case da_rinfo: {
        da_int rinfo_size = 6;
        if (*dim < rinfo_size) {
            *dim = rinfo_size;
            return da_status_invalid_array_dimension;
        }
        result[0] = n_samples;
        result[1] = n_features;
        result[2] = n_components;
        result[3] = n_iter_performed;
        result[4] = kl_divergence;
        result[5] = (T)lp_n_iter;
        return da_status_success;
    }
    case da_tsne_embedding: {
        da_int required = n_samples * n_components;
        if (*dim < required) {
            *dim = required;
            return da_status_invalid_array_dimension;
        }
        if (this->order == row_major) {
            da_std::copy(embedding.begin(), embedding.end(), result);
        } else {
            ARCH::da_utils::copy_transpose_2D_array_row_to_column_major(
                n_samples, n_components, embedding.data(), n_components, result,
                n_samples);
        }
        return da_status_success;
    }
    default:
        return da_error(this->err, da_status_unknown_query,
                        "The requested result is not available for t-SNE.");
    }
}

template <typename T>
da_status tsne<T>::get_result(da_result query, da_int *dim, da_int *result) {
    (void)query;
    (void)dim;
    (void)result;
    return da_error(this->err, da_status_unknown_query,
                    "There are no integer results available for this API.");
}

template class tsne<double>;
template class tsne<float>;

template void compute_row_probabilities<double>(const double *, da_int, double, double *);
template void compute_row_probabilities<float>(const float *, da_int, float, float *);
template da_status compute_affinities<float>(float, bool, da_int, da_int, const float *,
                                             da_errors::da_error_t *,
                                             std::vector<da_int> &, std::vector<da_int> &,
                                             std::vector<float> &);
template da_status compute_affinities<double>(double, bool, da_int, da_int,
                                              const double *, da_errors::da_error_t *,
                                              std::vector<da_int> &,
                                              std::vector<da_int> &,
                                              std::vector<double> &);
template float compute_kl_divergence<float>(da_int, da_int, const std::vector<da_int> &,
                                            const std::vector<da_int> &,
                                            const std::vector<float> &,
                                            const std::vector<float> &, float,
                                            std::vector<float> &);
template double compute_kl_divergence<double>(da_int, da_int, const std::vector<da_int> &,
                                              const std::vector<da_int> &,
                                              const std::vector<double> &,
                                              const std::vector<double> &, double,
                                              std::vector<double> &);
template void compute_attractive_forces<float>(
    da_int, da_int, float, const std::vector<da_int> &, const std::vector<da_int> &,
    const std::vector<float> &, const std::vector<float> &, const std::vector<float> &,
    float, std::vector<float> &, attractive_forces_kernel_fn<float>);
template void compute_attractive_forces<double>(
    da_int, da_int, double, const std::vector<da_int> &, const std::vector<da_int> &,
    const std::vector<double> &, const std::vector<double> &, const std::vector<double> &,
    double, std::vector<double> &, attractive_forces_kernel_fn<double>);
template void update_embedding<float>(da_int, da_int, float, float, std::vector<float> &,
                                      std::vector<float> &, std::vector<float> &,
                                      std::vector<float> &);
template void update_embedding<double>(da_int, da_int, double, double,
                                       std::vector<double> &, std::vector<double> &,
                                       std::vector<double> &, std::vector<double> &);

template da_status symmetrize_to_csr<double>(da_int, da_int, const da_int *,
                                             const double *, std::vector<da_int> &,
                                             std::vector<da_int> &,
                                             std::vector<double> &);
template da_status symmetrize_to_csr<float>(da_int, da_int, const da_int *, const float *,
                                            std::vector<da_int> &, std::vector<da_int> &,
                                            std::vector<float> &);

template void compute_repulsive_forces<float, 1>(BarnesHutTree<float, 1> &, da_int,
                                                 float *, float &, std::vector<float> &);
template void compute_repulsive_forces<float, 2>(BarnesHutTree<float, 2> &, da_int,
                                                 float *, float &, std::vector<float> &);
template void compute_repulsive_forces<float, 3>(BarnesHutTree<float, 3> &, da_int,
                                                 float *, float &, std::vector<float> &);
template void compute_repulsive_forces<double, 1>(BarnesHutTree<double, 1> &, da_int,
                                                  double *, double &,
                                                  std::vector<double> &);
template void compute_repulsive_forces<double, 2>(BarnesHutTree<double, 2> &, da_int,
                                                  double *, double &,
                                                  std::vector<double> &);
template void compute_repulsive_forces<double, 3>(BarnesHutTree<double, 3> &, da_int,
                                                  double *, double &,
                                                  std::vector<double> &);

} // namespace da_tsne

} // namespace ARCH
