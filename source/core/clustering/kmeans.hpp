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

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "aoclda.h"
#include "basic_handle.hpp"
#include "da_error.hpp"
#include "kmeans_types.hpp"
#include "macros.h"
#include <random>
#include <string>

namespace ARCH {

namespace da_kmeans {

using namespace da_kmeans_types;

/* k-means class */
template <typename T> class kmeans : public basic_handle<T> {
  public:
    ~kmeans();

  private:
    // n x p (samples x features)
    da_int n_samples = 0;
    da_int n_features = 0;

    // Set true when initialization is complete
    bool initdone = false;

    // Set true if set_init_centres is called
    bool centres_supplied = false;

    // Set true when k-means is computed successfully
    bool iscomputed = false;

    // Underlying algorithm
    da_int algorithm = lloyd;

    // Initialization method
    da_int init_method = random_samples;

    // Number of clusters requested
    da_int n_clusters = 1;

    // Number of runs to perform
    da_int n_init = 1;

    // Max iterations
    da_int max_iter = 1;

    // Actual number of iterations performed
    da_int best_n_iter = 0, current_n_iter = 0;

    // Do we need to warn the user that the best run of k-means ended after the maximum number of iterations?
    bool warn_maxit_reached = false;

    // This will be used to record the convergence status of the current/latest k-means run
    da_int converged = 0;

    // Convergence tolerance
    T tol = 1.0;

    // Random number generation
    da_int seed = 0;
    std::mt19937_64 mt_gen;

    // Norm of previous cluster centre array, for use in convergence testing
    T normc = 0.0;

    // User's data
    const T *A = nullptr;
    const T *C = nullptr;
    da_int lda = 0;
    da_int ldc = 0;

    //Utility pointers to column major allocated copies of user's data
    T *A_temp = nullptr;
    T *C_temp = nullptr;

    // Maximum size of data blocks for Elkan, Lloyd and MacQueen algorithms
    da_int max_block_size = 0;
    da_int n_blocks = 0;
    da_int block_rem = 0;

    // Leading dimension of workcs1
    da_int ldworkcs1 = 0;

    // Arrays used internally, and to store results
    T best_inertia = 0.0, current_inertia = 0.0; // Inertia
    std::vector<T> workcc1, workcs1, works1, works2, works3, works4, works5, workc1,
        workc2, workc3, thread_cluster_centres;
    std::vector<da_int> work_int1, work_int2, work_int3, work_int4, cluster_count;

    // For multiple runs we want to use pointers to point to the current best results
    std::unique_ptr<std::vector<T>> best_cluster_centres =
        std::make_unique<std::vector<T>>();
    std::unique_ptr<std::vector<T>> current_cluster_centres =
        std::make_unique<std::vector<T>>();
    std::unique_ptr<std::vector<T>> previous_cluster_centres =
        std::make_unique<std::vector<T>>();
    std::unique_ptr<std::vector<da_int>> best_labels =
        std::make_unique<std::vector<da_int>>();
    std::unique_ptr<std::vector<da_int>> current_labels =
        std::make_unique<std::vector<da_int>>();
    std::unique_ptr<std::vector<da_int>> previous_labels =
        std::make_unique<std::vector<da_int>>();

    // Lloyd algorithm functions, including various unrolled versions of the blocked part of the Lloyd iteration

    void init_lloyd();

    void lloyd_iteration(bool update_centres, da_int n_threads);

    void lloyd_iteration_block_no_unroll(bool update_centres, da_int block_size,
                                         const T *data, da_int lddata, T *cluster_centres,
                                         T *new_cluster_centres, T *centre_norms,
                                         da_int *cluster_count, da_int *labels, T *work,
                                         da_int ldwork);

    void lloyd_iteration_block_unroll_2(bool update_centres, da_int block_size,
                                        const T *data, da_int lddata, T *cluster_centres,
                                        T *new_cluster_centres, T *centre_norms,
                                        da_int *cluster_count, da_int *labels, T *work,
                                        da_int ldwork);

    void lloyd_iteration_block_unroll_4_T(bool update_centres, da_int block_size,
                                          const T *data, da_int lddata,
                                          T *cluster_centres, T *new_cluster_centres,
                                          T *centre_norms, da_int *cluster_count,
                                          da_int *labels, T *work, da_int ldwork);

    void lloyd_iteration_block_unroll_4(bool update_centres, da_int block_size,
                                        const T *data, da_int lddata, T *cluster_centres,
                                        T *new_cluster_centres, T *centre_norms,
                                        da_int *cluster_count, da_int *labels, T *work,
                                        da_int ldwork);

    void lloyd_iteration_block_unroll_8(bool update_centres, da_int block_size,
                                        const T *data, da_int lddata, T *cluster_centres,
                                        T *new_cluster_centres, T *centre_norms,
                                        da_int *cluster_count, da_int *labels, T *work,
                                        da_int ldwork);

    // Elkan algorithm functions, including various unrolled versions of the blocked part of the iteration

    void init_elkan();

    void init_elkan_bounds();

    void elkan_iteration(bool update_centres, da_int n_threads);

    void elkan_iteration_assign_block(bool update_centres, da_int block_size,
                                      const T *data, da_int lddata,
                                      T *old_cluster_centres, T *new_cluster_centres,
                                      T *u_bounds, T *l_bounds, da_int ldl_bounds,
                                      da_int *old_labels, da_int *new_labels,
                                      T *centre_half_distances, T *next_centre_distances,
                                      da_int *cluster_counts);

    void elkan_iteration_update_block_no_unroll(da_int block_size, T *l_bound,
                                                da_int ldl_bound, T *u_bound,
                                                T *centre_shift, da_int *labels);

    void elkan_iteration_update_block_unroll_4(da_int block_size, T *l_bound,
                                               da_int ldl_bound, T *u_bound,
                                               T *centre_shift, da_int *labels);

    void elkan_iteration_update_block_unroll_8(da_int block_size, T *l_bound,
                                               da_int ldl_bound, T *u_bound,
                                               T *centre_shift, da_int *labels);

    // Function pointers which will be set when the algorithm has been chosen

    void (kmeans<T>::*single_iteration)(bool, da_int);

    void (kmeans<T>::*initialize_algorithm)();

    void (kmeans<T>::*lloyd_iteration_block)(bool, da_int, const T *, da_int, T *, T *,
                                             T *, da_int *, da_int *, T *, da_int);

    void (kmeans<T>::*predict_block)(bool, da_int, const T *, da_int, T *, T *, T *,
                                     da_int *, da_int *, T *, da_int);

    void (kmeans<T>::*elkan_iteration_update_block)(da_int, T *, da_int, T *, T *,
                                                    da_int *);

    // MacQueen algorithm functions

    void init_macqueen();

    void init_macqueen_block(da_int block_size, da_int block_index);

    void macqueen_iteration(bool update_centres, da_int n_threads);

    // Miscellaneous functions and functions used by multiple algorithms

    void initialize_centres();

    void initialize_rng();

    void perform_kmeans();

    da_int convergence_test();

    void compute_current_inertia();

    void compute_centre_half_distances();

    void compute_centre_shift();

    void scale_current_cluster_centres();

    void kmeans_plusplus();

    void perform_hartigan_wong();

  public:
    kmeans(da_errors::da_error_t &err);

    da_status get_result(da_result query, da_int *dim, T *result);

    da_status get_result(da_result query, da_int *dim, da_int *result);

    /* Store details about user's data matrix in preparation for k-means computation */
    da_status set_data(da_int n_samples, da_int n_features, const T *A_in, da_int lda_in);

    da_status set_init_centres(const T *C_in, da_int ldc_in);

    /* Compute the k-means clusters */
    da_status compute();

    da_status transform(da_int m_samples, da_int m_features, const T *X, da_int ldx,
                        T *X_transform, da_int ldx_transform);

    da_status predict(da_int k_samples, da_int k_features, const T *Y, da_int ldy,
                      da_int *Y_labels);

    void refresh();
};

} // namespace da_kmeans

} // namespace ARCH