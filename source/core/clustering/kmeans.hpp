/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
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

#ifndef KMEANS_HPP
#define KMEANS_HPP

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "aoclda.h"
#include "aoclda_kmeans.h"
#include "basic_handle.hpp"
#include "callbacks.hpp"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "euclidean_distance.hpp"
#include "kmeans_options.hpp"
#include "kmeans_types.hpp"
#include "lapack_templates.hpp"
#include "options.hpp"
#include <iostream>
//#include <omp.h>
#include <random>
#include <string>

namespace da_kmeans {

/* k-means class */
template <typename T> class da_kmeans : public basic_handle<T> {
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

    // Pointer to error trace
    da_errors::da_error_t *err = nullptr;

    // User's data
    const T *A;
    const T *C;
    da_int lda = 0;
    da_int ldc = 0;

    //double t_iteration = 0.0;
    //double t_euclidean_it = 0.0;
    //double t_centre_half_distances = 0.0;
    //double t_init_bounds = 0.0;
    //double t_assign = 0.0;
    //double t_update = 0.0;
    //double t_loop361 = 0.0;
    //double t_assign1 = 0.0;
    //double t_assign2 = 0.0;
    //double t_assign3 = 0.0;
    //double t_assign4 = 0.0;
    //double t_assign5 = 0.0;

    // Maximum size of data blocks for elkan, lloyd and macqueen algorithms
    da_int max_block_size = 0;
    da_int n_blocks = 0;
    da_int block_rem = 0;

    // Leading dimension of workcs1
    da_int ldworkcs1 = 0;

    // This will point to the function to perform k-means iterations depending on algorithm choice
    void (da_kmeans<T>::*single_iteration)(bool);

    // Arrays used internally, and to store results
    T best_inertia = 0.0, current_inertia = 0.0; // Inertia
    std::vector<T> workcc1, workcs1, works1, works2, works3, works4, works5, workc1,
        workc2, workc3;
    std::vector<da_int> work_int1, work_int2, work_int3, work_int4;

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

    // Lloyd algorithm functions, including various unrolled versions of the blocked part of the lloyd iteration

    void init_lloyd();

    void lloyd_iteration(bool update_centres);

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

    void elkan_iteration(bool update_centres);

    void elkan_iteration_assign_block(bool update_centres, da_int block_size,
                                      const T *data, da_int lddata, T *u_bounds,
                                      T *l_bounds, da_int ldl_bounds, da_int *old_labels,
                                      da_int *new_labels, T *centre_half_distances,
                                      T *next_centre_distances, da_int *cluster_counts);

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

    void (da_kmeans<T>::*initialize_algorithm)();

    void (da_kmeans<T>::*lloyd_iteration_block)(bool, da_int, const T *, da_int, T *, T *,
                                                T *, da_int *, da_int *, T *, da_int);

    void (da_kmeans<T>::*predict_block)(bool, da_int, const T *, da_int, T *, T *, T *,
                                        da_int *, da_int *, T *, da_int);

    void (da_kmeans<T>::*elkan_iteration_update_block)(da_int, T *, da_int, T *, T *,
                                                       da_int *);

    // Macqueen algorithm functions

    void init_macqueen();

    void init_macqueen_block(da_int block_size, da_int block_index);

    void macqueen_iteration(bool update_centres);

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

    void get_blocking_scheme(da_int n_samples);

  public:
    da_options::OptionRegistry opts;

    da_kmeans(da_errors::da_error_t &err) {
        this->err = &err;
        register_kmeans_options<T>(opts);
    };

    da_status get_result(da_result query, da_int *dim, T *result) {
        // Don't return anything if k-means has not been computed
        if (!iscomputed) {
            return da_warn(err, da_status_no_data,
                           "k-means clustering has not yet been computed. Please call "
                           "da_kmeans_compute_s "
                           "or da_kmeans_compute_d before extracting results.");
        }

        da_int rinfo_size = 5;

        switch (query) {
        case da_result::da_rinfo:
            if (*dim < rinfo_size) {
                *dim = rinfo_size;
                return da_warn(err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(rinfo_size) + ".");
            }
            result[0] = (T)n_samples;
            result[1] = (T)n_features;
            result[2] = (T)n_clusters;
            result[3] = (T)best_n_iter;
            result[4] = best_inertia;
            break;
        case da_result::da_kmeans_cluster_centres:
            if (*dim < n_clusters * n_features) {
                *dim = n_clusters * n_features;
                return da_warn(err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(n_clusters * n_features) + ".");
            }
            for (da_int i = 0; i < n_clusters * n_features; i++)
                result[i] = (*best_cluster_centres)[i];
            break;
        default:
            return da_warn(err, da_status_unknown_query,
                           "The requested result could not be found.");
        }
        return da_status_success;
    };

    da_status get_result(da_result query, da_int *dim, da_int *result) {
        // Don't return anything if k-means has not been computed
        if (!iscomputed) {
            return da_warn(err, da_status_no_data,
                           "k-means clustering has not yet been computed. Please call "
                           "da_kmeans_compute_s "
                           "or da_kmeans_compute_d before extracting results.");
        }

        switch (query) {
        case da_result::da_kmeans_labels:
            if (*dim < n_samples) {
                *dim = n_samples;
                return da_warn(err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(n_samples) + ".");
            }
            for (da_int i = 0; i < n_samples; i++)
                result[i] = (*best_labels)[i];
            break;
        default:
            return da_warn(err, da_status_unknown_query,
                           "The requested result could not be found.");
        }

        return da_status_success;
    };

    /* Store details about user's data matrix in preparation for k-means computation */
    da_status set_data(da_int n_samples, da_int n_features, const T *A_in,
                       da_int lda_in) {
        // Check for illegal arguments and function calls
        if (n_samples < 1)
            return da_error(
                err, da_status_invalid_input,
                "The function was called with n_samples = " + std::to_string(n_samples) +
                    ". Constraint: n_samples >= 1.");
        if (n_features < 1)
            return da_error(err, da_status_invalid_input,
                            "The function was called with n_features = " +
                                std::to_string(n_features) +
                                ". Constraint: n_features >= 1.");
        if (lda_in < n_samples)
            return da_error(
                err, da_status_invalid_input,
                "The function was called with n_samples = " + std::to_string(n_samples) +
                    " and lda = " + std::to_string(lda) +
                    ". Constraint: lda >= n_samples.");
        if (A_in == nullptr)
            return da_error(err, da_status_invalid_pointer, "The array A is null.");

        // Store dimensions of A
        this->n_samples = n_samples;
        this->n_features = n_features;
        this->lda = lda_in;

        // Pointer to user's data - no need to copy it
        A = A_in;

        // Record that initialization is complete but computation has not yet been performed
        initdone = true;
        iscomputed = false;

        // Now that we have a data matrix we can re-register the n_clusters option with new constraints
        da_int temp_clusters;
        opts.get("n_clusters", temp_clusters);

        reregister_kmeans_option<T>(opts, n_samples);

        opts.set("n_clusters", std::min(temp_clusters, n_samples));

        if (temp_clusters > n_samples)
            return da_warn(err, da_status_incompatible_options,
                           "The requested number of clusters has been decreased from " +
                               std::to_string(temp_clusters) + " to " +
                               std::to_string(n_samples) +
                               " due to the size of the data array.");

        return da_status_success;
    }

    da_status set_init_centres(const T *C_in, da_int ldc_in) {

        if (initdone == false)
            return da_error(err, da_status_no_data,
                            "No data has been passed to the handle. Please call "
                            "da_kmeans_set_data_s or da_kmeans_set_data_d.");

        /* Check for illegal arguments */
        this->opts.get("n_clusters", n_clusters);
        if (ldc_in < n_clusters)
            return da_error(err, da_status_invalid_input,
                            "The function was called ldc = " + std::to_string(ldc_in) +
                                " and n_clusters is currently set to = " +
                                std::to_string(n_clusters) +
                                ". Constraint: ldc >= n_clusters.");

        if (C_in == nullptr)
            return da_error(err, da_status_invalid_pointer, "The array C is null.");

        C = C_in;
        ldc = ldc_in;

        // Record that centres have been set
        centres_supplied = true;

        return da_status_success;
    }

    /* Compute the k-means clusters */
    da_status compute() {

        da_status status = da_status_success;
        //double tt0 = omp_get_wtime();
        if (initdone == false)
            return da_error(err, da_status_no_data,
                            "No data has been passed to the handle. Please call "
                            "da_kmeans_set_data_s or da_kmeans_set_data_d.");

        // Read in options and store in class
        this->opts.get("n_clusters", n_clusters);

        std::string opt_method;
        this->opts.get("initialization method", opt_method, init_method);

        std::string opt_alg;
        this->opts.get("algorithm", opt_alg, algorithm);

        this->opts.get("n_init", n_init);

        this->opts.get("max_iter", max_iter);

        this->opts.get("convergence tolerance", tol);

        this->opts.get("seed", seed);

        // Remove the constraint on n_clusters, in case the user re-uses the handle with different data
        reregister_kmeans_option<T>(opts, std::numeric_limits<da_int>::max());
        opts.set("n_clusters", n_samples);

        // Check for conflicting options
        if (n_init > 1 && init_method == supplied) {
            std::string buff =
                "n_init was set to " + std::to_string(n_init) +
                " but the initialization method was set to 'supplied'. The "
                "k-means algorithm will only be run once.";
            n_init = 1;
            da_warn(err, da_status_incompatible_options, buff);
        }

        if (algorithm == hartigan_wong && (n_clusters == 1 || n_clusters >= n_samples)) {
            return da_error(err, da_status_incompatible_options,
                            "The Hartigan-Wong algorithm requires 1 < k < n_samples.");
        }

        // This can only be triggered if the user re-uses the handle, otherwise the option handling should catch it
        if (n_clusters > n_samples) {
            return da_error(err, da_status_incompatible_options,
                            "n_clusters = " + std::to_string(n_clusters) +
                                ", and n_samples = " + std::to_string(n_samples) +
                                ". Constraint: n_clusters <= n_samples.");
        }

        if (init_method == supplied && centres_supplied == false) {
            return da_error(
                err, da_status_no_data,
                "The initialization method was set to 'supplied' but no initial "
                "centres have been provided.");
        }

        switch (algorithm) {
        case lloyd:
            max_block_size = KMEANS_LLOYD_BLOCK_SIZE;
            initialize_algorithm = &da_kmeans<T>::init_lloyd;
            break;
        case elkan:
            max_block_size = KMEANS_ELKAN_BLOCK_SIZE;
            initialize_algorithm = &da_kmeans<T>::init_elkan;
            break;
        case macqueen:
            max_block_size = KMEANS_MACQUEEN_BLOCK_SIZE;
            initialize_algorithm = &da_kmeans<T>::init_macqueen;
            break;
        default:
            max_block_size = n_samples;
            break;
        }
        max_block_size = std::min(max_block_size, n_samples);

        // Initialize some arrays
        try {
            current_cluster_centres->resize(n_clusters * n_features, 0.0);
            previous_cluster_centres->resize(n_clusters * n_features, 0.0);
            work_int1.resize(n_clusters, 0);
            work_int2.resize(n_samples, 0);
            // Extra bit on workc1 just to enable some padding to be done if loop unrolling occurs
            workc1.resize(n_clusters + 8, 0.0);
            current_labels->resize(n_samples, 0);
            previous_labels->resize(n_samples, 0);
            if (n_init > 1) {
                best_cluster_centres->resize(n_clusters * n_features);
                best_labels->resize(n_samples);
            }
        } catch (std::bad_alloc const &) {
            return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }

        // Ensure the extra padding in workc1 (for vectorization) won't interfere with any computation
        std::fill(workc1.end() - 8, workc1.end(), std::numeric_limits<T>::infinity());

        // Based on what algorithms we are using, allocate the remaining memory
        try {

            switch (algorithm) {
            case elkan:
                workcc1.resize(n_clusters * n_clusters, 0.0);
                workcs1.resize(n_samples * (n_clusters + 8), 0.0);
                works1.resize(n_samples, 0.0);
                break;
            case macqueen:
                workcs1.resize(max_block_size * n_clusters, 0.0);
                workc2.resize(n_clusters, 0.0);
                break;
            case lloyd:
                workcs1.resize(max_block_size * (n_clusters + 8), 0.0);
                works1.resize(n_samples, 0.0);
                break;
            case hartigan_wong:
                works1.resize(n_samples, 0.0);
                workc2.resize(n_clusters, 0.0);
                workc3.resize(n_clusters, 0.0);
                work_int3.resize(n_clusters, 0);
                work_int4.resize(n_clusters, 0);
                break;
            }

            if (init_method == kmeanspp) {
                works1.resize(n_samples, 0.0);
                works2.resize(n_samples, 0.0);
                works3.resize(n_samples, 0.0);
                works4.resize(n_samples, 0.0);
                works5.resize(n_samples, 0.0);
            }

        } catch (std::bad_alloc const &) {
            return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }

        if (centres_supplied && init_method == supplied) {
            // Copy the initial centres matrix into internal matrix buffer
            for (da_int j = 0; j < n_features; j++) {
                for (da_int i = 0; i < n_clusters; i++) {
                    (*current_cluster_centres)[i + j * n_clusters] = C[i + ldc * j];
                }
            }
        }

        // If needed, initialize random number generation
        da_kmeans<T>::initialize_rng();

        // Set the initial best_inertia over all the runs to something large
        best_inertia = std::numeric_limits<T>::infinity();

        // Run k-means algorithm n_init times and select the run with the lowest inertia
        //double tt2 = omp_get_wtime();
        for (da_int run = 0; run < n_init; run++) {

            // Initialize the centres if needed
            da_kmeans<T>::initialize_centres();

            // Perform k-means using current_inertia, current_cluster_centres and current_labels
            da_kmeans<T>::perform_kmeans();

            // Check if it's the best run yet
            if (current_inertia < best_inertia) {
                best_inertia = current_inertia;
                best_n_iter = current_n_iter;
                // If this run hit the maximum number of iterations, a warning is required
                warn_maxit_reached = (converged == 0) ? true : false;
                std::swap(best_cluster_centres, current_cluster_centres);
                std::swap(best_labels, current_labels);
            }
        }

        // Compute the squared norms of the cluster centres in preparation for the predict phase of the algorithm; store in workc1
        for (da_int i = 0; i < n_clusters; i++) {
            workc1[i] = (T)0.0;
        }

        T tmp;
        for (da_int j = 0; j < n_features; j++) {
            for (da_int i = 0; i < n_clusters; i++) {
                tmp = (*best_cluster_centres)[i + j * n_clusters];
                workc1[i] += tmp * tmp;
            }
        }

        iscomputed = true;

        //double tt1 = omp_get_wtime();
        //std::cout << "t_iteration: " << t_iteration << std::endl;
        //std::cout << "t_euclidean_it: " << t_euclidean_it << std::endl;
        //std::cout << "t_centre_half_distances: " << t_centre_half_distances << std::endl;
        //std::cout << "t_init_bounds: " << t_init_bounds << std::endl;
        //std::cout << "t_assign: " << t_assign << std::endl;
        //std::cout << "t_update: " << t_update << std::endl;
        //std::cout << "t_loop361: " << t_loop361 << std::endl;
        //std::cout << "t_assign1: " << t_assign1 << std::endl;
        //std::cout << "t_assign2: " << t_assign2 << std::endl;
        //std::cout << "t_assign3: " << t_assign3 << std::endl;
        //std::cout << "t_assign4: " << t_assign4 << std::endl;
        //std::cout << "t_assign5: " << t_assign5 << std::endl;

        if (warn_maxit_reached)
            return da_warn(err, da_status_maxit,
                           "The maximum number of iterations was reached.");

        return status;
    }

    da_status transform(da_int m_samples, da_int m_features, const T *X, da_int ldx,
                        T *X_transform, da_int ldx_transform) {

        if (!iscomputed) {
            return da_warn(
                err, da_status_no_data,
                "The k-means has not been computed. Please call da_kmeans_compute_s or "
                "da_kmeans_compute_d.");
        }

        /* Check for illegal arguments */
        if (m_samples < 1)
            return da_error(
                err, da_status_invalid_input,
                "The function was called with m_samples = " + std::to_string(m_samples) +
                    ". Constraint: m_samples >= 1.");
        if (m_features != n_features)
            return da_error(err, da_status_invalid_input,
                            "The function was called with m_features = " +
                                std::to_string(m_features) +
                                " but the k-means has been computed with " +
                                std::to_string(n_features) + " features.");
        if (ldx < m_samples)
            return da_error(
                err, da_status_invalid_input,
                "The function was called with m_samples = " + std::to_string(m_samples) +
                    " and ldx = " + std::to_string(ldx) +
                    ". Constraint: ldx >= m_samples.");

        if (ldx_transform < m_samples)
            return da_error(
                err, da_status_invalid_input,
                "The function was called with m_samples = " + std::to_string(m_samples) +
                    " and ldx_transform = " + std::to_string(ldx_transform) +
                    ". Constraint: ldx_transform >= m_samples.");

        if (X == nullptr)
            return da_error(err, da_status_invalid_pointer, "The array X is null.");

        if (X_transform == nullptr)
            return da_error(err, da_status_invalid_pointer,
                            "The array X_transform is null.");

        std::vector<T> x_work;

        try {
            x_work.resize(m_samples);
        } catch (std::bad_alloc const &) {
            return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }

        // Compute m_samples x n_clusters matrix of distances to cluster centres
        euclidean_distance(m_samples, n_clusters, n_features, X, ldx,
                           (*best_cluster_centres).data(), n_clusters, X_transform,
                           ldx_transform, x_work.data(), 2, workc1.data(), 1, false,
                           false);

        return da_status_success;
    }

    da_status predict(da_int k_samples, da_int k_features, const T *Y, da_int ldy,
                      da_int *Y_labels) {

        if (!iscomputed) {
            return da_warn(
                err, da_status_no_data,
                "The k-means has not been computed. Please call da_kmeans_compute_s or "
                "da_kmeans_compute_d.");
        }

        /* Check for illegal arguments */
        if (k_samples < 1)
            return da_error(
                err, da_status_invalid_input,
                "The function was called with k_samples = " + std::to_string(k_samples) +
                    ". Constraint: k_samples >= 1.");
        if (k_features != n_features)
            return da_error(err, da_status_invalid_input,
                            "The function was called with k_features = " +
                                std::to_string(k_features) +
                                " but the k-means has been computed with " +
                                std::to_string(n_features) + " features.");
        if (ldy < k_samples)
            return da_error(
                err, da_status_invalid_input,
                "The function was called with k_samples = " + std::to_string(k_samples) +
                    " and ldy = " + std::to_string(ldy) +
                    ". Constraint: ldy >= k_samples.");

        if (Y == nullptr)
            return da_error(err, da_status_invalid_pointer, "The array Y is null.");

        if (Y_labels == nullptr)
            return da_error(err, da_status_invalid_pointer,
                            "The array Y_labels is null.");

        // Compute nearest cluster centre for each sample in Y; essentially a single blocked step of the Lloyd iteration
        std::vector<T> y_work;

        max_block_size = std::min(KMEANS_LLOYD_BLOCK_SIZE, k_samples);

        try {
            y_work.resize(max_block_size * (n_clusters + 8));
        } catch (std::bad_alloc const &) {
            return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }

        get_blocking_scheme(k_samples);

        da_int ldy_work;
        if (n_clusters < 4) {
            ldy_work = n_clusters + 8;
            predict_block = &da_kmeans<T>::lloyd_iteration_block_no_unroll;
        } else if (n_clusters < 6) {
            ldy_work = max_block_size;
            predict_block = &da_kmeans<T>::lloyd_iteration_block_unroll_4_T;
        } else if (n_clusters < 16) {
            predict_block = &da_kmeans<T>::lloyd_iteration_block_unroll_4;
            ldy_work = n_clusters + 8;
        } else {
            predict_block = &da_kmeans<T>::lloyd_iteration_block_unroll_8;
            ldy_work = n_clusters + 8;
        }

        T *dummy = nullptr;
        da_int *dummy_int = nullptr;
        da_int block_index;
        da_int block_size = max_block_size;
        for (da_int i = 0; i < n_blocks; i++) {
            if (i == n_blocks - 1 && block_rem > 0) {
                block_index = k_samples - block_rem;
                block_size = block_rem;
            } else {
                block_index = i * max_block_size;
            }
            (this->*predict_block)(false, block_size, &Y[block_index], ldy,
                                   (*best_cluster_centres).data(), dummy, workc1.data(),
                                   dummy_int, &Y_labels[block_index], y_work.data(),
                                   ldy_work);
        }

        return da_status_success;
    }
};

} // namespace da_kmeans

#include "kmeans_aux.hpp"

#endif //KMEANS_HPP
