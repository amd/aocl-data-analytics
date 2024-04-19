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

#ifndef KMEANS_AUX_HPP
#define KMEANS_AUX_HPP

#include "euclidean_distance.hpp"
#include "hartigan_wong.hpp"
#include <cstdlib>
#include <numeric>

namespace da_kmeans {

/* Initialize the upper and lower bounds for Elkan's method */
template <typename T> void da_kmeans<T>::init_elkan_bounds() {

    compute_centre_half_distances();

    // For every sample, set upper bound (works1) to be distance to closest centre and update label
    // Lower bound (worksc1) will contain distance from each sample to each cluster centre, if computed
    for (da_int i = 0; i < n_samples; i++) {

        T smallest_dist = std::numeric_limits<T>::infinity();
        da_int label = 0;
        da_int tmp_int;

        for (da_int j = 0; j < n_clusters; j++) {
            // Compute distance between the ith sample and the jth centre only if needed
            worksc1[i + j * ldworksc1] = (T)0.0;
            tmp_int = (label < j) ? label + j * n_clusters : label * n_clusters + j;
            if (smallest_dist > workcc1[tmp_int]) {

                T dist = 0.0;
                for (da_int k = 0; k < n_features; k++) {
                    dist +=
                        (A[i + k * lda] -
                         (*current_cluster_centres)[j + k * n_clusters]) *
                        (A[i + k * lda] - (*current_cluster_centres)[j + k * n_clusters]);
                }
                dist = std::sqrt(dist);
                worksc1[i + j * ldworksc1] = dist;

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

/* Compute the half distances between centres in current_cluster_centres and distance to next closest centre
   This matrix is symmetric so only the upper triangle is computed and stored */
template <typename T> void da_kmeans<T>::compute_centre_half_distances() {

    T *dummy = nullptr;

    euclidean_distance(
        n_clusters, n_clusters, n_features, (*current_cluster_centres).data(), n_clusters,
        dummy, 0, workcc1.data(), n_clusters, workc1.data(), 2, dummy, 0, false, true);

    // For each centre, compute half distance to next closest centre and store in workc1
    std::fill(workc1.begin(), workc1.end(), std::numeric_limits<T>::infinity());

    for (da_int j = 0; j < n_clusters; j++) {
        for (da_int i = 0; i < j; i++) {
            T tmp = (T)0.5 * workcc1[i + j * n_clusters];
            // Update so we have centre half distances since euclidean_distance gave us whole distances
            workcc1[i + j * n_clusters] = tmp;
            if (tmp < workc1[i])
                workc1[i] = tmp;
            if (tmp < workc1[j])
                workc1[j] = tmp;
        }
    }
}

/* Perform a single run of k-means */
template <typename T> void da_kmeans<T>::perform_kmeans() {

    // Note, iterations are not chunked currently - if OpenMP is to be added this willneed to be done

    switch (algorithm) {
    case lloyd:
        single_iteration = &da_kmeans<T>::lloyd_iteration;
        break;
    case macqueen:
        init_macqueen();
        single_iteration = &da_kmeans<T>::macqueen_iteration;
        break;
    case elkan:
        // For Elkan we also need to initialize the upper and lower bounds
        init_elkan_bounds();
        single_iteration = &da_kmeans<T>::elkan_iteration;
        break;
    case hartigan_wong:
        // Based on MIT licensed open-source implementation in another file
        da_int ifault;

        kmns(A, n_samples, n_features, lda, &(*current_cluster_centres)[0], n_clusters,
             &(*current_labels)[0], work_int1.data(), max_iter, workc1.data(), &ifault,
             &current_n_iter, work_int2.data(), workc2.data(), workc3.data(),
             &(*previous_labels)[0], works1.data(), work_int3.data(), work_int4.data());
        // Record if it converged or ran into maximum number of iterations
        converged = (ifault == 2) ? 0 : 1;
        current_inertia = 0.0;
        // Hartigan-Wong implementation counts from 1 rather than 0, so correct this
        for (da_int i = 0; i < n_samples; i++)
            (*current_labels)[i] -= 1;
        for (da_int i = 0; i < n_clusters; i++)
            current_inertia += workc1[i];
        return;
        break;
    }

    for (current_n_iter = 1; current_n_iter <= max_iter; current_n_iter++) {
        // Start with the 'old' centres stored in previous_cluster_centres
        // Note using swap quite a lot here, but it's only swapping pointers
        std::swap(previous_cluster_centres, current_cluster_centres);
        std::swap(previous_labels, current_labels);

        (this->*single_iteration)(true);

        // Check for convergence
        converged = convergence_test();
        if (converged > 0) {
            break;
        }
    }

    if (converged == 1) {
        // Tolerance-based convergence: means we should rerun labelling step without recomputing centres
        std::swap(previous_labels, current_labels);
        std::swap(previous_cluster_centres, current_cluster_centres);
        (this->*single_iteration)(false);
        std::swap(previous_cluster_centres, current_cluster_centres);
    }

    // Finished this run, so compute current_inertia
    compute_current_inertia();
}

/* Compute current_inertia based on the current_cluster_centres */
template <typename T> void da_kmeans<T>::compute_current_inertia() {
    current_inertia = 0;
    T tmp;
    for (da_int j = 0; j < n_features; j++) {
        for (da_int i = 0; i < n_samples; i++) {
            da_int label = (*current_labels)[i];
            tmp = A[i + j * lda] - (*current_cluster_centres)[label + j * n_clusters];
            current_inertia += tmp * tmp;
        }
    }
}

/* Compute the difference between the current and previous centres and store in previous_cluster_centres */
template <typename T> void da_kmeans<T>::compute_centre_shift() {

    // Before overwriting previous_cluster_centres, compute and store its norm, for use in convergence test
    char norm = 'F';
    normc = da::lange(&norm, &n_clusters, &n_features, (*previous_cluster_centres).data(),
                      &n_clusters, nullptr);

    for (da_int i = 0; i < n_clusters * n_features; i++) {
        (*previous_cluster_centres)[i] -= (*current_cluster_centres)[i];
    }
}

/* Perform single iteration of Elkan's method */
template <typename T> void da_kmeans<T>::elkan_iteration(bool update_centres) {

    if (update_centres) {
        for (da_int j = 0; j < n_clusters; j++) {
            work_int1[j] = 0;
        }
        for (da_int j = 0; j < n_clusters * n_features; j++)
            (*current_cluster_centres)[j] = (T)0.0;
    }

    // At this point workc1 contains distance of each cluster centre to the next nearest
    // The latest labels and centres are in 'previous' so we can update them to current

    // Go through each sample
    da_int tmp_int;
    for (da_int i = 0; i < n_samples; i++) {

        // New labels remain the same until we change them
        (*current_labels)[i] = (*previous_labels)[i];

        // This will be true if the upper and lower bounds are equal
        bool tight_bounds = false;

        // Only proceed if distance to closest centre (works1) exceeds 0.5* distance to next centre
        if (works1[i] > workc1[(*previous_labels)[i]]) {

            for (da_int j = 0; j < n_clusters; j++) {

                // Check if this centre is a good candidate for relabelling the sample - need to account for the fact
                // that only upper triangle of workcc1 is stored
                tmp_int = ((*previous_labels)[i] < j)
                              ? (*previous_labels)[i] + j * n_clusters
                              : (*previous_labels)[i] * n_clusters + j;

                if (j != (*previous_labels)[i] &&
                    works1[i] > worksc1[i + j * ldworksc1] &&
                    works1[i] > workcc1[tmp_int]) {

                    if (tight_bounds == false) {
                        // Get distance from sample point to currently assigned centre
                        works1[i] = 0.0;
                        for (da_int k = 0; k < n_features; k++) {
                            T tmp = A[i + k * lda] -
                                    (*previous_cluster_centres)[(*previous_labels)[i] +
                                                                k * n_clusters];
                            works1[i] += tmp * tmp;
                        }
                        works1[i] = std::sqrt(works1[i]);
                        worksc1[i + (*previous_labels)[i] * ldworksc1] = works1[i];
                        tight_bounds = true;
                    }

                    // If condition still holds then compute distance to candidate centre and check
                    if (works1[i] > worksc1[i + j * ldworksc1] ||
                        works1[i] > workcc1[tmp_int]) {

                        T dist = 0.0;
                        for (da_int k = 0; k < n_features; k++) {
                            T tmp = A[i + k * lda] -
                                    (*previous_cluster_centres)[j + k * n_clusters];
                            dist += tmp * tmp;
                        }
                        dist = std::sqrt(dist);
                        worksc1[i + j * ldworksc1] = dist;
                        if (dist < works1[i]) {
                            works1[i] = dist;
                            (*current_labels)[i] = j;
                        }
                    }
                }
            }
        }

        if (update_centres) {
            work_int1[(*current_labels)[i]] += 1;
            // Add this sample to the cluster mean
            for (da_int j = 0; j < n_features; j++) {
                (*current_cluster_centres)[(*current_labels)[i] + j * n_clusters] +=
                    A[i + j * lda];
            }
        }
    }

    if (update_centres) {
        //Guard against empty clusters - avoid division by zero below
        for (da_int i = 0; i < n_clusters; i++) {
            if (work_int1[i] == 0)
                work_int1[i] = 1;
        }

        // Scale to get proper column means
        for (da_int j = 0; j < n_features; j++) {
            for (da_int i = 0; i < n_clusters; i++) {
                (*current_cluster_centres)[i + j * n_clusters] /= work_int1[i];
            }
        }

        // Update upper and lower bounds and compute shift in centres
        compute_centre_shift();
        for (da_int i = 0; i < n_clusters; i++) {
            workc1[i] = (T)0.0;
            for (da_int j = 0; j < n_features; j++) {
                workc1[i] += (*previous_cluster_centres)[i + j * n_clusters] *
                             (*previous_cluster_centres)[i + j * n_clusters];
            }
            workc1[i] = std::sqrt(workc1[i]);
        }
        // workc1 now contains the distance moved by each centre during the iteration
        for (da_int i = 0; i < n_samples; i++) {
            works1[i] += workc1[(*current_labels)[i]];
            for (da_int j = 0; j < n_clusters; j++) {
                worksc1[i + j * ldworksc1] =
                    std::max(worksc1[i + j * ldworksc1] - workc1[j], (T)0.0);
            }
        }
    }

    compute_centre_half_distances();
}

/* Within lloyd iteration update a chunk of the distance matrix*/
template <typename T>
void da_kmeans<T>::lloyd_iteration_chunk(bool update_centres, da_int chunk_size,
                                         da_int chunk_index) {

    // Compute the matrix D where D_{ij} = ||C_j||^2 - 2 A C^T

    T *dummy = nullptr;
    euclidean_distance(chunk_size, n_clusters, n_features, &A[chunk_index], lda,
                       (*previous_cluster_centres).data(), n_clusters, worksc1.data(),
                       ldworksc1, dummy, 0, workc1.data(), 2, true, false);

    // Go through each sample (row) in worksc1 and find argmin
    for (da_int i = chunk_index; i < chunk_index + chunk_size; i++) {
        T smallest_dist = std::numeric_limits<T>::infinity();
        da_int label = 0;
        for (da_int j = 0; j < n_clusters; j++) {
            if (worksc1[i - chunk_index + ldworksc1 * j] < smallest_dist) {
                label = j;
                smallest_dist = worksc1[i - chunk_index + ldworksc1 * j];
            }
        }
        (*current_labels)[i] = label;
        if (update_centres) {
            work_int1[label] += 1;
            // Add this sample to the cluster mean
            for (da_int j = 0; j < n_features; j++) {
                (*current_cluster_centres)[label + j * n_clusters] += A[i + j * lda];
            }
        }
    }
}

/* Perform single iteration of Lloyd's method */
template <typename T> void da_kmeans<T>::lloyd_iteration(bool update_centres) {

    if (update_centres) {
        for (da_int j = 0; j < n_clusters; j++) {
            work_int1[j] = 0;
        }
        for (da_int j = 0; j < n_clusters * n_features; j++)
            (*current_cluster_centres)[j] = (T)0.0;
    }

    // Distance matrix part of the computation needs to be done in chunks since it is memory intensive
    da_int n_chunks = n_samples / max_chunk_size;
    da_int chunk_rem = n_samples % max_chunk_size;
    // Count the remainder in the number of chunks
    if (chunk_rem > 0)
        n_chunks += 1;

    for (da_int i = 0; i < n_chunks; i++) {
        if (i == n_chunks - 1 && chunk_rem > 0) {
            lloyd_iteration_chunk(update_centres, chunk_rem, n_samples - chunk_rem);
        } else {
            lloyd_iteration_chunk(update_centres, max_chunk_size, i * max_chunk_size);
        }
    }

    if (update_centres) {
        //Guard against empty clusters - avoid division by zero below
        for (da_int i = 0; i < n_clusters; i++) {
            if (work_int1[i] == 0)
                work_int1[i] = 1;
        }
        // Scale to get proper column means
        for (da_int j = 0; j < n_features; j++) {
            for (da_int i = 0; i < n_clusters; i++) {
                (*current_cluster_centres)[i + j * n_clusters] /= work_int1[i];
            }
        }

        // Compute change in centres in this iteration
        compute_centre_shift();
    }
}

/* Chunked part of MacQueen's method initialization */
template <typename T>
void da_kmeans<T>::init_macqueen_chunk(da_int chunk_size, da_int chunk_index) {

    T *dummy = nullptr;

    euclidean_distance(chunk_size, n_clusters, n_features, &A[chunk_index], lda,
                       (*previous_cluster_centres).data(), n_clusters, worksc1.data(),
                       ldworksc1, dummy, 0, workc1.data(), 2, true, false);

    for (da_int i = chunk_index; i < chunk_index + chunk_size; i++) {
        T smallest_dist = std::numeric_limits<T>::infinity();
        da_int label = 0;
        for (da_int j = 0; j < n_clusters; j++) {
            if (worksc1[i - chunk_index + ldworksc1 * j] < smallest_dist) {
                label = j;
                smallest_dist = worksc1[i - chunk_index + ldworksc1 * j];
            }
        }
        (*current_labels)[i] = label;
        // Also want to be counting number of points in each initial cluster
        work_int1[label] += 1;

        // And update clusters now that we have assigned points to them
        for (da_int j = 0; j < n_features; j++) {
            (*current_cluster_centres)[label + j * n_clusters] += A[i + j * lda];
        }
    }
}

/* Initialization for MacQueen's method */
template <typename T> void da_kmeans<T>::init_macqueen() {
    // Form in workc1 the cluster centre norms squared
    for (da_int j = 0; j < n_clusters; j++) {
        work_int1[j] = 0; // Initialize to zero for use later
    }

    for (da_int i = 0; i < n_clusters * n_features; i++)
        (*previous_cluster_centres)[i] = (*current_cluster_centres)[i];

    for (da_int i = 0; i < n_clusters * n_features; i++)
        (*current_cluster_centres)[i] = 0;

    // Distance matrix computation needs to be done in chunks due to memory use
    da_int n_chunks = n_samples / max_chunk_size;
    da_int chunk_rem = n_samples % max_chunk_size;
    // Count the remainder in the number of chunks
    if (chunk_rem > 0)
        n_chunks += 1;

    for (da_int i = 0; i < n_chunks; i++) {
        if (i == n_chunks - 1 && chunk_rem > 0) {
            init_macqueen_chunk(chunk_rem, n_samples - chunk_rem);
        } else {
            init_macqueen_chunk(max_chunk_size, i * max_chunk_size);
        }
    }

    // Finish updating cluster centres - being careful to guard against zero division in empty clusters
    for (da_int j = 0; j < n_features; j++) {
        for (da_int i = 0; i < n_clusters; i++) {
            if (work_int1[i] > 0)
                (*current_cluster_centres)[i + j * n_clusters] /= work_int1[i];
        }
    }

    // Rezero previous clusters, which were used temporarily here
    for (da_int i = 0; i < n_clusters * n_features; i++)
        (*previous_cluster_centres)[i] = 0;
}

/* Perform single iteration of MacQueen's method */
template <typename T> void da_kmeans<T>::macqueen_iteration(bool update_centres) {

    // Copy data from previous iteration since it's updated in place; no way round this since we need previous iteration for convergence test
    for (da_int i = 0; i < n_clusters * n_features; i++)
        (*current_cluster_centres)[i] = (*previous_cluster_centres)[i];

    for (da_int i = 0; i < n_samples; i++)
        (*current_labels)[i] = (*previous_labels)[i];

    for (da_int i = 0; i < n_samples; i++) {
        // For sample point i, compute the cluster centre distances in workc2

        T *dummy = nullptr;
        euclidean_distance(1, n_clusters, n_features, &A[i], lda,
                           (*current_cluster_centres).data(), n_clusters, workc2.data(),
                           1, dummy, 0, workc1.data(), 1, true, false);

        T smallest_dist = std::numeric_limits<T>::infinity();
        da_int closest_centre = 0;
        for (da_int j = 0; j < n_clusters; j++) {
            if (workc2[j] < smallest_dist) {
                smallest_dist = workc2[j];
                closest_centre = j;
            }
        }

        if ((*current_labels)[i] != closest_centre) {
            da_int old_centre = (*current_labels)[i];
            (*current_labels)[i] = closest_centre;

            if (update_centres) {
                // Now need to update the two affected centres: closest_centre and old_centre
                work_int1[closest_centre] += 1;
                work_int1[old_centre] -= 1;

                // Clear closest_centre and old_centre cluster centres ahead of recomputation
                for (da_int j = 0; j < n_features; j++) {
                    (*current_cluster_centres)[old_centre + j * n_clusters] = 0.0;
                    (*current_cluster_centres)[closest_centre + j * n_clusters] = 0.0;
                }

                for (da_int k = 0; k < n_samples; k++) {
                    if ((*current_labels)[k] == closest_centre) {
                        for (da_int j = 0; j < n_features; j++) {
                            (*current_cluster_centres)[closest_centre + j * n_clusters] +=
                                A[k + j * lda];
                        }
                    } else if ((*current_labels)[k] == old_centre) {
                        for (da_int j = 0; j < n_features; j++) {
                            (*current_cluster_centres)[old_centre + j * n_clusters] +=
                                A[k + j * lda];
                        }
                    }
                }

                // Scale to get proper mean
                for (da_int j = 0; j < n_features; j++) {
                    if (work_int1[old_centre] > 0)
                        (*current_cluster_centres)[old_centre + j * n_clusters] /=
                            work_int1[old_centre];
                    if (work_int1[closest_centre] > 0)
                        (*current_cluster_centres)[closest_centre + j * n_clusters] /=
                            work_int1[closest_centre];
                }
            }
        }
    }

    if (update_centres) {
        // Compute change in centres in this iteration
        compute_centre_shift();
    }
}

/* Check if the k-means iteration has converged */
/* 0 means no convergence, 1 is tol-based convergence, 2 is strict convergence (labels didn't change) */
template <typename T> da_int da_kmeans<T>::convergence_test() {

    da_int convergence_test = 0;

    // Check if labels have changed, but only after we've done at least one complete
    if (current_n_iter > 1) {
        convergence_test = 0;
        for (da_int i = 0; i < n_samples; i++) {
            if ((*current_labels)[i] != (*previous_labels)[i]) {
                convergence_test = 0;
                break;
            }
        }
    }

    if (convergence_test > 0)
        return convergence_test;

    // Recall that that the end of each iteration previous_cluster_centres contains the shift made in that particular iteration

    // Note Scikit-Learn seem to use an absolute test here
    char norm = 'F';
    if (da::lange(&norm, &n_clusters, &n_features, (*previous_cluster_centres).data(),
                  &n_clusters, nullptr) < tol * normc)
        convergence_test = 1;

    return convergence_test;
}

/* Initialize the centres, if needed, for the start of k-means computation*/
template <typename T> void da_kmeans<T>::initialize_centres() {
    std::fill(previous_cluster_centres->begin(), previous_cluster_centres->end(), 0.0);
    switch (init_method) {
    case random_samples: {
        // Select randomly (without replacement) from the data points
        std::iota(work_int2.begin(), work_int2.end(), 0);
        std::sample(work_int2.begin(), work_int2.end(), std::begin(work_int1), n_clusters,
                    mt_gen);
        for (da_int j = 0; j < n_clusters; j++) {
            for (da_int i = 0; i < n_features; i++) {
                (*current_cluster_centres)[i * n_clusters + j] =
                    A[i * lda + work_int1[j]];
            }
        }

        break;
    }
    case random_partitions: { // Zero out relevant arrays
        for (da_int i = 0; i < n_clusters; i++) {
            work_int1[i] = 0;
        }
        for (da_int j = 0; j < n_clusters * n_features; j++)
            (*current_cluster_centres)[j] = (T)0.0;

        // Assign each sample point to a random cluster
        std::uniform_int_distribution<> dis_int(0, n_clusters - 1);
        for (da_int i = 0; i < n_samples; i++) {
            da_int tmp_int = dis_int(mt_gen);
            (*current_labels)[i] = tmp_int;
            work_int1[tmp_int] += 1;
            // Add this sample to the relevant cluster mean
            for (da_int j = 0; j < n_features; j++) {
                (*current_cluster_centres)[tmp_int + j * n_clusters] += A[i + j * lda];
            }
        }

        //Guard against empty clusters - avoid division by zero below
        for (da_int i = 0; i < n_clusters; i++) {
            if (work_int1[i] == 0)
                work_int1[i] = 1;
        }

        // Scale to get proper column means
        for (da_int j = 0; j < n_features; j++) {
            for (da_int i = 0; i < n_clusters; i++) {
                (*current_cluster_centres)[i + j * n_clusters] /= work_int1[i];
            }
        }

        break;
    }
    case kmeanspp:
        kmeans_plusplus();
        break;
    default:
        // No need to do anything as initial centres were provided and have been stored in current_cluster_centres already
        break;
    }
}

/* Initialize centres using k-means++ */
template <typename T> void da_kmeans<T>::kmeans_plusplus() {

    // Compute squared norms of the data points and store in works1
    for (da_int j = 0; j < n_features; j++) {
        for (da_int i = 0; i < n_samples; i++) {
            works1[i] += A[j * lda + i] * A[j * lda + i];
        }
    }

    da_int n_trials = 2 + (da_int)std::log(n_clusters);

    // Pick first centre randomly from the sample data points and store which one it was in work_int1
    std::uniform_int_distribution<> dis_int(0, n_samples - 1);
    da_int tmp_int = dis_int(mt_gen);
    work_int1[0] = tmp_int;
    for (da_int i = 0; i < n_features; i++) {
        (*current_cluster_centres)[i * n_clusters] = A[i * lda + work_int1[0]];
    }

    T dummy = 0.0;
    euclidean_distance(n_samples, 1, n_features, A, lda,
                       (*current_cluster_centres).data(), n_clusters, works3.data(),
                       n_samples, works1.data(), 1, &dummy, 2, true, false);

    // Numerical errors could cause one of the distances to be slightly negative, leading to undefined behaviour in std::discrete_distribution
    works3[tmp_int] = 0.0;

    // Need to catch an edge case where all points are the same
    bool coincident_points = true;

    for (int i = 0; i < n_samples; i++) {
        if (works3[i] > 0.0) {
            coincident_points = false;
            break;
        }
    }

    if (coincident_points) {
        // Doesn't matter which ones we choose, this is just to prevent exceptions later, so just use the first ones
        for (da_int j = 0; j < n_features; j++) {
            for (da_int k = 0; k < n_clusters; k++) {
                (*current_cluster_centres)[j * n_clusters + k] = A[j * lda + k];
            }
        }
    } else {

        for (da_int k = 1; k < n_clusters; k++) {

            // Choose n_trials new sample points as the next centre, randomly, weighted by works3, the min distance
            // Don't need to worry about replacement because probability of zero of picking previously chosen point

            da_int best_candidate = 0;
            T best_candidate_cost = std::numeric_limits<T>::infinity();

            std::discrete_distribution<> weighted_dis(works3.begin(), works3.end());
            for (da_int trials = 0; trials < n_trials; trials++) {
                // Our candidate points are stored in work_int2
                work_int2[trials] = weighted_dis(mt_gen);
            }

            for (da_int trials = 0; trials < n_trials; trials++) {

                // It's worth checking in case we've selected a candidate point twice, in which case ignore it
                bool repeat_sample = false;
                for (da_int j = 0; j < trials; j++) {
                    if (work_int2[j] == work_int2[trials]) {
                        repeat_sample = true;
                        break;
                    }
                }
                if (repeat_sample)
                    break;

                // Calculate cost function for this candidate point
                T current_cost = 0.0;
                da_int current_candidate = work_int2[trials];

                for (da_int i = 0; i < n_features; i++) {
                    for (da_int j = 0; j < n_clusters; j++) {
                        // Also store extended copy in previous cluster centres for use in dgemm
                        (*previous_cluster_centres)[i * n_clusters + j] =
                            A[i * lda + current_candidate];
                    }
                }

                // Compute the distance from each point to the candidate centre and store in works4

                euclidean_distance(n_samples, 1, n_features, A, lda,
                                   &A[current_candidate], lda, works4.data(), n_samples,
                                   works1.data(), 1, &works1[current_candidate], 1, true,
                                   false);

                // Get minimum squared distance of each sample point to potential centre
                current_cost = 0;
                for (da_int j = 0; j < n_samples; j++) {
                    works5[j] = std::min(works3[j], works4[j]);
                    current_cost += works5[j];
                }

                if (current_cost < best_candidate_cost) {
                    best_candidate_cost = current_cost;
                    best_candidate = work_int2[trials];
                    for (da_int j = 0; j < n_samples; j++) {
                        works2[j] = works5[j];
                    }
                }
            }

            // Place the best candidate as the next cluster centre
            for (da_int i = 0; i < n_features; i++) {
                (*current_cluster_centres)[i * n_clusters + k] =
                    A[i * lda + best_candidate];
            }
            work_int1[k] = best_candidate;
            for (da_int j = 0; j < n_samples; j++) {
                works3[j] = works2[j];
            }
            // Guard against negative probabilities again
            works3[best_candidate] = 0.0;
        }
    }
    // Now we have n_clusters entries in current_cluster_centres
}

/* Initialize the random number generator, if needed */
template <typename T> void da_kmeans<T>::initialize_rng() {
    if (init_method != supplied) {
        if (seed == -1) {
            std::random_device r;
            seed = std::abs((da_int)r());
        }
        mt_gen.seed(seed);
    }
}

} // namespace da_kmeans

#endif //KMEANS_AUX_HPP
