/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */
#ifndef BINARY_TREE_HPP
#define BINARY_TREE_HPP

#include "da_error.hpp"
#include "da_utils.hpp"
#include "da_vector.hpp"
#include "model_persistence.hpp"
#include "nearest_neighbors_types.hpp"
#include <algorithm>
#include <memory>
#include <vector>

namespace ARCH {

namespace da_binary_tree {

/* Node structure for the binary tree class */

template <typename T> struct node {

    using value_type = T;

    // The depth of this node
    da_int depth;

    // Indices of the points in the dataset that are in this node and its children
    da_int *indices = nullptr;
    da_int n_indices;

    // Is this a leaf node
    bool is_leaf = false;

    // Constructor
    node(da_int depth, da_int *indices, da_int n_indices);
    // Constructor for when loaded from memory
    node() = default;

    da_status serialize(da_model_persistence::serialization_buffer &buffer);
};

/* Node structure for the k-d tree class - needs to be here to enable instantiation*/

template <typename T> struct kd_node : public node<T> {

    // Which dimension this node splits on
    da_int dim;

    // For a non-leaf node, the index of the point that splits the node
    da_int point;

    std::unique_ptr<kd_node<T>> left_child = nullptr;
    std::unique_ptr<kd_node<T>> right_child = nullptr;

    // bounding box for the node
    std::vector<T> min_bounds;
    std::vector<T> max_bounds;

    // Constructors - only the top level node need have min_bounds and max_bounds supplied
    kd_node(da_int dim, da_int depth, da_int *indices, da_int n_indices,
            std::vector<T> min_bounds, std::vector<T> max_bounds);
    kd_node(da_int dim, da_int depth, da_int *indices, da_int n_indices);
    // Constructor for when loaded from memory
    kd_node() = default;

    da_status serialize(da_model_persistence::serialization_buffer &buffer);
};

/* Node structure for the ball tree class - needs to be here to enable instantiation */

template <typename T> struct ball_node : public node<T> {

    std::unique_ptr<ball_node<T>> left_child = nullptr;
    std::unique_ptr<ball_node<T>> right_child = nullptr;

    // center and radius of the ball
    std::vector<T> centroid; // centroid of the ball
    T radius;                // radius of the ball

    // Constructor - only the top level node needs to have centroid and radius supplied
    ball_node(da_int depth, da_int *indices, da_int n_indices);
    ball_node(da_int depth, da_int *indices, da_int n_indices, std::vector<T> centroid,
              T radius);
    // Constructor for when loaded from memory
    ball_node() = default;

    da_status serialize(da_model_persistence::serialization_buffer &buffer);
};

// Lightweight partial MaxHeap implementation to keep track of tree searches
template <typename T> class MaxHeap {
  public:
    // Default constructor for use with std::vector
    MaxHeap() : indices(nullptr), distances(nullptr), capacity(0), size(0) {}

    // Constructor from existing arrays of indices and distances
    MaxHeap(da_int capacity, da_int *indices, T *distances);

    // Get the current maximum distance in the heap
    T GetMaxDist();

    da_int GetSize();

    // Insert a new point to the heap, if it is small enough, maintaining the max-heap property
    void Insert(da_int index, T distance);

  private:
    da_int *indices = nullptr; // Indices of the points in the dataset
    T *distances = nullptr;    // Distances of the points in the dataset
    da_int capacity = 0;       // Maximum capacity of the heap
    da_int size = 0;           // Current size of the heap

    void heapify_up(da_int index);
    void heapify_down(da_int index);
};

// Forward declaration of the kd_tree and ball_tree classes, which is are specializations of the binary_tree class
// This is required because we are using CRTP (Curiously Recurring Template Pattern) to implement static polymorphism:
// the binary_tree class is a template that takes the derived class and the node type as template parameters, so forward
// declarations of derived classes are required.
template <typename T> class kd_tree;
template <typename T> class ball_tree;

/* Binary tree class, use CRTP for static compile-time polymorphism which performs better and avoids
   virtual functions in the base class, which in this case must be explicitly instantiated */
template <typename Derived, typename NodeType> class binary_tree {
  public:
    using T = typename NodeType::value_type;

    da_status radius_neighbors(da_int m_samples_in, da_int m_features_in, const T *X_in,
                               da_int ldx_in, T eps,
                               std::vector<da_vector::da_vector<da_int>> &neighbors,
                               std::vector<da_vector::da_vector<T>> &distances,
                               bool return_distance, da_errors::da_error_t *err);

    da_status k_neighbors(da_int m_samples_in, da_int m_features_in, const T *X_in,
                          da_int ldx_in, da_int k, da_int *k_ind, T *k_dist,
                          da_errors::da_error_t *err);

    // Get the indices, for testing purposes
    const std::vector<da_int> &get_indices();

    da_status serialize(da_model_persistence::serialization_buffer &buffer);

  protected:
    // Internal functions used in tree construction and tree traversal to find neighbors

    void store_data(da_int n_samples_in, da_int n_features_in, const T *A_in,
                    da_int lda_in, da_int leaf_size_in, da_metric metric_in, T p_in);

    da_status preprocess_data(const T *X_in, da_int m_samples_in, da_int m_features_in,
                              da_int ldx_in, const T **X, bool &X_is_A, da_int &m_samples,
                              da_int &ldx, da_errors::da_error_t *err);

    da_status compute_distance(T &dist, da_int index_A, T *X, T X_norm);

    da_int leaf_size = 30;

    // Indices of points in the dataset
    std::vector<da_int> indices;

    // Dataset on which to build the tree
    const T *A;
    da_int lda;
    da_int n_samples;
    da_int n_features;

    // The metric to use for distance calculations (for kd- tree this isn't actually needed until we perform nearest neighbour searches)
    da_metric metric = da_euclidean;
    // Internal metric used for distance computation. In case euclidean is used, we can avoid
    // computing the squares by using da_sqeuclidean.
    da_metric metric_internal = da_euclidean;

    T p = 2.0;     // Default p for Minkowski distance is 2.0
    T p_inv = 0.5; // Inverse of p, used for Minkowski distance

    // Row norms of the dataset - only used for da_euclidean
    std::vector<T> A_norms;

    // Root node of the tree
    std::unique_ptr<NodeType> root = nullptr;

  private:
    da_status tree_serialization(da_model_persistence::serialization_buffer &buffer,
                                 std::unique_ptr<NodeType> &node);
    template <bool ReturnDistances>
    da_status radius_neighbors_loop(da_int m_samples, const T *X, da_int ldx, T eps,
                                    T eps_internal,
                                    std::vector<da_vector::da_vector<da_int>> &neighbors,
                                    std::vector<da_vector::da_vector<T>> &distances,
                                    bool X_is_A, std::vector<T> &X_row);
};

template <typename T>
da_status parallel_variance(da_int n_samples, da_int n_features, const T *A, da_int lda,
                            std::vector<T> &means, std::vector<T> &variances);

/* k-d tree class */
template <typename T> class kd_tree : public binary_tree<kd_tree<T>, kd_node<T>> {
  public:
    kd_tree(da_int n_samples_in, da_int n_features_in, const T *A_in, da_int lda_in,
            da_int leaf_size_in = 30, da_metric metric_in = da_sqeuclidean, T p_in = 2.0);
    // Constructor for when being loaded from memory
    kd_tree(const T *A_in, da_int lda_in);

    T single_pass_variance(da_int *indices, da_int n_indices, da_int dim);

    // Inherited functions used in tree construction and tree traversal to find neighbours
    da_status radius_neighbors_recursive(kd_node<T> *current_node, T *X, T eps,
                                         T eps_internal,
                                         da_vector::da_vector<da_int> &neighbors,
                                         da_vector::da_vector<T> &distances,
                                         bool return_distance, bool X_is_A,
                                         da_int index_X, T X_norm);

    da_status k_neighbors_recursive(kd_node<T> *current_node, T *X, da_int k, bool X_is_A,
                                    da_int index_X, T X_norm, MaxHeap<T> &heap);

  private:
    // Build the k-d tree from the dataset
    std::unique_ptr<kd_node<T>> build_tree(da_int depth, da_int *indices,
                                           da_int n_indices,
                                           std::vector<T> *min_bounds = nullptr,
                                           std::vector<T> *max_bounds = nullptr,
                                           da_int root_dim = 0);

    da_neighbors_types::nn_check_region check_bounding_box(T *X, T eps,
                                                           std::vector<T> &min_bounds,
                                                           std::vector<T> &max_bounds);
};

template <typename T>
da_status parallel_centroid_radius(da_int n_samples, da_int n_features, const T *A,
                                   da_int lda, da_metric metric, T p,
                                   std::vector<T> &centroid, T &radius);

/* ball tree class */
template <typename T> class ball_tree : public binary_tree<ball_tree<T>, ball_node<T>> {
  public:
    ball_tree(da_int n_samples_in, da_int n_features_in, const T *A_in, da_int lda_in,
              da_int leaf_size_in = 30, da_metric metric_in = da_sqeuclidean,
              T p_in = 2.0);
    // Constructor for when being loaded from memory
    ball_tree(const T *A_in, da_int lda_in);

    // Inherited functions used in tree construction and tree traversal to find neighbours
    da_status radius_neighbors_recursive(ball_node<T> *current_node, T *X, T eps,
                                         T eps_internal,
                                         da_vector::da_vector<da_int> &neighbors,
                                         da_vector::da_vector<T> &distances,
                                         bool return_distance, bool X_is_A,
                                         da_int index_X, T X_norm);

    da_status k_neighbors_recursive(ball_node<T> *current_node, T *X, da_int k,
                                    bool X_is_A, da_int index_X, T X_norm,
                                    MaxHeap<T> &heap);

  private:
    // Build the ball tree from the dataset
    std::unique_ptr<ball_node<T>> build_tree(da_int depth, da_int *indices,
                                             da_int n_indices,
                                             std::vector<T> *centroid = nullptr,
                                             T radius = 0.0);

    void node_centroid_radius(da_int *indices, da_int n_indices, std::vector<T> &centroid,
                              T &radius);

    bool choose_centroid(T *A, T A_norm, da_int index_A, T *B, T B_norm, da_int index_B,
                         da_int i);

    void furthest_point(da_int *indices, da_int n_indices, da_int index,
                        da_int &furthest_index);

    da_neighbors_types::nn_check_region check_ball(T *X, T eps, std::vector<T> &centroid,
                                                   T radius, T &dist);

    std::vector<T> A_row1, A_row2;
};

} // namespace da_binary_tree

} // namespace ARCH

#endif