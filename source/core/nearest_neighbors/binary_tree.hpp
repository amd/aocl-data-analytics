/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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

#include "aoclda.h"
#include "basic_handle.hpp"
#include "da_error.hpp"
#include "da_vector.hpp"
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
};

/* Node structure for the k-d tree class - needs to be here to enable instantiation*/

template <typename T> struct kd_node : public node<T> {

    // Which dimension this node splits on
    da_int dim;

    // For a non-leaf node, the index of the point that splits the node
    da_int point;

    // shared_ptr to the child nodes means we don't have to worry about memory management
    std::shared_ptr<kd_node<T>> left_child = nullptr;
    std::shared_ptr<kd_node<T>> right_child = nullptr;

    // bounding box for the node
    std::vector<T> min_bounds;
    std::vector<T> max_bounds;

    // Constructors - only the top level node need have min_bounds and max_bounds supplied
    kd_node(da_int dim, da_int depth, da_int *indices, da_int n_indices,
            std::vector<T> min_bounds, std::vector<T> max_bounds);
    kd_node(da_int dim, da_int depth, da_int *indices, da_int n_indices);
};

/* Node structure for the ball tree class - needs to be here to enable instantiation */

template <typename T> struct ball_node : public node<T> {

    // shared_ptr to the child nodes means we don't have to worry about memory management
    std::shared_ptr<ball_node<T>> left_child = nullptr;
    std::shared_ptr<ball_node<T>> right_child = nullptr;

    // center and radius of the ball
    std::vector<T> centroid; // centroid of the ball
    T radius;                // radius of the ball

    // Constructor - only the top level node needs to have centroid and radius supplied
    ball_node(da_int depth, da_int *indices, da_int n_indices);
    ball_node(da_int depth, da_int *indices, da_int n_indices, std::vector<T> centroid,
              T radius);
};

// Lightweight partial MaxHeap implementation to keep track of tree searches
template <typename T> class MaxHeap {
  public:
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
                               da_errors::da_error_t *err);

    da_status k_neighbors(da_int m_samples_in, da_int m_features_in, const T *X_in,
                          da_int ldx_in, da_int k, da_int *k_ind, T *k_dist,
                          da_errors::da_error_t *err);

    // Get the indices, for testing purposes
    const std::vector<da_int> &get_indices();

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
    std::shared_ptr<NodeType> root = nullptr;
};

template <typename T>
da_status parallel_variance(da_int n_samples, da_int n_features, const T *A, da_int lda,
                            std::vector<T> &means, std::vector<T> &variances);

/* k-d tree class */
template <typename T> class kd_tree : public binary_tree<kd_tree<T>, kd_node<T>> {
  public:
    kd_tree(da_int n_samples_in, da_int n_features_in, const T *A_in, da_int lda_in,
            da_int leaf_size_in = 30, da_metric metric_in = da_sqeuclidean, T p_in = 2.0);

    T single_pass_variance(da_int *indices, da_int n_indices, da_int dim);

    // Inherited functions used in tree construction and tree traversal to find neighbours
    da_status radius_neighbors_recursive(std::shared_ptr<kd_node<T>> current_node, T *X,
                                         T eps, T eps_internal,
                                         da_vector::da_vector<da_int> &neighbors,
                                         bool X_is_A, da_int index_X, T X_norm);

    da_status k_neighbors_recursive(std::shared_ptr<kd_node<T>> current_node, T *X,
                                    da_int k, bool X_is_A, da_int index_X, T X_norm,
                                    MaxHeap<T> &heap);

  private:
    // Build the k-d tree from the dataset
    std::shared_ptr<kd_node<T>> build_tree(da_int depth, da_int *indices,
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

    // Inherited functions used in tree construction and tree traversal to find neighbours
    da_status radius_neighbors_recursive(std::shared_ptr<ball_node<T>> current_node, T *X,
                                         T eps, T eps_internal,
                                         da_vector::da_vector<da_int> &neighbors,
                                         bool X_is_A, da_int index_X, T X_norm);

    da_status k_neighbors_recursive(std::shared_ptr<ball_node<T>> current_node, T *X,
                                    da_int k, bool X_is_A, da_int index_X, T X_norm,
                                    MaxHeap<T> &heap);

  private:
    // Build the ball tree from the dataset
    std::shared_ptr<ball_node<T>> build_tree(da_int depth, da_int *indices,
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