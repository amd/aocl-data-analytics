# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

"""
Approximate Nearest Neighbors Python test script
"""

import numpy as np
import pytest
from aoclda.neighbors import approximate_neighbors
from aoclda.clustering import kmeans


@pytest.mark.parametrize(
    "numpy_precision",
    [np.float16, np.float32, np.float64, np.int16, np.int32, np.int64, 'object'])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_approx_nn_all_dtypes(numpy_precision, numpy_order):
    """
    Test it runs when supported/unsupported C-interface type is provided.
    """

    x_train = np.array([[-1.5, -1.2, 2.6],
                        [-2.5, -1.3, 3.8],
                        [-3.6, -2.8, -1.7],
                        [1.6, 3.4, 1.2],
                        [2.8, 5.3, 1.3],
                        [3.0, -1.2, 2.4]],
                       dtype=numpy_precision, order=numpy_order)

    x_test = np.array([[-2.5, 2.2, 3.6],
                       [-1.4, -2.7, -1.0],
                       [2.6, 1.2, -3.7]],
                      dtype=numpy_precision, order=numpy_order)

    ann = approximate_neighbors(n_neighbors=3, n_list=2, n_probe=2, seed=42)
    ann.train_and_add(x_train)
    k_dist, k_ind = ann.kneighbors(x_test, return_distance=True)
    k_ind_only = ann.kneighbors(x_test, return_distance=False)


@pytest.mark.parametrize("numpy_precision", [np.float32])
@pytest.mark.parametrize("numpy_orders", [("C", "F"), ("F", "C")])
def test_approx_nn_multiple_orders(numpy_precision, numpy_orders):
    """
    Test it warns when arrays of multiple orders are provided.
    """

    x_train = np.array([[-1.5, -1.2, 2.6],
                        [-2.5, -1.3, 3.8],
                        [-3.6, -2.8, -1.7],
                        [1.6, 3.4, 1.2],
                        [2.8, 5.3, 1.3],
                        [3.0, -1.2, 2.4]],
                       dtype=numpy_precision, order=numpy_orders[0])

    x_test = np.array([[-2.5, 2.2, 3.6],
                       [-1.4, -2.7, -1.0],
                       [2.6, 1.2, -3.7]],
                      dtype=numpy_precision, order=numpy_orders[1])

    ann = approximate_neighbors(n_neighbors=3, n_list=2, n_probe=2, seed=42)
    ann.train_and_add(x_train)
    with pytest.warns(UserWarning):
        k_dist, k_ind = ann.kneighbors(x_test, return_distance=True)
    with pytest.warns(UserWarning):
        k_ind_only = ann.kneighbors(x_test, return_distance=False)

    # Test warning when adding data with different order
    ann.train(x_train)
    with pytest.warns(UserWarning):
        ann.add(x_test)

    # Test warning when re-training with different order (same object)
    x_train = np.array(x_train, order=numpy_orders[1])
    with pytest.warns(UserWarning):
        ann.train_and_add(x_train)


@pytest.mark.parametrize(
    "numpy_precisions", [('float32', 'float64'),
                         ('float64', 'float32')])
@pytest.mark.parametrize("numpy_order", ["C"])
def test_approx_nn_multiple_dtypes(numpy_precisions, numpy_order):
    """
    Test it runs when arrays of multiple dtypes are provided.
    """

    x_train = np.array([[-1.5, -1.2, 2.6],
                        [-2.5, -1.3, 3.8],
                        [-3.6, -2.8, -1.7],
                        [1.6, 3.4, 1.2],
                        [2.8, 5.3, 1.3],
                        [3.0, -1.2, 2.4]],
                       dtype=numpy_precisions[0], order=numpy_order)

    x_test = np.array([[-2.5, 2.2, 3.6],
                       [-1.4, -2.7, -1.0],
                       [2.6, 1.2, -3.7]],
                      dtype=numpy_precisions[1], order=numpy_order)

    ann = approximate_neighbors(n_neighbors=3, n_list=2, n_probe=2, seed=42)
    ann.train_and_add(x_train)
    k_dist, k_ind = ann.kneighbors(x_test, return_distance=True)
    k_ind_only = ann.kneighbors(x_test, return_distance=False)

    ann.train(x_train)
    ann.add(x_train)
    k_dist, k_ind = ann.kneighbors(x_test, return_distance=True)
    k_ind_only = ann.kneighbors(x_test, return_distance=False)

    x_train = np.array(x_train, dtype=numpy_precisions[1])
    ann = approximate_neighbors(n_neighbors=3, n_list=2, n_probe=2, seed=42)
    ann.train_and_add(x_train)


# Add more tests here if desired
functionality_tests = [
    # From ann_tests.hpp RowSqEuclidean
    # Data has 3 well separated clusters, n_neighbors=3
    {
        "name": "sqeuclidean",
        "metric": "sqeuclidean",
        "algorithm": "ivfflat",
        "n_list": 3,
        "n_probe": 2,
        "n_neighbors": 3,
        "kmeans_iter": 10,
        "train_fraction": 1.0,
        "seed": 0,
        "x_train": [[10.1, 10.2, 10.4],
                    [11.1, 12.2, 11.4],
                    [9.1, 9.2, 9.4],
                    [-20.2, -20.3, -20.3],
                    [-21.2, -21.3, -21.3],
                    [-19.2, -19.3, -18.3],
                    [5.0, -10.1, 5.2],
                    [5.0, -11.1, 11.2],
                    [5.0, -9.1, 9.2]],
        "x_test": [[10.1, 11.2, 9.4],
                   [-20.2, -21.3, -19.3],
                   [6.0, -10.1, 9.2]],
        "expected_indices": [[0, 2, 1], [3, 4, 5], [8, 7, 6]],
        "expected_distances": [[2.0, 5.0, 6.0], [2.0, 5.0, 6.0], [2.0, 6.0, 17.0]],
    }
]


def get_test_case_id(test_case):
    """Generate a readable test ID from the test case."""
    return test_case["name"]


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
@pytest.mark.parametrize("test_case", functionality_tests, ids=get_test_case_id)
def test_approx_nn_functionality(numpy_precision, numpy_order, test_case):
    """
    Test the functionality of the Python wrapper.
    """
    x_train = np.array(test_case["x_train"], dtype=numpy_precision, order=numpy_order)
    x_test = np.array(test_case["x_test"], dtype=numpy_precision, order=numpy_order)
    expected_indices = np.array(test_case["expected_indices"])
    expected_distances = np.array(test_case["expected_distances"], dtype=numpy_precision)

    n_samples = x_train.shape[0]
    n_features = x_train.shape[1]
    n_queries = x_test.shape[0]
    n_neighbors = test_case["n_neighbors"]
    n_list = test_case["n_list"]

    ann = approximate_neighbors(
        n_neighbors=n_neighbors,
        algorithm=test_case["algorithm"],
        metric=test_case["metric"],
        n_list=n_list,
        n_probe=test_case["n_probe"],
        kmeans_iter=test_case["kmeans_iter"],
        seed=test_case["seed"],
        train_fraction=test_case["train_fraction"]
    )

    ann.train_and_add(x_train)

    k_dist, k_ind = ann.kneighbors(x_test, return_distance=True)

    # Validate output shapes
    assert k_dist.shape == (n_queries, n_neighbors)
    assert k_ind.shape == (n_queries, n_neighbors)

    # Validate memory layout matches input
    assert k_dist.flags.f_contiguous == x_test.flags.f_contiguous
    assert k_ind.flags.f_contiguous == x_test.flags.f_contiguous

    k_ind_only = ann.kneighbors(x_test, return_distance=False)
    assert k_ind_only.shape == (n_queries, n_neighbors)

    # Test cluster_centroids property
    centroids = ann.cluster_centroids
    assert centroids.shape == (n_list, n_features)

    tol = np.sqrt(np.finfo(numpy_precision).eps) * 100

    km = kmeans(
        n_clusters=n_list,
        n_init=1,
        max_iter=test_case["kmeans_iter"],
        tol=np.finfo(numpy_precision).eps,
        seed=test_case["seed"],
        initialization_method='random')
    km.fit(x_train)
    assert list(centroids.flatten()) == pytest.approx(
        list(km.cluster_centres.flatten()), rel=tol)

    # Test list_sizes property
    list_sizes = ann.list_sizes
    assert list_sizes.shape == (n_list,)
    # Total should equal number of training samples
    assert np.sum(list_sizes) == n_samples

    # Test n_list property
    assert ann.n_list == n_list

    # Test n_index property
    assert ann.n_index == n_samples

    # Test n_features property
    assert ann.n_features == n_features

    # Test kmeans_iter property
    assert ann.kmeans_iter >= 1

    # Validate expected indices
    assert list(k_ind.flatten()) == list(expected_indices.flatten())

    # Validate distances within tolerance (need abs to deal with some cos
    # distances being 0)
    assert list(k_dist.flatten()) == pytest.approx(
        list(expected_distances.flatten()), rel=tol, abs=tol)


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_approx_nn_train_add_separate(numpy_precision, numpy_order):
    """
    Test using train() and add() separately produces same centroids as train_and_add()
    """
    x_train = np.array([[-1, -1, 2],
                        [-2, -1, 3],
                        [-3, -2, -1],
                        [1, 3, 1],
                        [2, 5, 1],
                        [3, -1, 2]],
                       dtype=numpy_precision, order=numpy_order)

    x_add = np.array([[0, 0, 0],
                      [1, 1, 1],
                      [-1, -1, -1]],
                     dtype=numpy_precision, order=numpy_order)

    x_test = np.array([[-2, 2, 3],
                       [-1, -2, -1],
                       [2, 1, -3]],
                      dtype=numpy_precision, order=numpy_order)

    # Use train_and_add
    ann_combined = approximate_neighbors(n_neighbors=3, n_list=2, n_probe=2, seed=42)
    ann_combined.train_and_add(x_train)
    centroids_combined = ann_combined.cluster_centroids

    # Use train() then add() separately
    ann_separate = approximate_neighbors(n_neighbors=3, n_list=2, n_probe=2, seed=42)
    ann_separate.train(x_train)
    ann_separate.add(x_train)
    centroids_separate = ann_separate.cluster_centroids

    # Centroids should be identical since same seed and same training data
    np.testing.assert_array_equal(centroids_combined, centroids_separate)

    # Add more data and verify we can still query
    ann_separate.add(x_add)

    # Query
    k_dist, k_ind = ann_separate.kneighbors(x_test, return_distance=True)
    assert k_dist.shape == (3, 3)
    assert k_ind.shape == (3, 3)


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_approx_nn_n_probe_setter(numpy_precision, numpy_order):
    """
    Test the n_probe property setter
    """
    x_train = np.array([[-1, -1, 2],
                        [-2, -1, 3],
                        [-3, -2, -1],
                        [1, 3, 1],
                        [2, 5, 1],
                        [3, -1, 2]],
                       dtype=numpy_precision, order=numpy_order)

    x_test = np.array([[-2, 2, 3],
                       [-1, -2, -1],
                       [2, 1, -3]],
                      dtype=numpy_precision, order=numpy_order)

    ann = approximate_neighbors(n_neighbors=3, n_list=2, n_probe=1, seed=42)
    ann.train_and_add(x_train)

    # Initial n_probe should be 1
    assert ann.n_probe == 1

    # Query with n_probe=1
    k_dist1, k_ind1 = ann.kneighbors(x_test, return_distance=True)

    # Change n_probe to 3
    ann.n_probe = 2
    assert ann.n_probe == 2

    # Query with n_probe=3 (should probe all lists)
    k_dist2, k_ind2 = ann.kneighbors(x_test, return_distance=True)

    # When probing more lists, distances should be <= (can find closer neighbors)
    assert np.all(k_dist2 <= k_dist1)


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_approx_nn_train_fraction(numpy_precision, numpy_order):
    """
    Test the train_fraction parameter produces different centroids
    """
    # Create larger dataset for subsampling to make sense
    np.random.seed(42)
    x_train = np.random.randn(100, 5).astype(numpy_precision)
    if numpy_order == "F":
        x_train = np.asfortranarray(x_train)

    # Use train_fraction=0.5
    ann_half = approximate_neighbors(
        n_neighbors=3,
        n_list=5,
        n_probe=2,
        seed=42,
        train_fraction=0.5
    )
    ann_half.train_and_add(x_train)
    centroids_half = ann_half.cluster_centroids

    # Use train_fraction=1.0 (default)
    ann_full = approximate_neighbors(
        n_neighbors=3,
        n_list=5,
        n_probe=2,
        seed=42,
        train_fraction=1.0
    )
    ann_full.train_and_add(x_train)
    centroids_full = ann_full.cluster_centroids

    # Centroids should be different since different subsets are used for training
    assert not np.allclose(centroids_half, centroids_full)


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
def test_approx_nn_error_exits(numpy_precision):
    """
    Test error exits in the Python wrapper
    """
    x_train = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
                       dtype=numpy_precision)
    x_test = np.array([[1, 1, 1]], dtype=numpy_precision)

    # Invalid n_neighbors (less than 1)
    with pytest.raises(RuntimeError):
        ann = approximate_neighbors(n_neighbors=0)

    # Invalid algorithm
    with pytest.raises(RuntimeError):
        ann = approximate_neighbors(algorithm="nonexistent")

    # Invalid metric
    with pytest.raises(RuntimeError):
        ann = approximate_neighbors(metric="nonexistent")

    # Invalid n_list (less than 1)
    with pytest.raises(RuntimeError):
        ann = approximate_neighbors(n_list=0)

    # Invalid n_probe (less than 1)
    with pytest.raises(RuntimeError):
        ann = approximate_neighbors(n_probe=0)

    # Invalid kmeans_iter (less than 1)
    with pytest.raises(RuntimeError):
        ann = approximate_neighbors(kmeans_iter=0)

    with pytest.raises(RuntimeError):
        # need to actually train to raise train_fraction error since the
        # option is not set in the C++ layer until then
        ann = approximate_neighbors(train_fraction=1.3)
        ann.train(np.random.randn(100, 5).astype(numpy_precision))

    # n_list greater than n_samples
    ann = approximate_neighbors(n_neighbors=2, n_list=10, seed=42)
    with pytest.raises(RuntimeError):
        ann.train_and_add(x_train)

    # Add without training
    ann = approximate_neighbors(n_neighbors=2, n_list=2, seed=42)
    with pytest.raises(RuntimeError):
        ann.add(x_train)

    # kneighbors without training and adding
    ann = approximate_neighbors(n_neighbors=2, n_list=2, seed=42)
    with pytest.raises(RuntimeError):
        with pytest.warns(RuntimeWarning):
            ann.kneighbors(x_test)

    # Access properties without training
    ann = approximate_neighbors(n_neighbors=2, n_list=2, seed=42)
    with pytest.warns(RuntimeWarning):
        _ = ann.cluster_centroids

    # Mismatched n_features in add
    ann = approximate_neighbors(n_neighbors=2, n_list=2, seed=42)
    ann.train(x_train)
    ann.add(x_train)
    x_add_wrong = np.array([[1, 1], [2, 2]], dtype=numpy_precision)
    with pytest.raises(RuntimeError):
        ann.add(x_add_wrong)

    # Mismatched n_features in kneighbors
    ann = approximate_neighbors(n_neighbors=2, n_list=2, seed=42)
    ann.train_and_add(x_train)
    x_test_wrong = np.array([[1, 1], [2, 2]], dtype=numpy_precision)
    with pytest.raises(RuntimeError):
        ann.kneighbors(x_test_wrong)


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_approx_nn_retraining(numpy_precision, numpy_order):
    """
    Test that the model can be retrained with different data
    """
    x_train1 = np.array([[-1, -1, 2],
                         [-2, -1, 3],
                         [-3, -2, -1],
                         [1, 3, 1]],
                        dtype=numpy_precision, order=numpy_order)

    x_train2 = np.array([[10, 10, 10],
                         [20, 20, 20],
                         [30, 30, 30],
                         [40, 40, 40]],
                        dtype=numpy_precision, order=numpy_order)

    x_test = np.array([[15, 15, 15]],
                      dtype=numpy_precision, order=numpy_order)

    ann = approximate_neighbors(n_neighbors=2, n_list=2, n_probe=2, seed=42)

    # First training
    ann.train_and_add(x_train1)
    k_dist1, k_ind1 = ann.kneighbors(x_test, return_distance=True)

    # Retrain with different data
    ann.train_and_add(x_train2)
    k_dist2, k_ind2 = ann.kneighbors(x_test, return_distance=True)

    # Results should be different because training data changed
    # The indices should refer to the new training set
    assert k_dist2.shape == (1, 2)
    assert k_ind2.shape == (1, 2)
