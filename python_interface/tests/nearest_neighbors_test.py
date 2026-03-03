# Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
Nearest Neighbors Python test script
"""

import numpy as np
import pytest
from aoclda.neighbors import nearest_neighbors
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


@pytest.mark.parametrize(
    "numpy_precision",
    [np.float16, np.float32, np.float64, np.int16, np.int32, np.int64, 'object'])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
@pytest.mark.parametrize("algo", ["auto", "brute", "kd_tree", "ball_tree"])
def test_nn_all_dtypes(numpy_precision, numpy_order, algo):

    # Test it runs when supported/unsupported C-interface type is provided.

    x_train = np.array([[-1.5, -1.2, 2.6],
                        [-2.5, -1.3, 3.8],
                        [-3.6, -2.8, -1.7],
                        [1.6, 3.4, 1.2],
                        [2.8, 5.3, 1.3],
                        [3.0, -1.2, 2.4]],
                       dtype=numpy_precision, order=numpy_order)

    y_train = np.array([1, 2, 0, 1, 2, 2],
                       dtype=numpy_precision, order=numpy_order)

    x_test = np.array([[-2.5, 2.2, 3.6],
                       [-1.4, -2.7, -1.0],
                       [2.6, 1.2, -3.7]],
                      dtype=numpy_precision, order=numpy_order)

    nn = nearest_neighbors(algorithm=algo, radius=10.5)
    nn.fit(x_train, y_train)
    k_dist, k_ind = nn.kneighbors(x_test, n_neighbors=3, return_distance=True)
    r_dist, r_ind = nn.radius_neighbors(x_test, return_distance=True, radius=2.0)
    proba = nn.classifier_predict_proba(x_test, "knn")
    y_test_class = nn.classifier_predict(x_test, "radius_neighbors")
    y_test_reg = nn.regressor_predict(x_test, "knn")


@pytest.mark.parametrize("algo", ["auto", "brute", "kd_tree"])
@pytest.mark.parametrize("numpy_precision", [np.float32])
@pytest.mark.parametrize("numpy_orders",
                         [("C", "F"), ("F", "C")])
def test_nearest_neighbors_multiple_orders(numpy_precision, numpy_orders, algo):

    # Test it runs when arrays of multiple orders are provided.

    x_train = np.array([[-1.5, -1.2, 2.6],
                        [-2.5, -1.3, 3.8],
                        [-3.6, -2.8, -1.7],
                        [1.6, 3.4, 1.2],
                        [2.8, 5.3, 1.3],
                        [3.0, -1.2, 2.4]],
                       dtype=numpy_precision, order=numpy_orders[0])

    y_train = np.array([1, 2, 0, 1, 2, 2],
                       dtype=numpy_precision, order=numpy_orders[1])

    x_test = np.array([[-2.5, 2.2, 3.6],
                       [-1.4, -2.7, -1.0],
                       [2.6, 1.2, -3.7]],
                      dtype=numpy_precision, order=numpy_orders[1])

    nn = nearest_neighbors(algorithm=algo, radius=10.5)
    nn.fit(x_train, y_train)
    with pytest.warns(UserWarning):
        k_dist, k_ind = nn.kneighbors(x_test, n_neighbors=3, return_distance=True)
    with pytest.warns(UserWarning):
        r_dist, r_ind = nn.radius_neighbors(x_test, return_distance=True, radius=2.0)
    with pytest.warns(UserWarning):
        proba = nn.classifier_predict_proba(x_test, "radius_neighbors")
    with pytest.warns(UserWarning):
        y_test = nn.classifier_predict(x_test, "knn")
    x_train = np.array(x_train, order=numpy_orders[1])
    with pytest.warns(UserWarning):
        nn.fit(x_train, y_train)

    x_train = np.array(x_train, order=numpy_orders[0])
    nn = nearest_neighbors(algorithm=algo, radius=10.5)
    nn.fit(x_train, y_train)
    with pytest.warns(UserWarning):
        k_dist, k_ind = nn.kneighbors(x_test, n_neighbors=3, return_distance=True)
    with pytest.warns(UserWarning):
        y_test = nn.regressor_predict(x_test, "radius_neighbors")
    x_train = np.array(x_train, order=numpy_orders[1])
    with pytest.warns(UserWarning):
        nn.fit(x_train, y_train)

    x_train = np.array(x_train, order=numpy_orders[0])
    nn = nearest_neighbors(algorithm=algo)
    nn.fit(x_train)
    with pytest.warns(UserWarning):
        k_dist, k_ind = nn.kneighbors(x_test, n_neighbors=3, return_distance=True)
    with pytest.warns(UserWarning):
        r_dist, r_ind = nn.radius_neighbors(x_test, return_distance=True, radius=2.0)


@pytest.mark.parametrize("algo", ["auto", "brute", "kd_tree"])
@pytest.mark.parametrize(
    "numpy_precisions", [('float32', 'float64'),
                         ('float64', 'float32')])
@pytest.mark.parametrize("numpy_order", ["C"])
def test_nearest_neighbors_multiple_dtypes(numpy_precisions, numpy_order, algo):

    # Test it runs when arrays of multiple orders are provided.

    x_train = np.array([[-1.5, -1.2, 2.6],
                        [-2.5, -1.3, 3.8],
                        [-3.6, -2.8, -1.7],
                        [1.6, 3.4, 1.2],
                        [2.8, 5.3, 1.3],
                        [3.0, -1.2, 2.4]],
                       dtype=numpy_precisions[0], order=numpy_order)

    y_train = np.array([1, 2, 0, 1, 2, 2],
                       dtype=numpy_precisions[1], order=numpy_order)

    x_test = np.array([[-2.5, 2.2, 3.6],
                       [-1.4, -2.7, -1.0],
                       [2.6, 1.2, -3.7]],
                      dtype=numpy_precisions[1], order=numpy_order)

    nn = nearest_neighbors(algorithm=algo, radius=10.5)
    nn.fit(x_train, y_train)
    k_dist, k_ind = nn.kneighbors(x_test, n_neighbors=3, return_distance=True)
    proba = nn.classifier_predict_proba(x_test, "radius_neighbors")
    y_test = nn.classifier_predict(x_test, "knn")
    x_train = np.array(x_train, dtype=numpy_precisions[1])
    nn.fit(x_train, y_train)

    x_train = np.array(x_train, dtype=numpy_precisions[0])
    nn = nearest_neighbors(algorithm=algo, radius=10.5)
    nn.fit(x_train, y_train)
    k_dist, k_ind = nn.kneighbors(x_test, n_neighbors=3, return_distance=True)
    y_test = nn.regressor_predict(x_test, "radius_neighbors")
    x_train = np.array(x_train, dtype=numpy_precisions[1])
    nn.fit(x_train, y_train)

    x_train = np.array(x_train, dtype=numpy_precisions[0])
    nn = nearest_neighbors(algorithm=algo)
    nn.fit(x_train)
    k_dist, k_ind = nn.kneighbors(x_test, n_neighbors=3, return_distance=True)
    r_dist, r_ind = nn.radius_neighbors(x_test, return_distance=True, radius=2.0)


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
@pytest.mark.parametrize("algo", ["auto", "brute", "kd_tree"])
def test_nearest_neighbors_functionality(numpy_precision, numpy_order, algo):

    # Test the functionality of the Python wrapper

    x_train = np.array([[-1, -1, 2],
                        [-2, -1, 3],
                        [-3, -2, -1],
                        [1, 3, 1],
                        [2, 5, 1],
                        [3, -1, 2]],
                       dtype=numpy_precision, order=numpy_order)

    y_train = np.array([1, 2, 0, 1, 2, 2],
                       dtype=numpy_precision, order=numpy_order)

    y_train_targets = np.array([1.5, 2.3, 0.6, 1, 2.8, 5.2],
                               dtype=numpy_precision, order=numpy_order)

    x_test = np.array([[-2, 2, 3],
                       [-1, -2, -1],
                       [2, 1, -3]],
                      dtype=numpy_precision, order=numpy_order)

    nn = nearest_neighbors(algorithm=algo, radius=5.0)
    nn.fit(x_train, y_train)
    k_dist, k_ind = nn.kneighbors(x_test, n_neighbors=3, return_distance=True)

    assert k_dist.flags.f_contiguous == x_test.flags.f_contiguous
    assert k_ind.flags.f_contiguous == x_test.flags.f_contiguous

    # Prediction using k-nearest neighbors
    proba_knn = nn.classifier_predict_proba(x_test, "knn")
    assert proba_knn.flags.f_contiguous == x_test.flags.f_contiguous
    y_test_knn = nn.classifier_predict(x_test, "knn")

    # Prediction using radius neighbors
    proba_radius = nn.classifier_predict_proba(x_test, "radius_neighbors")
    assert proba_radius.flags.f_contiguous == x_test.flags.f_contiguous
    y_test_radius = nn.classifier_predict(x_test, "radius_neighbors")

    expected_ind = np.array([[1, 0, 3],
                             [2, 0, 1],
                             [3, 5, 4]])
    expected_dist = np.array([[3., 3.31662479, 3.74165739],
                              [2., 3.16227766, 4.24264069],
                              [4.58257569, 5.47722558, 5.65685425]])
    expected_proba_knn = np.array([[0.2, 0.4, 0.4],
                                   [0.2, 0.4, 0.4],
                                   [0.2, 0.4, 0.4]])
    expected_labels_knn = np.array([[1, 1, 1,]])

    expected_targets_knn = np.array([1.64, 2.12, 2.22])

    expected_proba_radius = np.array([[0., 0.66666667, 0.33333333],
                                      [0.33333333, 0.33333333, 0.33333333],
                                      [0., 1., 0.]])
    expected_labels_radius = np.array([[1, 0, 1,]])

    expected_targets_radius = np.array([1.6, 1.46666667, 1.])

    # Check we have the right answer
    tol = np.sqrt(np.finfo(numpy_precision).eps)

    assert k_dist == pytest.approx(expected_dist, tol)
    assert not np.any(k_ind - expected_ind)
    # Test knn predictions
    assert proba_knn == pytest.approx(expected_proba_knn, tol)
    assert not np.any(y_test_knn - expected_labels_knn)
    # Test radius neighbors predictions
    assert proba_radius == pytest.approx(expected_proba_radius, tol)
    assert not np.any(y_test_radius - expected_labels_radius)

    # Regressor tests
    nn_reg = nearest_neighbors(algorithm=algo, radius=5.0)
    nn_reg.fit(x_train, y_train_targets)
    k_dist, k_ind = nn_reg.kneighbors(x_test, n_neighbors=3, return_distance=True)

    assert k_dist.flags.f_contiguous == x_test.flags.f_contiguous
    assert k_ind.flags.f_contiguous == x_test.flags.f_contiguous

    # Test knn predictions
    y_test_knn = nn_reg.regressor_predict(x_test, "knn")
    assert k_dist == pytest.approx(expected_dist, tol)
    assert not np.any(k_ind - expected_ind)
    assert y_test_knn == pytest.approx(expected_targets_knn, tol)
    # Test radius neighbors predictions
    y_test_radius = nn_reg.regressor_predict(x_test, "radius_neighbors")
    assert y_test_radius == pytest.approx(expected_targets_radius, tol)

    nn = nearest_neighbors(algorithm=algo)
    nn.fit(x_train)
    k_dist, k_ind = nn.kneighbors(x_test, n_neighbors=3, return_distance=True)
    assert k_dist == pytest.approx(expected_dist, tol)
    assert not np.any(k_ind - expected_ind)

    r_dist, r_ind = nn.radius_neighbors(
        x_test, return_distance=True, sort_results=True, radius=3.0)
    expected_r_dist = [np.array([3.]), np.array([2.]), np.array([])]
    expected_r_ind = [np.array([1]), np.array([2]), np.array([])]

    for i in range(len(r_ind)):
        assert r_dist[i] == pytest.approx(expected_r_dist[i], tol)
        assert not np.any(r_ind[i] - expected_r_ind[i])


@pytest.mark.parametrize("n_samples", [20, 500, 1234])
@pytest.mark.parametrize("n_features", [4, 15])
@pytest.mark.parametrize("n_classes", [3])
@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
@pytest.mark.parametrize("leaf_size", [1, 2, 10, 41])
def test_knn_tree_functionality(
        n_samples,
        n_features,
        n_classes,
        numpy_precision,
        numpy_order,
        leaf_size):

    # Test the functionality of the Python wrapper for different leaf sizes.
    # Since we expect the same results for tree and brute force algorithms,
    # we can compare against the brute force algorithm.

    x, y = make_classification(n_samples=2 * n_samples,
                               n_features=n_features,
                               n_informative=n_features,
                               n_repeated=0,
                               n_redundant=0,
                               n_classes=n_classes,
                               random_state=42)

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.5,
                                                        train_size=0.5,
                                                        random_state=42)

    if numpy_order == "F":
        x_train = np.asfortranarray(x_train, dtype=numpy_precision)
        x_test = np.asfortranarray(x_test, dtype=numpy_precision)
    else:
        # Only ensure numpy_precision is correct
        x_train = np.array(x_train, dtype=numpy_precision)
        x_test = np.array(x_test, dtype=numpy_precision)

    # Compute using k-d tree algorithm
    kdtree_knn = nearest_neighbors(
        algorithm="kd_tree",
        leaf_size=leaf_size,
        metric="euclidean_gemm")
    kdtree_knn.fit(x_train, y_train)
    kdtree_knn_k_dist, kdtree_knn_k_ind = kdtree_knn.kneighbors(
        x_test, n_neighbors=7, return_distance=True)

    assert kdtree_knn_k_dist.flags.f_contiguous == x_test.flags.f_contiguous
    assert kdtree_knn_k_ind.flags.f_contiguous == x_test.flags.f_contiguous

    kdtree_knn_proba = kdtree_knn.classifier_predict_proba(x_test)
    assert kdtree_knn_proba.flags.f_contiguous == x_test.flags.f_contiguous

    kdtree_knn_y_test = kdtree_knn.classifier_predict(x_test)

    # Compute using ball tree algorithm
    balltree_knn = nearest_neighbors(
        algorithm="ball_tree",
        leaf_size=leaf_size,
        metric="euclidean_gemm")
    balltree_knn.fit(x_train, y_train)
    balltree_knn_k_dist, balltree_knn_k_ind = balltree_knn.kneighbors(
        x_test, n_neighbors=7, return_distance=True)

    assert balltree_knn_k_dist.flags.f_contiguous == x_test.flags.f_contiguous
    assert balltree_knn_k_ind.flags.f_contiguous == x_test.flags.f_contiguous

    balltree_knn_proba = balltree_knn.classifier_predict_proba(x_test)
    assert balltree_knn_proba.flags.f_contiguous == x_test.flags.f_contiguous

    balltree_knn_y_test = balltree_knn.classifier_predict(x_test)

    # Now compute using brute force algorithm
    brute_knn = nearest_neighbors(
        algorithm="brute",
        leaf_size=leaf_size,
        metric="euclidean_gemm")
    brute_knn.fit(x_train, y_train)
    brute_knn_k_dist, brute_knn_k_ind = brute_knn.kneighbors(
        x_test, n_neighbors=7, return_distance=True)

    assert brute_knn_k_dist.flags.f_contiguous == x_test.flags.f_contiguous
    assert brute_knn_k_ind.flags.f_contiguous == x_test.flags.f_contiguous

    brute_knn_proba = brute_knn.classifier_predict_proba(x_test)
    assert brute_knn_proba.flags.f_contiguous == x_test.flags.f_contiguous

    brute_knn_y_test = brute_knn.classifier_predict(x_test)

    # Check we have the right answer
    tol = np.sqrt(np.finfo(numpy_precision).eps)

    assert kdtree_knn_k_dist == pytest.approx(brute_knn_k_dist, tol)
    assert not np.any(kdtree_knn_k_ind - brute_knn_k_ind)
    assert kdtree_knn_proba == pytest.approx(brute_knn_proba, tol)
    assert not np.any(kdtree_knn_y_test - brute_knn_y_test)

    assert balltree_knn_k_dist == pytest.approx(brute_knn_k_dist, tol)
    assert not np.any(balltree_knn_k_ind - brute_knn_k_ind)
    assert balltree_knn_proba == pytest.approx(brute_knn_proba, tol)
    assert not np.any(balltree_knn_y_test - brute_knn_y_test)


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
def test_nearest_neighbors_error_exits(numpy_precision):

    # Test error exits in the Python wrapper

    # Errors in nearest_neighbors constructor
    with pytest.raises(RuntimeError):
        nn = nearest_neighbors(n_neighbors=-1)
    with pytest.raises(ValueError):
        nn = nearest_neighbors(n_neighbors=1.5)
    with pytest.raises(RuntimeError):
        nn = nearest_neighbors(metric="nonexistent")
    with pytest.raises(RuntimeError):
        nn = nearest_neighbors(algorithm="nonexistent")
    with pytest.raises(ValueError):
        nn = nearest_neighbors(radius=-1.0)
    with pytest.raises(ValueError):
        nn = nearest_neighbors(radius="k")

    nn = nearest_neighbors()
    x_train = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                       dtype=numpy_precision)
    x_test = np.array([[1, 1, 1], [2, 2, 2]], dtype=numpy_precision)
    # Errors when input dimension is wrong
    y_train = np.array([[1, 2]], dtype=numpy_precision)
    nn.fit(x_train, y_train)
    # Tests for knn methods
    with pytest.raises(RuntimeError):
        nn.classifier_predict(X=x_test, search_mode="knn")
    with pytest.raises(RuntimeError):
        nn.classifier_predict_proba(X=x_test, search_mode="knn")
    with pytest.raises(RuntimeError):
        nn.regressor_predict(X=x_test, search_mode="knn")
    # Tests for radius neighbors methods
    with pytest.raises(RuntimeError):
        nn.classifier_predict(X=x_test, search_mode="radius_neighbors")
    with pytest.raises(RuntimeError):
        nn.classifier_predict_proba(X=x_test, search_mode="radius_neighbors")
    with pytest.raises(RuntimeError):
        nn.regressor_predict(X=x_test, search_mode="radius_neighbors")

    # Reset with correct y_train
    y_train = np.array([[1, 2, 3]], dtype=numpy_precision)
    x_test = np.array([[1, 1], [2, 2], [3, 3]], dtype=numpy_precision)
    nn.fit(x_train, y_train)

    # Errors in nearest_neighbors methods when input dimension is wrong
    with pytest.raises(RuntimeError):
        nn.kneighbors(X=x_test)
    with pytest.raises(RuntimeError):
        nn.radius_neighbors(X=x_test)
    # Tests for knn methods
    with pytest.raises(RuntimeError):
        nn.classifier_predict(X=x_test, search_mode="knn")
    with pytest.raises(RuntimeError):
        nn.classifier_predict_proba(X=x_test, search_mode="knn")
    with pytest.raises(RuntimeError):
        nn.regressor_predict(X=x_test, search_mode="knn")
    # Tests for radius neighbors methods
    with pytest.raises(RuntimeError):
        nn.classifier_predict(X=x_test, search_mode="radius_neighbors")
    with pytest.raises(RuntimeError):
        nn.classifier_predict_proba(X=x_test, search_mode="radius_neighbors")
    with pytest.raises(RuntimeError):
        nn.regressor_predict(X=x_test, search_mode="radius_neighbors")

    # Errors in fit method when input data has NaN values
    nn = nearest_neighbors(check_data=True)
    x_train = np.array([[1, 1, np.nan], [2, 2, 2], [3, 3, 3]],
                       dtype=numpy_precision, order="F")
    y_train = np.array([[1, 2, 3]], dtype=numpy_precision)
    with pytest.raises(RuntimeError):
        nn.fit(x_train, y_train)

    # Set up valid data
    x_train = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                       dtype=numpy_precision)
    y_train = np.array([[1, 2, 3]], dtype=numpy_precision)
    x_test = np.array([[1, 1, 2], [2, 2, 3]], dtype=numpy_precision)

    # Errors in nearest_neighbors when n_neighbors=None
    # for both the constructor and kneighbors method
    nn = nearest_neighbors(n_neighbors=None)
    nn.fit(x_train)
    with pytest.raises(ValueError):
        nn.kneighbors(x_test, n_neighbors=None)

    # Check other nearest_neighbors method errors
    nn = nearest_neighbors()
    nn.fit(x_train)
    with pytest.raises(ValueError):
        nn.kneighbors(X=None)
    with pytest.raises(ValueError):
        nn.kneighbors(x_test, n_neighbors=-1)
    with pytest.raises(ValueError):
        nn.kneighbors(x_test, n_neighbors=1.5)
    with pytest.raises(ValueError):
        nn.radius_neighbors(X=None)
    with pytest.raises(ValueError):
        nn.radius_neighbors(x_test, radius=-1.0)
    with pytest.raises(ValueError):
        nn.radius_neighbors(x_test, radius='k')
    with pytest.raises(ValueError):
        nn.radius_neighbors(x_test, return_distance=False, sort_results=True)
    # Tests for knn methods
    with pytest.raises(ValueError):
        nn.classifier_predict(X=None, search_mode="knn")
    with pytest.raises(ValueError):
        nn.classifier_predict_proba(X=None, search_mode="knn")
    with pytest.raises(ValueError):
        nn.regressor_predict(X=None, search_mode="knn")
    # Tests for radius neighbors methods
    with pytest.raises(ValueError):
        nn.classifier_predict(X=None, search_mode="radius_neighbors")
    with pytest.raises(ValueError):
        nn.classifier_predict_proba(X=None, search_mode="radius_neighbors")
    with pytest.raises(ValueError):
        nn.regressor_predict(X=None, search_mode="radius_neighbors")

    # Check other method errors if y_train was not provided
    # Tests for knn methods
    with pytest.raises(ValueError):
        nn.classifier_predict(X=x_test, search_mode="knn")
    with pytest.raises(ValueError):
        nn.classifier_predict_proba(X=x_test, search_mode="knn")
    with pytest.raises(ValueError):
        nn.regressor_predict(X=x_test, search_mode="knn")
    # Tests for radius neighbors methods
    with pytest.raises(ValueError):
        nn.classifier_predict(X=x_test, search_mode="radius_neighbors")
    with pytest.raises(ValueError):
        nn.classifier_predict_proba(X=x_test, search_mode="radius_neighbors")
    with pytest.raises(ValueError):
        nn.regressor_predict(X=x_test, search_mode="radius_neighbors")
