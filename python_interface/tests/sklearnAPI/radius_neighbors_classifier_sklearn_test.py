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
Radius neighbors classification tests, check output of skpatch versus sklearn
"""

# pylint: disable = import-outside-toplevel, reimported, no-member,
# no-value-for-parameter, too-many-positional-arguments

import numpy as np
import pytest
from aoclda.sklearn import skpatch, undo_skpatch


@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("weights", ['uniform', 'distance'])
@pytest.mark.parametrize("metric",
                         ['euclidean',
                          'l2',
                          'sqeuclidean',
                          'manhattan',
                          'l1',
                          'cityblock',
                          'cosine',
                          'minkowski',
                          'euclidean_gemm',
                          'sqeuclidean_gemm'])
@pytest.mark.parametrize("radius_constructor", [30.0])
@pytest.mark.parametrize("radius_compute", [6.0, None])
@pytest.mark.parametrize("algorithm", ['auto', 'kd_tree', 'ball_tree', 'brute'])
@pytest.mark.parametrize("sort_res", [True, False])
def test_radius_neighbors_classifier(
        precision,
        weights,
        metric,
        radius_constructor,
        radius_compute,
        algorithm,
        sort_res):
    """
    Solve a small problem
    """
    # Skip test for unsupported parameter combinations
    if (metric == 'cosine' or metric == 'sqeuclidean' or metric == 'sqeuclidean_gemm') and (
            algorithm == 'kd_tree' or algorithm == 'ball_tree'):
        pytest.skip("Cosine/sqeuclidean metrics are not supported with tree algorithms")

    # Define data arrays
    x_train = np.array([[-1, 3, 2],
                        [-2, -1, 4],
                        [-3, 2, -3],
                        [2, 5, -2],
                        [2, 6, -3],
                        [3, -1, 4]], dtype=precision)

    y_train = np.array([1, 2, 0, 1, 2, 2], dtype=precision)

    x_test = np.array([[-2, 5, 3],
                       [-1, -2, 4],
                       [4, 1, -3]], dtype=precision)

    tol = np.sqrt(10 * np.finfo(precision).eps)

    p = 3.2
    # patch and import scikit-learn
    skpatch()
    from sklearn import neighbors
    if metric == "minkowski":
        rnn_da = neighbors.RadiusNeighborsClassifier(weights=weights,
                                                     radius=radius_constructor,
                                                     metric=metric,
                                                     p=p,
                                                     algorithm=algorithm)
    else:
        rnn_da = neighbors.RadiusNeighborsClassifier(weights=weights,
                                                     radius=radius_constructor,
                                                     metric=metric,
                                                     algorithm=algorithm)
    rnn_da.fit(x_train, y_train)
    rn_dist_da, rn_ind_da = rnn_da.radius_neighbors(
        x_test, radius=radius_compute, return_distance=True, sort_results=sort_res)
    da_predict_proba = rnn_da.predict_proba(x_test)
    da_y_test = rnn_da.predict(x_test)
    da_params = rnn_da.get_params()
    assert rnn_da.aocl is True

    # unpatch and solve the same problem with sklearn
    undo_skpatch()
    sk_metric = metric

    if sk_metric == 'sqeuclidean_gemm':
        sk_metric = 'sqeuclidean'
    elif sk_metric == 'euclidean_gemm':
        sk_metric = 'euclidean'

    from sklearn import neighbors
    if sk_metric == "minkowski":
        rnn_sk = neighbors.RadiusNeighborsClassifier(
            weights=weights,
            radius=radius_constructor,
            p=p,
            metric=sk_metric,
            algorithm=algorithm)
    else:
        rnn_sk = neighbors.RadiusNeighborsClassifier(
            weights=weights,
            radius=radius_constructor,
            metric=sk_metric,
            algorithm=algorithm)

    rnn_sk.fit(x_train, y_train)
    rn_dist_sk, rn_ind_sk = rnn_sk.radius_neighbors(
        x_test, radius=radius_compute, return_distance=True, sort_results=sort_res)
    sk_predict_proba = rnn_sk.predict_proba(x_test)
    sk_y_test = rnn_sk.predict(x_test)
    sk_params = rnn_sk.get_params()
    assert not hasattr(rnn_sk, 'aocl')

    # Check shapes
    assert rn_dist_da.shape == rn_dist_sk.shape
    assert rn_ind_da.shape == rn_ind_sk.shape

    # Check if any arrays have elements before comparing values
    has_radius_neighbors = any(
        arr.size > 0 for arr in rn_dist_da) and any(
        arr.size > 0 for arr in rn_dist_sk)

    if has_radius_neighbors:
        # Compare distances element by element
        for da_arr, sk_arr in zip(rn_dist_da, rn_dist_sk):
            assert da_arr == pytest.approx(sk_arr, tol)

        # Compare indices element by element
        for da_arr, sk_arr in zip(rn_ind_da, rn_ind_sk):
            # Convert da_arr to int64 to match sk_arr dtype
            da_arr_int64 = da_arr.astype(np.int64)
            assert np.array_equal(da_arr_int64, sk_arr)
    else:
        # Both should be empty - verify all individual arrays are empty
        for r_dist_da, sk_dist in zip(rn_dist_da, rn_dist_sk):
            assert r_dist_da.shape == (0,) and sk_dist.shape == (0,)
        for r_ind_da, sk_ind in zip(rn_ind_da, rn_ind_sk):
            assert r_ind_da.shape == (0,) and sk_ind.shape == (0,)

    # Check if predict_proba results are close
    assert np.allclose(da_predict_proba, sk_predict_proba, atol=tol)

    # Check if predicted labels are identical
    assert np.array_equal(da_y_test, sk_y_test)

    # Normalize metric names for comparison
    if da_params.get('metric') == 'euclidean_gemm':
        da_params['metric'] = 'euclidean'
    elif da_params.get('metric') == 'sqeuclidean_gemm':
        da_params['metric'] = 'sqeuclidean'

    assert da_params == sk_params

    # print the results if pytest is invoked with the -rA option
    print("Indices of neighbors")
    print("     aoclda: \n", rn_ind_da)
    print("    sklearn: \n", rn_ind_sk)
    print("Distances to neighbors")
    print("     aoclda: \n", rn_dist_da)
    print("    sklearn: \n", rn_dist_sk)
    print("Class probabilities")
    print("     aoclda: \n", da_predict_proba)
    print("    sklearn: \n", sk_predict_proba)
    print("Predicted labels")
    print("     aoclda: \n", da_y_test)
    print("    sklearn: \n", sk_y_test)
    print("Parameters")
    print("     aoclda: \n", da_params)
    print("    sklearn: \n", sk_params)


def test_radius_neighbors_classifier_errors():
    '''
    Check we can catch errors in the sklearn neighbors patch
    '''
    # patch and import scikit-learn
    skpatch()
    from sklearn import neighbors
    # Check RadiusNeighborsClassifier constructor errors
    with pytest.raises(ValueError):
        rnn = neighbors.RadiusNeighborsClassifier(radius=-1)
    with pytest.raises(ValueError):
        rnn = neighbors.RadiusNeighborsClassifier(radius='k')
    with pytest.raises(ValueError):
        rnn = neighbors.RadiusNeighborsClassifier(weights="ones")
    with pytest.raises(ValueError):
        rnn = neighbors.RadiusNeighborsClassifier(metric="nonexistent")
    with pytest.raises(ValueError):
        rnn = neighbors.RadiusNeighborsClassifier(algorithm="nonexistent")
    rnn = neighbors.RadiusNeighborsClassifier()
    x_train = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float64)
    x_test = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float64)
    # Errors when input dimension is wrong
    y_train = np.array([[1, 2]], dtype=np.float64)
    rnn.fit(x_train, y_train)
    with pytest.raises(RuntimeError):
        rnn.predict(X=x_test)
    with pytest.raises(RuntimeError):
        rnn.predict_proba(X=x_test)
    # Reset with correct y_train and wrong x_test
    y_train = np.array([1, 2, 3], dtype=np.float64)
    x_test = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.float64)
    rnn.fit(x_train, y_train)
    # Errors in RadiusNeighborsClassifier methods when input dimension is wrong
    with pytest.raises(RuntimeError):
        rnn.predict(X=x_test)
    with pytest.raises(RuntimeError):
        rnn.predict_proba(X=x_test)
    # Set up valid data
    x_test = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float64)
    # Error in RadiusNeighborsClassifier methods when radius=None
    # for both the constructor and radius_neighbors call
    rnn = neighbors.RadiusNeighborsClassifier(radius=None)
    rnn.fit(x_train, y_train)
    with pytest.raises(ValueError):
        rnn.radius_neighbors(x_test, radius=None)
    # Check other RadiusNeighborsClassifier method errors
    rnn = neighbors.RadiusNeighborsClassifier()
    rnn.fit(x_train, y_train)
    with pytest.raises(ValueError):
        rnn.radius_neighbors(X=None)
    with pytest.raises(ValueError):
        rnn.radius_neighbors(x_test, radius=-1)
    with pytest.raises(ValueError):
        rnn.radius_neighbors(x_test, radius='k')
    with pytest.raises(ValueError):
        rnn.radius_neighbors(x_test, return_distance=False, sort_results=True)
    with pytest.raises(ValueError):
        rnn.predict_proba(X=None)
    with pytest.raises(ValueError):
        rnn.predict(X=None)
    # Check unimplemented RadiusNeighborsClassifier methods
    rnn = neighbors.RadiusNeighborsClassifier()
    rnn.fit(x_train, y_train)
    y_test = np.array([[1, 1]], dtype=np.float64)
    with pytest.raises(RuntimeError):
        rnn.score(x_test, y_test)
    with pytest.raises(RuntimeError):
        rnn.get_metadata_routing()
    with pytest.raises(RuntimeError):
        rnn.radius_neighbors_graph()
    with pytest.raises(RuntimeError):
        rnn.set_params()
    with pytest.raises(RuntimeError):
        rnn.set_score_request()


if __name__ == "__main__":
    test_radius_neighbors_classifier()  # pylint: disable=no-value-for-parameter
    test_radius_neighbors_classifier_errors()
