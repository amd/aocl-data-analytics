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
Nearest Neighbors tests, check output of skpatch versus sklearn
"""

# pylint: disable = import-outside-toplevel, reimported, no-member,
# no-value-for-parameter, too-many-positional-arguments

import numpy as np
import pytest
from aoclda.sklearn import skpatch, undo_skpatch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


@pytest.mark.parametrize("precision", [np.float64, np.float32])
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
@pytest.mark.parametrize("neigh_radius_constructor", [(3, 1.0), (5, 3.0)])
@pytest.mark.parametrize("neigh_radius_compute", [(3, 1.0), (2, 3.0)])
@pytest.mark.parametrize("algorithm", ['auto', 'kd_tree', 'ball_tree', 'brute'])
def test_nearest_neighbors(
        precision,
        metric,
        neigh_radius_constructor,
        neigh_radius_compute,
        algorithm):

    # Solve a small problem

    # Skip test for unsupported parameter combinations
    if (metric == 'cosine' or metric == 'sqeuclidean' or metric == 'sqeuclidean_gemm') and (
            algorithm == 'kd_tree' or algorithm == 'ball_tree'):
        pytest.skip("Cosine/sqeuclidean metrics are not supported with tree algorithms")

    # Define data arrays
    x_train = np.array([[-1, 3, 2],
                        [-2, -1, 4],
                        [-3, 2, -3],
                        [2, 5, -2],
                        [2, 5, -3],
                        [3, -1, 4]], dtype=precision)

    y_train = np.array([1, 2, 0, 1, 2, 2], dtype=precision)

    x_test = np.array([[-2, 5, 3],
                       [-1, -2, 4],
                       [4, 1, -3]], dtype=precision)

    tol = np.sqrt(10 * np.finfo(precision).eps)

    p = 3.2

    neigh_constructor, radius_constructor = neigh_radius_constructor
    neigh_compute, radius_compute = neigh_radius_compute

    # patch and import scikit-learn
    skpatch()
    from sklearn import neighbors

    # Check NearestNeighbors API
    if metric == "minkowski":
        nn_da = neighbors.NearestNeighbors(n_neighbors=neigh_constructor,
                                           radius=radius_constructor,
                                           algorithm=algorithm,
                                           leaf_size=3,
                                           metric=metric,
                                           p=p)
    else:
        nn_da = neighbors.NearestNeighbors(n_neighbors=neigh_constructor,
                                           radius=radius_constructor,
                                           algorithm=algorithm,
                                           leaf_size=3,
                                           metric=metric)
    # Call fit
    nn_da.fit(x_train)
    # Call kneighbors()
    kn_dist_da, kn_ind_da = nn_da.kneighbors(
        x_test, n_neighbors=neigh_compute, return_distance=True)
    # Call radius_neighbors()
    rn_dist_da, rn_ind_da = nn_da.radius_neighbors(
        x_test, radius=radius_compute, return_distance=True, sort_results=True)
    # Get parameters
    da_params = nn_da.get_params()
    # Check that the AOCL patch was applied
    assert nn_da.aocl is True

    # unpatch and solve the same problem with sklearn
    undo_skpatch()
    sk_metric = metric

    if sk_metric == 'sqeuclidean_gemm':
        sk_metric = 'sqeuclidean'
    elif sk_metric == 'euclidean_gemm':
        sk_metric = 'euclidean'

    from sklearn import neighbors
    if sk_metric == "minkowski":
        nn_sk = neighbors.NearestNeighbors(
            n_neighbors=neigh_constructor,
            radius=radius_constructor,
            algorithm=algorithm,
            leaf_size=3,
            metric=sk_metric,
            p=p)
    else:
        nn_sk = neighbors.NearestNeighbors(
            n_neighbors=neigh_constructor,
            radius=radius_constructor,
            algorithm=algorithm,
            leaf_size=3,
            metric=sk_metric)

    nn_sk.fit(x_train, y_train)
    # Call kneighbors()
    kn_dist_sk, kn_ind_sk = nn_sk.kneighbors(
        x_test, n_neighbors=neigh_compute, return_distance=True)
    # Call radius_neighbors()
    rn_dist_sk, rn_ind_sk = nn_sk.radius_neighbors(
        x_test, radius=radius_compute, return_distance=True, sort_results=True)
    # Get parameters
    sk_params = nn_sk.get_params()
    # Check that the AOCL patch was not applied
    assert not hasattr(nn_sk, 'aocl')

    # Normalize metric names for comparison
    if da_params.get('metric') == 'euclidean_gemm':
        da_params['metric'] = 'euclidean'
    elif da_params.get('metric') == 'sqeuclidean_gemm':
        da_params['metric'] = 'sqeuclidean'

    # Check shapes
    assert kn_dist_da.shape == kn_dist_sk.shape
    assert kn_ind_da.shape == kn_ind_sk.shape
    assert rn_dist_da.shape == rn_dist_sk.shape
    assert rn_ind_da.shape == rn_ind_sk.shape

    # Check results
    assert kn_dist_da == pytest.approx(kn_dist_sk, tol)

    if kn_ind_da.size > 0:  # kneighbors always returns fixed-size arrays
        assert not np.any(kn_ind_da - kn_ind_sk)

    # For radius neighbors, first check shapes match
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

    # Check parameters
    assert da_params == sk_params

    # print the results if pytest is invoked with the -rA option
    print("Indices of k-nearest neighbors")
    print("     aoclda: \n", kn_ind_da)
    print("    sklearn: \n", kn_ind_sk)
    print("Distances to k-nearest neighbors")
    print("     aoclda: \n", kn_dist_da)
    print("    sklearn: \n", kn_dist_sk)
    print("Indices of r neighbors")
    print("     aoclda: \n", rn_ind_da)
    print("    sklearn: \n", rn_ind_sk)
    print("Distances to r neighbors")
    print("     aoclda: \n", rn_dist_da)
    print("    sklearn: \n", rn_dist_sk)
    print("Parameters")
    print("     aoclda: \n", da_params)
    print("    sklearn: \n", sk_params)


def test_nearest_neighbors_errors():
    '''
    Check we can catch errors in the sklearn neighbors patch
    '''
    # patch and import scikit-learn
    skpatch()
    from sklearn import neighbors

    # Check NearestNeighbors constructor errors
    with pytest.raises(ValueError):
        nn = neighbors.NearestNeighbors(n_neighbors=-1)
    with pytest.raises(ValueError):
        nn = neighbors.NearestNeighbors(n_neighbors=1.5)
    with pytest.raises(ValueError):
        nn = neighbors.NearestNeighbors(metric="nonexistent")
    with pytest.raises(ValueError):
        nn = neighbors.NearestNeighbors(algorithm="nonexistent")
    with pytest.raises(ValueError):
        nn = neighbors.NearestNeighbors(radius=-1.0)
    with pytest.raises(ValueError):
        nn = neighbors.NearestNeighbors(radius="k")

    x_train = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float64)
    y_train = np.array([[1, 2, 3]], dtype=np.float64)
    x_test = np.array([[1, 2, 3], [3, 2, 1]], dtype=np.float64)
    y_test = np.array([[1, 1]], dtype=np.float64)

    # Check NearestNeighbors cannot have n_neighbors=None
    # for both the constructor and kneighbors()
    nn = neighbors.NearestNeighbors(n_neighbors=None)
    nn.fit(x_train, y_train)
    with pytest.raises(ValueError):
        nn.kneighbors(n_neighbors=None)

    # Check other NearestNeighbors method errors
    nn = neighbors.NearestNeighbors()
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
    # Check non-implemented methods raise RuntimeError
    with pytest.raises(RuntimeError):
        nn.get_metadata_routing()
    with pytest.raises(RuntimeError):
        nn.kneighbors_graph()
    with pytest.raises(RuntimeError):
        nn.radius_neighbors_graph(x_test, y_test)
    with pytest.raises(RuntimeError):
        nn.set_params()


if __name__ == "__main__":
    test_nearest_neighbors()  # pylint: disable=no-value-for-parameter
    test_nearest_neighbors_errors()
