# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
k-NN classification tests, check output of skpatch versus sklearn
"""

# pylint: disable = import-outside-toplevel, reimported, no-member, no-value-for-parameter

import numpy as np
import pytest
from aoclda.sklearn import skpatch, undo_skpatch

@pytest.mark.parametrize("precision", [np.float64,  np.float32])
@pytest.mark.parametrize("weights", ['uniform',  'distance'])
@pytest.mark.parametrize("metric", ['euclidean',  'sqeuclidean'])
@pytest.mark.parametrize("n_neigh_constructor", [3, 5])
@pytest.mark.parametrize("n_neigh_kneighbors", [3])

def test_knn_classifier(precision, weights, metric, n_neigh_constructor, n_neigh_kneighbors):
    """
    Solve a small problem
    """
    # Define data arrays
    x_train = np.array([[-1, -1, 2],
                        [-2, -1, 3],
                        [-3, -2, -1],
                        [1, 3, 1],
                        [2, 5, 1],
                        [3, -1, 2]], dtype=precision)

    y_train = np.array([1, 2, 0, 1, 2, 2], dtype=precision)

    x_test = np.array([[-2, 2, 3],
                       [-1, -2, -1],
                       [2, 1, -3]], dtype=precision)

    tol = np.sqrt(np.finfo(precision).eps)

    # patch and import scikit-learn
    skpatch()
    from sklearn import neighbors
    knn_da = neighbors.KNeighborsClassifier(weights=weights,
                                            n_neighbors=n_neigh_constructor,
                                            metric=metric)
    knn_da.fit(x_train, y_train)
    da_dist, da_ind = knn_da.kneighbors(x_test, n_neighbors=n_neigh_kneighbors, return_distance=True)
    da_predict_proba = knn_da.predict_proba(x_test)
    da_y_test = knn_da.predict(x_test)
    da_params = knn_da.get_params()
    assert knn_da.aocl is True

    # unpatch and solve the same problem with sklearn
    undo_skpatch()
    from sklearn import neighbors
    knn_sk = neighbors.KNeighborsClassifier(weights=weights, n_neighbors=n_neigh_constructor, metric=metric)
    knn_sk.fit(x_train, y_train)
    sk_dist, sk_ind = knn_sk.kneighbors(x_test, n_neighbors=n_neigh_kneighbors, return_distance=True)
    sk_predict_proba = knn_sk.predict_proba(x_test)
    sk_y_test = knn_sk.predict(x_test)
    sk_params = knn_sk.get_params()
    assert not hasattr(knn_sk, 'aocl')

    # Check results
    assert da_dist == pytest.approx(sk_dist, tol)
    assert not np.any(da_ind - sk_ind)
    assert da_predict_proba == pytest.approx(sk_predict_proba, tol)
    assert not np.any(da_y_test - sk_y_test)
    assert da_params == sk_params

    # print the results if pytest is invoked with the -rA option
    print("Indices of neighbors")
    print("     aoclda: \n", da_ind)
    print("    sklearn: \n", sk_ind)
    print("Distances to neighbors")
    print("     aoclda: \n", da_dist)
    print("    sklearn: \n", sk_dist)
    print("Class probabilities")
    print("     aoclda: \n", da_predict_proba)
    print("    sklearn: \n", sk_predict_proba)
    print("Predicted labels")
    print("     aoclda: \n", da_y_test)
    print("    sklearn: \n", sk_y_test)
    print("Parameters")
    print("     aoclda: \n", da_params)
    print("    sklearn: \n", sk_params)

def test_knn_errors():
    '''
    Check we can catch errors in the sklearn neighbors patch
    '''
    # patch and import scikit-learn
    skpatch()
    from sklearn import neighbors

    with pytest.raises(RuntimeError):
        knn = neighbors.KNeighborsClassifier(n_neighbors = -1)
    with pytest.raises(ValueError):
        knn = neighbors.KNeighborsClassifier(weights = "ones")
    with pytest.raises(ValueError):
        knn = neighbors.KNeighborsClassifier(metric = "manhattan")

    x_train = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float64)
    y_train = np.array([[1, 2, 3]], dtype=np.float64)
    x_test = np.array([[1, 2, 3], [3, 2, 1]], dtype=np.float64)
    y_test = np.array([[1, 1]], dtype=np.float64)
    knn = neighbors.KNeighborsClassifier()
    knn.fit(x_train, y_train)
    with pytest.raises(RuntimeError):
        knn.score(x_test, y_test)
    with pytest.raises(RuntimeError):
        knn.get_metadata_routing()
    with pytest.raises(RuntimeError):
        knn.kneighbors_graph()
    with pytest.raises(RuntimeError):
        knn.set_params()
    with pytest.raises(RuntimeError):
        knn.set_score_request()

if __name__ == "__main__":
    test_knn_classifier()
    test_knn_errors()
