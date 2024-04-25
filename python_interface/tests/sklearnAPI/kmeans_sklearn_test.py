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
KMeans tests, check output of skpatch versus sklearn
"""

# pylint: disable = import-outside-toplevel, reimported, no-member

import numpy as np
import pytest
from aoclda.sklearn import skpatch, undo_skpatch

def dummy():
    return 0

def test_kmeans():
    """
    Basic 8 x 2 problem
    """
    # Define data arrays
    a = np.array([[2., 1.],
                  [-1., -2.],
                  [3., 2.],
                  [2., 3.],
                  [-3., -2.],
                  [-2., -1.],
                  [-2., -3.],
                  [1., 2.]])

    c = np.array([[1., 1.],
                  [-3., -3.]])

    x = np.array([[0., 1.],
                  [0., -1.]])

    # patch and import scikit-learn
    skpatch()
    from sklearn.cluster import KMeans
    kmeans_da = KMeans(n_clusters = 2, init = c)
    kmeans_da.fit(a)
    da_centres = kmeans_da.cluster_centers_
    da_labels = kmeans_da.labels_
    da_inertia = kmeans_da.inertia_
    da_x_transform = kmeans_da.transform(x)
    da_predict = kmeans_da.predict(x)
    kmeans_da = KMeans(n_clusters = 2, init = c)
    da_a_transform = kmeans_da.fit_transform(a)
    kmeans_da = KMeans(n_clusters = 2, init = c)
    da_a_predict = kmeans_da.fit_predict(a)
    assert kmeans_da.aocl is True

    # unpatch and solve the same problem with sklearn
    undo_skpatch()
    from sklearn.cluster import KMeans
    kmeans_sk = KMeans(n_clusters = 2, init = c)
    kmeans_sk.fit(a)
    sk_centres = kmeans_sk.cluster_centers_
    sk_labels = kmeans_sk.labels_
    sk_inertia = kmeans_sk.inertia_
    sk_x_transform = kmeans_sk.transform(x)
    sk_predict = kmeans_sk.predict(x)
    kmeans_sk = KMeans(n_clusters = 2, init = c)
    sk_a_transform = kmeans_sk.fit_transform(a)
    kmeans_sk = KMeans(n_clusters = 2, init = c)
    sk_a_predict = kmeans_sk.fit_predict(a)
    assert not hasattr(kmeans_sk, 'aocl')

    # Check results
    assert da_inertia == pytest.approx(sk_inertia, 1.0e-08)
    assert da_centres == pytest.approx(sk_centres, 1.0e-08)
    assert not np.any(da_labels - sk_labels)
    assert da_x_transform == pytest.approx(sk_x_transform, 1.0e-08)
    assert not np.any(da_predict - sk_predict)
    assert da_a_transform == pytest.approx(sk_a_transform, 1.0e-08)
    assert not np.any(da_a_predict - sk_a_predict)

    # print the results if pytest is invoked with the -rA option
    print("Centres")
    print("    aoclda: \n", da_centres)
    print("   sklearn: \n", sk_centres)
    print("Labels")
    print("    aoclda: \n", da_labels)
    print("   sklearn: \n", sk_labels)
    print("Predict")
    print("    aoclda: \n", da_predict)
    print("   sklearn: \n", sk_predict)


def test_kmeans_errors():
    '''
    Check we can catch errors in the sklearn pca patch
    '''
    a = np.array([[1, 2, 3], [0.22, 5, 4.1], [3, 6, 1]])

    skpatch()
    from sklearn.cluster import KMeans
    with pytest.raises(ValueError):
        kmeans = KMeans(init = dummy)

    with pytest.warns(RuntimeWarning):
        kmeans = KMeans(n_clusters = 1, copy_x = True)

    kmeans.fit(a)

    # Test unsupported functions

    with pytest.raises(RuntimeError):
        kmeans.set_output()

    with pytest.raises(RuntimeError):
        kmeans.set_params()

    with pytest.raises(RuntimeError):
        kmeans.get_feature_names_out()

    with pytest.raises(RuntimeError):
        kmeans.get_metadata_routing()

    with pytest.raises(RuntimeError):
        kmeans.score(1)

    with pytest.raises(RuntimeError):
        kmeans.set_score_request(1)

    with pytest.raises(RuntimeError):
        kmeans.set_predict_request(1)

    with pytest.raises(RuntimeError):
        kmeans.set_params()

    with pytest.raises(RuntimeError):
        kmeans.set_output()

    with pytest.raises(RuntimeError):
        kmeans.set_fit_request()

    assert kmeans.n_features_in_ is None
    assert kmeans.feature_names_in_ is None


if __name__ == "__main__":
    test_kmeans()
    test_kmeans_errors()
