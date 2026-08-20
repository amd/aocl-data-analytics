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
KernelPCA tests, check output of skpatch versus sklearn
"""

# pylint: disable = import-outside-toplevel, reimported, no-member

import numpy as np
import pytest
from aoclda.sklearn import skpatch, undo_skpatch


@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_kernel_pca(precision):
    """
    Basic 6 x 3 problem with RBF kernel
    """
    a = np.array([[1, 2, 3], [0.22, 5, 4.1], [3, 6, 1],
                  [2, 3, 1.5], [0.5, 1.2, 2.8], [4, 2, 0.6]], dtype=precision)
    b = np.array([[3, 2, 3], [1.22, 5, 4.1], [3, 3, 1]], dtype=precision)

    tol = np.sqrt(np.finfo(precision).eps)

    # patch and import scikit-learn
    skpatch()
    from sklearn.decomposition import KernelPCA
    nc = 2
    kpca_da = KernelPCA(n_components=nc, kernel='rbf', gamma=0.5,
                        fit_inverse_transform=True, eigen_solver='dense')
    kpca_da.fit(a)
    da_transform = kpca_da.transform(b)
    da_fit_transform = kpca_da.fit_transform(a)
    da_eigenvalues = kpca_da.eigenvalues_
    da_eigenvectors = kpca_da.eigenvectors_
    da_scores = kpca_da.X_transformed_fit_
    da_dual_coef = kpca_da.dual_coef_
    da_reconstructed = kpca_da.inverse_transform(da_scores)
    assert kpca_da.aocl is True

    # unpatch and solve the same problem with sklearn
    undo_skpatch()
    from sklearn.decomposition import KernelPCA
    kpca = KernelPCA(n_components=nc, kernel='rbf', gamma=0.5,
                     fit_inverse_transform=True, eigen_solver='dense')
    kpca.fit(a)
    sk_transform = kpca.transform(b)
    sk_fit_transform = kpca.fit_transform(a)
    sk_eigenvalues = kpca.eigenvalues_
    sk_eigenvectors = kpca.eigenvectors_
    sk_scores = kpca.X_transformed_fit_
    sk_dual_coef = kpca.dual_coef_
    sk_reconstructed = kpca.inverse_transform(sk_scores)
    assert not hasattr(kpca, 'aocl')

    # Check results
    assert da_eigenvalues == pytest.approx(sk_eigenvalues, abs=tol)
    assert np.abs(da_eigenvectors) == pytest.approx(np.abs(sk_eigenvectors), abs=tol)
    assert np.abs(da_scores) == pytest.approx(np.abs(sk_scores), abs=tol)
    assert np.abs(da_transform) == pytest.approx(np.abs(sk_transform), abs=tol)
    assert np.abs(da_fit_transform) == pytest.approx(np.abs(sk_fit_transform), abs=tol)
    assert da_dual_coef.shape == sk_dual_coef.shape
    assert da_reconstructed == pytest.approx(sk_reconstructed, abs=tol)
    assert kpca_da.n_features_in_ == kpca.n_features_in_
    assert kpca_da.n_components_ == nc
    assert kpca_da.n_samples_ == a.shape[0]

    # print the results if pytest is invoked with the -rA option
    print("Eigenvalues")
    print("    aoclda: \n", da_eigenvalues)
    print("   sklearn: \n", sk_eigenvalues)


@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_double_solve(precision):
    """
    Check that solving the model twice doesn't fail
    """
    a = np.array([[1, 2, 3], [0.22, 5, 4.1], [3, 6, 1],
                  [2, 3, 1.5], [0.5, 1.2, 2.8], [4, 2, 0.6]], dtype=precision)
    skpatch()
    from sklearn.decomposition import KernelPCA
    kpca_da = KernelPCA(n_components=2, kernel='rbf', gamma=0.5,
                        eigen_solver='randomized', random_state=42)
    kpca_da.fit(a)
    kpca_da.fit(a)
    undo_skpatch()


def test_kernel_pca_errors():
    """
    Check we can catch errors in the sklearn kernel_pca patch
    """
    a = np.array([[1, 2, 3], [0.22, 5, 4.1], [3, 6, 1],
                  [2, 3, 1.5], [0.5, 1.2, 2.8], [4, 2, 0.6]])

    skpatch()
    from sklearn.decomposition import KernelPCA

    with pytest.raises(ValueError):
        KernelPCA(n_components=0)

    with pytest.raises(ValueError):
        KernelPCA(kernel=lambda x, y: x @ y.T)

    with pytest.raises(ValueError):
        KernelPCA(kernel='cosine')

    with pytest.raises(ValueError):
        KernelPCA(kernel_params={'gamma': 0.5})

    with pytest.raises(ValueError):
        KernelPCA(eigen_solver='arpack')

    with pytest.raises(ValueError):
        KernelPCA(n_components=2, random_state=np.random.RandomState(42))

    with pytest.warns(RuntimeWarning):
        kpca = KernelPCA(n_components=2, kernel='rbf', tol=1e-3)

    kpca.fit(a)

    # Test unsupported functions
    with pytest.raises(RuntimeError):
        kpca.get_feature_names_out()

    with pytest.raises(RuntimeError):
        kpca.get_metadata_routing()

    with pytest.raises(RuntimeError):
        kpca.set_output()

    with pytest.raises(RuntimeError):
        kpca.set_params()

    with pytest.raises(AttributeError):
        _ = kpca.feature_names_in_

    undo_skpatch()


if __name__ == "__main__":
    test_kernel_pca()
    test_kernel_pca_errors()
