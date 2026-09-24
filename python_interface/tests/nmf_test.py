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
Nonnegative matrix factorization Python tests
"""

import numpy as np
import pytest
from aoclda.factorization import NMF


@pytest.mark.parametrize(
    "numpy_precision",
    [np.float16, np.float32, np.float64, np.int16, np.int32, np.int64, 'object'])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_nmf_all_dtypes(numpy_precision, numpy_order):
    """
    Test it runs when supported/unsupported C-interface type is provided.
    """
    a = np.array([[1.3, 2.53, 3.86], [2.4, 5.5, 4.5], [3.33, 6.21, 1.76]],
                 dtype=numpy_precision, order=numpy_order)
    x = np.array([[1.2, 1.1, 4.3], [3.333, 2.6, 3.4], [0.3, 2.2, 3.8],
                  [1.8, 0.7, 1.3]], dtype=numpy_precision, order=numpy_order)

    nmf = NMF(n_components=2, random_state=0)
    nmf.fit(a)
    nmf.inverse_transform(nmf.transform(x))
    nmf.fit_transform(a)
    nmf.transform(a)


@pytest.mark.parametrize("numpy_precision", [np.float32])
@pytest.mark.parametrize("numpy_orders", [("C", "F"), ("F", "C")])
def test_nmf_multiple_orders(numpy_precision, numpy_orders):
    """
    Test it runs when arrays of multiple orders are provided.
    """
    a = np.array([[1.3, 2.53, 3.86], [2.4, 5.5, 4.5], [3.33, 6.21, 1.76]],
                 dtype=numpy_precision, order=numpy_orders[0])
    x = np.array([[1.2, 1.1, 4.3], [3.333, 2.6, 3.4], [0.3, 2.2, 3.8],
                  [1.8, 0.7, 1.3]], dtype=numpy_precision, order=numpy_orders[1])

    nmf = NMF(n_components=2, random_state=0)
    nmf.fit(a)
    with pytest.warns(UserWarning):
        nmf.inverse_transform(nmf.transform(x))
    with pytest.warns(UserWarning):
        nmf.fit_transform(x)
    a = np.array(a, order=numpy_orders[1])
    with pytest.warns(UserWarning):
        nmf.fit(a)
    nmf.transform(a)


@pytest.mark.parametrize(
    "numpy_precisions", [('float32', 'float64'), ('float64', 'float32')])
@pytest.mark.parametrize("numpy_order", ["C"])
def test_nmf_multiple_dtypes(numpy_precisions, numpy_order):
    """
    Test it runs when arrays of multiple dtypes are provided.
    """
    a = np.array([[1.3, 2.53, 3.86], [2.4, 5.5, 4.5], [3.33, 6.21, 1.76]],
                 dtype=numpy_precisions[0], order=numpy_order)
    x = np.array([[1.2, 1.1, 4.3], [3.333, 2.6, 3.4], [0.3, 2.2, 3.8],
                  [1.8, 0.7, 1.3]], dtype=numpy_precisions[1], order=numpy_order)

    nmf = NMF(n_components=2, random_state=0)
    nmf.fit(a)
    nmf.inverse_transform(nmf.transform(x))
    nmf.fit_transform(x)
    a = np.array(a, dtype=numpy_precisions[1])
    nmf.fit(a)
    nmf.transform(a)


@pytest.mark.parametrize("numpy_precision", [np.float32, np.float64])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
@pytest.mark.parametrize("init", ['nndsvd', 'nndsvda', 'nndsvdar', 'random', 'custom'])
def test_nmf_functionality(numpy_precision, numpy_order, init):
    """
    Test the functionality of the Python wrapper for each supported init method.
    """
    # Start by constructing the factors W and H and use them to create matrix A.
    W_true = np.array([[2.0, 0.0], [2.0, 0.0], [0.0, 3.0], [0.0, 3.0]])
    H_true = np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    A = (W_true @ H_true).astype(numpy_precision)
    if numpy_order == 'F':
        A = np.asfortranarray(A)
    solve_tol = np.finfo(numpy_precision).eps * 1000
    # The solver is iterative so need a separate tolerance for checking the results.
    tol = 1e-4 if numpy_precision == np.float32 else 5e-7
    num_tol = np.sqrt(np.finfo(numpy_precision).eps) * 10

    W0, H0 = None, None
    if init == 'custom':
        rng = np.random.default_rng(42)
        W0 = rng.random((4, 2)).astype(numpy_precision)
        H0 = rng.random((2, 3)).astype(numpy_precision)

    # X_new = [1, 1] @ H_true = [1, 2, 1] lies in the column space of H_true,
    # so inverse_transform(transform(X_new)) recovers X_new exactly.
    X_new = np.array([[1.0, 2.0, 1.0]], dtype=numpy_precision, order=numpy_order)
    expected_rec_new = np.array([[1.0, 2.0, 1.0]])

    # (for custom init fit_transform is the only entry point that accepts W/H)
    nmf_1 = NMF(n_components=2, solver='mu', beta_loss='frobenius',
                max_iter=500, tol=solve_tol, init=init, random_state=0)
    nmf_1.fit(A, W=W0, H=H0)
    # Get results
    H = nmf_1.h
    W = nmf_1.w
    WH = W @ H
    assert np.linalg.norm(WH - A.astype(np.float64)) < tol
    assert abs(
        nmf_1.reconstruction_err -
        np.linalg.norm(
            A.astype(
                np.float64) -
            WH)) < num_tol
    assert H.shape == (2, 3)
    assert W.shape == (4, 2)
    assert np.all(H >= 0.0)
    assert np.all(W >= 0.0)
    assert nmf_1.n_components == 2
    assert nmf_1.n_samples == 4
    assert nmf_1.n_features == 3

    # transform(A) re-solves for W with H fixed and must converge close to stored W.
    Xt = nmf_1.transform(A)
    assert Xt.shape == (4, 2)
    assert np.all(Xt >= 0.0)
    assert np.linalg.norm(Xt - W) / np.linalg.norm(W) < tol

    A_rec = nmf_1.inverse_transform(Xt)
    assert A_rec.shape == A.shape
    assert np.all(A_rec >= 0.0)

    Xt_new = nmf_1.transform(X_new)
    assert Xt_new.shape == (1, 2)
    assert np.all(Xt_new >= 0.0)
    X_rec = nmf_1.inverse_transform(Xt_new)
    assert X_rec.shape == X_new.shape
    assert np.all(X_rec >= 0.0)
    assert np.linalg.norm(X_rec - expected_rec_new) / \
        np.linalg.norm(expected_rec_new) < tol

    # Check that fit_transform() + inverse_transform() is consistent with
    # fit() + transform() + inverse_transform().
    nmf_2 = NMF(n_components=2, solver='mu', beta_loss='frobenius',
                max_iter=500, tol=solve_tol, init=init, random_state=0)
    W_ft = nmf_2.fit_transform(A, W=W0, H=H0)
    assert W_ft.shape == (4, 2)
    assert np.all(W_ft >= 0.0)
    # fit_transform must return exactly the stored transform_matrix
    assert np.array_equal(W_ft, nmf_2.w)
    assert nmf_2.h.shape == (2, 3)
    assert np.all(nmf_2.h >= 0.0)
    WH_2 = W_ft @ nmf_2.h
    assert np.linalg.norm(WH_2 - A.astype(np.float64)) < tol
    assert abs(
        nmf_2.reconstruction_err -
        np.linalg.norm(
            A.astype(
                np.float64) -
            WH_2)) < num_tol

    # pipeline: fit_transform output feeds directly into inverse_transform
    A_rec_2 = nmf_2.inverse_transform(W_ft)
    assert A_rec_2.shape == A.shape
    assert np.all(A_rec_2 >= 0.0)

    Xt_new_2 = nmf_2.transform(X_new)
    assert Xt_new_2.shape == (1, 2)
    assert np.all(Xt_new_2 >= 0.0)
    X_rec_2 = nmf_2.inverse_transform(Xt_new_2)
    assert X_rec_2.shape == X_new.shape
    assert np.all(X_rec_2 >= 0.0)
    assert np.linalg.norm(X_rec_2 - expected_rec_new) < tol


# ---------------------------------------------------------------------------
# n_components='auto' / None
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_components", ['auto', None])
def test_nmf_auto_n_components(n_components):
    """
    Test that n_components='auto'/None resolves to min(n_samples, n_features).
    Both branches of min() are tested: n_features < n_samples and n_samples < n_features.
    """
    # Tall matrix: min(n_samples=4, n_features=3) = 3
    A_tall = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9],
                       [10, 11, 12]], dtype=np.float64)
    nmf = NMF(n_components=n_components)
    nmf.fit(A_tall)
    assert nmf.n_components == 3
    assert nmf.n_samples == 4
    assert nmf.n_features == 3
    # w is W stored by fit
    W = nmf.w
    assert W.shape == (4, 3)
    assert np.all(W >= 0.0)
    assert nmf.h.shape == (3, 3)
    # transform() re-solves for W given fixed H; shape must match transform_matrix
    Xt = nmf.transform(A_tall)
    assert Xt.shape == (4, 3)
    assert np.all(Xt >= 0.0)

    # Wide matrix: min(n_samples=2, n_features=5) = 2
    A_wide = np.array([[1, 2, 3, 4, 5],
                       [6, 7, 8, 9, 10]], dtype=np.float64)
    nmf = NMF(n_components=n_components)
    nmf.fit(A_wide)
    assert nmf.n_components == 2
    assert nmf.n_samples == 2
    assert nmf.n_features == 5
    # w is W stored by fit
    W = nmf.w
    assert W.shape == (2, 2)
    assert np.all(W >= 0.0)
    assert nmf.h.shape == (2, 5)
    # transform() on a held-out row must return shape (1, n_components)
    X_new = np.array([[2, 4, 6, 8, 10]], dtype=np.float64)
    Xt = nmf.transform(X_new)
    assert Xt.shape == (1, 2)
    assert np.all(Xt >= 0.0)


def test_nmf_error_exits():
    """
    Test error exits in the Python wrapper.
    """
    # Invalid n_components
    with pytest.raises(ValueError):
        NMF(n_components=0)
    with pytest.raises(ValueError):
        NMF(n_components=-1)
    with pytest.raises(ValueError):
        NMF(n_components=1.5)
    with pytest.raises(ValueError):
        NMF(n_components='bad')

    # Invalid random_state
    with pytest.raises(ValueError):
        NMF(random_state=1.5)
    with pytest.raises(ValueError):
        NMF(random_state='seed')

    # Invalid check_every
    with pytest.raises(ValueError):
        NMF(check_every=0)
    with pytest.raises(ValueError):
        NMF(check_every=-1)
    with pytest.raises(ValueError):
        NMF(check_every=1.5)

    # Supplying only W or only H to fit_transform must raise ValueError
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    nmf = NMF(n_components=1)
    with pytest.raises(ValueError):
        nmf.fit_transform(A, W=np.ones((2, 1)))
    with pytest.raises(ValueError):
        nmf.fit_transform(A, H=np.ones((1, 2)))

    # Same check via fit()
    with pytest.raises(ValueError):
        nmf.fit(A, W=np.ones((2, 1)))
    with pytest.raises(ValueError):
        nmf.fit(A, H=np.ones((1, 2)))


@pytest.mark.parametrize("a_order", ["C", "F"])
@pytest.mark.parametrize("w_order", ["C", "F"])
@pytest.mark.parametrize("h_order", ["C", "F"])
def test_nmf_custom_wh_mixed_orders(a_order, w_order, h_order):
    """
    Test that custom W and H with different memory orders from A are handled.
    """
    rng = np.random.default_rng(42)
    A = np.array([[1.3, 2.53, 3.86], [2.4, 5.5, 4.5],
                  [3.33, 6.21, 1.76], [1.0, 2.0, 3.0]],
                 dtype=np.float64, order=a_order)
    W0 = np.array(rng.random((4, 2)), dtype=np.float64, order=w_order)
    H0 = np.array(rng.random((2, 3)), dtype=np.float64, order=h_order)

    nmf = NMF(n_components=2, init='custom', random_state=0, max_iter=500)
    nmf.fit(A, W=W0, H=H0)
    assert nmf.w.shape == (4, 2)
    assert nmf.h.shape == (2, 3)
    assert np.all(nmf.w >= 0.0)
    assert np.all(nmf.h >= 0.0)

    W_ft = nmf.fit_transform(A, W=W0, H=H0)
    assert np.array_equal(W_ft, nmf.w)


@pytest.mark.parametrize("a_dtype", ["float32", "float64"])
@pytest.mark.parametrize("w_dtype", ["float32", "float64"])
@pytest.mark.parametrize("h_dtype", ["float32", "float64"])
def test_nmf_custom_wh_mixed_dtypes(a_dtype, w_dtype, h_dtype):
    """
    Test that custom W and H with different dtypes from A are handled.
    """
    rng = np.random.default_rng(42)
    A = np.array([[1.3, 2.53, 3.86], [2.4, 5.5, 4.5],
                  [3.33, 6.21, 1.76], [1.0, 2.0, 3.0]], dtype=a_dtype)
    W0 = np.array(rng.random((4, 2)), dtype=w_dtype)
    H0 = np.array(rng.random((2, 3)), dtype=h_dtype)

    nmf = NMF(n_components=2, init='custom', random_state=0, max_iter=500)
    nmf.fit(A, W=W0, H=H0)
    assert nmf.w.shape == (4, 2)
    assert nmf.h.shape == (2, 3)
    assert np.all(nmf.w >= 0.0)
    assert np.all(nmf.h >= 0.0)

    W_ft = nmf.fit_transform(A, W=W0, H=H0)
    assert np.array_equal(W_ft, nmf.w)
