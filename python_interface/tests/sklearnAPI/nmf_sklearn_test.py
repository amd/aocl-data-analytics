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
NMF tests, check output of skpatch versus sklearn
"""

# pylint: disable = import-outside-toplevel, reimported, no-member

import numpy as np
import pytest
from aoclda.sklearn import skpatch, undo_skpatch


@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("init", ['nndsvd', 'nndsvda', 'nndsvdar', 'random', 'custom'])
def test_nmf(precision, init):

    a = np.array(
        [[1, 2, 3, 4],
         [2, 4, 6, 8],
         [3, 6, 9, 12],
         [4, 1, 5, 2],
         [5, 3, 8, 4]],
        dtype=precision,
    )
    max_iter = 200
    tol = 10 * np.sqrt(np.finfo(precision).eps)

    if init == 'custom':
        W0 = np.array(
            [[0.5, 1.0],
             [1.0, 0.5],
             [1.5, 1.0],
             [2.0, 0.5],
             [1.0, 1.5]],
            dtype=precision,
        )
        H0 = np.array(
            [[1.0, 0.5, 1.5, 1.0],
             [0.5, 1.0, 0.5, 1.5]],
            dtype=precision,
        )

    skpatch()
    from sklearn.decomposition import NMF
    nmf_da = NMF(n_components=2, init=init, solver='mu', beta_loss='frobenius',
                 max_iter=max_iter, tol=1e-6, random_state=0)
    if init == 'custom':
        da_W = nmf_da.fit_transform(a, W=W0.copy(), H=H0.copy())
    else:
        da_W = nmf_da.fit_transform(a)
    da_H = nmf_da.components_
    da_WH = da_W @ da_H
    da_err = nmf_da.reconstruction_err_
    da_n_iter = nmf_da.n_iter_
    da_params = nmf_da.get_params()
    da_Xt = nmf_da.inverse_transform(da_W)
    assert nmf_da.aocl is True

    undo_skpatch()
    from sklearn.decomposition import NMF
    nmf_sk = NMF(n_components=2, init=init, solver='mu', beta_loss='frobenius',
                 max_iter=max_iter, tol=1e-6, random_state=0)
    if init == 'custom':
        sk_W = nmf_sk.fit_transform(a, W=W0.copy(), H=H0.copy())
    else:
        sk_W = nmf_sk.fit_transform(a)
    sk_H = nmf_sk.components_
    sk_WH = sk_W @ sk_H
    sk_err = nmf_sk.reconstruction_err_
    sk_n_iter = nmf_sk.n_iter_
    sk_params = nmf_sk.get_params()
    sk_Xt = nmf_sk.inverse_transform(sk_W)
    assert not hasattr(nmf_sk, 'aocl')

    n_rows, n_cols = a.shape

    # Shapes
    assert da_W.shape == sk_W.shape == (n_rows, 2)
    assert da_H.shape == sk_H.shape == (2, n_cols)
    assert da_Xt.shape == a.shape
    assert nmf_da.n_components_ == nmf_sk.n_components_
    assert nmf_da.n_features_in_ == nmf_sk.n_features_in_

    # Nonnegativity
    assert np.all(da_W >= 0.0)
    assert np.all(da_H >= 0.0)
    assert np.all(da_Xt >= 0.0)
    assert da_err >= 0.0
    assert 1 <= da_n_iter <= max_iter

    # Parameters
    assert da_params == sk_params

    a_fro = np.linalg.norm(a)
    if init in ('nndsvd', 'nndsvda', 'custom'):
        # Deterministic init: AOCL-DA and sklearn follow the same MU trajectory.
        assert da_WH == pytest.approx(sk_WH, rel=tol)
        assert da_err == pytest.approx(sk_err, abs=tol)
        assert da_Xt == pytest.approx(da_WH, rel=tol)
        assert da_Xt == pytest.approx(sk_Xt, rel=tol)
    elif init == 'nndsvdar':
        # Random perturbation of zeros. Different RNGs cause different cross-backend tol.
        assert da_WH == pytest.approx(sk_WH, abs=0.01 * a_fro)
        assert da_Xt == pytest.approx(da_WH, rel=tol)
    else:
        # 'random': fully random init with different RNGs, so only check self-consistency.
        assert da_Xt == pytest.approx(da_WH, rel=tol)

    print(f"init={init}  reconstruction_err_  aocl: {da_err}  sklearn: {sk_err}")
    print(f"init={init}  n_iter_              aocl: {da_n_iter}  sklearn: {sk_n_iter}")

    if init != 'custom':
        # Also test the fit() + transform() API: in-sample and out-of-sample.
        # H is frozen after fit(), transform() re-solves W with H fixed.
        x_new = np.array([[2.0, 3.0, 5.0, 4.0],
                          [1.0, 1.0, 2.0, 1.0]], dtype=precision)

        skpatch()
        from sklearn.decomposition import NMF
        nmf_da2 = NMF(n_components=2, init=init, solver='mu', beta_loss='frobenius',
                      max_iter=max_iter, tol=1e-6, random_state=0)
        nmf_da2.fit(a)
        da_W_t = nmf_da2.transform(a)
        da_WH_t = da_W_t @ nmf_da2.components_
        da_xt_new = nmf_da2.transform(x_new)
        da_Xt_new = nmf_da2.inverse_transform(da_xt_new)
        undo_skpatch()

        from sklearn.decomposition import NMF
        nmf_sk2 = NMF(n_components=2, init=init, solver='mu', beta_loss='frobenius',
                      max_iter=max_iter, tol=1e-6, random_state=0)
        nmf_sk2.fit(a)
        sk_W_t = nmf_sk2.transform(a)
        sk_WH_t = sk_W_t @ nmf_sk2.components_
        sk_xt_new = nmf_sk2.transform(x_new)
        sk_Xt_new = nmf_sk2.inverse_transform(sk_xt_new)

        # In-sample transform
        assert da_W_t.shape == (n_rows, 2)
        assert np.all(da_W_t >= 0.0)
        assert np.all(da_WH_t >= 0.0)
        if init in ('nndsvd', 'nndsvda'):
            assert da_WH_t == pytest.approx(sk_WH_t, rel=tol)
        elif init == 'nndsvdar':
            assert da_WH_t == pytest.approx(sk_WH_t, abs=0.01 * a_fro)

        # Out-of-sample transform + inverse_transform
        x_fro = np.linalg.norm(x_new)
        assert da_xt_new.shape == (x_new.shape[0], 2)
        assert da_Xt_new.shape == x_new.shape
        assert np.all(da_xt_new >= 0.0)
        assert np.all(da_Xt_new >= 0.0)
        if init in ('nndsvd', 'nndsvda'):
            assert da_Xt_new == pytest.approx(sk_Xt_new, abs=tol * x_fro + 5e-2 * x_fro)
        print(
            f"init={init}  fit+transform  da_WH_t[0]: {da_WH_t[0]}  sk_WH_t[0]: {sk_WH_t[0]}")


@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_double_solve(precision):
    """Calling fit twice on the same estimator must give the same result."""
    a = np.array(
        [[1, 2, 3, 4],
         [2, 4, 6, 8],
         [3, 6, 9, 12],
         [4, 1, 5, 2],
         [5, 3, 8, 4]],
        dtype=precision,
    )
    tol = np.sqrt(np.finfo(precision).eps)

    skpatch()
    from sklearn.decomposition import NMF
    nmf_da = NMF(
        n_components=2,
        init='nndsvda',
        solver='mu',
        beta_loss='frobenius',
        max_iter=50,
        tol=1e-6,
        random_state=0,
    )
    nmf_da.fit(a)
    err1 = nmf_da.reconstruction_err_
    n1 = nmf_da.n_iter_
    nmf_da.fit(a)
    err2 = nmf_da.reconstruction_err_
    n2 = nmf_da.n_iter_
    undo_skpatch()

    assert err1 == pytest.approx(err2, rel=tol, abs=tol)
    assert n1 == n2

    print("err1:", err1, "err2:", err2)
    print("n1:", n1, "n2:", n2)


def test_nmf_partial_wh_error():
    """Supplying only one of W / H to fit_transform with init='custom' must raise ValueError."""
    a = np.array(
        [[1, 2, 3, 4],
         [2, 4, 6, 8],
         [3, 6, 9, 12],
         [4, 1, 5, 2],
         [5, 3, 8, 4]],
        dtype=np.float64,
    )
    W0 = np.ones((5, 2))

    skpatch()
    from sklearn.decomposition import NMF
    nmf_da = NMF(n_components=2, init='custom', solver='mu')

    with pytest.raises(ValueError):
        nmf_da.fit_transform(a, W=W0)
    with pytest.raises(ValueError):
        nmf_da.fit_transform(a, H=W0.T)
    undo_skpatch()


def test_nmf_errors():
    """Argument validation paths exposed by the skpatched NMF."""
    skpatch()
    from sklearn.decomposition import NMF

    # Any solver other than 'mu' must be rejected at construction.
    with pytest.raises(ValueError):
        NMF(n_components=2, solver='cd')
    with pytest.raises(ValueError):
        NMF(n_components=2, solver='unknown')
    # Wrong type for solver (e.g. int) must also raise ValueError.
    with pytest.raises(ValueError):
        NMF(n_components=2, solver=1)
    # None is not accepted; only the string 'mu' is valid.
    with pytest.raises(ValueError):
        NMF(n_components=2, solver=None)

    # Invalid beta_loss — only the string 'frobenius' is accepted.
    with pytest.raises(ValueError):
        NMF(n_components=2, solver='mu', beta_loss=None)
    with pytest.raises(ValueError):
        NMF(n_components=2, solver='mu', beta_loss='kullback-leibler')
    with pytest.raises(ValueError):
        NMF(n_components=2, solver='mu', beta_loss='itakura-saito')
    # Wrong type for beta_loss must also raise ValueError.
    with pytest.raises(ValueError):
        NMF(n_components=2, solver='mu', beta_loss=2)

    # ---- init -------------------------------------------------------------
    # All supported init values construct without error.
    for init_val in (None, 'nndsvd', 'nndsvda', 'nndsvdar', 'random', 'custom'):
        NMF(n_components=2, solver='mu', init=init_val, random_state=0)

    # 'auto' is not a valid sklearn init option and must be rejected.
    with pytest.raises(ValueError):
        NMF(n_components=2, solver='mu', init='auto')
    # Unsupported init string is rejected.
    with pytest.raises(ValueError):
        NMF(n_components=2, solver='mu', init='unknown')
    # Wrong type for init must also raise ValueError.
    with pytest.raises(ValueError):
        NMF(n_components=2, solver='mu', init=0)

    # ---- unsupported-but-silent parameters --------------------------------
    # Each individually emits a RuntimeWarning and does not raise.
    with pytest.warns(RuntimeWarning):
        NMF(n_components=2, solver='mu', alpha_W=0.1)
    with pytest.warns(RuntimeWarning):
        NMF(n_components=2, solver='mu', alpha_H=0.1)
    with pytest.warns(RuntimeWarning):
        NMF(n_components=2, solver='mu', l1_ratio=0.5)
    with pytest.warns(RuntimeWarning):
        NMF(n_components=2, solver='mu', verbose=1)
    with pytest.warns(RuntimeWarning):
        NMF(n_components=2, solver='mu', shuffle=True)

    # ---- unimplemented API surface ----------------------------------------
    a = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0]])
    nmf = NMF(n_components=2, solver='mu')
    nmf.fit(a)

    with pytest.raises(RuntimeError):
        nmf.set_params()
    with pytest.raises(RuntimeError):
        nmf.set_output(transform='pandas')
    with pytest.raises(RuntimeError):
        nmf.get_feature_names_out()
    with pytest.raises(RuntimeError):
        nmf.get_metadata_routing()
    with pytest.raises(RuntimeError):
        nmf.get_feature_names_out()
    with pytest.raises(RuntimeError):
        nmf.get_metadata_routing()

    # feature_names_in_ is intentionally a no-op accessor.
    assert nmf.feature_names_in_ is None

    undo_skpatch()


if __name__ == "__main__":
    for _init in ['nndsvd', 'nndsvda', 'nndsvdar', 'random', 'custom']:
        test_nmf(np.float64, _init)
        test_nmf(np.float32, _init)
    test_double_solve(np.float64)
    test_double_solve(np.float32)
    test_nmf_partial_wh_error()
    test_nmf_errors()
