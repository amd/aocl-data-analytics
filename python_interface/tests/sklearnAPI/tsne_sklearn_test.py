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
t-SNE tests, check output of skpatch versus sklearn.
"""

# pylint: disable = import-outside-toplevel, reimported, no-member

import numpy as np
import pytest
from aoclda.sklearn import skpatch, undo_skpatch


def get_tsne_data(precision):
    """Load a compact deterministic dataset for TSNE patch checks."""
    from sklearn.datasets import load_iris

    iris = load_iris()
    return iris.data.astype(precision)


@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("method", ["exact", "barnes_hut"])
def test_tsne(precision, method):
    """
    Compare AOCL-DA patched TSNE against sklearn TSNE on identical input.
    """
    from sklearn.manifold import trustworthiness

    X = get_tsne_data(precision)
    tol = 0.02

    # patch and import scikit-learn
    skpatch("TSNE")
    from sklearn.manifold import TSNE as TSNE_model
    tsne_da = TSNE_model(
        n_components=2,
        perplexity=30.0,
        max_iter=350,
        init="random",
        random_state=42,
        method=method,
        angle=0.5)
    da_emb = tsne_da.fit_transform(X)
    da_kl = tsne_da.kl_divergence_
    da_trust = trustworthiness(X, da_emb, n_neighbors=10)
    da_params = tsne_da.get_params()
    da_n_iter = tsne_da.n_iter_
    da_n_features = tsne_da.n_features_in_
    assert tsne_da.aocl is True

    # unpatch and solve the same problem with sklearn
    undo_skpatch("TSNE")
    from sklearn.manifold import TSNE as TSNE_model
    tsne_sk = TSNE_model(
        n_components=2,
        perplexity=30.0,
        max_iter=350,
        init="random",
        random_state=42,
        method=method,
        angle=0.5)
    sk_emb = tsne_sk.fit_transform(X)
    sk_kl = tsne_sk.kl_divergence_
    sk_trust = trustworthiness(X, sk_emb, n_neighbors=10)
    sk_params = tsne_sk.get_params()
    sk_n_iter = tsne_sk.n_iter_
    sk_n_features = tsne_sk.n_features_in_
    assert not hasattr(tsne_sk, "aocl")

    # Check results
    assert da_emb.shape == sk_emb.shape
    assert np.all(np.isfinite(da_emb))
    assert np.all(np.isfinite(sk_emb))
    assert da_trust == pytest.approx(sk_trust, abs=tol)
    assert da_kl == pytest.approx(sk_kl, abs=tol)
    assert da_params == sk_params  # This might break if sklearn changes the params
    assert da_n_iter > 0
    assert abs(da_n_iter - sk_n_iter) <= 1
    assert da_n_features == sk_n_features


@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_double_solve(precision):
    """
    Check that solving the model twice doesn't fail.
    """
    X = get_tsne_data(precision)

    skpatch("TSNE")
    from sklearn.manifold import TSNE as TSNE_model
    tsne_da = TSNE_model(
        n_components=2,
        perplexity=25.0,
        max_iter=300,
        init="random",
        random_state=7,
        method="exact")
    emb1 = tsne_da.fit_transform(X)
    emb2 = tsne_da.fit_transform(X)

    assert tsne_da.aocl is True
    assert emb1.shape == emb2.shape
    assert np.allclose(emb1, emb2, atol=1e-6)
    assert np.all(np.isfinite(emb2))
    undo_skpatch("TSNE")


@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_tsne_errors(precision):
    """
    Check we can catch errors in the sklearn tsne patch.
    """
    X = get_tsne_data(precision)
    skpatch("TSNE")
    from sklearn.manifold import TSNE as TSNE_model

    with pytest.raises(ValueError):
        TSNE_model(n_components=4)

    with pytest.raises(ValueError):
        TSNE_model(init="unsupported")

    # Init arrays must be 2D with shape (n_samples, n_components).
    with pytest.raises(ValueError):
        TSNE_model(init=np.array([1.0, 2.0]))

    # random_state must be an integer or None.
    with pytest.raises(ValueError):
        TSNE_model(random_state=np.random.RandomState(1))

    # Runtime validation errors delegated to AOCL-DA tsne backend/wrapper.
    with pytest.raises(RuntimeError):
        TSNE_model(perplexity=-1.0).fit(X)

    with pytest.raises(ValueError):
        TSNE_model(learning_rate="invalid").fit(X)

    # Unsupported parameters should trigger warnings.
    with pytest.warns(RuntimeWarning):
        TSNE_model(metric="manhattan")

    with pytest.warns(RuntimeWarning):
        TSNE_model(verbose=1)

    with pytest.warns(RuntimeWarning):
        TSNE_model(metric_params={"p": 1})

    with pytest.warns(RuntimeWarning):
        TSNE_model(n_jobs=2)

    with pytest.warns(RuntimeWarning):
        TSNE_model(method="invalid_method")

    with pytest.warns(RuntimeWarning):
        TSNE_model(perplexity=1000.0).fit(X)

    # Not implemented
    tsne_da = TSNE_model(random_state=0, init="random")
    tsne_da.fit(X)

    with pytest.raises(RuntimeError):
        tsne_da.set_params()

    with pytest.raises(RuntimeError):
        tsne_da.set_output()
    undo_skpatch("TSNE")


def test_tsne_mixed_precision_sklearn():
    """
    Verify that mixed precision options pass through the sklearn wrapper.
    """
    rng = np.random.default_rng(123)
    X = rng.standard_normal((20, 10)).astype(np.float64)

    skpatch("TSNE")
    from sklearn.manifold import TSNE as TSNE_model
    model = TSNE_model(
        n_components=2,
        perplexity=5.0,
        max_iter=300,
        init="random",
        random_state=42,
        method="exact",
        mixed_precision=True,
        low_precision_max_iter=100,
        low_precision_min_grad_norm=1e-3)
    assert model.aocl is True
    emb = model.fit_transform(X)
    assert emb.shape == (20, 2)
    assert np.all(np.isfinite(emb))
    undo_skpatch("TSNE")
