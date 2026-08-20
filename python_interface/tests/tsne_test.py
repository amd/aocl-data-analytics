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
t-SNE Python test script
"""

import numpy as np
import pytest
from aoclda.dimension_reduction import tsne


def get_tsne_data(dtype=np.float64, order="C"):
    """Full iris dataset: 150 samples, 4 features, 3 classes."""
    from sklearn.datasets import load_iris
    return np.array(load_iris().data, dtype=dtype, order=order)


@pytest.mark.parametrize(
    "numpy_precision",
    [np.float16, np.float32, np.float64, np.int16, np.int32, np.int64, "object"])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_tsne_all_dtypes(numpy_precision, numpy_order):
    """
    Test it runs when supported/unsupported C-interface type is provided.
    """
    x = get_tsne_data(dtype=numpy_precision, order=numpy_order)
    model = tsne(n_components=2, perplexity=30.0,
                 max_iter=150, theta=0.5, seed=7)

    model.fit(x)

    embedding_fit = model.embedding
    assert embedding_fit.shape == (x.shape[0], 2)
    assert np.all(np.isfinite(embedding_fit))
    expected_dtype = np.float32 if model.dtype == "float32" else np.float64
    assert embedding_fit.dtype == np.dtype(expected_dtype)
    assert model.n_samples == x.shape[0]
    assert model.n_features == x.shape[1]
    assert model.n_components == 2
    assert model.n_iter == 150
    assert model.lp_n_iter == 0
    assert np.isfinite(model.kl_divergence)

    # Exercise fit_transform() path as well.
    embedding_ft = model.fit_transform(x)
    assert embedding_ft.shape == (x.shape[0], 2)
    assert np.all(np.isfinite(embedding_ft))
    assert embedding_ft.dtype == np.dtype(expected_dtype)


@pytest.mark.parametrize("numpy_precision", [np.float32])
@pytest.mark.parametrize("numpy_orders", [("C", "F"), ("F", "C")])
def test_tsne_multiple_orders(numpy_precision, numpy_orders):
    """
    Test it runs when arrays of multiple orders are provided.
    """
    x1 = get_tsne_data(dtype=numpy_precision, order=numpy_orders[0])
    x2 = get_tsne_data(dtype=numpy_precision, order=numpy_orders[1])
    model = tsne(n_components=2, perplexity=30.0,
                 max_iter=150, theta=0.5, seed=7)
    emb1 = model.fit_transform(x1)
    with pytest.warns(UserWarning):
        emb2 = model.fit_transform(x2)
    assert np.all(np.isfinite(emb2))
    assert np.allclose(emb1, emb2)


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
@pytest.mark.parametrize(
    "tsne_method,theta",
    [("exact", 0.0), ("barnes_hut", 0.5)],
    ids=["exact", "barnes_hut"])
def test_tsne_functionality(numpy_precision, numpy_order, tsne_method, theta):
    """
    Test the functionality of the Python wrapper
    """
    x = get_tsne_data(dtype=numpy_precision, order=numpy_order)

    model = tsne(n_components=2, perplexity=30.0,
                 max_iter=300, theta=theta, seed=42)
    embedding = model.fit_transform(x)

    assert embedding.shape == (x.shape[0], 2)
    assert np.all(np.isfinite(embedding))
    # sklearn values
    expected_kl = {
        "exact": {
            np.float64: 0.15038418722650815,
            np.float32: 0.2142138562542804,
        },
        "barnes_hut": {
            np.float64: 0.16197189688682556,
            np.float32: 0.16674014925956726,
        },
    }[tsne_method][numpy_precision]
    assert model.kl_divergence <= 1.6 * expected_kl
    assert model.n_samples == 150
    assert model.n_features == 4
    assert model.n_components == 2
    assert model.n_iter == 300

    # Rectangle corners: after centering [(-1,-½),(-1,½),(1,-½),(1,½)].
    # PCA init and gradient descent preserve the point symmetry, so the
    # output must have the form [(c0,c1),(c0,-c1),(-c0,c1),(-c0,-c1)]
    # where |c0|,|c1|>1.  The sign convention depends on PCA eigenvector
    # orientation, which is inherently ambiguous.
    x_rect = np.array([[0.0, 0.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]],
                      dtype=numpy_precision, order=numpy_order)
    rect_model = tsne(n_components=2, perplexity=1.0, max_iter=50, theta=theta,
                      seed=0, learning_rate=200.0, init="pca")
    rect_emb = rect_model.fit_transform(x_rect)
    c0, c1 = rect_emb[0]
    assert abs(c0) > 1.0 and abs(c1) > 1.0
    expected_pattern = np.array([[c0, c1], [c0, -c1], [-c0, c1], [-c0, -c1]],
                                dtype=numpy_precision)
    assert np.allclose(rect_emb, expected_pattern, atol=1e-3, rtol=0.0)


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
@pytest.mark.parametrize(
    "tsne_method,theta",
    [("exact", 0.0), ("barnes_hut", 0.5)],
    ids=["exact", "barnes_hut"])
def test_tsne_seed_reproducibility(numpy_precision, numpy_order, tsne_method, theta):
    """
    Same seed should produce identical embeddings.
    """
    x = get_tsne_data(dtype=numpy_precision, order=numpy_order)
    model1 = tsne(n_components=2, init="random", perplexity=30.0, max_iter=300,
                  theta=theta, seed=7)
    model2 = tsne(n_components=2, init="random", perplexity=30.0, max_iter=300,
                  theta=theta, seed=7)
    emb1 = model1.fit_transform(x)
    emb2 = model2.fit_transform(x)
    assert np.allclose(emb1, emb2)


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_tsne_supplied_init(numpy_precision, numpy_order):
    """
    Verify the supplied-initialization path: embedding shape and dtype are
    correct and optimization proceeds from the user-provided starting point.
    """
    n_samples = 4
    n_components = 2
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                 dtype=numpy_precision, order=numpy_order)
    y_init = np.array([[2.0, -3.0]] * n_samples, dtype=numpy_precision,
                      order=numpy_order)

    model = tsne(n_components=n_components, perplexity=1.0, max_iter=50,
                 theta=0.0, seed=0, learning_rate=200.0,
                 init=y_init)
    embedding = model.fit_transform(x)

    assert embedding.shape == (n_samples, n_components)
    assert embedding.dtype == np.dtype(numpy_precision)
    # Supplied init with all identical points has zero gradient, so embedding
    # should remain at the supplied initial positions.
    assert np.allclose(embedding, y_init)
    assert model.n_samples == n_samples
    assert model.n_components == n_components


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
def test_tsne_error_exits(numpy_precision):
    """
    Test error exits in the Python wrapper
    """
    x = np.array([[1.0, 2.0], [3.0, 4.0], [4.0, 5.0], [6.0, 7.0]],
                 dtype=numpy_precision)

    with pytest.raises(RuntimeError):
        tsne(n_components=4)

    with pytest.raises(ValueError):
        tsne(n_components=2, perplexity=2.0,
             init=np.array([1.0, 2.0, 3.0])).fit_transform(x)

    with pytest.raises(ValueError):
        tsne(n_components=2, perplexity=2.0,
             init=np.ones((x.shape[0], 3), dtype=numpy_precision))

    with pytest.raises(ValueError):
        tsne(n_components=2, perplexity=2.0,
             init=np.ones((x.shape[0] + 5, 2),
                          dtype=numpy_precision)).fit_transform(x)

    with pytest.raises(ValueError):
        tsne(perplexity=2.0, learning_rate="bad-token").fit_transform(x)

    with pytest.warns(RuntimeWarning):
        tsne(n_components=2, perplexity=1000.0, max_iter=50,
             theta=0.0, seed=7).fit_transform(x)


def test_tsne_mixed_precision_double():
    """
    Mixed precision on float64 data should run and produce a valid embedding.
    """
    rng = np.random.default_rng(123)
    x = rng.standard_normal((20, 10)).astype(np.float64)
    model = tsne(n_components=2, perplexity=5.0, max_iter=300, theta=0.0,
                 seed=42, mixed_precision=True, low_precision_max_iter=100,
                 low_precision_min_grad_norm=1e-3)
    embedding = model.fit_transform(x)
    assert embedding.shape == (20, 2)
    assert embedding.dtype == np.float64
    assert np.all(np.isfinite(embedding))
    assert np.isfinite(model.kl_divergence)
    assert model.lp_n_iter >= 1
