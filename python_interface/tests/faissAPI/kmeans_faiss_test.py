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

import pickle
import warnings
import numpy as np
import pytest
import faiss
from conftest import D, NQ, X_CORPUS, X_QUERY
from aoclda.faiss import faiss_patch, undo_faiss_patch
from aoclda.faiss._kmeans import Kmeans as Kmeans_da

N_CLUSTERS = 8
NITER = 50
SEED = 42


@pytest.fixture
def patched_faiss_kmeans():
    faiss_patch("Kmeans", print_patched=False)
    yield
    undo_faiss_patch("Kmeans", print_patched=False)


def test_kmeans_construction(patched_faiss_kmeans):
    km = faiss.Kmeans(D, N_CLUSTERS)
    assert km.d == D
    assert km.k == N_CLUSTERS
    assert km.is_trained is False
    assert km.centroids is None
    assert km.aocl is True


def test_kmeans_train(patched_faiss_kmeans):
    km = faiss.Kmeans(D, N_CLUSTERS, niter=NITER, seed=SEED)
    km.train(X_CORPUS)
    assert km.centroids.shape == (N_CLUSTERS, D)
    assert km.is_trained is True


def test_kmeans_assign(patched_faiss_kmeans):
    km = faiss.Kmeans(D, N_CLUSTERS, niter=NITER, seed=SEED)
    km.train(X_CORPUS)
    D_out, I_out = km.assign(X_QUERY)
    assert D_out.shape == (NQ,)
    assert I_out.shape == (NQ,)
    assert I_out.dtype == np.int64
    assert (I_out >= 0).all() and (I_out < N_CLUSTERS).all()
    np.testing.assert_allclose(D_out, km._kmeans.transform(X_QUERY).min(axis=1))


def test_kmeans_patch_lifecycle():
    native_Kmeans = faiss.Kmeans
    assert faiss.Kmeans is not Kmeans_da
    faiss_patch("Kmeans", print_patched=False)
    assert faiss.Kmeans is Kmeans_da
    undo_faiss_patch("Kmeans", print_patched=False)
    assert faiss.Kmeans is not Kmeans_da
    assert faiss.Kmeans is native_Kmeans


def test_kmeans_spherical(patched_faiss_kmeans):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        km = faiss.Kmeans(D, N_CLUSTERS, niter=NITER, spherical=True)
    assert km.cp.spherical is True
    km.train(X_CORPUS)
    assert km.centroids.shape == (N_CLUSTERS, D)


def test_kmeans_da_specific_options(patched_faiss_kmeans):
    # The AOCL-DA backend options exposed on the constructor (in addition to the
    # native faiss parameters) must be accepted without warning and reach the
    # backend.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        km = faiss.Kmeans(
            D, N_CLUSTERS, niter=NITER, seed=SEED,
            algorithm='elkan',
            initialization_method='random',
            tol=1.0e-3,
            empty_clusters='ignore',
            afk_mcmc_samples=20,
            mixed_precision=True,
            low_precision_max_iter=100,
            low_precision_tol=1.0e-1,
        )
    assert km._algorithm == 'elkan'
    assert km._initialization_method == 'random'
    assert km._tol == 1.0e-3
    assert km._mixed_precision is True
    assert km._low_precision_max_iter == 100


def test_kmeans_da_specific_options_train(patched_faiss_kmeans):
    # A representative set of DA-specific options must reach the backend and
    # produce a valid result.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        km = faiss.Kmeans(
            D, N_CLUSTERS, niter=NITER, seed=SEED,
            algorithm='elkan',
            initialization_method='random',
            tol=1.0e-3,
            empty_clusters='ignore',
        )
    km.train(X_CORPUS)
    assert km.centroids.shape == (N_CLUSTERS, D)
    assert km.is_trained is True


def test_kmeans_normalize_data_with_spherical(patched_faiss_kmeans):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        km = faiss.Kmeans(
            D, N_CLUSTERS, niter=NITER, spherical=True, normalize_data=True)
    assert km._normalize_data is True
    km.train(X_CORPUS)
    assert km.centroids.shape == (N_CLUSTERS, D)


def test_kmeans_unsupported_kwargs_warn(patched_faiss_kmeans):
    with pytest.warns(UserWarning, match="update_index"):
        faiss.Kmeans(D, N_CLUSTERS, update_index=True)


def test_kmeans_unsupported_attr_warn(patched_faiss_kmeans):
    km = faiss.Kmeans(D, N_CLUSTERS, niter=NITER, seed=SEED)
    km.train(X_CORPUS)
    with pytest.warns(UserWarning, match="frozen_centroids"):
        km.frozen_centroids = True
    assert km.frozen_centroids is True


def test_kmeans_pickle_with_patch(patched_faiss_kmeans, tmp_path):
    km = faiss.Kmeans(D, N_CLUSTERS, niter=NITER, seed=SEED)
    km.train(X_CORPUS)
    D_before, I_before = km.assign(X_QUERY)

    filepath = tmp_path / "kmeans.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(km, f)
    del km

    with open(filepath, 'rb') as f:
        km_loaded = pickle.load(f)

    assert km_loaded.aocl is True
    assert km_loaded.is_trained is True
    assert km_loaded.centroids.shape == (N_CLUSTERS, D)
    D_after, I_after = km_loaded.assign(X_QUERY)
    assert np.array_equal(I_before, I_after)


def test_kmeans_pickle_cross_patch(tmp_path):
    faiss_patch("Kmeans", print_patched=False)
    km = faiss.Kmeans(D, N_CLUSTERS, niter=NITER, seed=SEED)
    km.train(X_CORPUS)
    D_before, I_before = km.assign(X_QUERY)

    filepath = tmp_path / "kmeans_cross.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(km, f)
    undo_faiss_patch("Kmeans", print_patched=False)

    with open(filepath, 'rb') as f:
        km_loaded = pickle.load(f)

    assert km_loaded.aocl is True
    D_after, I_after = km_loaded.assign(X_QUERY)
    assert np.array_equal(I_before, I_after)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
