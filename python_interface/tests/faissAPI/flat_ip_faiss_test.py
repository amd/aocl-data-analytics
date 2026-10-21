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

# pylint: disable = missing-module-docstring, missing-function-docstring
# pylint: disable = unused-argument, no-value-for-parameter
# pylint: disable = import-outside-toplevel, no-member, protected-access

import pickle
import warnings
import numpy as np
import pytest
import faiss
from conftest import _native_IndexFlatIP, D, N, NQ, K, X_CORPUS, X_QUERY
from aoclda.faiss import faiss_patch, undo_faiss_patch
from aoclda.faiss._index_flat_ip import IndexFlatIP as _AoclIndexFlatIP

DTYPES_ORDERS = [(np.float32, 'C'), (np.float32, 'F'),
                 (np.float64, 'C'), (np.float64, 'F')]


def test_flat_ip_construction(patched_faiss):
    idx = faiss.IndexFlatIP(D)
    assert idx.d == D
    assert idx.ntotal == 0
    assert idx.is_trained


def test_flat_ip_is_aocl_wrapper(patched_faiss):
    idx = faiss.IndexFlatIP(D)
    assert isinstance(idx, _AoclIndexFlatIP)
    assert not isinstance(idx, _native_IndexFlatIP)


@pytest.mark.parametrize("dtype,order", DTYPES_ORDERS)
def test_flat_ip_two_adds(patched_faiss, dtype, order):
    corpus = np.array(X_CORPUS, dtype=dtype, order=order)
    query = np.array(X_QUERY, dtype=dtype, order=order)
    idx = faiss.IndexFlatIP(D)
    idx.add(corpus[:250])
    assert idx.ntotal == 250
    idx.add(corpus[250:])
    assert idx.ntotal == N
    D_arr, I_arr = idx.search(query, K)
    assert D_arr.shape == (NQ, K)
    assert I_arr.shape == (NQ, K)
    assert D_arr.dtype == dtype
    assert I_arr.dtype == np.int64
    assert (I_arr < N).all()


@pytest.mark.parametrize("dtype,order", DTYPES_ORDERS)
def test_flat_ip_distance_correctness(patched_faiss, dtype, order):
    from aoclda.neighbors import nearest_neighbors
    corpus = np.array(X_CORPUS, dtype=dtype, order=order)
    query = np.array(X_QUERY, dtype=dtype, order=order)
    ref = nearest_neighbors(metric='inner_product')
    ref.fit(corpus)
    D_ref, I_ref = ref.kneighbors(query, n_neighbors=K)

    idx = faiss.IndexFlatIP(D)
    idx.add(corpus)
    D_da, I_da = idx.search(query, K)

    np.testing.assert_array_equal(I_ref, I_da)
    np.testing.assert_allclose(D_ref, D_da, rtol=1e-5, atol=1e-5)


def test_flat_ip_add_copies_input(patched_faiss):
    corpus = np.asfortranarray(X_CORPUS)
    expected = corpus.copy()
    idx = faiss.IndexFlatIP(D)
    idx.add(corpus)

    corpus[:] = 0

    np.testing.assert_array_equal(idx.reconstruct_n(0, N), expected)


def test_flat_ip_search_pads_when_k_exceeds_ntotal(patched_faiss):
    idx = faiss.IndexFlatIP(D)
    idx.add(X_CORPUS[:3])

    D_arr, I_arr, R = idx.search_and_reconstruct(X_QUERY[:2], 5)

    assert D_arr.shape == (2, 5)
    np.testing.assert_array_equal(I_arr[:, 3:], -1)
    np.testing.assert_array_equal(
        D_arr[:, 3:], -np.finfo(D_arr.dtype).max)
    np.testing.assert_array_equal(R[:, 3:], 0)


def test_flat_ip_reset(patched_faiss):
    idx = faiss.IndexFlatIP(D)
    idx.add(X_CORPUS)
    idx.reset()
    assert idx.ntotal == 0
    assert idx._nn is None


def test_flat_ip_train_is_noop(patched_faiss):
    idx = faiss.IndexFlatIP(D)
    idx.train(X_CORPUS)
    assert idx.is_trained


def test_flat_ip_metric_type(patched_faiss):
    idx = faiss.IndexFlatIP(D)
    assert idx.metric_type == faiss.METRIC_INNER_PRODUCT


def test_flat_ip_unsupported_attrs(patched_faiss):
    idx = faiss.IndexFlatIP(D)
    with pytest.raises(NotImplementedError):
        _ = idx.range_search
    with pytest.raises(AttributeError):
        _ = idx.__copy__
    with pytest.warns(UserWarning, match="verbose"):
        idx.verbose = True
    assert idx.verbose is True


def test_index_flat_ip_pickle_with_patch(tmp_path):
    faiss_patch('IndexFlatIP', print_patched=False)
    idx = faiss.IndexFlatIP(D)
    assert idx.aocl is True
    idx.add(X_CORPUS)
    D_before, I_before = idx.search(X_QUERY, K)
    filepath = tmp_path / "flatip.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(idx, f)
    del idx
    with open(filepath, 'rb') as f:
        idx_loaded = pickle.load(f)
    assert idx_loaded.aocl is True
    assert idx_loaded.ntotal == len(X_CORPUS)
    D_after, I_after = idx_loaded.search(X_QUERY, K)
    assert np.array_equal(I_before, I_after)
    assert np.array_equal(D_before, D_after)
    undo_faiss_patch('IndexFlatIP', print_patched=False)


def test_index_flat_ip_pickle_cross_patch(tmp_path):
    faiss_patch('IndexFlatIP', print_patched=False)
    idx = faiss.IndexFlatIP(D)
    assert idx.aocl is True
    idx.add(X_CORPUS)
    D_before, I_before = idx.search(X_QUERY, K)
    filepath = tmp_path / "flatip.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(idx, f)
    del idx
    undo_faiss_patch('IndexFlatIP', print_patched=False)
    with open(filepath, 'rb') as f:
        idx_loaded = pickle.load(f)
    assert idx_loaded.aocl is True
    D_after, I_after = idx_loaded.search(X_QUERY, K)
    assert np.array_equal(I_before, I_after)
    assert np.array_equal(D_before, D_after)


def test_index_flat_ip_reconstruct_after_unpickle(tmp_path):
    faiss_patch('IndexFlatIP', print_patched=False)
    idx = faiss.IndexFlatIP(D)
    idx.add(X_CORPUS)
    filepath = tmp_path / "flatip.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(idx, f)
    del idx
    with open(filepath, 'rb') as f:
        idx_loaded = pickle.load(f)
    np.testing.assert_array_equal(idx_loaded.reconstruct(0), X_CORPUS[0])
    np.testing.assert_array_equal(idx_loaded.reconstruct_n(0, 3), X_CORPUS[0:3])
    undo_faiss_patch('IndexFlatIP', print_patched=False)


def test_flat_ip_assign(patched_faiss):
    idx = faiss.IndexFlatIP(D)
    idx.add(X_CORPUS)
    _, I_search = idx.search(X_QUERY, K)
    I_assign = idx.assign(X_QUERY, K)
    np.testing.assert_array_equal(I_assign, I_search)
    assert I_assign.dtype == np.int64


def test_flat_ip_reconstruct(patched_faiss):
    idx = faiss.IndexFlatIP(D)
    idx.add(X_CORPUS)
    np.testing.assert_array_equal(idx.reconstruct(0), X_CORPUS[0])
    np.testing.assert_array_equal(idx.reconstruct(N - 1), X_CORPUS[N - 1])
    with pytest.raises(IndexError):
        idx.reconstruct(N)
    with pytest.raises(IndexError):
        idx.reconstruct(-1)


def test_flat_ip_reconstruct_n(patched_faiss):
    idx = faiss.IndexFlatIP(D)
    idx.add(X_CORPUS)
    np.testing.assert_array_equal(idx.reconstruct_n(10, 5), X_CORPUS[10:15])


def test_flat_ip_reconstruct_batch(patched_faiss):
    idx = faiss.IndexFlatIP(D)
    idx.add(X_CORPUS)
    ids = [3, 1, 42, 7]
    np.testing.assert_array_equal(idx.reconstruct_batch(ids), X_CORPUS[ids])


def test_flat_ip_search_and_reconstruct(patched_faiss):
    idx = faiss.IndexFlatIP(D)
    idx.add(X_CORPUS)
    D_arr, I_arr, R = idx.search_and_reconstruct(X_QUERY, K)
    assert D_arr.shape == (NQ, K)
    assert R.shape == (NQ, K, D)
    assert R.dtype == np.float32
    np.testing.assert_array_equal(R, X_CORPUS[I_arr])


def test_setattr(patched_faiss):
    idx = faiss.IndexFlatIP(D)
    with pytest.warns(UserWarning) as record:
        idx.use_residual = True
    msg = str(record[0].message)
    assert "IndexFlatIP" in msg
    assert "use_residual" in msg
    assert idx.use_residual is True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        idx.aocl = True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
