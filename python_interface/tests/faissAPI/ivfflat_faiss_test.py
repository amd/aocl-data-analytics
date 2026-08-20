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
import pytest
import numpy as np
import faiss
from conftest import _native_IndexFlatL2, _native_IndexIVFFlat, D, N, NQ, K, NLIST, X_CORPUS, X_QUERY
from aoclda.faiss import faiss_patch, undo_faiss_patch
from aoclda.faiss._index_ivfflat import IndexIVFFlat as _AoclIndexIVFFlat

DTYPES_ORDERS = [(np.float32, 'C'), (np.float32, 'F'),
                 (np.float64, 'C'), (np.float64, 'F')]


def test_ivfflat_construction(patched_faiss):
    q = faiss.IndexFlatL2(D)
    idx = faiss.IndexIVFFlat(q, D, NLIST)
    assert idx.d == D
    assert idx.nlist == NLIST
    assert not idx.is_trained


def test_ivfflat_is_aocl_wrapper(patched_faiss):
    q = faiss.IndexFlatL2(D)
    idx = faiss.IndexIVFFlat(q, D, NLIST)
    assert isinstance(idx, _AoclIndexIVFFlat)
    assert not isinstance(idx, _native_IndexIVFFlat)


@pytest.mark.parametrize("dtype,order", DTYPES_ORDERS)
def test_ivfflat_train_add_search(patched_faiss, dtype, order):
    corpus = np.array(X_CORPUS, dtype=dtype, order=order)
    query = np.array(X_QUERY, dtype=dtype, order=order)
    q = faiss.IndexFlatL2(D)
    idx = faiss.IndexIVFFlat(q, D, NLIST)
    idx.train(corpus)
    assert idx.is_trained
    idx.add(corpus)
    D_arr, I_arr = idx.search(query, K)
    assert D_arr.shape == (NQ, K)
    assert I_arr.shape == (NQ, K)
    assert D_arr.dtype == dtype
    assert I_arr.dtype == np.int64
    assert (D_arr >= 0).all()


def test_ivfflat_nprobe_readwrite(patched_faiss):
    q = faiss.IndexFlatL2(D)
    idx = faiss.IndexIVFFlat(q, D, NLIST)
    idx.nprobe = 4
    assert idx.nprobe == 4


def test_ivfflat_metric_type_l2(patched_faiss):
    q = faiss.IndexFlatL2(D)
    idx = faiss.IndexIVFFlat(q, D, NLIST, faiss.METRIC_L2)
    assert idx.metric_type == faiss.METRIC_L2


def test_ivfflat_is_trained_state_machine(patched_faiss):
    q = faiss.IndexFlatL2(D)
    idx = faiss.IndexIVFFlat(q, D, NLIST)
    assert not idx.is_trained
    idx.train(X_CORPUS)
    assert idx.is_trained


def test_ivfflat_reset(patched_faiss):
    q = faiss.IndexFlatL2(D)
    idx = faiss.IndexIVFFlat(q, D, NLIST)
    idx.train(X_CORPUS)
    idx.add(X_CORPUS)
    idx.reset()
    assert not idx.is_trained


def test_ivfflat_quantizer_dimension_mismatch_raises(patched_faiss):
    q = faiss.IndexFlatL2(D + 1)
    with pytest.raises(ValueError):
        faiss.IndexIVFFlat(q, D, NLIST)


def test_ivfflat_non_flat_quantizer_warns(patched_faiss):
    inner_q = faiss.IndexFlatL2(D)
    outer_q = faiss.IndexIVFFlat(inner_q, D, NLIST)
    with pytest.warns(UserWarning):
        faiss.IndexIVFFlat(outer_q, D, NLIST)


def test_ivfflat_unsupported_method_raises(patched_faiss):
    q = faiss.IndexFlatL2(D)
    idx = faiss.IndexIVFFlat(q, D, NLIST)
    with pytest.raises(NotImplementedError):
        idx.add_with_ids(X_CORPUS, np.arange(N, dtype=np.int64))


def test_ivfflat_recall_fixed_seed(patched_faiss):
    rng = np.random.default_rng(123)
    X = rng.random((1300, D)).astype(np.float32)
    Q = rng.random((50, D)).astype(np.float32)
    nlist = 32

    ref = _native_IndexFlatL2(D)
    ref.add(X)
    _, I_exact = ref.search(Q, K)

    q = faiss.IndexFlatL2(D)
    idx = faiss.IndexIVFFlat(q, D, nlist, faiss.METRIC_L2)
    idx.train(X)
    idx.add(X)
    idx.nprobe = 8
    _, I_ann = idx.search(Q, K)

    recall = np.mean([
        len(set(I_ann[i]) & set(I_exact[i])) / K
        for i in range(len(Q))
    ])
    assert recall >= 0.8, f"Recall too low: {recall:.3f}"


def test_ivfflat_nprobe_monotonic_recall(patched_faiss):
    rng = np.random.default_rng(456)
    X = rng.random((1300, D)).astype(np.float32)
    Q = rng.random((50, D)).astype(np.float32)
    nlist = 32

    ref = _native_IndexFlatL2(D)
    ref.add(X)
    _, I_exact = ref.search(Q, K)

    q = faiss.IndexFlatL2(D)
    idx = faiss.IndexIVFFlat(q, D, nlist, faiss.METRIC_L2)
    idx.train(X)
    idx.add(X)

    nprobe_values = [1, 2, 4, 8, 16, nlist]
    recalls = []
    mean_distances = []

    for np_val in nprobe_values:
        idx.nprobe = np_val
        D_arr, I_arr = idx.search(Q, K)
        recall = np.mean([
            len(set(I_arr[i]) & set(I_exact[i])) / K
            for i in range(len(Q))
        ])
        recalls.append(recall)
        mean_distances.append(float(D_arr.mean()))

    for i in range(len(recalls) - 1):
        assert recalls[i + 1] >= recalls[i] - 1e-9, (
            f"Recall decreased from nprobe={nprobe_values[i]} to "
            f"nprobe={nprobe_values[i + 1]}: {recalls[i]:.4f} -> {recalls[i + 1]:.4f}"
        )

    for i in range(len(mean_distances) - 1):
        assert mean_distances[i + 1] <= mean_distances[i] + 1e-5, (
            f"Mean distance increased from nprobe={nprobe_values[i]} to "
            f"nprobe={nprobe_values[i + 1]}: {mean_distances[i]:.6f} -> {mean_distances[i + 1]:.6f}"
        )


def test_ivfflat_cp_lifecycle(patched_faiss):
    rng = np.random.default_rng(99)
    X = rng.random((800, D)).astype(np.float32)
    q = faiss.IndexFlatL2(D)
    idx = faiss.IndexIVFFlat(q, D, NLIST)

    assert idx.cp.niter == 25
    assert idx.cp.seed == 0
    assert idx.cp.max_points_per_centroid is None
    assert idx.ntotal == 0

    idx.cp.niter = 5
    idx.cp.seed = 42
    idx.cp.max_points_per_centroid = 20
    idx.nprobe = 4
    idx.train(X)
    assert idx.is_trained
    assert idx.ntotal == 0
    idx.add(X)
    assert idx.ntotal == len(X)

    idx.reset()
    assert not idx.is_trained
    assert idx.cp.niter == 5
    assert idx.cp.seed == 42
    assert idx.nprobe == 4


def test_ivfflat_nprobe_lifecycle(patched_faiss):
    q = faiss.IndexFlatL2(D)
    idx = faiss.IndexIVFFlat(q, D, NLIST)
    idx.nprobe = 7
    assert idx.nprobe == 7
    idx.train(X_CORPUS)
    assert idx.nprobe == 7
    idx.add(X_CORPUS)
    D_arr, I_arr = idx.search(X_QUERY, K)
    assert D_arr.shape == (NQ, K)

    idx.nprobe = 3
    assert idx.nprobe == 3
    D_arr, I_arr = idx.search(X_QUERY, K)
    assert D_arr.shape == (NQ, K)


def test_ivfflat_unsupported_attrs(patched_faiss):
    q = faiss.IndexFlatL2(D)
    idx = faiss.IndexIVFFlat(q, D, NLIST)
    with pytest.raises(NotImplementedError):
        _ = idx.range_search
    # cp fields shared with Kmeans (e.g. spherical) exist but are ignored by IVF
    assert idx.cp.spherical is False
    with pytest.raises(AttributeError):
        _ = idx.__copy__


def test_index_ivf_flat_pickle_with_patch(tmp_path):
    faiss_patch('IndexIVFFlat', print_patched=False)

    idx = faiss.IndexIVFFlat(faiss.IndexFlatL2(D), D, NLIST)
    assert idx.aocl is True
    idx.train(X_CORPUS)
    idx.add(X_CORPUS)
    D_before, I_before = idx.search(X_QUERY, K)

    filepath = tmp_path / "ivfflat.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(idx, f)
    del idx

    with open(filepath, 'rb') as f:
        idx_loaded = pickle.load(f)

    assert idx_loaded.aocl is True
    assert idx_loaded.is_trained
    assert idx_loaded.ntotal == len(X_CORPUS)
    assert idx_loaded.nprobe == 1
    D_after, I_after = idx_loaded.search(X_QUERY, K)
    assert np.array_equal(I_before, I_after)

    undo_faiss_patch('IndexIVFFlat', print_patched=False)


def test_index_ivf_flat_pickle_cross_patch(tmp_path):
    faiss_patch('IndexIVFFlat', print_patched=False)

    idx = faiss.IndexIVFFlat(faiss.IndexFlatL2(D), D, NLIST)
    assert idx.aocl is True
    idx.train(X_CORPUS)
    idx.add(X_CORPUS)
    D_before, I_before = idx.search(X_QUERY, K)

    filepath = tmp_path / "ivfflat.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(idx, f)
    del idx

    undo_faiss_patch('IndexIVFFlat', print_patched=False)

    with open(filepath, 'rb') as f:
        idx_loaded = pickle.load(f)

    assert idx_loaded.aocl is True
    D_after, I_after = idx_loaded.search(X_QUERY, K)
    assert np.array_equal(I_before, I_after)


def test_index_ivf_flat_pickle_untrained(tmp_path):
    faiss_patch('IndexIVFFlat', print_patched=False)

    idx = faiss.IndexIVFFlat(faiss.IndexFlatL2(D), D, NLIST)
    assert idx.aocl is True
    assert not idx.is_trained

    filepath = tmp_path / "ivfflat_untrained.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(idx, f)
    del idx

    with open(filepath, 'rb') as f:
        idx_loaded = pickle.load(f)

    assert idx_loaded.aocl is True
    assert not idx_loaded.is_trained
    idx_loaded.train(X_CORPUS)
    assert idx_loaded.is_trained
    idx_loaded.add(X_CORPUS)
    assert idx_loaded.ntotal == len(X_CORPUS)

    undo_faiss_patch('IndexIVFFlat', print_patched=False)


def test_setattr(patched_faiss):
    idx = faiss.IndexIVFFlat(faiss.IndexFlatL2(D), D, NLIST)
    with pytest.warns(UserWarning) as record:
        idx.use_residual = True
    msg = str(record[0].message)
    assert "IndexIVFFlat" in msg
    assert "use_residual" in msg
    assert idx.use_residual is True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        idx.nprobe = 5


def test_setattr_cp(patched_faiss):
    idx = faiss.IndexIVFFlat(faiss.IndexFlatL2(D), D, NLIST)
    with pytest.warns(UserWarning) as record:
        idx.cp.not_a_real_param = True
    msg = str(record[0].message)
    assert "ClusteringParameters" in msg
    assert "not_a_real_param" in msg
    # supported fields never warn, even ones IVF ignores (e.g. spherical)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        idx.cp.niter = 5
        idx.cp.spherical = True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
