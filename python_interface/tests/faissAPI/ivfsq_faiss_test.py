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
import pytest
import numpy as np
import faiss
from conftest import (
    _native_IndexFlatL2,
    _native_IndexIVFScalarQuantizer,
    D,
    N,
    NQ,
    K,
    NLIST,
    X_CORPUS,
    X_QUERY,
)
from aoclda.faiss import faiss_patch, undo_faiss_patch
from aoclda.faiss._index_ivfsq import (
    IndexIVFScalarQuantizer as _AoclIndexIVFScalarQuantizer,
)


DTYPES_ORDERS = [(np.float32, 'C'), (np.float32, 'F'),
                 (np.float64, 'C'), (np.float64, 'F')]
QTYPE_CASES = [
    ("QT_4bit", True),
    ("QT_4bit_uniform", True),
    ("QT_8bit", True),
    ("QT_8bit_uniform", True),
    ("QT_8bit_direct", False),
    ("QT_8bit_direct_signed", False),
    ("QT_bf16", True),
]
METRICS = [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]


def make_test_data(rng, qtype_name, dtype, order):
    if qtype_name == "QT_8bit_direct":
        values = rng.integers(0, 256, size=(N, D))
    elif qtype_name == "QT_8bit_direct_signed":
        values = rng.integers(-128, 128, size=(N, D))
    else:
        values = rng.normal(size=(N, D))
    return np.array(values, dtype=dtype, order=order)


def make_index(qtype=None, metric=faiss.METRIC_L2, by_residual=True):
    if qtype is None:
        qtype = faiss.ScalarQuantizer.QT_8bit
    return faiss.IndexIVFScalarQuantizer(
        faiss.IndexFlatL2(D), D, NLIST, qtype,
        metric, by_residual)


def test_ivfsq_construction(patched_faiss):
    idx = make_index()
    assert isinstance(idx, _AoclIndexIVFScalarQuantizer)
    assert not isinstance(idx, _native_IndexIVFScalarQuantizer)
    assert idx.d == D
    assert idx.nlist == NLIST
    assert idx.metric_type == faiss.METRIC_L2
    assert idx.by_residual is True
    assert not idx.is_trained
    assert idx.aocl is True


@pytest.mark.parametrize(
    "qtype_name,by_residual", QTYPE_CASES)
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("dtype,order", DTYPES_ORDERS)
def test_ivfsq_train_add_search(
        patched_faiss, qtype_name, by_residual, metric, dtype, order):
    rng = np.random.default_rng(7)
    corpus = make_test_data(rng, qtype_name, dtype, order)
    query = np.array(corpus[:NQ], dtype=dtype, order=order)
    idx = make_index(
        getattr(faiss.ScalarQuantizer, qtype_name),
        metric=metric, by_residual=by_residual)
    idx.nprobe = 4
    idx.train(corpus)
    idx.add(corpus)
    distances, indices = idx.search(query, K)
    assert distances.shape == (NQ, K)
    assert indices.shape == (NQ, K)
    assert distances.dtype == dtype
    assert indices.dtype == np.int64
    assert np.isfinite(distances).all()
    assert ((indices >= 0) & (indices < N)).all()
    assert idx.ntotal == N
    assert idx.nprobe == 4
    assert idx.metric_type == metric


def test_ivfsq_by_residual_lifecycle(patched_faiss):
    idx = make_index()
    idx.by_residual = False
    assert idx.by_residual is False
    idx.train(X_CORPUS)
    with pytest.raises(RuntimeError, match="cannot be changed after training"):
        idx.by_residual = True
    idx.reset()
    idx.by_residual = True
    assert idx.by_residual is True


@pytest.mark.parametrize(
    "unsupported_qtype",
    [faiss.ScalarQuantizer.QT_fp16])
def test_ivfsq_validation_and_state_errors(patched_faiss, unsupported_qtype):
    with pytest.raises(ValueError, match="Unsupported scalar quantizer type"):
        make_index(unsupported_qtype)
    with pytest.raises(ValueError, match="quantizer dimension"):
        faiss.IndexIVFScalarQuantizer(
            faiss.IndexFlatL2(D + 1), D, NLIST,
            faiss.ScalarQuantizer.QT_8bit)
    idx = make_index()
    with pytest.raises(RuntimeError, match="IndexIVFScalarQuantizer.add"):
        idx.add(X_CORPUS)
    with pytest.raises(RuntimeError, match="IndexIVFScalarQuantizer.search"):
        idx.search(X_QUERY, K)


def test_ivfsq_recall_fixed_seed(patched_faiss):
    rng = np.random.default_rng(123)
    corpus = rng.random((1300, D)).astype(np.float32)
    query = rng.random((50, D)).astype(np.float32)

    reference = _native_IndexFlatL2(D)
    reference.add(corpus)
    _, exact_indices = reference.search(query, K)

    idx = faiss.IndexIVFScalarQuantizer(
        faiss.IndexFlatL2(D), D, 32, faiss.ScalarQuantizer.QT_8bit)
    idx.train(corpus)
    idx.add(corpus)
    idx.nprobe = 8
    _, approximate_indices = idx.search(query, K)

    recall = np.mean([
        len(set(approximate_indices[i]) & set(exact_indices[i])) / K
        for i in range(len(query))
    ])
    assert recall >= 0.8, f"Recall too low: {recall:.3f}"


@pytest.mark.parametrize(
    "qtype", [faiss.ScalarQuantizer.QT_4bit,
              faiss.ScalarQuantizer.QT_4bit_uniform,
              faiss.ScalarQuantizer.QT_8bit,
              faiss.ScalarQuantizer.QT_bf16])
def test_ivfsq_pickle_with_patch(tmp_path, qtype):
    faiss_patch('IndexIVFScalarQuantizer', print_patched=False)
    idx = make_index(qtype)
    idx.nprobe = 4
    idx.train(X_CORPUS)
    idx.add(X_CORPUS)
    distances_before, indices_before = idx.search(X_QUERY, K)

    filepath = tmp_path / "ivfsq.pkl"
    with open(filepath, 'wb') as file:
        pickle.dump(idx, file)
    with open(filepath, 'rb') as file:
        loaded = pickle.load(file)

    distances_after, indices_after = loaded.search(X_QUERY, K)
    assert loaded.aocl is True
    assert loaded.is_trained
    assert loaded.ntotal == N
    assert loaded.nprobe == 4
    assert np.array_equal(distances_before, distances_after)
    assert np.array_equal(indices_before, indices_after)
    undo_faiss_patch('IndexIVFScalarQuantizer', print_patched=False)


def test_ivfsq_pickle_cross_patch(tmp_path):
    faiss_patch('IndexIVFScalarQuantizer', print_patched=False)
    idx = make_index()
    idx.train(X_CORPUS)
    idx.add(X_CORPUS)
    _, indices_before = idx.search(X_QUERY, K)

    filepath = tmp_path / "ivfsq_cross_patch.pkl"
    with open(filepath, 'wb') as file:
        pickle.dump(idx, file)
    undo_faiss_patch('IndexIVFScalarQuantizer', print_patched=False)
    with open(filepath, 'rb') as file:
        loaded = pickle.load(file)

    _, indices_after = loaded.search(X_QUERY, K)
    assert loaded.aocl is True
    assert np.array_equal(indices_before, indices_after)


def test_ivfsq_pickle_untrained(tmp_path):
    faiss_patch('IndexIVFScalarQuantizer', print_patched=False)
    filepath = tmp_path / "ivfsq_untrained.pkl"
    with open(filepath, 'wb') as file:
        pickle.dump(make_index(by_residual=False), file)
    with open(filepath, 'rb') as file:
        loaded = pickle.load(file)

    assert not loaded.is_trained
    assert loaded.by_residual is False
    loaded.train(X_CORPUS)
    loaded.add(X_CORPUS)
    assert loaded.ntotal == N
    undo_faiss_patch('IndexIVFScalarQuantizer', print_patched=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
