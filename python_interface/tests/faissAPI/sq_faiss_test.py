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

# pylint: disable = missing-function-docstring, missing-module-docstring
# pylint: disable = protected-access, redefined-outer-name
# pylint: disable = unused-argument, no-member, no-value-for-parameter

import pickle

import faiss
import numpy as np
import pytest

from aoclda.faiss import faiss_patch, undo_faiss_patch
from aoclda.faiss._index_sq import IndexScalarQuantizer as _AoclIndexScalarQuantizer
from conftest import (
    _native_IndexFlatL2,
    _native_IndexScalarQuantizer,
    D,
    K,
    N,
    NQ,
    X_CORPUS,
    X_QUERY,
)


DTYPES_ORDERS = [(np.float32, 'C'), (np.float32, 'F'),
                 (np.float64, 'C'), (np.float64, 'F')]
QTYPE_CASES = [
    "QT_4bit",
    "QT_4bit_uniform",
    "QT_8bit",
    "QT_8bit_uniform",
    "QT_8bit_direct",
    "QT_8bit_direct_signed",
    "QT_bf16",
]


def make_test_data(qtype_name, dtype=np.float32, order='C'):
    rng = np.random.default_rng(7)
    if qtype_name == "QT_8bit_direct":
        values = rng.integers(0, 256, size=(N, D))
    elif qtype_name == "QT_8bit_direct_signed":
        values = rng.integers(-128, 128, size=(N, D))
    else:
        values = rng.normal(size=(N, D))
    return np.array(values, dtype=dtype, order=order)


def make_index(qtype=None, metric=faiss.METRIC_L2):
    if qtype is None:
        qtype = faiss.ScalarQuantizer.QT_8bit
    return faiss.IndexScalarQuantizer(D, qtype, metric)


def test_sq_construction(patched_faiss):
    index = make_index()
    assert isinstance(index, _AoclIndexScalarQuantizer)
    assert not isinstance(index, _native_IndexScalarQuantizer)
    assert index.d == D
    assert index.ntotal == 0
    assert not index.is_trained
    assert index.metric_type == faiss.METRIC_L2
    assert index.aocl is True


@pytest.mark.parametrize("qtype_name", QTYPE_CASES)
@pytest.mark.parametrize("metric", [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT])
def test_sq_supported_qtypes_train_add_search(patched_faiss, qtype_name, metric):
    corpus = make_test_data(qtype_name)
    index = make_index(getattr(faiss.ScalarQuantizer, qtype_name), metric)
    index.train(corpus)
    index.add(corpus)
    distances, indices = index.search(corpus[:NQ], K)

    assert distances.shape == (NQ, K)
    assert indices.shape == (NQ, K)
    assert distances.dtype == np.float32
    assert indices.dtype == np.int64
    assert np.isfinite(distances).all()
    assert ((indices >= 0) & (indices < N)).all()
    assert index.ntotal == N
    assert index.metric_type == metric
    assert index._ann.n_list == 1
    assert index._ann.n_probe == 1


@pytest.mark.parametrize("dtype,order", DTYPES_ORDERS)
def test_sq_dtype_and_layout(patched_faiss, dtype, order):
    corpus = np.array(X_CORPUS, dtype=dtype, order=order)
    query = np.array(X_QUERY, dtype=dtype, order=order)
    index = make_index(faiss.ScalarQuantizer.QT_bf16)
    index.train(corpus)
    index.add(corpus[:N // 2])
    index.add(corpus[N // 2:])
    distances, indices = index.search(query, K)

    assert distances.dtype == dtype
    assert indices.dtype == np.int64
    assert index.ntotal == N


def test_sq_state_and_validation(patched_faiss):
    with pytest.raises(ValueError, match="Unsupported scalar quantizer type"):
        make_index(faiss.ScalarQuantizer.QT_fp16)
    with pytest.raises(ValueError, match="Unknown metric integer"):
        make_index(metric=99)

    index = make_index(faiss.ScalarQuantizer.QT_bf16)
    with pytest.raises(ValueError, match="expected array with 16 columns"):
        index.train(np.zeros((10, D + 1), dtype=np.float32))
    with pytest.raises(RuntimeError, match="IndexScalarQuantizer.add"):
        index.add(X_CORPUS)
    with pytest.raises(RuntimeError, match="IndexScalarQuantizer.search"):
        index.search(X_QUERY, K)


def test_sq_reset(patched_faiss):
    index = make_index(faiss.ScalarQuantizer.QT_bf16)
    index.train(X_CORPUS)
    index.add(X_CORPUS)
    index.reset()
    assert index.ntotal == 0
    assert not index.is_trained
    assert index._ann is None


def test_sq_recall_fixed_seed(patched_faiss):
    rng = np.random.default_rng(123)
    corpus = rng.random((1300, D)).astype(np.float32)
    query = rng.random((50, D)).astype(np.float32)

    reference = _native_IndexFlatL2(D)
    reference.add(corpus)
    _, exact_indices = reference.search(query, K)

    index = make_index()
    index.train(corpus)
    index.add(corpus)
    _, quantized_indices = index.search(query, K)

    recall = np.mean([
        len(set(quantized_indices[i]) & set(exact_indices[i])) / K
        for i in range(len(query))
    ])
    assert recall >= 0.9, f"Recall too low: {recall:.3f}"


def test_sq_pickle_with_patch(tmp_path):
    faiss_patch('IndexScalarQuantizer', print_patched=False)
    index = make_index(faiss.ScalarQuantizer.QT_bf16)
    index.train(X_CORPUS)
    index.add(X_CORPUS)
    distances_before, indices_before = index.search(X_QUERY, K)

    filepath = tmp_path / "sq.pkl"
    with open(filepath, 'wb') as file:
        pickle.dump(index, file)
    with open(filepath, 'rb') as file:
        loaded = pickle.load(file)

    distances_after, indices_after = loaded.search(X_QUERY, K)
    assert loaded.aocl is True
    assert loaded.is_trained
    assert loaded.ntotal == N
    assert np.array_equal(distances_before, distances_after)
    assert np.array_equal(indices_before, indices_after)
    undo_faiss_patch('IndexScalarQuantizer', print_patched=False)


def test_sq_pickle_cross_patch(tmp_path):
    faiss_patch('IndexScalarQuantizer', print_patched=False)
    index = make_index(faiss.ScalarQuantizer.QT_bf16)
    index.train(X_CORPUS)
    index.add(X_CORPUS)
    _, indices_before = index.search(X_QUERY, K)

    filepath = tmp_path / "sq_cross_patch.pkl"
    with open(filepath, 'wb') as file:
        pickle.dump(index, file)
    undo_faiss_patch('IndexScalarQuantizer', print_patched=False)
    with open(filepath, 'rb') as file:
        loaded = pickle.load(file)

    _, indices_after = loaded.search(X_QUERY, K)
    assert loaded.aocl is True
    assert np.array_equal(indices_before, indices_after)


def test_sq_pickle_untrained(tmp_path):
    faiss_patch('IndexScalarQuantizer', print_patched=False)
    filepath = tmp_path / "sq_untrained.pkl"
    with open(filepath, 'wb') as file:
        pickle.dump(make_index(faiss.ScalarQuantizer.QT_bf16), file)
    with open(filepath, 'rb') as file:
        loaded = pickle.load(file)

    assert not loaded.is_trained
    loaded.train(X_CORPUS)
    loaded.add(X_CORPUS)
    assert loaded.ntotal == N
    undo_faiss_patch('IndexScalarQuantizer', print_patched=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
