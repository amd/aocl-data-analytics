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
# pylint: disable = unused-argument

import pytest
import faiss
from aoclda.faiss import faiss_patch, undo_faiss_patch
from aoclda.faiss._index_ivfflat import IndexIVFFlat as IndexIVFFlat_da
from aoclda.faiss._index_ivfsq import IndexIVFScalarQuantizer as IndexIVFScalarQuantizer_da
from aoclda.faiss._index_sq import IndexScalarQuantizer as IndexScalarQuantizer_da
from aoclda.faiss._index_flat_l2 import IndexFlatL2 as IndexFlatL2_da
from conftest import (
    _native_IndexFlatL2,
    _native_IndexIVFFlat,
    _native_IndexIVFScalarQuantizer,
    _native_IndexScalarQuantizer,
    D,
    NLIST,
)


# ---------------------------------------------------------------------------
# Patch/unpatch lifecycle
# ---------------------------------------------------------------------------

def test_patch_installs(capsys):
    faiss_patch(print_patched=True)
    assert faiss.IndexIVFFlat is IndexIVFFlat_da
    assert faiss.IndexIVFScalarQuantizer is IndexIVFScalarQuantizer_da
    assert faiss.IndexScalarQuantizer is IndexScalarQuantizer_da
    assert faiss.IndexFlatL2 is IndexFlatL2_da
    out = capsys.readouterr().out
    assert "AOCL Extension for faiss enabled" in out
    undo_faiss_patch(print_patched=False)


def test_unpatch_restores(capsys):
    faiss_patch(print_patched=False)
    undo_faiss_patch(print_patched=True)
    assert faiss.IndexIVFFlat is _native_IndexIVFFlat
    assert faiss.IndexIVFScalarQuantizer is _native_IndexIVFScalarQuantizer
    assert faiss.IndexScalarQuantizer is _native_IndexScalarQuantizer
    assert faiss.IndexFlatL2 is _native_IndexFlatL2
    out = capsys.readouterr().out
    assert "AOCL Extension for faiss disabled" in out


def test_double_patch_idempotent():
    faiss_patch(print_patched=False)
    faiss_patch(print_patched=False)
    assert faiss.IndexIVFFlat is IndexIVFFlat_da
    assert faiss.IndexIVFScalarQuantizer is IndexIVFScalarQuantizer_da
    assert faiss.IndexScalarQuantizer is IndexScalarQuantizer_da
    undo_faiss_patch(print_patched=False)


def test_selective_patch_by_string():
    faiss_patch('IndexFlatL2', print_patched=False)
    assert faiss.IndexFlatL2 is IndexFlatL2_da
    assert faiss.IndexIVFFlat is _native_IndexIVFFlat
    undo_faiss_patch(print_patched=False)


def test_selective_patch_by_list():
    faiss_patch(
        ['IndexIVFFlat', 'IndexIVFScalarQuantizer',
         'IndexScalarQuantizer', 'IndexFlatL2'],
        print_patched=False)
    assert faiss.IndexIVFFlat is IndexIVFFlat_da
    assert faiss.IndexIVFScalarQuantizer is IndexIVFScalarQuantizer_da
    assert faiss.IndexScalarQuantizer is IndexScalarQuantizer_da
    assert faiss.IndexFlatL2 is IndexFlatL2_da
    undo_faiss_patch(print_patched=False)


def test_selective_unpatch_by_string():
    faiss_patch(print_patched=False)
    undo_faiss_patch('IndexFlatL2', print_patched=False)
    assert faiss.IndexFlatL2 is _native_IndexFlatL2
    assert faiss.IndexIVFFlat is IndexIVFFlat_da
    undo_faiss_patch(print_patched=False)


def test_patch_unknown_symbol(capsys):
    faiss_patch('NonExistentIndex', print_patched=False)
    undo_faiss_patch(print_patched=False)
    assert "NonExistentIndex was not found" in capsys.readouterr().out


def test_patch_bad_type_raises():
    with pytest.raises(TypeError):
        faiss_patch(42)
    with pytest.raises(TypeError):
        undo_faiss_patch(42)


def test_da_index_rejected_by_native_composite(patched_faiss):
    # DA IndexIVFFlat is a plain Python object, not SWIG-wrapped; FAISS
    # type-checks C++ index arguments and rejects the wrapper with TypeError.
    q = faiss.IndexFlatL2(D)
    da_ivf = faiss.IndexIVFFlat(q, D, NLIST)
    with pytest.raises(TypeError):
        faiss.IndexHNSW2Level(da_ivf, NLIST, 2, 16)


# ---------------------------------------------------------------------------
# index_factory dispatch
# ---------------------------------------------------------------------------

def test_index_factory_ivfflat_string(patched_faiss):
    idx = faiss.index_factory(D, "IVF64,Flat")
    assert isinstance(idx, IndexIVFFlat_da)
    assert idx.nlist == 64


def test_index_factory_flat_string_returns_da_wrapper(patched_faiss):
    # _aoclda_index_factory routes "Flat" + METRIC_L2 to IndexFlatL2_da
    idx = faiss.index_factory(D, "Flat")
    assert isinstance(idx, IndexFlatL2_da)


def test_index_factory_metric_inner_product(patched_faiss):
    idx = faiss.index_factory(D, "IVF32,Flat", faiss.METRIC_INNER_PRODUCT)
    assert isinstance(idx, IndexIVFFlat_da)
    assert idx.metric_type == faiss.METRIC_INNER_PRODUCT


@pytest.mark.parametrize(
    "description",
    ["IVF32,SQ4", "IVF32,SQ8", "IVF32,SQ8_direct", "IVF32,SQ8_direct_signed",
     "IVF32,SQbf16"])
def test_index_factory_ivfsq_string(patched_faiss, description):
    idx = faiss.index_factory(D, description)
    assert isinstance(idx, IndexIVFScalarQuantizer_da)
    assert idx.nlist == 32
    assert idx.by_residual is True


def test_index_factory_ivfsq_inner_product(patched_faiss):
    idx = faiss.index_factory(D, "IVF32,SQ8", faiss.METRIC_INNER_PRODUCT)
    assert isinstance(idx, IndexIVFScalarQuantizer_da)
    assert idx.metric_type == faiss.METRIC_INNER_PRODUCT


@pytest.mark.parametrize(
    "description",
    ["SQ4", "SQ8", "SQ8_direct", "SQ8_direct_signed", "SQbf16"])
def test_index_factory_sq_string(patched_faiss, description):
    idx = faiss.index_factory(D, description)
    assert isinstance(idx, IndexScalarQuantizer_da)


def test_index_factory_sq_inner_product(patched_faiss):
    idx = faiss.index_factory(D, "SQ8", faiss.METRIC_INNER_PRODUCT)
    assert isinstance(idx, IndexScalarQuantizer_da)
    assert idx.metric_type == faiss.METRIC_INNER_PRODUCT


@pytest.mark.parametrize("description", ["SQfp16", "SQ6"])
def test_index_factory_native_sq_falls_through(patched_faiss, description):
    idx = faiss.index_factory(D, description)
    assert isinstance(idx, _native_IndexScalarQuantizer)


@pytest.mark.parametrize("description", ["SQ4_uniform", "SQ8_uniform"])
def test_index_factory_invalid_flat_sq_raises(patched_faiss, description):
    with pytest.raises(RuntimeError):
        faiss.index_factory(D, description)


@pytest.mark.parametrize("description", ["IVF32,SQfp16"])
def test_index_factory_unsupported_sq_falls_through(patched_faiss, description):
    idx = faiss.index_factory(D, description)
    assert isinstance(idx, _native_IndexIVFScalarQuantizer)


@pytest.mark.parametrize(
    "description",
    [pytest.param("IVF32,SQ4_uniform", id="uniform-has-no-factory-token"),
     pytest.param("IVF32,SQ4_direct", id="direct-qtype-does-not-exist"),
     pytest.param("IVF32,SQ4_direct_signed",
                  id="direct-signed-qtype-does-not-exist")])
def test_index_factory_invalid_sq4_falls_through(patched_faiss, description):
    with pytest.raises(RuntimeError):
        faiss.index_factory(D, description)


def test_index_factory_unpatched_algo_falls_through_to_native(patched_faiss):
    # IVFPQ is not intercepted; _aoclda_index_factory falls through to native faiss
    idx = faiss.index_factory(D, "IVF64,PQ8")
    assert not isinstance(idx, IndexIVFFlat_da)
    assert not isinstance(idx, IndexFlatL2_da)
    assert isinstance(idx, faiss.IndexIVFPQ)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
