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

import pytest
import faiss
from aoclda.faiss import faiss_patch, undo_faiss_patch
from aoclda.faiss._index_ivfflat import IndexIVFFlat as IndexIVFFlat_da
from aoclda.faiss._index_flat_l2 import IndexFlatL2 as IndexFlatL2_da
from conftest import _native_IndexFlatL2, _native_IndexIVFFlat, D, NLIST


# ---------------------------------------------------------------------------
# Patch/unpatch lifecycle
# ---------------------------------------------------------------------------

def test_patch_installs(capsys):
    faiss_patch(print_patched=True)
    assert faiss.IndexIVFFlat is IndexIVFFlat_da
    assert faiss.IndexFlatL2 is IndexFlatL2_da
    out = capsys.readouterr().out
    assert "AOCL Extension for faiss enabled" in out
    undo_faiss_patch(print_patched=False)


def test_unpatch_restores(capsys):
    faiss_patch(print_patched=False)
    undo_faiss_patch(print_patched=True)
    assert faiss.IndexIVFFlat is _native_IndexIVFFlat
    assert faiss.IndexFlatL2 is _native_IndexFlatL2
    out = capsys.readouterr().out
    assert "AOCL Extension for faiss disabled" in out


def test_double_patch_idempotent():
    faiss_patch(print_patched=False)
    faiss_patch(print_patched=False)
    assert faiss.IndexIVFFlat is IndexIVFFlat_da
    undo_faiss_patch(print_patched=False)


def test_selective_patch_by_string():
    faiss_patch('IndexFlatL2', print_patched=False)
    assert faiss.IndexFlatL2 is IndexFlatL2_da
    assert faiss.IndexIVFFlat is _native_IndexIVFFlat
    undo_faiss_patch(print_patched=False)


def test_selective_patch_by_list():
    faiss_patch(['IndexIVFFlat', 'IndexFlatL2'], print_patched=False)
    assert faiss.IndexIVFFlat is IndexIVFFlat_da
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
    assert type(idx) is IndexIVFFlat_da
    assert idx.nlist == 64


def test_index_factory_flat_string_returns_da_wrapper(patched_faiss):
    # _aoclda_index_factory routes "Flat" + METRIC_L2 to IndexFlatL2_da
    idx = faiss.index_factory(D, "Flat")
    assert type(idx) is IndexFlatL2_da


def test_index_factory_metric_inner_product(patched_faiss):
    idx = faiss.index_factory(D, "IVF32,Flat", faiss.METRIC_INNER_PRODUCT)
    assert type(idx) is IndexIVFFlat_da
    assert idx.metric_type == faiss.METRIC_INNER_PRODUCT


def test_index_factory_unpatched_algo_falls_through_to_native(patched_faiss):
    # IVFPQ is not intercepted; _aoclda_index_factory falls through to native faiss
    idx = faiss.index_factory(D, "IVF64,PQ8")
    assert type(idx) is not IndexIVFFlat_da
    assert type(idx) is not IndexFlatL2_da
    assert type(idx) is faiss.IndexIVFPQ


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
