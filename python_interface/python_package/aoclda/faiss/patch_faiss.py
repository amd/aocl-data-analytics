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

"""Contains the functions to replace FAISS index classes with AOCL-DA optimized equivalents."""

# pylint: disable = duplicate-code

import re
import faiss
from .._patching import register_patch, requested_patches
from ._index_ivfflat import IndexIVFFlat as IndexIVFFlat_da
from ._index_ivfsq import IndexIVFScalarQuantizer as IndexIVFScalarQuantizer_da
from ._index_sq import IndexScalarQuantizer as IndexScalarQuantizer_da
from ._index_flat_l2 import IndexFlatL2 as IndexFlatL2_da
from ._index_flat_ip import IndexFlatIP as IndexFlatIP_da
from ._kmeans import Kmeans as Kmeans_da

_orig_index_factory = getattr(faiss, 'index_factory', None)

# Map FAISS's canonical factory tokens to the qtypes accepted by the wrapper.
# QT_4bit_uniform is constructor-only in FAISS; there is no SQ4_uniform token.
_SQ_FACTORY_QTYPES = {
    'SQ4': faiss.ScalarQuantizer.QT_4bit,
    'SQ8': faiss.ScalarQuantizer.QT_8bit,
    'SQ8_direct': faiss.ScalarQuantizer.QT_8bit_direct,
    'SQ8_direct_signed': faiss.ScalarQuantizer.QT_8bit_direct_signed,
    'SQbf16': faiss.ScalarQuantizer.QT_bf16,
}


def _aoclda_index_factory(d, description, metric=faiss.METRIC_L2):
    desc = description.strip()
    m = re.fullmatch(r'IVF(\d+),Flat', desc)
    if m:
        nlist = int(m.group(1))
        # The AOCL-DA backend only reads the dimension from the quantizer, so an
        # IndexFlatL2 coarse quantizer is fine regardless of the search metric.
        quantizer = faiss.IndexFlatL2(d)
        return IndexIVFFlat_da(quantizer, d, nlist, metric)
    m = re.fullmatch(
        r'IVF(\d+),(SQ4|SQ8|SQ8_direct|SQ8_direct_signed|SQbf16)', desc)
    if m and m.group(2) in _SQ_FACTORY_QTYPES:
        nlist = int(m.group(1))
        quantizer = faiss.IndexFlatL2(d)
        return IndexIVFScalarQuantizer_da(
            quantizer, d, nlist, _SQ_FACTORY_QTYPES[m.group(2)], metric)
    if desc in _SQ_FACTORY_QTYPES:
        return IndexScalarQuantizer_da(d, _SQ_FACTORY_QTYPES[desc], metric)
    if desc == 'Flat' and metric == faiss.METRIC_L2:
        return IndexFlatL2_da(d)
    if desc == 'Flat' and metric == faiss.METRIC_INNER_PRODUCT:
        return IndexFlatIP_da(d)
    return _orig_index_factory(d, description, metric)


AMD_SYMBOLS = {}
UNAVAILABLE_SYMBOLS = {}

_PATCHES = (
    ('IndexIVFFlat', IndexIVFFlat_da),
    ('IndexIVFScalarQuantizer', IndexIVFScalarQuantizer_da),
    ('IndexScalarQuantizer', IndexScalarQuantizer_da),
    ('IndexFlatL2', IndexFlatL2_da),
    ('IndexFlatIP', IndexFlatIP_da),
    ('index_factory', _aoclda_index_factory),
    ('Kmeans', Kmeans_da),
)

for _name, _da_symbol in _PATCHES:
    register_patch(AMD_SYMBOLS, UNAVAILABLE_SYMBOLS,
                   _name, faiss, _da_symbol)


def faiss_patch(*args, print_patched=True):
    """Replace specified faiss index classes with AOCL-DA optimized equivalents."""

    packages = requested_patches(
        args, AMD_SYMBOLS, UNAVAILABLE_SYMBOLS)

    successfully_patched = []

    for package in packages:
        pack = AMD_SYMBOLS[package]['pack']
        sym = AMD_SYMBOLS[package]['da_sym']
        setattr(pack, package, sym)
        successfully_patched.append(package)

    if successfully_patched and print_patched:
        print("AOCL Extension for faiss enabled for the following symbols:")
        print(', '.join(successfully_patched))


def undo_faiss_patch(*args, print_patched=True):
    """Reinstate faiss index classes with their original symbols."""

    packages = requested_patches(
        args, AMD_SYMBOLS, UNAVAILABLE_SYMBOLS)

    successfully_unpatched = []

    for package in packages:
        pack = AMD_SYMBOLS[package]['pack']
        sym = AMD_SYMBOLS[package]['orig_sym']
        setattr(pack, package, sym)
        successfully_unpatched.append(package)

    if successfully_unpatched and print_patched:
        print("AOCL Extension for faiss disabled for the following symbols:")
        print(', '.join(successfully_unpatched))
