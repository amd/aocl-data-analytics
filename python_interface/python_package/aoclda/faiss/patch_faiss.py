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

import re
import faiss
from ._index_ivfflat import IndexIVFFlat as IndexIVFFlat_da
from ._index_flat_l2 import IndexFlatL2 as IndexFlatL2_da
from ._kmeans import Kmeans as Kmeans_da

_orig_index_factory = faiss.index_factory
_orig_kmeans = faiss.Kmeans


def _aoclda_index_factory(d, description, metric=faiss.METRIC_L2):
    desc = description.strip()
    m = re.fullmatch(r'IVF(\d+),Flat', desc)
    if m:
        nlist = int(m.group(1))
        # The AOCL-DA backend only reads the dimension from the quantizer, so an
        # IndexFlatL2 coarse quantizer is fine regardless of the search metric.
        quantizer = faiss.IndexFlatL2(d)
        return IndexIVFFlat_da(quantizer, d, nlist, metric)
    if desc == 'Flat' and metric == faiss.METRIC_L2:
        return IndexFlatL2_da(d)
    return _orig_index_factory(d, description, metric)


AMD_FAISS_SYMBOLS = {
    'IndexIVFFlat': {
        'pack': faiss,
        'faiss_sym': getattr(faiss, 'IndexIVFFlat'),
        'da_sym': IndexIVFFlat_da,
    },
    'IndexFlatL2': {
        'pack': faiss,
        'faiss_sym': getattr(faiss, 'IndexFlatL2'),
        'da_sym': IndexFlatL2_da,
    },
    'index_factory': {
        'pack': faiss,
        'faiss_sym': _orig_index_factory,
        'da_sym': _aoclda_index_factory,
    },
    'Kmeans': {
        'pack': faiss,
        'faiss_sym': _orig_kmeans,
        'da_sym': Kmeans_da,
    },
}


def faiss_patch(*args, print_patched=True):
    """Replace specified faiss index classes with AOCL-DA optimized equivalents."""

    if not args:
        packages = AMD_FAISS_SYMBOLS.keys()
    elif isinstance(args[0], str):
        packages = [args[0]]
    elif isinstance(args[0], (list, tuple)):
        packages = args[0]
    else:
        raise TypeError("Unrecognized argument")

    successfully_patched = []

    for package in packages:
        try:
            pack = AMD_FAISS_SYMBOLS[package]['pack']
            sym = AMD_FAISS_SYMBOLS[package]['da_sym']
            setattr(pack, package, sym)
            successfully_patched.append(package)
        except KeyError:
            print(f"The symbol {package} was not found.")

    if successfully_patched and print_patched:
        print("AOCL Extension for faiss enabled for the following symbols:")
        print(', '.join(successfully_patched))


def undo_faiss_patch(*args, print_patched=True):
    """Reinstate faiss index classes with their original symbols."""

    if not args:
        packages = AMD_FAISS_SYMBOLS.keys()
    elif isinstance(args[0], str):
        packages = [args[0]]
    elif isinstance(args[0], (list, tuple)):
        packages = args[0]
    else:
        raise TypeError("Unrecognized argument")

    successfully_unpatched = []

    for package in packages:
        try:
            pack = AMD_FAISS_SYMBOLS[package]['pack']
            sym = AMD_FAISS_SYMBOLS[package]['faiss_sym']
            setattr(pack, package, sym)
            successfully_unpatched.append(package)
        except KeyError:
            print(f"The symbol {package} was not found.")

    if successfully_unpatched and print_patched:
        print("AOCL Extension for faiss disabled for the following symbols:")
        print(', '.join(successfully_unpatched))
