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

"""Patching faiss IndexIVFScalarQuantizer."""

import faiss
from ._index_ivfflat import IndexIVFFlat


# Map FAISS qtypes to the corresponding AOCL-DA quantization options.
_QTYPE_OPTIONS = {
    faiss.ScalarQuantizer.QT_4bit: ('int4', 'per_dimension'),
    faiss.ScalarQuantizer.QT_4bit_uniform: ('int4', 'uniform'),
    faiss.ScalarQuantizer.QT_8bit: ('int8', 'per_dimension'),
    faiss.ScalarQuantizer.QT_8bit_uniform: ('int8', 'uniform'),
    faiss.ScalarQuantizer.QT_8bit_direct: ('int8', 'direct_unsigned'),
    faiss.ScalarQuantizer.QT_8bit_direct_signed: ('int8', 'direct_signed'),
    faiss.ScalarQuantizer.QT_bf16: ('bfloat16', 'direct_signed'),
}
_QTYPE_NAMES = {
    faiss.ScalarQuantizer.QT_4bit: 'QT_4bit',
    faiss.ScalarQuantizer.QT_4bit_uniform: 'QT_4bit_uniform',
    faiss.ScalarQuantizer.QT_8bit: 'QT_8bit',
    faiss.ScalarQuantizer.QT_8bit_uniform: 'QT_8bit_uniform',
    faiss.ScalarQuantizer.QT_8bit_direct: 'QT_8bit_direct',
    faiss.ScalarQuantizer.QT_8bit_direct_signed: 'QT_8bit_direct_signed',
    faiss.ScalarQuantizer.QT_bf16: 'QT_bf16',
}


# Reuse IVFFlat's train/add/search lifecycle, adding only quantized storage options.
class IndexIVFScalarQuantizer(IndexIVFFlat):
    """AOCL-DA backed replacement for faiss.IndexIVFScalarQuantizer.

    Supported qtypes are ``QT_4bit``, ``QT_4bit_uniform``, ``QT_8bit``,
    ``QT_8bit_uniform``, ``QT_8bit_direct``, ``QT_8bit_direct_signed`` and
    ``QT_bf16``.
    """

    # Extend the inherited public-attribute allowlist with the IVFSQ setting.
    _SUPPORTED_ATTRS = IndexIVFFlat._SUPPORTED_ATTRS | frozenset({'by_residual'})

    def __init__(self, quantizer, d, nlist, qtype, metric=1, by_residual=True):
        super().__init__(quantizer, d, nlist, metric)
        if qtype not in _QTYPE_OPTIONS:
            supported = ', '.join(_QTYPE_NAMES.values())
            raise ValueError(
                f"Unsupported scalar quantizer type {qtype}; expected one of {supported}")

        self._qtype = qtype
        quantization, self._quantization_scheme = _QTYPE_OPTIONS[qtype]
        self._by_residual = bool(by_residual)
        self._backend_options.update({
            'algorithm': 'ivfsq',
            'quantization': quantization,
            'quantization_scheme': self._quantization_scheme,
            'quantization_residual': self._by_residual,
        })

    @property
    def by_residual(self):
        return self._by_residual

    @by_residual.setter
    def by_residual(self, value):
        if self._is_trained:
            raise RuntimeError(
                "IndexIVFScalarQuantizer.by_residual cannot be changed after training")
        self._by_residual = bool(value)
        self._backend_options['quantization_residual'] = self._by_residual
