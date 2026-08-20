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

"""Shared pytest fixtures and test data for the FAISS extension tests.

"""

import warnings
import pytest
import numpy as np

# Suppress faiss' SWIG DeprecationWarnings, emitted at import time.
# import in the indented block so autopep doesn't move it
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", message="builtin type Swig", category=DeprecationWarning)
    warnings.filterwarnings(
        "ignore", message="builtin type swigvarlink", category=DeprecationWarning)
    import faiss
    from aoclda.faiss import faiss_patch, undo_faiss_patch


# Grab the genuine faiss classes now, before any test patches them, so tests
# can compare AOCL-DA results against native faiss.
_native_IndexFlatL2 = faiss.IndexFlatL2
_native_IndexIVFFlat = faiss.IndexIVFFlat

# Small, fixed random dataset
RNG = np.random.default_rng(0)
D = 16
N = 500
NQ = 20
K = 5
NLIST = 16
X_CORPUS = RNG.random((N, D)).astype(np.float32)
X_QUERY = RNG.random((NQ, D)).astype(np.float32)


# Fixture that tests opt into to run with AOCL-DA patched in; native faiss is
# restored afterwards.
@pytest.fixture(autouse=False)
def patched_faiss():
    faiss_patch(print_patched=False)
    yield
    undo_faiss_patch(print_patched=False)
