# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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


from aoclda.factorization import PCA
import aoclda as da
import numpy as np
import pytest

# Run these tests by typing 'python -m pytest' in the command prompt or 'pytohn -m pytest -rA' to print out stuff in the tests

@pytest.mark.parametrize("da_precision, numpy_precision", [
    (da.double, np.float64), (da.single, np.float32),
])

def test_pca_double(da_precision, numpy_precision):

    a = np.array([[1,2,3],[0.2,5,4.1],[3,6,1]], dtype=numpy_precision)

    pca = PCA(n_components=3, precision = da_precision)
    pca.fit(a)

    # Check we look to have got the right answer
    norm = np.linalg.norm(pca.inverse_transform(pca.transform(a)) - a)
    tol =  np.finfo(numpy_precision).eps * 1000

    print("data type: {}, norm: {}, tol: {} ".format(numpy_precision, norm, tol))

    assert norm < tol