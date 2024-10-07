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

"""
PCA tests, check output of skpatch versus sklearn
"""

# pylint: disable = import-outside-toplevel, reimported, no-member

import numpy as np
import pytest
from aoclda.sklearn import skpatch, undo_skpatch

def test_patching():
    """
    Test we can patch using string arguments to skpatch
    """

    skpatch()
    from sklearn.decomposition import PCA
    pca_da = PCA(n_components=3)
    assert pca_da.aocl is True

    undo_skpatch()
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    assert not hasattr(pca, 'aocl')

    skpatch("Lasso")
    from sklearn.linear_model import Lasso
    lasso_da = Lasso(alpha=0.1)
    assert lasso_da.aocl is True

    undo_skpatch("Lasso")
    from sklearn.linear_model import Lasso
    lasso = Lasso(alpha=0.1)
    assert not hasattr(lasso, 'aocl')

    skpatch(["LinearRegression", "Ridge", "Wibble"])
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.decomposition import PCA
    pca_da2 = PCA(n_components=3)
    assert not hasattr(pca_da2, 'aocl')
    ridge_da = Ridge()
    assert ridge_da.aocl is True
    linreg_da = LinearRegression()
    assert linreg_da.aocl is True

    undo_skpatch(["LinearRegression", "Wobble", "Ridge"])
    from sklearn.linear_model import LinearRegression, Ridge
    ridge = Ridge()
    linreg = LinearRegression()
    assert not hasattr(ridge, 'aocl')
    assert not hasattr(linreg, 'aocl')

    with pytest.raises(TypeError):
        skpatch(1)

    with pytest.raises(TypeError):
        undo_skpatch(1)

if __name__ == "__main__":
    test_patching()
