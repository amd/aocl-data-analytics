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
from aoclda.sklearn import skpatch, undo_skpatch
import pytest

def test_pca():
    """
    Basic 3 x 2 problem
    """
    a = np.array([[1, 2, 3], [0.22, 5, 4.1], [3, 6, 1]])

    # patch and import scikit-learn
    skpatch()
    from sklearn.decomposition import PCA
    pca_da = PCA(n_components=3)
    pca_da.fit(a)
    assert pca_da.aocl is True

    # unpatch and solve the same problem with sklearn
    undo_skpatch()
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca.fit(a)
    assert not hasattr(pca, 'aocl')

    # Check results
    da_components = pca_da.components_
    components = pca.components_
    assert da_components == pytest.approx(components, 1.0e-08)
    da_mean = pca_da.mean_
    mean = pca.mean_
    assert da_mean == pytest.approx(mean, 1.0e-08)
    da_singval = pca_da.singular_values_
    singval = pca.singular_values_
    assert da_singval == pytest.approx(singval, 1.0e-08)

    # print the results if pytest is invoked with the -rA option
    print("Components")
    print("    aoclda: \n", da_components)
    print("   sklearn: \n", components)


if __name__ == "__main__":
    test_pca()
