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
#from sklearn.decomposition import PCA as sklearnPCA
import numpy as np

import time


def test_pca():

    a = np.array([[1, 2, 3], [0.22, 5, 4.1], [3, 6, 1]])

    pca = PCA(n_components=3)
    pca.fit(a)
    print(pca.principal_components)
    print(pca.u)
    print(pca.variance)
    print(pca.total_variance)

    print(pca.__doc__)
    #help(pca)

    # Check we look to have got the right answer
    print(np.linalg.norm(pca.inverse_transform(pca.transform(a)) - a))

    # Now try a single precision PCA
    a = np.array([[1, 2, 3], [0.22, 5, 4.1], [3, 6, 1]], dtype=np.float32)
    pca = PCA(n_components=3, precision=da.single)
    pca.fit(a)
    print(np.linalg.norm(pca.inverse_transform(pca.transform(a)) - a))

    n = 100

    a = np.random.rand(n, n)

    t0 = time.time()
    pca = PCA(n_components=n)
    pca.fit(a)
    t1 = time.time()

    print(t1-t0)

    #t0 = time.time()
    #sk_pca = sklearnPCA(n_components=n)
    #sk_pca.fit(a)
    #t1 = time.time()

    #print(t1-t0)

    # Check we can create an exception
    #b = np.array([[3,22],[1,1]])
    #pca.transform(b)

if __name__ == "__main__":
    test_pca()