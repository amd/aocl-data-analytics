# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
train_test_split tests, check output of skpatch versus sklearn
"""

import numpy as np
import pytest
from aoclda.sklearn import skpatch, undo_skpatch


@pytest.mark.parametrize("np_precision",
                         [np.float64, np.float32, np.int32, np.int64])
def test_train_test_split(np_precision):
    """
    Basic test to see that the skpatch for train_test_split works.
    """

    X = np.random.rand(30, 30).astype(np_precision)
    y = np.random.rand(30, 1).astype(np_precision)

    skpatch()
    from sklearn.model_selection import train_test_split

    X_train_da, X_test_da, y_train_da, y_test_da = train_test_split(
        X, y, shuffle=False)

    undo_skpatch()

    X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split(
        X, y, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)
    assert np.array_equal(y_train_da, y_train_sk)
    assert np.array_equal(y_test_da, y_test_sk)
