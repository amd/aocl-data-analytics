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
Decision tree Python test script
"""

import numpy as np
import pytest
from aoclda.decision_tree import decision_tree

@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_decision_tree_functionality(numpy_precision, numpy_order):
    """
    Test the functionality of the Python wrapper
    """

    X_train = np.array([[0.0, 0.0], [1.0, 1.0]],
                       dtype=numpy_precision, order=numpy_order)
    y_train = np.array([0, 1],
                       dtype=numpy_precision, order=numpy_order)
    X_test = np.array([[2., 2.]],
                      dtype=numpy_precision, order=numpy_order)
    y_test = np.array([[1]],
                      dtype=numpy_precision, order=numpy_order)

    clf = decision_tree()
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    score = clf.score(X_test, y_test)

    assert pred[0] == 1
    assert score == 1.0
    assert clf.n_nodes == 3
    assert clf.n_leaves == 2

    print(f"predictions: [{pred[0]:d}]")
    print(f"probabilities: [{proba[0,0]:.3f}, {proba[0,1]:.3f}]")
    print(f"score: {score:.3f}")
