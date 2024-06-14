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
Decision forest example Python script
"""

import sys
import numpy as np
from aoclda.decision_forest import decision_forest

def decision_forest_example():
    """
    Decision forest classification
    """
    X_train = np.array([[0.0, 0.0], [1.0, 1.0]])
    y_train = np.array([0, 1])
    X_test = np.array([[2., 2.]])
    y_test = np.array([[1]])

    clf = decision_forest(seed = 988,
                          n_obs_per_tree = 100,
                          n_features_to_select = 1,
                          n_trees = 20,
                          score_criteria = "cross-entropy")
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    print(f"predictions: [{pred[0]:d}]")

    score = clf.score(X_test, y_test)
    print(f"score: {score:.3f}")

if __name__ == "__main__":
    try:
        decision_forest_example()
    except RuntimeError:
        sys.exit(1)