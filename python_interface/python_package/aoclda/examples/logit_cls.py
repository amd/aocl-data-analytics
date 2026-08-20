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
# pylint: disable = invalid-name,unbalanced-tuple-unpacking

"""
Linear logistic classifier example Python script
"""

import sys
from aoclda.linear_model import linmod
from aoclda.utils import train_test_split as split
import numpy as np


def logit_classifier():
    """
    Linear logistic classifier with an intercept for three classes
    """
    X = np.array([[-0.73018489, 1.99695757],
                  [-0.77691935, 2.88362735],
                  [-0.5531103, -0.45059391],
                  [-0.53005191, 1.60981248],
                  [-0.29156667, -0.72399731],
                  [0.1168031, 0.79898074],
                  [-1.40480427, 2.11187681],
                  [-0.33364746, -1.43869773],
                  [-1.12263088, -1.76823514],
                  [-1.48629103, 2.48784366],
                  [-0.86543502, 2.44513227],
                  [-1.12070302, 0.07709601],
                  [-0.9318949, 2.87190709],
                  [0.07586872, 0.05356797],
                  [-0.66288856, 0.5230911],
                  [0.18171725, 1.01352685],
                  [-0.737435, -1.11759934],
                  [0.22143158, -0.75510393],
                  [-0.37855445, 0.36976656],
                  [0.99452555, -1.71651458],
                  [-0.11916944, -0.67448552],
                  [0.42473491, 1.98517725],
                  [1.08249098, 0.56222941],
                  [0.3675828, 0.07899979],
                  [0.45179015, 1.76713307],])

    labels = np.array([1, 1, 0, 1, 0, 2, 0, 0, 0, 1, 1, 0,
                      1, 2, 1, 2, 0, 0, 2, 2, 0, 2, 2, 2, 1])
    X_train, X_test, labels_train, labels_test = split(
        X, labels, test_size=0.2, seed=2001)

    clss = linmod("logistic", intercept=True)
    clss.fit(X_train, labels_train)

    # Get trained model accuracy
    acc = clss.score(X_train, labels_train)
    print(f"Accuracy: {acc:.3f}")  # expected 0.75

    # Evaluate model on new data
    pred = clss.predict(X_test).astype(int)
    print("Predictions:")

    for p, e in zip(pred, labels_test):
        print(f"predicted {p}    expected {e}")

    return clss.score(X_test, labels_test) < acc


if __name__ == "__main__":
    try:
        flag = 2 if logit_classifier() else 0
        sys.exit(flag)
    except RuntimeError:
        sys.exit(1)
