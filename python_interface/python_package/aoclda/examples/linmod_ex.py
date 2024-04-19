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
# pylint: disable = invalid-name

"""
Linear regression example Python script
"""

import sys
from aoclda.linear_model import linmod
import numpy as np


def linmod_example():
    """
    Linear regression with an intercept variable
    """
    X = np.array([[1, 1], [2, 3], [3, 5], [4, 8], [5, 7], [6, 9]])
    y = np.array([3., 6.5, 10., 12., 13., 19.])
    lmod = linmod("mse", intercept=True)
    lmod.fit(X, y)

    # Extract coefficients
    coef = lmod.coef
    print(f"coefficients: [{coef[0]:.3f}, {coef[1]:.3f}, {coef[2]:.3f}]")
    print('expected    : [2.350, 0.350, 0.433]\n')

    # Evaluate model on new data
    X_test = np.array([[1, 1.1], [2.5, 3], [7, 9]])
    pred = lmod.predict(X_test)
    print(f"predictions: [{pred[0]:.3f}, {pred[1]:.3f}, {pred[2]:.3f}]")
    print('expected   : [3.168  7.358 20.0333]')


if __name__ == "__main__":
    try:
        linmod_example()
    except RuntimeError:
        sys.exit(1)
