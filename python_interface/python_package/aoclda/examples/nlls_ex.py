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
# pylint: disable = invalid-name, import-error, unused-argument, missing-function-docstring, unused-variable

"""
Nonlinear Data Fitting example Python script
"""

import sys
from aoclda.nonlinear_model import nlls
import numpy as np

def res(x, residuals, data=None) -> int:
    (t, y) = data
    x1 = x[0]
    x2 = x[1]
    residuals[:] = x1 * np.exp(x2 * t) - y
    return 0

def jac(x, jacobian, data) -> int:
    x1 = x[0]
    x2 = x[1]
    (t, y) = data
    jacobian[:] = np.column_stack((
        np.exp(x2*t),
        t * x1 * np.exp(x2*t)
    ))
    return 0


def hes(x, r, Hr, data) -> int:
    (t, y) = data
    x1 = x[0]
    x2 = x[1]
    Hr[:] = np.zeros((2, 2))
    Hr[0, 0] = 0.0                    # H_11
    v = t * np.exp(x2*t)
    Hr[1, 0] = np.dot(r, v)           # H_21
    Hr[1, 1] = np.dot(r, (t*x1)*v)    # H_22
    return 0


def nlls_example():
    """
    Nonlinear Data Fitting of a convolution model, the solution
    provides the isolated parameters for the model:
        yi = x1 e^(x2 ti)
    for data vectors y : yi and t : ti (i=1:5)
    """
    # Data to be fitted
    t = np.array([1.0, 2.0, 4.0,  5.0,  8.0])
    y = np.array([3.0, 4.0, 6.0, 11.0, 20.0])

    n_coef = 2
    n_res = 5
    xexp = np.array([2.54104549, 0.25950481], dtype=np.float64)
    x = np.array([2.5, 0.25], dtype=np.float64)
    w = 0.12 * np.array([1, 1, 1, 1, 1], dtype=np.float64)
    blx = np.array([0.0,  0.0], dtype=np.float64)
    bux = np.array([5.0,  3.0], dtype=np.float64)
    ndf = nlls(n_coef, n_res, weights=w, lower_bounds=blx, upper_bounds=bux)
    ndf.fit(x, res, jac, hes, data=(t, y), abs_gtol=1e-7, gtol=1.e-9, maxit=20)

    print(f"Solution found in {ndf.n_iter} iterations")
    print(f"Residual norm at solution: {ndf.metrics['obj']:.4f}")
    print("Solution:")
    for i in range(2):
        ok = np.abs(x[i]-xexp[i]) <= 1.e-5
        print(f"x[{i}]={x[i]:.5f} expected: ({xexp[i]:.5f}) OK? {ok}")

if __name__ == "__main__":
    try:
        nlls_example()
    except RuntimeError:
        print("Something unexpected happened while running the example.")
        sys.exit(1)
