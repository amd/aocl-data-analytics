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
Nonlinear data fitting example Python script
"""

import sys
import math
from aoclda.nonlinear_model import nlls
import numpy as np

n_res = 64

# Define Normal and log-Normal distributions


def lognormal(d, a, b, Al):
    return Al/(d*b*np.sqrt(2*math.pi))*np.exp(-((np.log(d)-a)**2)/(2*b**2))


def gaussian(d, mu, sigma, Ag):
    return Ag*np.exp(-0.5*((d-mu)/sigma)**2)/(sigma*np.sqrt(2*math.pi))


def res(x, residuals, data=None) -> int:
    d = data['d']
    y = data['y']
    a = x[0]
    b = x[1]
    Al = x[2]
    mu = x[3]
    sigma = x[4]
    Ag = x[5]
    for i in range(n_res):
        residuals[i] = lognormal(d[i], a, b, Al) + \
            gaussian(d[i], mu, sigma, Ag) - y[i]
    return 0


def nlls_fd_example():
    """
    Nonlinear data fitting for the convolution model
    y ~ mixture of normal and log-normal
    for data vectors y : yi and d : di (i=1:5)
    using finite-differences
    """
    # Observations / data
    diameter = range(1, n_res+1)
    density = [
        0.0722713864, 0.0575221239, 0.0604719764, 0.0405604720, 0.0317109145,
        0.0309734513, 0.0258112094, 0.0228613569, 0.0213864307, 0.0213864307,
        0.0147492625, 0.0213864307, 0.0243362832, 0.0169616519, 0.0095870206,
        0.0147492625, 0.0140117994, 0.0132743363, 0.0147492625, 0.0140117994,
        0.0140117994, 0.0132743363, 0.0117994100, 0.0132743363, 0.0110619469,
        0.0103244838, 0.0117994100, 0.0117994100, 0.0147492625, 0.0110619469,
        0.0132743363, 0.0206489676, 0.0169616519, 0.0169616519, 0.0280235988,
        0.0221238938, 0.0235988201, 0.0221238938, 0.0206489676, 0.0228613569,
        0.0184365782, 0.0176991150, 0.0132743363, 0.0132743363, 0.0088495575,
        0.0095870206, 0.0073746313, 0.0110619469, 0.0036873156, 0.0051622419,
        0.0058997050, 0.0014749263, 0.0022123894, 0.0029498525, 0.0014749263,
        0.0007374631, 0.0014749263, 0.0014749263, 0.0007374631, 0.0000000000,
        0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000
    ]
    # Define the data structure to be passed to the callback functions
    data = {'d': diameter, 'y': density}

    n_coef = 6
    xexp = np.array([2.02, 1.39, 0.693, 36.672,
                     6.96, 0.3368], dtype=np.float64)

    x = np.array([1.63, 0.88, 1.0, 30, 1.52, 0.24], dtype=np.float64)
    w = np.ones(n_res, dtype=np.float64)
    w[55:63] = 5.0
    w /= w.sum()
    blx = np.zeros(n_coef, dtype=np.float64)
    ndf = nlls(n_coef, n_res, weights=w, lower_bounds=blx)
    ndf.fit(x, res, data=data, abs_gtol=1e-7, gtol=1.e-7, fd_step=5.e-7)

    print(f"Solution found in {ndf.n_iter} iterations")
    print(f"Residual norm at solution: {ndf.metrics['obj']:.4e}")
    print("Solution:")
    for i in range(n_coef):
        ok = np.abs(x[i]-xexp[i]) <= 1.e-2
        print(f"x[{i}]={x[i]:.3f} expected: ({xexp[i]:.3f}) OK? {ok}")


if __name__ == "__main__":
    try:
        nlls_fd_example()
    except RuntimeError:
        print("Something unexpected happened while running the example.")
        sys.exit(1)
