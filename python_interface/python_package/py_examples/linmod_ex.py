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


from aoclda.linear_model import linmod, linmod_model
import aoclda as da
import numpy as np

if __name__ == "__main__":

    np.set_printoptions(precision=117)

    # Small standard linra regression
    lmod = linmod(linmod_model.mse)
    X = np.array([[1, 1], [2, 3], [3, 5], [4, 1], [5, 1]], dtype=np.float64)
    y = np.array([1, 1, 1, 1, 1], dtype=np.float64)
    lmod.fit(X, y)
    coef = lmod.get_coef()
    print("data type: {0}".format(coef.dtype))
    print("coefficients: [{:6f}, {:6f}]".format(coef[0], coef[1]))
    print('expected    : [0.199256, 0.130354]')

    # The same test in single precision
    lmod_s = linmod(linmod_model.mse, precision=da.single)
    Xs = np.array([[1, 1], [2, 3], [3, 5], [4, 1], [5, 1]], dtype=np.float32)
    ys = np.array([1, 1, 1, 1, 1], dtype=np.float32)
    lmod_s.fit(Xs, ys)
    coef_s = lmod_s.get_coef()
    print("data type: {0}".format(coef_s.dtype))
    print("coefficients: [{:6f}, {:6f}]".format(coef_s[0], coef_s[1]))
    print('expected    : [0.199256, 0.130354]')

    # another test with intercept defined
    lmod2 = linmod(linmod_model.mse, intercept=True)
    lmod2.fit(X, y)
    coef = lmod2.get_coef()
    print("coefficients: ", end='')
    for c in coef:
        print("{:6f}  ".format(c), end='')
    print()

    # test invalid input in define model
    lmod_inv = linmod(linmod_model.mse)
    try:
        coef = lmod_inv.get_coef()  # coef not available
    except:
        lmod_inv.print_error_message()
