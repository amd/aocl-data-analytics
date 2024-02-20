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
Basic statistics example python script
"""

from aoclda.basic_stats import harmonic_mean, mean, variance, quantile, covariance_matrix, standardize
import numpy as np
import sys


def basic_stats_example():
    """
    Basic statistics examples
    """
    a = np.array([[1.1, 2.213, 3.3], [4, 5.013, 6]],
                 dtype=np.float32,
                 order='F')
    print("Dataset:\n")
    print(a)

    means = mean(a, axis="row")
    print(f"\nRow means: {means}")

    harmonic_means = harmonic_mean(a, axis="col")
    print(f"\nColumn harmonic means: {harmonic_means}")

    var = variance(a, axis="all")
    print(f"\nOverall data variance: {var}")

    medians = quantile(a, 0.5, axis="row")
    print(f"\nRow medians: {medians}")

    covar = covariance_matrix(a)
    print(f"\nCovariance matrix:\n{covar}")

    standardized = standardize(a)
    print(f"\nStandardized matrix:\n{standardized}")

    print("\nBasic statistics calculations succesful")
    print("---------------------------")


if __name__ == "__main__":
    try:
        basic_stats_example()
    except RuntimeError:
        sys.exit(1)