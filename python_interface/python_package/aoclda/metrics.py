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
# pylint: disable = missing-module-docstring, unused-import
"""
aoclda.metrics module
"""

from ._aoclda.metrics import (pybind_pairwise_distances)

def pairwise_distances(X, Y=None, metric="euclidean", force_all_finite="allow_infinite"):
    """
    Pairwise distance metrics.

    Args:
        X (numpy.ndarray): The feature matrix for which the distance matrix needs to be computed.
            Its shape is (n_samples_X, n_features).

        Y (numpy.ndarray, optional): The optional second feature matrix for which the distance
            matrix needs to be computed. Its shape is (n_samples_Y, n_features).

        metric (str, optional): The type of metric used to compute the distance matrix. It can take
            the values 'euclidean' or 'sqeuclidean'. Default: 'euclidean'.

        force_all_finite (str, optional): Denotes whether infinite values are allowed in input data.
            Placeholder for adding options in the future.

    Returns:
        numpy.ndarray with shape (n_samples_X, n_samples_Y) if Y is provided, or shape
        (n_samples_X, n_samples_X), if Y is None and the distance matrix for the rows
        of X is required.
    """
    return pybind_pairwise_distances(X, Y, metric, force_all_finite)
