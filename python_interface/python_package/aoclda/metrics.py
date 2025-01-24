# Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
# pylint: disable = import-error, invalid-name, missing-module-docstring, unused-import
"""
aoclda.metrics module
"""

import numpy as np
from ._aoclda.metrics import (pybind_pairwise_distances)

def pairwise_distances(X, Y=None, metric="euclidean", p=2.0):
    """
    Pairwise distance metrics.

    Args:
        X (numpy.ndarray): The feature matrix for which the distance matrix needs to be computed.
            Its shape is (n_samples_X, n_features).

        Y (numpy.ndarray, optional): The optional second feature matrix for which the distance
            matrix needs to be computed. Its shape is (n_samples_Y, n_features).

        metric (str, optional): The type of metric used to compute the distance matrix. It can take
            the values 'euclidean', 'l2', 'sqeuclidean', 'manhattan', 'l1', 'cityblock', 'cosine',
            or 'minkowski'. Default: 'euclidean'.

        p (float, optional): The power parameter used for the Minkowski metric. For p = 1.0, 
            this defaults to 'manhattan' metric and for p = 2.0 this defaults to 'euclidean' metric.
            p is only used for Miknowski distance and will be ignored otherwise. Will return an 
            error when p is not positive. Default p = 2.0.

    Returns:
        numpy.ndarray with shape (n_samples_X, n_samples_Y) if Y is provided, or shape
        (n_samples_X, n_samples_X), if Y is None and the distance matrix for the rows
        of X is required.
    """
    if X.dtype == "float32":
        p = np.float32(p)
    else:
        p = np.float64(p)
    return pybind_pairwise_distances(X, Y, metric, p)
