# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

"""
Train test split example Python script
"""

import sys
import numpy as np
from aoclda.utils import train_test_split


def train_test_split_example():
    """
    Demonstrate basic usage of train test split with stratification.
    """

    X = np.array(
        [[-2.0, 4.2, 1.5],
         [3.1, -1.2, 0.0],
         [5.5, 2.3, -3.3],
         [0.0, -4.4, 2.2],
         [1.1, 3.3, -1.1],
         [-3.3, 0.0, 4.4],
         [2.2, -2.2, 3.3],
         [4.4, 1.1, -0.5],
         [-1.1, 5.5, 0.0],
         [3.3, -3.3, 1.1]],
        dtype='float32')

    y = np.array(
        [[-2.5, 3.5],
         [3.7, -4.7],
         [-1.1, 2.1],
         [4.4, -5.4],
         [-3.3, 4.3],
         [2.2, -3.2],
         [-4.8, 5.8],
         [1.5, -2.5],
         [0.0, 1.0],
         [-6.6, 7.6]],
        dtype='float32')

    classes = [2, 2, 0, 0, 0, 0, 0, 0, 1, 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, train_size=0.6, seed=42, shuffle=True,
        stratify=classes)

    X_train_expected = np.array(
        [[-2.0, 4.2, 1.5],
         [5.5, 2.3, -3.3],
         [-1.1, 5.5, 0.0],
         [0.0, -4.4, 2.2],
         [4.4, 1.1, -0.5],
         [2.2, -2.2, 3.3]],
        dtype='float32')
    X_test_expected = np.array(
        [[3.1, -1.2, 0.0],
         [3.3, -3.3, 1.1],
         [-3.3, 0.0, 4.4],
         [1.1, 3.3, -1.1]],
        dtype='float32')
    y_train_expected = np.array(
        [[-2.5, 3.5],
         [-1.1, 2.1],
         [0.0, 1.0],
         [4.4, -5.4],
         [1.5, -2.5],
         [-4.8, 5.8]],
        dtype='float32')
    y_test_expected = np.array(
        [[3.7, -4.7], [-6.6, 7.6], [2.2, -3.2], [-3.3, 4.3]], dtype='float32')

    print("X_train matrix:")
    print(X_train)
    print("X_test matrix:")
    print(X_test)

    print("y_train matrix:")
    print(y_train)
    print("y_test matrix:")
    print(y_test)

    if not np.array_equal(
            X_train, X_train_expected) or not np.array_equal(
            X_test, X_test_expected) or not np.array_equal(
            y_train, y_train_expected) or not np.array_equal(
            y_test, y_test_expected):
        print("Train Test Split with stratify gave an invalid result.")
    else:
        print("Train Test Split with stratify successful.")


if __name__ == "__main__":
    try:
        train_test_split_example()
    except RuntimeError:
        print("Something unexpected happened while running the example.")
        sys.exit(1)
