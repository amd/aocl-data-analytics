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
#
# pylint: disable=unbalanced-tuple-unpacking,unused-variable

"""
Train Test Split Python test scrpit
"""

from aoclda.utils import train_test_split as train_test_split_da
from sklearn.model_selection import train_test_split as train_test_split_sk
import numpy as np
import pytest


def get_data2D(numpy_precision, numpy_order):
    """
    Pytest function that defines our 2D matrix input for tests
    """
    data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                     [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                     [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                     [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]],
                    dtype=numpy_precision,
                    order=numpy_order)
    return data


def get_data1D(numpy_precision, numpy_order):
    """
    Pytest function that defines our 1D matrix input for tests
    """
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    dtype=numpy_precision,
                    order=numpy_order)
    return data


@pytest.mark.parametrize("numpy_precision",
                         [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_valid_input_no_shuffle(numpy_precision, numpy_order):
    """
    Test train test split with no shuffle for valid inputs against sklearn
    """

    X2D = get_data2D(numpy_precision, numpy_order)
    X1D = get_data1D(numpy_precision, numpy_order)

    X_train_da, X_test_da = train_test_split_da(
        X2D, test_size=0.3, train_size=0.7, shuffle=False)
    X_train_sk, X_test_sk = train_test_split_sk(
        X2D, test_size=0.3, train_size=0.7, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)

    X_train_da, X_test_da = train_test_split_da(
        X2D, test_size=0.5, train_size=0.5, shuffle=False)
    X_train_sk, X_test_sk = train_test_split_sk(
        X2D, test_size=0.5, train_size=0.5, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)

    X_train_da, X_test_da = train_test_split_da(
        X2D, test_size=0.7, train_size=0.3, shuffle=False)
    X_train_sk, X_test_sk = train_test_split_sk(
        X2D, test_size=0.7, train_size=0.3, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)

    X_train_da, X_test_da = train_test_split_da(
        X2D, test_size=2, train_size=2, shuffle=False)
    X_train_sk, X_test_sk = train_test_split_sk(
        X2D, test_size=2, train_size=2, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)

    X_train_da, X_test_da = train_test_split_da(
        X2D, test_size=0.3, train_size=None, shuffle=False)
    X_train_sk, X_test_sk = train_test_split_sk(
        X2D, test_size=0.3, train_size=None, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)

    X_train_da, X_test_da = train_test_split_da(
        X2D, test_size=None, train_size=0.3, shuffle=False)
    X_train_sk, X_test_sk = train_test_split_sk(
        X2D, test_size=None, train_size=0.3, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)

    X_train_da, X_test_da = train_test_split_da(
        X2D, test_size=None, train_size=2, shuffle=False)
    X_train_sk, X_test_sk = train_test_split_sk(
        X2D, test_size=None, train_size=2, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)

    X_train_da, X_test_da = train_test_split_da(
        X2D, test_size=3, train_size=None, shuffle=False)
    X_train_sk, X_test_sk = train_test_split_sk(
        X2D, test_size=3, train_size=None, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)

    X_train_da, X_test_da = train_test_split_da(
        X2D, test_size=None, train_size=None, shuffle=False)
    X_train_sk, X_test_sk = train_test_split_sk(
        X2D, test_size=None, train_size=None, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)

    X_train_da, X_test_da = train_test_split_da(
        X1D, test_size=0.3, train_size=0.7, shuffle=False)
    X_train_sk, X_test_sk = train_test_split_sk(
        X1D, test_size=0.3, train_size=0.7, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)

    X_train_da, X_test_da = train_test_split_da(
        X1D, test_size=0.5, train_size=0.5, shuffle=False)
    X_train_sk, X_test_sk = train_test_split_sk(
        X1D, test_size=0.5, train_size=0.5, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)

    X_train_da, X_test_da = train_test_split_da(
        X1D, test_size=0.7, train_size=0.3, shuffle=False)
    X_train_sk, X_test_sk = train_test_split_sk(
        X1D, test_size=0.7, train_size=0.3, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)

    X_train_da, X_test_da = train_test_split_da(
        X1D, test_size=2, train_size=2, shuffle=False)
    X_train_sk, X_test_sk = train_test_split_sk(
        X1D, test_size=2, train_size=2, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)

    X_train_da, X_test_da = train_test_split_da(
        X1D, test_size=0.3, train_size=None, shuffle=False)
    X_train_sk, X_test_sk = train_test_split_sk(
        X1D, test_size=0.3, train_size=None, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)

    X_train_da, X_test_da = train_test_split_da(
        X1D, test_size=None, train_size=0.3, shuffle=False)
    X_train_sk, X_test_sk = train_test_split_sk(
        X1D, test_size=None, train_size=0.3, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)

    X_train_da, X_test_da = train_test_split_da(
        X1D, test_size=None, train_size=2, shuffle=False)
    X_train_sk, X_test_sk = train_test_split_sk(
        X1D, test_size=None, train_size=2, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)

    X_train_da, X_test_da = train_test_split_da(
        X1D, test_size=3, train_size=None, shuffle=False)
    X_train_sk, X_test_sk = train_test_split_sk(
        X1D, test_size=3, train_size=None, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)

    X_train_da, X_test_da = train_test_split_da(
        X1D, test_size=None, train_size=None, shuffle=False)
    X_train_sk, X_test_sk = train_test_split_sk(
        X1D, test_size=None, train_size=None, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)

    X_train_da, X_test_da, y_train_da, y_test_da = train_test_split_da(
        X2D, X1D[:5], test_size=0.3, train_size=0.7, shuffle=False)
    X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split_sk(
        X2D, X1D[:5], test_size=0.3, train_size=0.7, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)
    assert np.array_equal(y_train_da, y_train_sk)
    assert np.array_equal(y_test_da, y_test_sk)

    X_train_da, X_test_da, y_train_da, y_test_da = train_test_split_da(
        X2D, X1D[:5], test_size=0.5, train_size=0.5, shuffle=False)
    X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split_sk(
        X2D, X1D[:5], test_size=0.5, train_size=0.5, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)
    assert np.array_equal(y_train_da, y_train_sk)
    assert np.array_equal(y_test_da, y_test_sk)

    X_train_da, X_test_da, y_train_da, y_test_da = train_test_split_da(
        X2D, X1D[:5], test_size=0.7, train_size=0.3, shuffle=False)
    X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split_sk(
        X2D, X1D[:5], test_size=0.7, train_size=0.3, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)
    assert np.array_equal(y_train_da, y_train_sk)
    assert np.array_equal(y_test_da, y_test_sk)

    X_train_da, X_test_da, y_train_da, y_test_da = train_test_split_da(
        X2D, X1D[:5], test_size=2, train_size=2, shuffle=False)
    X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split_sk(
        X2D, X1D[:5], test_size=2, train_size=2, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)
    assert np.array_equal(y_train_da, y_train_sk)
    assert np.array_equal(y_test_da, y_test_sk)

    X_train_da, X_test_da, y_train_da, y_test_da = train_test_split_da(
        X2D, X1D[:5], test_size=0.3, train_size=None, shuffle=False)
    X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split_sk(
        X2D, X1D[:5], test_size=0.3, train_size=None, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)
    assert np.array_equal(y_train_da, y_train_sk)
    assert np.array_equal(y_test_da, y_test_sk)

    X_train_da, X_test_da, y_train_da, y_test_da = train_test_split_da(
        X2D, X1D[:5], test_size=None, train_size=0.3, shuffle=False)
    X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split_sk(
        X2D, X1D[:5], test_size=None, train_size=0.3, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)
    assert np.array_equal(y_train_da, y_train_sk)
    assert np.array_equal(y_test_da, y_test_sk)

    X_train_da, X_test_da, y_train_da, y_test_da = train_test_split_da(
        X2D, X1D[:5], test_size=None, train_size=2, shuffle=False)
    X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split_sk(
        X2D, X1D[:5], test_size=None, train_size=2, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)
    assert np.array_equal(y_train_da, y_train_sk)
    assert np.array_equal(y_test_da, y_test_sk)

    X_train_da, X_test_da, y_train_da, y_test_da = train_test_split_da(
        X2D, X1D[:5], test_size=3, train_size=None, shuffle=False)
    X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split_sk(
        X2D, X1D[:5], test_size=3, train_size=None, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)
    assert np.array_equal(y_train_da, y_train_sk)
    assert np.array_equal(y_test_da, y_test_sk)

    X_train_da, X_test_da, y_train_da, y_test_da = train_test_split_da(
        X2D, X1D[:5], test_size=None, train_size=None, shuffle=False)
    X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split_sk(
        X2D, X1D[:5], test_size=None, train_size=None, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk)
    assert np.array_equal(y_train_da, y_train_sk)
    assert np.array_equal(y_test_da, y_test_sk)


@pytest.mark.parametrize("numpy_precision",
                         [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_valid_input_shuffle(numpy_precision, numpy_order):
    """
    Test train test split with shuffle for valid inputs
    """

    X2D = get_data2D(numpy_precision, numpy_order)
    X1D = get_data1D(numpy_precision, numpy_order)

    exp_2d_shuffle = np.array(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
         [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
         [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
         [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
         [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]],
        dtype=numpy_precision, order=numpy_order)

    exp_1d_shuffle = np.array(
        [2, 1, 7, 3, 5, 4, 8, 9, 10, 6],
        dtype=numpy_precision, order=numpy_order)

    exp_1d_shuffle_2 = np.array(
        [1, 2, 4, 5, 3],
        dtype=numpy_precision, order=numpy_order)

    X_train_da, X_test_da = train_test_split_da(
        X2D, test_size=0.3, train_size=0.7, shuffle=True, seed=0)

    assert np.array_equal(
        X_train_da, exp_2d_shuffle[:3])
    assert np.array_equal(
        X_test_da, exp_2d_shuffle[3:])

    X_train_da, X_test_da = train_test_split_da(
        X2D, test_size=0.5, train_size=0.5, shuffle=True, seed=0)

    assert np.array_equal(
        X_train_da, exp_2d_shuffle[:2])
    assert np.array_equal(
        X_test_da, exp_2d_shuffle[2:])

    X_train_da, X_test_da = train_test_split_da(
        X2D, test_size=0.7, train_size=0.3, shuffle=True, seed=0)

    assert np.array_equal(
        X_train_da, exp_2d_shuffle[:1])
    assert np.array_equal(
        X_test_da, exp_2d_shuffle[1:])

    X_train_da, X_test_da = train_test_split_da(
        X2D, test_size=2, train_size=2, shuffle=True, seed=0)

    assert np.array_equal(
        X_train_da, exp_2d_shuffle[:2])
    assert np.array_equal(
        X_test_da, exp_2d_shuffle[2:4])

    X_train_da, X_test_da = train_test_split_da(
        X1D, test_size=0.3, train_size=0.7, shuffle=True, seed=0)

    assert np.array_equal(
        X_train_da, exp_1d_shuffle[:7])
    assert np.array_equal(
        X_test_da, exp_1d_shuffle[7:])

    X_train_da, X_test_da = train_test_split_da(
        X1D, test_size=0.5, train_size=0.5, shuffle=True, seed=0)

    assert np.array_equal(
        X_train_da, exp_1d_shuffle[:5])
    assert np.array_equal(
        X_test_da, exp_1d_shuffle[5:])

    X_train_da, X_test_da = train_test_split_da(
        X1D, test_size=0.7, train_size=0.3, shuffle=True, seed=0)

    assert np.array_equal(
        X_train_da, exp_1d_shuffle[:3])
    assert np.array_equal(
        X_test_da, exp_1d_shuffle[3:])

    X_train_da, X_test_da = train_test_split_da(
        X1D, test_size=2, train_size=2, shuffle=True, seed=0)

    assert np.array_equal(
        X_train_da, exp_1d_shuffle[:2])
    assert np.array_equal(
        X_test_da, exp_1d_shuffle[2:4])

    X_2d_train_da, X_2d_test_da, X_1d_train_da, X_1d_test_da = train_test_split_da(
        X2D, X1D[:5], test_size=0.3, train_size=0.7, shuffle=True, seed=0)

    assert np.array_equal(
        X_2d_train_da, exp_2d_shuffle[:3])
    assert np.array_equal(
        X_2d_test_da, exp_2d_shuffle[3:])
    assert np.array_equal(
        X_1d_train_da, exp_1d_shuffle_2[:3])
    assert np.array_equal(
        X_1d_test_da, exp_1d_shuffle_2[3:])

    X_2d_train_da, X_2d_test_da, X_1d_train_da, X_1d_test_da = train_test_split_da(
        X2D, X1D[:5], test_size=0.5, train_size=0.5, shuffle=True, seed=0)

    assert np.array_equal(
        X_2d_train_da, exp_2d_shuffle[:2])
    assert np.array_equal(
        X_2d_test_da, exp_2d_shuffle[2:])
    assert np.array_equal(
        X_1d_train_da, exp_1d_shuffle_2[:2])
    assert np.array_equal(
        X_1d_test_da, exp_1d_shuffle_2[2:])

    X_2d_train_da, X_2d_test_da, X_1d_train_da, X_1d_test_da = train_test_split_da(
        X2D, X1D[:5], test_size=0.7, train_size=0.3, shuffle=True, seed=0)

    assert np.array_equal(
        X_2d_train_da, exp_2d_shuffle[:1])
    assert np.array_equal(
        X_2d_test_da, exp_2d_shuffle[1:])
    assert np.array_equal(
        X_1d_train_da, exp_1d_shuffle_2[:1])
    assert np.array_equal(
        X_1d_test_da, exp_1d_shuffle_2[1:])

    X_2d_train_da, X_2d_test_da, X_1d_train_da, X_1d_test_da = train_test_split_da(
        X2D, X1D[:5], test_size=2, train_size=2, shuffle=True, seed=0)

    assert np.array_equal(
        X_2d_train_da, exp_2d_shuffle[:2])
    assert np.array_equal(
        X_2d_test_da, exp_2d_shuffle[2:4])
    assert np.array_equal(
        X_1d_train_da, exp_1d_shuffle_2[:2])
    assert np.array_equal(
        X_1d_test_da, exp_1d_shuffle_2[2:4])


@pytest.mark.parametrize("numpy_precision",
                         [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_valid_input_stratify(numpy_precision, numpy_order):
    """
    Test train test split with stratify
    """

    X2D = get_data2D(numpy_precision, numpy_order)
    X1D = get_data1D(numpy_precision, numpy_order)

    stratify_2d = np.array([-2, 5, -2, 5, 5], dtype=numpy_precision, order=numpy_order)
    stratify_1d = np.array(
        [-2, 5, -2, 5, 5, -2, 5, -2, 5, 5],
        dtype=numpy_precision, order=numpy_order)

    exp_2d_shuffle = np.array(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
         [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
         [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
         [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
         [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]],
        dtype=numpy_precision, order=numpy_order)

    exp_2d_shuffle_2 = np.array(
        [[31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
         [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
         [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
         [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]],
        dtype=numpy_precision, order=numpy_order)

    exp_1d_shuffle = np.array(
        [1, 5, 2, 10, 6, 4, 3, 9, 8, 7],
        dtype=numpy_precision, order=numpy_order)

    exp_1d_shuffle_2 = np.array(
        [4, 1, 5, 3, 2],
        dtype=numpy_precision, order=numpy_order)

    X_train_da, X_test_da = train_test_split_da(
        X2D, test_size=0.3, train_size=0.7, shuffle=True, seed=0,
        stratify=stratify_2d)

    assert np.array_equal(
        X_train_da, exp_2d_shuffle[:3])
    assert np.array_equal(
        X_test_da, exp_2d_shuffle[3:])

    X_train_da, X_test_da = train_test_split_da(
        X1D, test_size=0.7, train_size=0.3, shuffle=True, seed=0,
        stratify=stratify_1d)

    assert np.array_equal(
        X_train_da, exp_1d_shuffle[:3])
    assert np.array_equal(
        X_test_da, exp_1d_shuffle[3:])

    X_train_da, X_test_da, y_train_da, y_test_da = train_test_split_da(
        X2D, X1D[: 5],
        test_size=0.5, train_size=0.5, shuffle=True, seed=10,
        stratify=stratify_2d)

    assert np.array_equal(
        X_train_da, exp_2d_shuffle_2[:2])
    assert np.array_equal(
        X_test_da, exp_2d_shuffle_2[2:])
    assert np.array_equal(
        y_train_da, exp_1d_shuffle_2[:2])
    assert np.array_equal(
        y_test_da, exp_1d_shuffle_2[2:])


@pytest.mark.parametrize("numpy_precision",
                         [np.float64, np.float32, np.int32, np.int64])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_invalid_train_test_sizes(numpy_precision, numpy_order):
    """
    Test train test split with invalid arguments
    """

    X2D = get_data2D(numpy_precision, numpy_order)
    X1D = get_data1D(numpy_precision, numpy_order)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X2D, test_size=0, train_size=1, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X2D, test_size=1, train_size=0, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X2D, test_size=1., train_size=0, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X2D, test_size=0, train_size=1., shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X2D, test_size=1, train_size=-1, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X2D, test_size=-1, train_size=1, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X2D, test_size=0.5, train_size=4, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X2D, test_size=4, train_size=0.5, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X2D, test_size=1.1, train_size=0.2, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X2D, test_size=0.2, train_size=1.1, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X2D, test_size=0.7, train_size=0.5, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X2D, test_size=0.5, train_size=0.7, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X2D, test_size=1., train_size=None, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X2D, test_size=None, train_size=1., shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X2D, test_size=5, train_size=None, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X2D, test_size=None, train_size=5, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X1D, test_size=0, train_size=1, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X1D, test_size=1, train_size=0, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X1D, test_size=1., train_size=0, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X1D, test_size=0, train_size=1., shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X1D, test_size=1, train_size=-1, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X1D, test_size=-1, train_size=1, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X1D, test_size=0.5, train_size=6, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X1D, test_size=6, train_size=0.5, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X1D, test_size=1.1, train_size=0.2, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X1D, test_size=0.2, train_size=1.1, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X1D, test_size=0.7, train_size=0.5, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X1D, test_size=0.5, train_size=0.7, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X1D, test_size=1., train_size=None, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X1D, test_size=None, train_size=1., shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X1D, test_size=10, train_size=None, shuffle=False)

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X1D, test_size=None, train_size=10, shuffle=False)


@pytest.mark.parametrize("numpy_precision",
                         [np.float64, np.float32, np.int32, np.int64])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_invalid_dimensions_multiple_arrays(numpy_precision, numpy_order):
    """
    Test train test split with invalid dimenstions
    """

    X2D = get_data2D(numpy_precision, numpy_order)
    X1D = get_data2D(numpy_precision, numpy_order)
    x_short = np.array(
        [[1, 1, 1, 1, 1],
         [4, 4, 4, 4, 4],
         [2, 2, 2, 2, 2],
         [0, 0, 0, 0, 0]],
        dtype=numpy_precision, order=numpy_order)
    x_long = np.array(
        [[1, 1],
         [4, 4],
         [2, 2],
         [0, 0],
         [3, 3],
         [7, 7],
         [-5, -5],
         [-2, 3],
         [0, 109],
         [-12, -555],
         [22, -33]],
        dtype=numpy_precision, order=numpy_order)

    with pytest.raises(ValueError):
        X_train, X_test, y_train, y_test = train_test_split_da(
            X2D, x_short, test_size=1, train_size=2, seed=42, shuffle=True)

    with pytest.raises(ValueError):
        X_train, X_test, y_train, y_test = train_test_split_da(
            X2D, x_long,
            test_size=1, train_size=2, seed=42, shuffle=True)

    with pytest.raises(ValueError):
        X_train, X_test, y_train, y_test = train_test_split_da(
            X2D, x_short, test_size=1, train_size=2, seed=42, shuffle=False)

    with pytest.raises(ValueError):
        X_train, X_test, y_train, y_test = train_test_split_da(
            X2D, x_long,
            test_size=1, train_size=2, seed=42, shuffle=False)

    with pytest.raises(ValueError):
        X_train, X_test, y_train, y_test = train_test_split_da(
            X2D, x_short, test_size=1, train_size=2, seed=42, shuffle=True)

    with pytest.raises(ValueError):
        X_train, X_test, y_train, y_test = train_test_split_da(
            X2D, x_long,
            test_size=1, train_size=2, seed=42, shuffle=True)


def test_invalid_data():
    """
    Test train test split with when data has invalid values
    """

    X = np.array([[1, 2, 3], [1, "a", 3], [1, 2, 3]])

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X, test_size=1, train_size=1, shuffle=False)

    X = np.array([[1., 2., 3.]])

    with pytest.raises(ValueError):
        X_train_da, X_test_da = train_test_split_da(
            X, test_size=1, train_size=1, shuffle=False)


@pytest.mark.parametrize("numpy_precision",
                         [np.float64, np.float32, np.int32, np.int64])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_invalid_stratify(numpy_precision, numpy_order):
    """
    Test train test split when invalid stratify array has been provided
    """

    X2D = get_data2D(numpy_precision, numpy_order)
    X1D = get_data2D(numpy_precision, numpy_order)
    classes_2D = np.array([1, 1, 2, 2, 2], dtype='int32')
    classes_1D = np.array([-1, -1, 2, 2, 2, -1, -1, 2, 2, 2], dtype='int32')

    with pytest.raises(ValueError):
        X_train, X_test = train_test_split_da(
            X2D, test_size=1, train_size=2, seed=42, shuffle=True,
            stratify=classes_2D)

    with pytest.raises(ValueError):
        X_train, X_test = train_test_split_da(
            X2D, test_size=2, train_size=1, seed=42, shuffle=True,
            stratify=classes_2D)

    with pytest.raises(ValueError):
        X_train, X_test = train_test_split_da(
            X2D, test_size=2, train_size=2, seed=42, shuffle=False,
            stratify=classes_2D)

    with pytest.raises(ValueError):
        X_train, X_test = train_test_split_da(
            X2D, test_size=2, train_size=2, seed=42, shuffle=True,
            stratify=np.array([1, 2, 2, 2, 2],
                              dtype='int32'))

    with pytest.raises(ValueError):
        X_train, X_test = train_test_split_da(
            X2D, test_size=2, train_size=2, seed=42, shuffle=True,
            stratify=np.array([2, 2, 2, 2, 2],
                              dtype='int32'))

    with pytest.raises(ValueError):
        X_train, X_test = train_test_split_da(
            X2D, test_size=2, train_size=2, seed=42, shuffle=True,
            stratify=np.array([1, 1, 2, 2],
                              dtype='int32'))

    with pytest.raises(ValueError):
        X_train, X_test = train_test_split_da(
            X1D, test_size=1, train_size=2, seed=42, shuffle=True,
            stratify=classes_1D)

    with pytest.raises(ValueError):
        X_train, X_test = train_test_split_da(
            X1D, test_size=2, train_size=1, seed=42, shuffle=True,
            stratify=classes_2D)

    with pytest.raises(ValueError):
        X_train, X_test = train_test_split_da(
            X1D, test_size=2, train_size=2, seed=42, shuffle=False,
            stratify=classes_2D)

    with pytest.raises(ValueError):
        X_train, X_test = train_test_split_da(
            X1D, test_size=2, train_size=2, seed=42, shuffle=True,
            stratify=np.array([-1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                              dtype='int32'))

    with pytest.raises(ValueError):
        X_train, X_test = train_test_split_da(
            X1D, test_size=2, train_size=2, seed=42, shuffle=True,
            stratify=np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                              dtype='int32'))

    with pytest.raises(ValueError):
        X_train, X_test = train_test_split_da(
            X1D, test_size=2, train_size=2, seed=42, shuffle=True,
            stratify=np.array([-1, -1, 2, 2, 2, 2, 2, 2, 2],
                              dtype='int32'))


def test_nan():
    """
    Test train test split when nan values are in the data
    """

    X = np.array(
        [[1, 2, 3],
         [1, np.nan, 3],
         [1, 2, 3]],
        dtype='float32')

    X_train_da, X_test_da = train_test_split_da(
        X, test_size=1, train_size=1, shuffle=False)
    X_train_sk, X_test_sk = train_test_split_sk(
        X, test_size=1, train_size=1, shuffle=False)

    assert np.array_equal(X_train_da, X_train_sk)
    assert np.array_equal(X_test_da, X_test_sk, equal_nan=True)


@pytest.mark.parametrize("numpy_precision",
                         [np.float64, np.float32, np.int32, np.int64])
@pytest.mark.parametrize("numpy_orders",
                         [("C", "C", "F"),
                          ("C", "F", "C"),
                          ("C", "F", "F"),
                          ("F", "C", "C"),
                          ("F", "C", "F"),
                          ("F", "F", "C")])
def test_train_test_split_multiple_orders(numpy_orders, numpy_precision):
    """
    Test it runs when arrays of multiple orders are provided.
    """

    X2D = get_data2D(numpy_precision, numpy_orders[0])
    X1D = get_data1D(numpy_precision, numpy_orders[1])

    stratify_2d = np.array(
        [-2, 5, -2, 5, 5],
        dtype=numpy_precision, order=numpy_orders[2])

    x2d_train, x2d_test, x1d_train, x1d_test = train_test_split_da(
        X2D, X1D[:5], stratify=stratify_2d)


@pytest.mark.parametrize(
    "numpy_precisions", [('float32', 'float32', 'float64'),
                         ('float32', 'float64', 'float32'),
                         ('float32', 'float64', 'float64'),
                         ('float64', 'float32', 'float32'),
                         ('float64', 'float32', 'float64'),
                         ('float64', 'float64', 'float32')])
@pytest.mark.parametrize("numpy_order", ["C"])
def test_train_test_split_multiple_dtypes(numpy_order, numpy_precisions):
    """
    Test it runs when arrays of multiple dtypes are provided.
    """

    X2D = get_data2D(numpy_precisions[0], numpy_order)
    X1D = get_data1D(numpy_precisions[1], numpy_order)

    stratify_2d = np.array(
        [-2, 5, -2, 5, 5],
        dtype=numpy_precisions[2], order=numpy_order)

    x2d_train, x2d_test, x1d_train, x1d_test = train_test_split_da(
        X2D, X1D[:5], stratify=stratify_2d)
